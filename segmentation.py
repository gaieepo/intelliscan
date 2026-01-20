#!/usr/bin/env python3
"""
3D Segmentation Inference Module

Provides inference capabilities using MONAI BasicUNet for 3D semantic segmentation.
Designed for integration with detection + metrology pipelines.

Workflow:
1. Load trained BasicUNet checkpoint
2. For each detected 3D bounding box:
   - Expand bbox with margin (handles edge cases)
   - Extract crop from full volume
   - Apply ClipZScoreNormalize preprocessing
   - Run sliding window inference
   - Apply post-processing (void filling)
3. Assemble all predictions into full volume
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai.networks.nets import BasicUNet

# =============================================================================
# Post-processing utilities
# =============================================================================


def fill_internal_voids(
    prediction: np.ndarray,
    background_class: int = 0,
    void_class: int = 3,
) -> np.ndarray:
    """Fill internal voids (holes) in a 3D segmentation volume.

    Finds all connected regions of background_class that do NOT touch
    the volume boundary and relabels them as void_class.

    Args:
        prediction: 3D integer segmentation array (D, H, W)
        background_class: Label considered as true background
        void_class: Label to assign to internal holes

    Returns:
        Modified prediction array with internal holes relabeled
    """
    # Lazy import - only needed when apply_void_fill is True
    from scipy.ndimage import label as scipy_label

    bg_mask = prediction == background_class
    bg_labels, n_labels = scipy_label(bg_mask)

    # Identify which labels touch the volume boundary (6 faces of cuboid)
    boundary = np.zeros_like(bg_mask, dtype=bool)
    boundary[0, :, :] = True
    boundary[-1, :, :] = True
    boundary[:, 0, :] = True
    boundary[:, -1, :] = True
    boundary[:, :, 0] = True
    boundary[:, :, -1] = True

    external_labels = set(np.unique(bg_labels[boundary & bg_mask]).tolist())
    external_labels.discard(0)

    # Relabel internal (non-boundary-touching) background as void
    for lab in range(1, n_labels + 1):
        if lab not in external_labels:
            prediction[bg_labels == lab] = void_class

    return prediction


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class SegmentationConfig:
    """Configuration for segmentation inference pipeline."""

    roi_size: tuple[int, int, int] = (112, 112, 80)
    overlap: float = 0.5
    sw_batch_size: int = 1
    margin: int = 15
    num_classes: int = 5
    apply_normalization: bool = True
    apply_void_fill: bool = False  # Disabled by default for performance
    device: str = "cuda"

    # Normalization parameters
    clip_percentile_low: float = 1.0
    clip_percentile_high: float = 99.0

    # Post-processing parameters
    void_class: int = 3
    background_class: int = 0


# =============================================================================
# Core inference class
# =============================================================================


class SegmentationInference:
    """Segmentation inference engine using MONAI BasicUNet.

    Provides methods for:
    - Single crop inference
    - Multi-bbox batch inference
    - Full volume assembly
    """

    def __init__(
        self,
        model_path: str | Path,
        config: SegmentationConfig | None = None,
    ):
        """Initialize inference engine.

        Args:
            model_path: Path to trained model checkpoint (.ckpt)
            config: Configuration object (uses defaults if None)
        """
        self.model_path = Path(model_path)
        self.config = config or SegmentationConfig()
        self.device = torch.device(self.config.device)
        self.model: BasicUNet | None = None

    def load_model(self) -> None:
        """Load BasicUNet model and weights from checkpoint."""
        if self.model is not None:
            return  # Already loaded

        self.model = BasicUNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=self.config.num_classes,
        ).to(self.device)

        state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)

        # Handle different checkpoint formats
        if isinstance(state_dict, dict) and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        print(f"Loaded segmentation model from: {self.model_path}")

    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Apply ClipZScoreNormalize preprocessing.

        Clips to [p_low, p_high] percentile, then z-score normalizes.
        """
        cfg = self.config
        flat = data.ravel()
        p_low = np.percentile(flat, cfg.clip_percentile_low)
        p_high = np.percentile(flat, cfg.clip_percentile_high)
        clipped = np.clip(data, p_low, p_high)
        mean, std = clipped.mean(), clipped.std()
        return ((clipped - mean) / (std + 1e-8)).astype(np.float32)

    def infer_crop(self, crop: np.ndarray) -> np.ndarray:
        """Run inference on a single 3D crop.

        Args:
            crop: 3D numpy array (X, Y, Z) - raw intensity values

        Returns:
            Prediction array (X, Y, Z) with class labels
        """
        if self.model is None:
            self.load_model()

        # Preprocess
        if self.config.apply_normalization:
            crop = self.normalize(crop)

        # To tensor [1, 1, X, Y, Z]
        tensor = torch.from_numpy(crop).unsqueeze(0).unsqueeze(0).to(self.device)

        # Sliding window inference
        with torch.no_grad():
            logits = sliding_window_inference(
                inputs=tensor,
                roi_size=self.config.roi_size,
                sw_batch_size=self.config.sw_batch_size,
                predictor=self.model,
                overlap=self.config.overlap,
                mode="gaussian",
            )

        prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        # Post-process
        if self.config.apply_void_fill:
            prediction = fill_internal_voids(
                prediction,
                background_class=self.config.background_class,
                void_class=self.config.void_class,
            )

        return prediction


# =============================================================================
# Result dataclass and utility functions
# =============================================================================


@dataclass
class BBoxSegmentationResult:
    """Result from segmenting a single bounding box region."""

    bbox_index: int
    prediction: np.ndarray
    original_bbox: np.ndarray
    expanded_bbox: list[int]
    crop_shape: tuple[int, ...]


def expand_bbox(
    bbox: np.ndarray,
    volume_shape: tuple[int, ...],
    margin: int,
) -> list[int]:
    """Expand bounding box with margin, clamped to volume boundaries.

    Args:
        bbox: [x_min, x_max, y_min, y_max, z_min, z_max]
        volume_shape: Shape of the full volume (X, Y, Z)
        margin: Expansion margin in voxels

    Returns:
        Expanded bbox as [x_min, x_max, y_min, y_max, z_min, z_max]
    """
    return [
        max(int(bbox[0]) - margin, 0),
        min(int(bbox[1]) + margin, volume_shape[0]),
        max(int(bbox[2]) - margin, 0),
        min(int(bbox[3]) + margin, volume_shape[1]),
        max(int(bbox[4]) - margin, 0),
        min(int(bbox[5]) + margin, volume_shape[2]),
    ]


# =============================================================================
# High-level inference functions
# =============================================================================


def segment_bboxes(
    engine: SegmentationInference,
    volume: np.ndarray,
    bboxes: np.ndarray,
    save_dir: Path | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[BBoxSegmentationResult]:
    """Segment multiple bounding box regions from a volume.

    Args:
        engine: Initialized SegmentationInference instance
        volume: Full 3D volume array
        bboxes: Array of bboxes [N, 6] format [x_min, x_max, y_min, y_max, z_min, z_max]
        save_dir: Optional directory to save per-bbox crops and predictions.
                  If provided, saves to save_dir/img/ and save_dir/pred/
        progress_callback: Optional callback(current_idx, total) for progress updates

    Returns:
        List of BBoxSegmentationResult for each bbox
    """
    results = []
    cfg = engine.config

    # Ensure model is loaded
    engine.load_model()

    for idx, bbox in enumerate(bboxes):
        if progress_callback:
            progress_callback(idx, len(bboxes))

        # Expand bbox with margin
        expanded = expand_bbox(bbox, volume.shape, cfg.margin)

        # Check for valid region size
        region_size = (
            expanded[1] - expanded[0],
            expanded[3] - expanded[2],
            expanded[5] - expanded[4],
        )
        if any(s < 5 for s in region_size):
            print(f"Skipping bbox {idx}: region too small {region_size}")
            continue

        print(f"Processing bbox {idx}: shape x={region_size[0]}, y={region_size[1]}, z={region_size[2]}")

        # Extract crop
        crop = volume[
            expanded[0] : expanded[1],
            expanded[2] : expanded[3],
            expanded[4] : expanded[5],
        ]

        # Run inference
        prediction = engine.infer_crop(crop)

        # Save per-bbox outputs if directory provided
        if save_dir is not None:
            img_path = save_dir / "img" / f"img_{idx}.nii.gz"
            pred_path = save_dir / "pred" / f"pred_{idx}.nii.gz"
            nib.save(nib.Nifti1Image(crop.astype(np.float32), np.eye(4)), str(img_path))
            nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), str(pred_path))

        results.append(
            BBoxSegmentationResult(
                bbox_index=idx,
                prediction=prediction,
                original_bbox=bbox,
                expanded_bbox=expanded,
                crop_shape=crop.shape,
            )
        )

    return results


def assemble_full_volume(
    results: list[BBoxSegmentationResult],
    volume_shape: tuple[int, ...],
) -> np.ndarray:
    """Assemble per-bbox predictions into a full volume.

    Args:
        results: List of BBoxSegmentationResult from segment_bboxes
        volume_shape: Shape of the output volume

    Returns:
        Full volume with predictions placed at their bbox locations
    """
    output = np.zeros(volume_shape, dtype=np.float32)

    for result in results:
        bbox = result.expanded_bbox
        pred = result.prediction

        # Direct placement - prediction shape matches expanded bbox
        output[
            bbox[0] : bbox[0] + pred.shape[0],
            bbox[2] : bbox[2] + pred.shape[1],
            bbox[4] : bbox[4] + pred.shape[2],
        ] = pred

    return output


if __name__ == "__main__":
    print("3D Segmentation Inference Module")
    print("\nUsage:")
    print("  engine = SegmentationInference(model_path, config)")
    print("  results = segment_bboxes(engine, volume, bboxes, save_dir)")
    print("  full_vol = assemble_full_volume(results, volume.shape)")
