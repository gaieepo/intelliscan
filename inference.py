"""
Standalone 3D Segmentation Inference Module

A deployment-ready module for running segmentation inference on 3D volumes.
Designed to be imported into larger workflows as a pure segmentation procedure.

Usage:
    from inference import SegmentationInferencer, InferenceConfig

    # Initialize once
    config = InferenceConfig.from_toml("configs/inference.toml")
    inferencer = SegmentationInferencer(config)

    # Run on multiple volumes
    for volume in volumes:
        prediction = inferencer(volume)
        # or with score map
        prediction, scores = inferencer(volume, return_scores=True)
"""

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class InferenceConfig:
    """Configuration for segmentation inference."""

    # Model settings
    checkpoint_path: str = ""
    num_classes: int = 5
    num_channels: int = 1

    # Inference settings
    patch_size: tuple[int, int, int] = (112, 112, 80)
    stride_xy: int = 18
    stride_z: int = 4

    # Normalization settings
    normalize: bool = True
    clip_percentile_low: float = 0.5
    clip_percentile_high: float = 99.5

    # Device settings
    device: str = "cuda"
    gpu_id: str = "0"

    @classmethod
    def from_toml(cls, config_path: str | Path) -> "InferenceConfig":
        """Load configuration from a TOML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        # Flatten nested structure if present
        flat_data = {}
        for section in ["model", "inference", "normalization", "device"]:
            if section in data:
                flat_data.update(data[section])

        # Handle patch_size tuple conversion
        if "patch_size" in flat_data and isinstance(flat_data["patch_size"], list):
            flat_data["patch_size"] = tuple(flat_data["patch_size"])

        # Filter to only valid fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in flat_data.items() if k in valid_fields}

        return cls(**filtered_data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InferenceConfig":
        """Create configuration from a dictionary."""
        if "patch_size" in data and isinstance(data["patch_size"], list):
            data["patch_size"] = tuple(data["patch_size"])

        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}

        return cls(**filtered_data)


class SegmentationInferencer:
    """
    3D Segmentation inference engine with sliding window support.

    Handles model loading, normalization, and patch-based inference
    for deployment in production workflows.

    Args:
        config: InferenceConfig with model and inference settings.
        model: Optional pre-loaded model (skips loading from checkpoint).

    Example:
        >>> config = InferenceConfig(checkpoint_path="model.pth", num_classes=5)
        >>> inferencer = SegmentationInferencer(config)
        >>>
        >>> # Single volume inference
        >>> prediction = inferencer(volume)
        >>>
        >>> # With probability scores
        >>> prediction, scores = inferencer(volume, return_scores=True)
        >>>
        >>> # Batch processing
        >>> for vol in volumes:
        ...     pred = inferencer(vol)
    """

    def __init__(self, config: InferenceConfig, model: torch.nn.Module | None = None):
        self.config = config
        self._setup_device()

        if model is not None:
            self.model = model.to(self.device)
        else:
            self.model = self._load_model()

        self.model.eval()

    def _setup_device(self) -> None:
        """Configure CUDA device."""
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.gpu_id

        if self.config.device == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def _load_model(self) -> torch.nn.Module:
        """Load model from checkpoint."""
        from networks.vnet_pyramid import VNet

        if not self.config.checkpoint_path:
            raise ValueError("No checkpoint path specified in config")

        checkpoint_path = Path(self.config.checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Create model
        model = VNet(
            n_channels=self.config.num_channels,
            n_classes=self.config.num_classes,
            normalization="batchnorm",
            has_dropout=False,
            pyramid_has_dropout=False,
        )

        # Load weights
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict, strict=True)
        model = model.to(self.device)

        return model

    def normalize(self, volume: np.ndarray) -> np.ndarray:
        """
        Apply clip-minmax normalization to volume.

        Args:
            volume: Input volume [W, H, D] or [C, W, H, D].

        Returns:
            Normalized volume with values in [0, 1].
        """
        if not self.config.normalize:
            return volume.astype(np.float32)

        volume = volume.astype(np.float32)

        # Clip to percentile range
        low = np.percentile(volume, self.config.clip_percentile_low)
        high = np.percentile(volume, self.config.clip_percentile_high)
        volume = np.clip(volume, low, high)

        # Min-max normalize to [0, 1]
        v_min, v_max = volume.min(), volume.max()
        if v_max - v_min > 1e-8:
            volume = (volume - v_min) / (v_max - v_min)
        else:
            volume = volume - v_min

        return volume

    def _pad_volume(self, volume: np.ndarray) -> tuple[np.ndarray, tuple[int, ...], tuple[int, int, int]]:
        """
        Pad volume if smaller than patch size.

        Returns:
            Tuple of (padded_volume, pad_widths, original_shape)
        """
        original_shape = volume.shape
        pw, ph, pd = self.config.patch_size
        w, h, d = volume.shape

        w_pad = max(pw - w, 0)
        h_pad = max(ph - h, 0)
        d_pad = max(pd - d, 0)

        if w_pad == 0 and h_pad == 0 and d_pad == 0:
            return volume, (0, 0, 0, 0, 0, 0), original_shape

        pad_widths = (
            w_pad // 2,
            w_pad - w_pad // 2,
            h_pad // 2,
            h_pad - h_pad // 2,
            d_pad // 2,
            d_pad - d_pad // 2,
        )

        padded = np.pad(
            volume,
            [(pad_widths[0], pad_widths[1]), (pad_widths[2], pad_widths[3]), (pad_widths[4], pad_widths[5])],
            mode="constant",
            constant_values=0,
        )

        return padded, pad_widths, original_shape

    def _unpad_volume(
        self,
        volume: np.ndarray,
        original_shape: tuple[int, int, int],
        pad_widths: tuple[int, ...],
    ) -> np.ndarray:
        """Remove padding from volume."""
        w, h, d = original_shape
        wl, _, hl, _, dl, _ = pad_widths

        if volume.ndim == 3:
            return volume[wl : wl + w, hl : hl + h, dl : dl + d]
        elif volume.ndim == 4:
            return volume[:, wl : wl + w, hl : hl + h, dl : dl + d]
        else:
            raise ValueError(f"Unexpected volume dimensions: {volume.ndim}")

    def _run_sliding_window(self, volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Run sliding window inference.

        Args:
            volume: Normalized input volume [W, H, D].

        Returns:
            Tuple of (prediction, score_map)
        """
        original_shape = volume.shape
        volume, pad_widths, _ = self._pad_volume(volume)

        ww, hh, dd = volume.shape
        pw, ph, pd = self.config.patch_size
        stride_xy = self.config.stride_xy
        stride_z = self.config.stride_z
        num_classes = self.config.num_classes

        # Compute number of patches
        sx = math.ceil((ww - pw) / stride_xy) + 1
        sy = math.ceil((hh - ph) / stride_xy) + 1
        sz = math.ceil((dd - pd) / stride_z) + 1

        # Initialize accumulators
        score_map = np.zeros((num_classes, ww, hh, dd), dtype=np.float32)
        count_map = np.zeros((ww, hh, dd), dtype=np.float32)

        # Sliding window inference
        with torch.no_grad():
            for x in range(sx):
                xs = min(stride_xy * x, ww - pw)
                for y in range(sy):
                    ys = min(stride_xy * y, hh - ph)
                    for z in range(sz):
                        zs = min(stride_z * z, dd - pd)

                        # Extract patch
                        patch = volume[xs : xs + pw, ys : ys + ph, zs : zs + pd]
                        patch_tensor = torch.from_numpy(patch[np.newaxis, np.newaxis, ...].astype(np.float32)).to(
                            self.device
                        )

                        # Forward pass
                        outputs = self.model(patch_tensor)
                        main_out = outputs[0] if isinstance(outputs, tuple) else outputs

                        # Accumulate softmax scores
                        probs = F.softmax(main_out, dim=1).cpu().numpy()[0]
                        score_map[:, xs : xs + pw, ys : ys + ph, zs : zs + pd] += probs
                        count_map[xs : xs + pw, ys : ys + ph, zs : zs + pd] += 1

        # Average overlapping regions
        score_map = score_map / np.expand_dims(count_map, axis=0)
        prediction = np.argmax(score_map, axis=0)

        # Remove padding
        needs_unpad = any(p > 0 for p in pad_widths)
        if needs_unpad:
            prediction = self._unpad_volume(prediction, original_shape, pad_widths)
            score_map = self._unpad_volume(score_map, original_shape, pad_widths)

        return prediction.astype(np.int32), score_map

    def __call__(
        self,
        volume: np.ndarray,
        normalize: bool | None = None,
        return_scores: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """
        Run segmentation inference on a 3D volume.

        Args:
            volume: Input volume [W, H, D], any numeric dtype.
            normalize: Whether to apply normalization (None = use config setting).
            return_scores: If True, also return probability score map.

        Returns:
            If return_scores=False: prediction array [W, H, D] with integer labels.
            If return_scores=True: tuple of (prediction, score_map) where
                score_map is [num_classes, W, H, D] with probabilities.
        """
        # Normalize if needed
        if normalize is None:
            normalize = self.config.normalize

        if normalize:
            volume = self.normalize(volume)
        else:
            volume = volume.astype(np.float32)

        # Run inference
        prediction, score_map = self._run_sliding_window(volume)

        if return_scores:
            return prediction, score_map
        return prediction

    def predict_batch(
        self,
        volumes: list[np.ndarray],
        normalize: bool | None = None,
        return_scores: bool = False,
    ) -> list[np.ndarray] | list[tuple[np.ndarray, np.ndarray]]:
        """
        Run inference on a batch of volumes.

        Args:
            volumes: List of input volumes [W, H, D].
            normalize: Whether to apply normalization.
            return_scores: If True, also return probability score maps.

        Returns:
            List of predictions (or tuples of (prediction, scores) if return_scores=True).
        """
        results = []
        for volume in volumes:
            result = self(volume, normalize=normalize, return_scores=return_scores)
            results.append(result)
        return results


def create_inferencer(
    checkpoint_path: str,
    num_classes: int = 5,
    patch_size: tuple[int, int, int] = (112, 112, 80),
    stride_xy: int = 18,
    stride_z: int = 4,
    device: str = "cuda",
    gpu_id: str = "0",
) -> SegmentationInferencer:
    """
    Convenience function to create an inferencer with common settings.

    Args:
        checkpoint_path: Path to model checkpoint.
        num_classes: Number of segmentation classes.
        patch_size: Patch size for sliding window.
        stride_xy: Stride in x/y directions.
        stride_z: Stride in z direction.
        device: "cuda" or "cpu".
        gpu_id: GPU device ID.

    Returns:
        Configured SegmentationInferencer instance.

    Example:
        >>> inferencer = create_inferencer("checkpoints/best_model.pth")
        >>> prediction = inferencer(volume)
    """
    config = InferenceConfig(
        checkpoint_path=checkpoint_path,
        num_classes=num_classes,
        patch_size=patch_size,
        stride_xy=stride_xy,
        stride_z=stride_z,
        device=device,
        gpu_id=gpu_id,
    )
    return SegmentationInferencer(config)


# =============================================================================
# CLI for standalone testing
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run segmentation inference")
    parser.add_argument("--config", "-c", type=str, help="Path to inference config TOML")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--input", "-i", type=str, required=True, help="Input volume (numpy .npy or NIfTI .nii.gz)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output prediction path")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device ID")
    args = parser.parse_args()

    # Load config
    if args.config:
        config = InferenceConfig.from_toml(args.config)
    else:
        config = InferenceConfig()

    if args.checkpoint:
        config.checkpoint_path = args.checkpoint
    config.gpu_id = args.gpu

    # Load input
    input_path = Path(args.input)
    if input_path.suffix == ".npy":
        volume = np.load(input_path)
    elif input_path.suffix in [".nii", ".gz"]:
        import nibabel as nib

        volume = nib.load(input_path).get_fdata()
    else:
        raise ValueError(f"Unsupported input format: {input_path.suffix}")

    print(f"Input shape: {volume.shape}")
    print(f"Checkpoint: {config.checkpoint_path}")

    # Run inference
    inferencer = SegmentationInferencer(config)
    prediction = inferencer(volume)

    print(f"Prediction shape: {prediction.shape}")
    print(f"Unique labels: {np.unique(prediction)}")

    # Save output
    output_path = Path(args.output)
    if output_path.suffix == ".npy":
        np.save(output_path, prediction)
    elif output_path.suffix in [".nii", ".gz"]:
        import nibabel as nib

        nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), output_path)
    else:
        raise ValueError(f"Unsupported output format: {output_path.suffix}")

    print(f"Saved to: {output_path}")
