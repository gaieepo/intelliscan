#!/usr/bin/env python3
"""
2D Object Detection Inference for 3D Semiconductor Inspection

This module performs batch 2D object detection on image slices extracted from 3D
volumes. It uses YOLO models to detect semiconductor components (copper pillars,
solder bumps, etc.) in horizontal and vertical view slices.

The detection workflow processes all images in a directory, generates bounding box
predictions, and saves both coordinate files and visualizations.

Key Features:
- YOLO-based object detection for semiconductor components
- Batch processing of image directories (horizontal/vertical views)
- Automatic bounding box coordinate extraction (class_id, x1, y1, x2, y2)
- Visualization generation for quality control
- Multi-class detection with confidence scoring

Output Format:
- Detection files: <image_name>.txt with format: class_id x1 y1 x2 y2 (one bbox per line)
- Visualizations: Annotated images saved to visualize/ subdirectory
"""

import os
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from utils import log


def run_yolo_detection_inmemory(
    model_path: str,
    input_folder: str,
    batch_size: int = 64,
) -> dict[str, np.ndarray]:
    """
    Perform batch YOLO detection and return results in memory.

    This function processes all images in the input folder and returns
    detection results as a dictionary, avoiding file I/O overhead.

    Args:
        model_path: Path to trained YOLO model weights (.pt file)
        input_folder: Path to folder containing input images (JPG/PNG)
        batch_size: Number of images to process in parallel (default: 64)

    Returns:
        Dictionary mapping image filename (without extension) to detection array.
        Each array has shape [N, 5] with columns [class_id, x1, y1, x2, y2].
        Empty detections return an empty array with shape [0, 5].

    Raises:
        FileNotFoundError: If model_path or input_folder doesn't exist
        ValueError: If input_folder contains no valid images

    Example:
        >>> detections = run_yolo_detection_inmemory(
        ...     model_path="models/detector.pt",
        ...     input_folder="data/view1/input_images"
        ... )
        >>> detections["image0"]  # array([[0, 100.5, 200.3, 150.2, 250.1], ...])
    """
    # Validate inputs
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    # Load YOLO model
    log(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)

    # Run batch prediction
    log(f"Running detection on images in: {input_folder}")
    results = model.predict(source=input_folder, batch=batch_size, verbose=True)

    if len(results) == 0:
        raise ValueError(f"No images processed from {input_folder}. Check folder contents.")

    # Build detection dictionary
    detections: dict[str, np.ndarray] = {}
    num_detections_total = 0

    for result in results:
        filename = Path(result.path).stem

        if result.boxes is None or len(result.boxes) == 0:
            # No detections - store empty array with correct shape
            detections[filename] = np.empty((0, 5), dtype=np.float32)
            continue

        # Extract detection information
        class_ids = result.boxes.cls.cpu().numpy()
        bboxes = result.boxes.xyxy.cpu().numpy()  # xyxy format: x1, y1, x2, y2
        num_detections_total += len(class_ids)

        # Store as [class_id, x1, y1, x2, y2]
        detection_data = np.column_stack((class_ids.reshape(-1, 1), bboxes)).astype(np.float32)
        detections[filename] = detection_data

    log(f"Processed {len(results)} images with {num_detections_total} total detections")
    return detections


def run_yolo_detection(
    model_path: str,
    input_folder: str,
    output_folder: str,
    batch_size: int = 64,
) -> None:
    """
    Perform batch YOLO object detection on all images in a folder.

    This function processes all images in the input folder, runs YOLO detection,
    and saves bounding box coordinates as text files.

    Args:
        model_path: Path to trained YOLO model weights (.pt file)
        input_folder: Path to folder containing input images (JPG/PNG)
        output_folder: Path to folder where detection results will be saved
        batch_size: Number of images to process in parallel (default: 16)

    Output Structure:
        output_folder/
            ├── image001.txt  (bounding boxes: class_id x1 y1 x2 y2)
            ├── image002.txt
            └── ...

    Raises:
        FileNotFoundError: If model_path or input_folder doesn't exist
        ValueError: If input_folder contains no valid images

    Example:
        >>> run_yolo_detection(
        ...     model_path="models/detector.pt",
        ...     input_folder="data/view1/input_images",
        ...     output_folder="data/view1/detections"
        ... )
    """
    # Validate inputs
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    log(f"Detection output folder: {output_folder}")

    # Load YOLO model
    log(f"Loading YOLO model from: {model_path}")
    model = YOLO(model_path)

    # Run batch prediction
    log(f"Running detection on images in: {input_folder}")
    results = model.predict(source=input_folder, batch=batch_size, verbose=True)

    if len(results) == 0:
        raise ValueError(f"No images processed from {input_folder}. Check folder contents.")

    # Process each result
    num_detections_total = 0
    for result in results:
        filename = Path(result.path).stem

        if result.boxes is None or len(result.boxes) == 0:
            # No detections for this image - create empty file
            output_path = os.path.join(output_folder, f"{filename}.txt")
            Path(output_path).touch()
            continue

        # Extract detection information
        class_ids = result.boxes.cls.cpu().numpy()
        bboxes = result.boxes.xyxy.cpu().numpy()  # xyxy format: x1, y1, x2, y2
        num_detections = len(class_ids)
        num_detections_total += num_detections

        # Save bounding box coordinates: [class_id, x1, y1, x2, y2]
        output_path = os.path.join(output_folder, f"{filename}.txt")
        detection_data = np.column_stack((class_ids.reshape(-1, 1), bboxes))
        np.savetxt(output_path, detection_data, fmt="%d %.2f %.2f %.2f %.2f")

    log(f"✓ Processed {len(results)} images with {num_detections_total} total detections")
    log(f"  Detection files saved to: {output_folder}")


# ============================================================================
# Convenience Functions
# ============================================================================


def detect_from_views(
    model_path: str,
    view1_folder: str,
    view2_folder: str,
    output_base_folder: str,
    batch_size: int = 16,
) -> tuple[str, str]:
    """
    Run detection on both horizontal and vertical view folders.

    Convenience function to process both views with a single model.

    Args:
        model_path: Path to YOLO model weights
        view1_folder: Path to horizontal view images
        view2_folder: Path to vertical view images
        output_base_folder: Base output folder (will create view1/view2 subfolders)
        batch_size: Batch size for inference

    Returns:
        Tuple of (view1_output_folder, view2_output_folder)

    Example:
        >>> detect_from_views(
        ...     model_path="models/detector.pt",
        ...     view1_folder="output/sample1/view1/input_images",
        ...     view2_folder="output/sample1/view2/input_images",
        ...     output_base_folder="output/sample1"
        ... )
    """
    view1_output = os.path.join(output_base_folder, "view1", "detections")
    view2_output = os.path.join(output_base_folder, "view2", "detections")

    log("\n" + "=" * 60)
    log("Running Object Detection - View 1 (Horizontal)")
    log("=" * 60)
    run_yolo_detection(model_path, view1_folder, view1_output, batch_size)

    log("\n" + "=" * 60)
    log("Running Object Detection - View 2 (Vertical)")
    log("=" * 60)
    run_yolo_detection(model_path, view2_folder, view2_output, batch_size)

    return view1_output, view2_output


# ============================================================================
# Test/Demo
# ============================================================================
if __name__ == "__main__":
    import sys

    log("2D Object Detection Module v2.0")
    log("=" * 60)

    if len(sys.argv) < 4:
        log("Usage: python infer_batch_new.py <model_path> <input_folder> <output_folder>")
        log("\nExample:")
        log("  python infer_batch_new.py models/detector.pt data/view1/images output/detections")
        sys.exit(1)

    model_path = sys.argv[1]
    input_folder = sys.argv[2]
    output_folder = sys.argv[3]

    try:
        run_yolo_detection(model_path, input_folder, output_folder)
        log("\n✓ Detection completed successfully!")
    except Exception as e:
        log(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
