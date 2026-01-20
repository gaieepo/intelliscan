"""
Pipeline utility classes and functions for 3D-IntelliScan.

Contains:
- PipelineMetrics: Timing and count metrics collection for pipeline phases
- PhaseRecord: Individual phase timing record
- create_folder_structure: Setup output folder hierarchy
- process_metrology: Compute metrology measurements from segmentation masks
"""

import json
import os
import time
from contextlib import contextmanager

import nibabel as nib
from PIL import Image


class PhaseRecord:
    """Record for a single pipeline phase execution."""

    def __init__(self, name: str, start_time: float):
        self.name = name
        self.start_time = start_time
        self.end_time = None
        self.elapsed = None
        self.count = None

    def complete(self, count: int = None):
        """Mark this phase as complete with optional item count."""
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        self.count = count

    @property
    def avg_time_per_item(self) -> float | None:
        """Return average time per item if count is set and > 0."""
        if self.count is not None and self.count > 0 and self.elapsed is not None:
            return self.elapsed / self.count
        return None

    def to_dict(self) -> dict:
        """Convert record to dictionary for serialization."""
        return {
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "elapsed_seconds": self.elapsed,
            "count": self.count,
            "avg_seconds_per_item": self.avg_time_per_item,
        }

    def __repr__(self):
        if self.count is not None and self.avg_time_per_item is not None:
            return f"[{self.name}] {self.elapsed:.2f}s (count: {self.count}, avg: {self.avg_time_per_item:.4f}s/item)"
        return f"[{self.name}] {self.elapsed:.2f}s"


class PipelineMetrics:
    """Collects timing and count metrics for pipeline execution.

    Each task can have its own PipelineMetrics instance to track
    per-phase timing and quantities for analysis and reporting.

    Example:
        >>> metrics = PipelineMetrics(task_id="sample_001")
        >>> with metrics.phase("NII to JPG") as p:
        ...     slices = convert_nii_to_jpg(...)
        ...     p.complete(count=len(slices))
        >>> metrics.save("output/timing.json")
    """

    def __init__(self, task_id: str = None):
        self.task_id = task_id
        self.created_at = time.time()
        self.phases: list[PhaseRecord] = []
        self._current_phase: PhaseRecord = None

    @contextmanager
    def phase(self, name: str):
        """Context manager for timing a pipeline phase.

        Args:
            name: Descriptive name for the phase

        Yields:
            PhaseRecord: Call record.complete(count=N) to register quantity
        """
        record = PhaseRecord(name, time.time())
        self._current_phase = record
        try:
            yield record
        finally:
            # Auto-complete if not already done (allows phases without count)
            if record.end_time is None:
                record.complete()
            self.phases.append(record)
            self._current_phase = None
            print(record)

    def summary(self) -> dict:
        """Return summary of all phases."""
        return {
            "task_id": self.task_id,
            "created_at": self.created_at,
            "total_elapsed": sum(p.elapsed for p in self.phases if p.elapsed),
            "phases": [p.to_dict() for p in self.phases],
        }

    def save(self, filepath: str):
        """Save metrics to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2)

    def write_log(self, filepath: str):
        """Write human-readable log file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"=== Pipeline Metrics for {self.task_id} ===\n")
            f.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.created_at))}\n")
            f.write("=" * 50 + "\n\n")
            for phase in self.phases:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(phase.start_time))
                f.write(f"{timestamp} - {phase}\n")
            f.write("\n" + "=" * 50 + "\n")
            total = sum(p.elapsed for p in self.phases if p.elapsed)
            f.write(f"Total elapsed: {total:.2f} seconds\n")


def create_folder_structure(base_folder, views):
    """
    Create and organize folder structure for processing pipeline.

    With single detection model, we no longer need per-class detection folders.
    Structure:
        base_folder/
            view1/
                input_images/  (2D slices for detection)
                detections/    (detection results)
                visualize/     (annotated images)
            view2/
                input_images/
                detections/
                visualize/

    Args:
        base_folder: Base output folder path
        views: List of view names (e.g., ['view1', 'view2'])

    Returns:
        Dictionary with folder paths for each view
    """
    folders = {}
    os.makedirs(base_folder, exist_ok=True)
    print(f"Created base folder: {base_folder}")

    for view in views:
        view_path = os.path.join(base_folder, view)
        folders[view] = {
            "input_images": os.path.join(view_path, "input_images"),
            "detections": os.path.join(view_path, "detections"),
            "visualize": os.path.join(view_path, "visualize"),
        }

        # Create all folders for this view
        for folder_type, folder_path in folders[view].items():
            os.makedirs(folder_path, exist_ok=True)
            print(f"Created {folder_type} folder for {view}: {folder_path}")

    return folders


def save_image(array, data_max, filename):
    """Convert to 8-bit, rotate, flip, and save as JPEG
    Normalize using the pre-computed maximum value and convert to 8-bit

    This function normalizes the input array using the pre-computed maximum value,
    converts it to an 8-bit image, rotates the image 90 degrees, flips it vertically,
    and saves it as a JPEG file.

    @param array The input array.
    @param data_max The maximum value used for normalization.
    @param filename The name of the file to save the image as, including the .jpeg extension.

    @details
    The function performs the following steps:
    - Normalizes the input array by dividing it by the pre-computed maximum value and
      scaling it to the 0-255 range.
    - Converts the normalized array to an 8-bit unsigned integer type.
    - Rotates the image 90 degrees clockwise.
    - Flips the image vertically (top to bottom).
    - Converts the image to RGB mode.
    - Saves the image as a JPEG file with the specified filename.

    @exception IOError Raised if the image cannot be saved to the specified filename.
    """
    array = (array / data_max) * 255

    img = Image.fromarray(array).rotate(90, expand=True).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    img = img.convert("RGB")
    img.save(filename)


def nii2jpg(inputfile, outputdir, view):
    """Convert a 3D or 4D image to a series of 2D JPEG images based on specified view.

    This function processes 3D or 4D medical imaging data, extracts slices based
    on the specified view (axial, coronal, or sagittal), normalizes each slice,
    and saves them as JPEG files in the given output directory.

    @param inputfile The path to the input image file.
    @param outputdir The directory where the output JPEG images will be saved.
    @param view View index specifying the slicing direction:
        - 0: Axial (XY plane, slices along Z)
        - 1: Coronal (XZ plane, slices along Y)

    @details
    The function performs the following steps:
    - Loads the input image using nibabel.
    - Extracts the image data.
    - If the image data is 4D, it processes only the fourth dimension.
    - Computes the maximum value of the data for normalization.
    - Iterates through each 2D slice of the 3D data.
    - Normalizes each slice and converts it to an 8-bit image.
    - Rotates and flips each image, then saves it as a JPEG file in the output directory.

    @exception IOError Raised if the input file cannot be loaded or if the output images cannot be saved.
    """
    # Load the input NIfTI file
    img = nib.load(inputfile)
    data = img.get_fdata()

    if data.ndim == 4:
        data = data[..., 3]  # If 4D, take the 4th dimension

    # Pre-compute max for normalization
    data_max = data.max()
    if data_max == 0:
        raise ValueError("The input data contains only zero values, normalization not possible.")

    # Determine slicing direction based on the view
    if view == 0:  # Axial view (slices along Z-axis)
        slice_dim = 2
        slice_func = lambda i: data[:, :, i]
    elif view == 1:  # Coronal view (slices along Y-axis)
        slice_dim = 1
        slice_func = lambda i: data[:, i, :]
    else:
        raise ValueError(f"Invalid view parameter: {view}. Must be 0 (axial) or 1 (coronal).")

    # Create output directory if it doesn't exist
    os.makedirs(outputdir, exist_ok=True)

    # Process and save each slice
    for i in range(data.shape[slice_dim]):
        slice_data = slice_func(i)
        filename = os.path.join(outputdir, f"image{i}.jpg")
        if not os.path.exists(filename):
            save_image(slice_data, data_max, filename)
