#!/usr/bin/env python3
"""
3D-IntelliScan Main Pipeline

This module serves as the main entry point for the 3D-IntelliScan semiconductor
metrology and defect detection pipeline. It orchestrates the complete workflow
from 3D NII to 2D conversion, object detection, 3D bounding box generation,
segmentation, and metrology analysis.

Pipeline Workflow:
1. Convert 3D NIfTI volumes to 2D slices (horizontal & vertical views)
2. Run YOLO object detection on 2D slices (single model for both views)
3. Generate 3D bounding boxes from 2D detections
4. Perform 3D semantic segmentation within bounding boxes
5. Compute metrology measurements (BLT, void ratio, defects)
6. Generate PDF reports with analysis

Key Changes (v2.0):
- Simplified to use single detection model for all views (was: multiple models)
- Streamlined folder structure (no per-class detection folders)
- Improved error handling and logging

Author: Wang Jie (original), Refactored by Claude
Date: 1st Aug 2025 (original), January 2026 (refactored)
Version: 2.0
"""

import csv
import fnmatch
import glob
import os
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from conv_nii_jpg import conv
from generate_bb3d_2 import process_images
from generate_report import generate_pdf_report
from infer_batch_new import run_yolo_detection
from metrology.post_process import MAKE_CLEAN_DEFAULT, compute_metrology_info
from mmt import test_calculate_metric


@contextmanager
def timer(section_name, log_file_path=None):
    """Context manager for timing and logging code execution blocks.

    Measures execution time of code blocks and optionally logs results
    to a specified file with timestamps.

    Args:
        section_name (str): Descriptive name for the timed section
        log_file_path (str, optional): Path to log file for timing records

    Yields:
        None: Control is passed to the with-block code

    Example:
        >>> with timer("Data Processing", "timing.log"):
        ...     # Code to be timed
        ...     process_data()
    """
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time

    # Format the timing message
    timing_message = f"[{section_name}] completed in {elapsed_time:.2f} seconds."

    # Print to console
    print(timing_message)

    # Write to log file if path is provided
    if log_file_path:
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

            # Append timing information to log file
            with open(log_file_path, "a", encoding="utf-8") as log_file:
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
                log_file.write(f"{timestamp} - {timing_message}\n")
        except Exception as e:
            print(f"Warning: Could not write to log file {log_file_path}: {e}")


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


def run_inference(detection_model, folder_structure, views):
    """
    Run YOLO object detection on both views using a single model.

    Args:
        detection_model: Path to YOLO model weights (.pt file)
        folder_structure: Dictionary containing folder paths for each view
        views: List of view names to process

    Returns:
        None (detections saved to disk)
    """
    for view in views:
        input_folder = folder_structure[view]["input_images"]
        output_folder = folder_structure[view]["detections"]

        print(f"\nRunning detection on {view}...")
        run_yolo_detection(
            model_path=detection_model,
            input_folder=input_folder,
            output_folder=output_folder,
        )


def process_metrology(input_folder, output_folder, clean_out_path, clean_mask, bb_3d_list):
    """
    Process 3D segmented files and compute metrology measurements.

    Iterates through all .nii.gz segmentation files, computes metrology
    measurements (BLT, void ratio, pad misalignment, etc.), and separates
    results into memory die and logic die categories.

    Args:
        input_folder: Path to folder containing segmented .nii.gz files
        output_folder: Path to save metrology CSV reports
        clean_out_path: Path to save cleaned/processed masks
        clean_mask: Whether to apply morphological cleaning to masks
        bb_3d_list: Array of 3D bounding boxes [num_classes, num_samples]

    Returns:
        Tuple of (memory_df, logic_df): DataFrames with metrology results

    Output Files:
        - memory.csv: Metrology data for memory dies
        - logic.csv: Metrology data for logic dies
    """
    segmented_files = sorted(glob.glob(os.path.join(input_folder, "**/*.nii.gz")))
    num_files = len(segmented_files)
    print(f"Number of segmented files to process: {num_files}")

    # Create output folders if they don't exist
    os.makedirs(output_folder, exist_ok=True)
    if clean_mask:
        os.makedirs(clean_out_path, exist_ok=True)

    # Initialize DataFrames with all expected columns
    memory_columns = [
        "filename",
        "void_ratio_defect",
        "solder_extrusion_defect",
        "pad_misalignment_defect",
        "bb",
        "BLT",
        "Pad_misalignment",
        "Void_to_solder_ratio",
        "solder_extrusion_copper_pillar",
        "pillar_width",
        "pillar_height",
    ]

    memory_df = pd.DataFrame(columns=memory_columns)
    logic_df = pd.DataFrame(columns=memory_columns)  # Using same columns for now

    for i, segmented_file in enumerate(segmented_files):
        print(f"Processing file {i + 1}/{num_files}: {segmented_file}")
        rel_path = os.path.relpath(segmented_file, input_folder)
        output_path = os.path.join(clean_out_path, rel_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            measurements = compute_metrology_info(
                nii_file=segmented_file,
                output_path=output_path if clean_mask else None,
                clean_mask=clean_mask,
            )

            # Create record with all measurements (values already in micrometers)
            record = {
                "filename": rel_path,
                "void_ratio_defect": measurements["void_ratio_defect"],
                "solder_extrusion_defect": measurements["solder_extrusion_defect"],
                "pad_misalignment_defect": measurements["pad_misalignment_defect"],
                "bb": bb_3d_list[0, i],
                "BLT": measurements["blt"],
                "Pad_misalignment": measurements["pad_misalignment"],
                "Void_to_solder_ratio": measurements["void_solder_ratio"],
                "solder_extrusion_copper_pillar": [
                    measurements["solder_extrusion_left"],
                    measurements["solder_extrusion_right"],
                    measurements["solder_extrusion_front"],
                    measurements["solder_extrusion_back"],
                ],
                "pillar_width": measurements["pillar_width"],
                "pillar_height": measurements["pillar_height"],
            }

            # Append to appropriate DataFrame
            if measurements["is_memory"]:
                memory_df = pd.concat([memory_df, pd.DataFrame([record])], ignore_index=True)
            else:
                logic_df = pd.concat([logic_df, pd.DataFrame([record])], ignore_index=True)

        except Exception as e:
            print(f"Error processing file {segmented_file}")
            print(f"Error details: {str(e)}")
            print("Stack trace:", traceback.format_exc())
            continue

    # Save DataFrames to CSV
    memory_df.to_csv(os.path.join(output_folder, "memory.csv"), index=False)
    logic_df.to_csv(os.path.join(output_folder, "logic.csv"), index=False)

    return memory_df, logic_df


def main():
    # Read input files from files.txt
    with open("files.txt") as f:
        input_files = [line.strip() for line in f.readlines()]

    print(f"Found {len(input_files)} files to process")

    # Each line in files.txt is a task
    for inputfile in input_files:
        print(f"\nProcessing file: {inputfile}")

        # Original path
        p = Path(inputfile)

        # Remove the extension
        p_no_ext = p.with_suffix("")  # yields folder1/folder2/filename

        # Remove the first folder by taking all parts except the first one
        input_base_name = Path(*p_no_ext.parts[-2:])  # yields folder2/filename
        outfolder = os.path.join("output", input_base_name)

        # Single detection model for both views
        detection_model = "models/detection_model.pt"
        segmentation_model = "models/segmentation_model.pth"

        views = ["view1", "view2"]

        # Initialize timing log file with header
        timing_log_path = os.path.join(outfolder, "timing.log")
        try:
            os.makedirs(outfolder, exist_ok=True)
            with open(timing_log_path, "w", encoding="utf-8") as log_file:
                log_file.write(f"=== Timing Log for {inputfile} ===\n")
                log_file.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write("=" * 50 + "\n\n")
        except Exception as e:
            print(f"Warning: Could not initialize log file {timing_log_path}: {e}")

        try:
            with timer("Folder Creation", timing_log_path):
                folder_structure = create_folder_structure(outfolder, views)

            with timer("Converting NII to JPG", timing_log_path):
                print("Converting NII to JPG...")
                conv(inputfile, folder_structure["view1"]["input_images"], 0)
                conv(inputfile, folder_structure["view2"]["input_images"], 1)

            with timer("Inference", timing_log_path):
                print("Running object detection inference...")
                run_inference(detection_model, folder_structure, views)

            with timer("Generating 3D bounding boxes", timing_log_path):
                print("Generating 3D bounding boxes...")
                view1_params = [
                    folder_structure["view1"]["detections"],
                    folder_structure["view1"]["input_images"],
                    0,
                    len(fnmatch.filter(os.listdir(folder_structure["view1"]["input_images"]), "*.jpg")),
                ]
                view2_params = [
                    folder_structure["view2"]["detections"],
                    folder_structure["view2"]["input_images"],
                    0,
                    len(fnmatch.filter(os.listdir(folder_structure["view2"]["input_images"]), "*.jpg")),
                ]
                bb_3d = process_images(
                    view1_params, view2_params, os.path.join(outfolder, "3d_bounding_boxes.npy"), is_normalized=False
                )
                # Save as single array (no longer list of arrays per class)
                np.save(os.path.join(outfolder, "bb3d.npy"), bb_3d)

            with timer("3D Segmentation", timing_log_path):
                bb_3d = np.load(os.path.join(outfolder, "bb3d.npy"))
                print(f"3D bounding boxes shape: {bb_3d.shape}")

                print("Running 3D segmentation...")
                img = nib.load(inputfile)
                data3d = img.get_fdata()

                # Run segmentation with single model
                avg_metric, new_data_3d = test_calculate_metric(
                    segmentation_model,
                    bb_3d,
                    outfolder,
                    "segmentation.nii.gz",
                    0,  # class index (only one class now)
                    num_classes=5,
                    data_3d=data3d,
                    new_data_3d=None,
                    input_im=folder_structure["view1"]["input_images"],
                )
                print(f"Segmentation completed, saved to {outfolder}")

            with timer("Metrology", timing_log_path):
                # Add diagnostic prints
                metrology_input_path = os.path.join(outfolder, "mmt", "pred")
                print(f"Checking metrology input folder: {metrology_input_path}")
                if os.path.exists(metrology_input_path):
                    files = glob.glob(os.path.join(metrology_input_path, "**/*.nii.gz"))
                    print(f"Found {len(files)} .nii.gz files to process")
                    if len(files) == 0:
                        print("No files found! Check if previous steps generated the required files.")
                else:
                    print(f"Error: Metrology input folder does not exist: {metrology_input_path}")

                bb_3d = np.load(os.path.join(outfolder, "bb3d.npy"))
                # Reshape to match expected format [1, N] for single model
                bb_3d_list = bb_3d.reshape(1, -1) if bb_3d.ndim == 1 else np.expand_dims(bb_3d, axis=0)

                memory_df, logic_df = process_metrology(
                    input_folder=metrology_input_path,
                    output_folder=os.path.join(outfolder, "metrology"),
                    clean_out_path=os.path.join(outfolder, "cleaned_metrology"),
                    clean_mask=MAKE_CLEAN_DEFAULT,
                    bb_3d_list=bb_3d_list,
                )

            outf = os.path.join(outfolder, "metrology", "memory_report.pdf")
            with timer("Analysis", timing_log_path):
                generate_pdf_report(
                    os.path.join(outfolder, "metrology", "memory.csv"),
                    outf,
                    input_filename=os.path.basename(inputfile),
                )

            print(f"Successfully processed {inputfile}")
            outf2 = os.path.dirname(outf)
            print(f"Report generated at {outf2}")

        except Exception as e:
            print(f"Error processing {inputfile}: {str(e)}")
            continue


if __name__ == "__main__":
    main()
