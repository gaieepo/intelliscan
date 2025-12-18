#!/usr/bin/env python3
"""
3D-IntelliScan Main Pipeline

This module serves as the main entry point for the 3D-IntelliScan semiconductor
metrology and defect detection pipeline. It orchestrates the complete workflow
from 3D NII to 2D conversion, object detection, 3D bounding box generation,
segmentation, and metrology analysis.

Author: Wang Jie
Date: 1st Aug 2025
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

from _version import get_version, print_version
from conv_nii_jpg import conv
from generate_bb3d_2 import process_images
from generate_report import generate_pdf_report
from infer_batch_new import infer_batch_l
from metrology.config import cfg
from metrology.post_process import compute_metrology_info
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


def try_cast(value):
    """Attempts to cast a string value to appropriate Python type.

    Converts string representations to boolean, float, or integer types
    when possible, otherwise returns the trimmed string.

    Args:
        value (str): String value to be cast

    Returns:
        bool|float|int|str: Appropriately typed value
    """
    if value.lower() in ["true", "false"]:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except (ValueError, TypeError):
        return value.strip()


def add_csv_file(filename, collection):
    """Loads CSV data into a ChromaDB collection for vector search.

    Processes CSV files and adds records to ChromaDB collection for
    similarity-based retrieval and analysis.

    Args:
        filename (str): Path to the CSV file
        collection: ChromaDB collection object

    Note:
        This function appears to be for ChromaDB integration but may not
        be actively used in the current pipeline.
    """
    with open(filename, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        ids = []
        documents = []
        metadatas = []

        for row in reader:
            if not row or all(v is None or str(v).strip() == "" for v in row.values()):
                continue

            row_id = str(row["id"]) if "id" in row else str(len(ids))
            document = row.get("summary", "")  # fallback if summary missing

            # Exclude id and document fields from metadata
            metadata = {k: try_cast(v) for k, v in row.items() if k not in ["id", "summary"]}

            ids.append(row_id)
            documents.append(document)
            metadatas.append(metadata)

        # Add to Chroma
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        print(f"Inserted {len(ids)} records.")


def create_folder_structure(base_folder, views, detection_classes):
    """Creates and organizes folders for output, views, and classes."""
    folders = {}
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
        print(f"Created base folder: {base_folder}")

    for view in views:
        view_path = os.path.join(base_folder, view)
        folders[view] = {
            "input_images": os.path.join(view_path, "input_images"),
            "detections": {},
            "visualize": os.path.join(view_path, "visualize"),  # Add visualize folder
        }
        if not os.path.exists(view_path):
            os.makedirs(view_path)
            print(f"Created view folder: {view_path}")

        # Create input images folder
        input_images_path = folders[view]["input_images"]
        if not os.path.exists(input_images_path):
            os.makedirs(input_images_path)
            print(f"Created input images folder for {view}: {input_images_path}")

        # Create visualize folder
        vis_path = folders[view]["visualize"]
        if not os.path.exists(vis_path):
            os.makedirs(vis_path)
            print(f"Created visualization folder for {view}: {vis_path}")

        # Create folders for detection classes
        for cls in range(detection_classes):
            class_path = os.path.join(view_path, f"detections/class_{cls}")
            folders[view]["detections"][cls] = class_path
            if not os.path.exists(class_path):
                os.makedirs(class_path)
                print(f"Created detection folder for {view}, class {cls}: {class_path}")

    return folders


def run_inference(det_models, folder_structure, views):
    """Run inference in parallel for each model-view combination."""
    for j, models in enumerate(det_models):
        for i, _ in enumerate(views):
            input_folder = folder_structure[views[i]]["input_images"]
            output_folder = folder_structure[views[i]]["detections"][j]
            infer_batch_l([models, input_folder, output_folder])


def process_metrology(input_folder, output_folder, clean_out_path, clean_mask, bb_3d_list):
    """
    Processes segmented files and computes metrology information.
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
                nii_file=segmented_file, clean_mask=clean_mask, output_path=output_path if clean_mask else None
            )

            # Create record with all measurements
            record = {
                "filename": rel_path,
                "void_ratio_defect": measurements["void_ratio_defect"],
                "solder_extrusion_defect": measurements["solder_extrusion_defect"],
                "pad_misalignment_defect": measurements["pad_misalignment_defect"],
                "bb": bb_3d_list[0, i],
                "BLT": round(measurements["blt"] * cfg.METROLOGY.PIXEL_SIZE_UM, cfg.METROLOGY.NUM_DECIMALS),
                "Pad_misalignment": round(
                    measurements["pad_misalignment"] * cfg.METROLOGY.PIXEL_SIZE_UM, cfg.METROLOGY.NUM_DECIMALS
                ),
                "Void_to_solder_ratio": round(measurements["void_solder_ratio"], cfg.METROLOGY.NUM_DECIMALS),
                "solder_extrusion_copper_pillar": [
                    round(
                        measurements["solder_extrusion_left"] * cfg.METROLOGY.PIXEL_SIZE_UM, cfg.METROLOGY.NUM_DECIMALS
                    ),
                    round(
                        measurements["solder_extrusion_right"] * cfg.METROLOGY.PIXEL_SIZE_UM, cfg.METROLOGY.NUM_DECIMALS
                    ),
                    round(
                        measurements["solder_extrusion_front"] * cfg.METROLOGY.PIXEL_SIZE_UM, cfg.METROLOGY.NUM_DECIMALS
                    ),
                    round(
                        measurements["solder_extrusion_back"] * cfg.METROLOGY.PIXEL_SIZE_UM, cfg.METROLOGY.NUM_DECIMALS
                    ),
                ],
                "pillar_width": round(
                    measurements["pillar_width"] * cfg.METROLOGY.PIXEL_SIZE_UM, cfg.METROLOGY.NUM_DECIMALS
                ),
                "pillar_height": round(
                    measurements["pillar_height"] * cfg.METROLOGY.PIXEL_SIZE_UM, cfg.METROLOGY.NUM_DECIMALS
                ),
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
    # Display version information
    print_version()
    print(f"3D-IntelliScan Pipeline v{get_version()}")
    print("=" * 60)

    # Read input files from files.txt
    with open("files.txt") as f:
        input_files = [line.strip() for line in f.readlines()]

    print(f"Found {len(input_files)} files to process")

    for inputfile in input_files:
        print(f"\nProcessing file: {inputfile}")

        # Original path
        p = Path(inputfile)

        # Remove the extension
        p_no_ext = p.with_suffix("")  # yields folder1/folder2/filename

        # Remove the first folder by taking all parts except the first one
        input_base_name = Path(*p_no_ext.parts[-2:])  # yields folder2/filename
        outfolder = os.path.join("output", input_base_name)

        detection_model_mem = ["models/detection_model.pt"]
        segmentation_model_mem = "models/segmentation_model.pth"

        det_models = [detection_model_mem]
        seg_models = [segmentation_model_mem]

        views = ["view1", "view2"]
        detection_classes = len(det_models)

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
                folder_structure = create_folder_structure(outfolder, views, detection_classes)

            with timer("Converting NII to JPG", timing_log_path):
                print("Converting NII to JPG...")
                conv(inputfile, folder_structure["view1"]["input_images"], 0)
                conv(inputfile, folder_structure["view2"]["input_images"], 1)

            with timer("Inference", timing_log_path):
                print("Running object detection inference...")
                run_inference(det_models, folder_structure, views)

            with timer("Generating 3D bounding boxes", timing_log_path):
                print("Generating 3D bounding boxes...")
                bb_3d_list = []
                for j, _ in enumerate(det_models):
                    view1_params = [
                        folder_structure["view1"]["detections"][j],
                        folder_structure["view1"]["input_images"],
                        0,
                        len(fnmatch.filter(os.listdir(folder_structure["view1"]["input_images"]), "*.jpg")),
                    ]
                    view2_params = [
                        folder_structure["view2"]["detections"][j],
                        folder_structure["view2"]["input_images"],
                        0,
                        len(fnmatch.filter(os.listdir(folder_structure["view2"]["input_images"]), "*.jpg")),
                    ]
                    bb = process_images(
                        view1_params, view2_params, os.path.join(outfolder, f"class_{j}_3d_bb.npy"), is_normalized=False
                    )
                    bb_3d_list.append(bb)
                np.save(os.path.join(outfolder, "bb3d.npy"), np.array(bb_3d_list))

            with timer("Testing MMT", timing_log_path):
                bb_3d_list = np.load(os.path.join(outfolder, "bb3d.npy"))
                print(f"3D bboxes has shape: {bb_3d_list.shape}")

                print("Testing MMT...")
                img = nib.load(inputfile)
                data3d = img.get_fdata()
                new_data_3d = None  # Initialize the combined 3D data

                for j, model in enumerate(seg_models):
                    avg_metric, new_data_3d_result = test_calculate_metric(
                        model,
                        bb_3d_list[j],
                        outfolder,
                        f"class_{j}_segmentation.nii.gz",
                        j,
                        num_classes=5,
                        data_3d=data3d,
                        new_data_3d=new_data_3d,
                        input_im=folder_structure[views[j]]["input_images"],
                    )
                print(f"Segmentation for class {j} completed, saved to {outfolder}")
                new_data_3d = new_data_3d_result

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

                bb_3d_list = np.load(os.path.join(outfolder, "bb3d.npy"))
                memory_df, logic_df = process_metrology(
                    input_folder=metrology_input_path,
                    output_folder=os.path.join(outfolder, "metrology"),
                    clean_out_path=os.path.join(outfolder, "cleaned_metrology"),
                    clean_mask=cfg.METROLOGY.MAKE_CLEAN,
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
