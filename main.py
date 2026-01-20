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
"""

import fnmatch
import glob
import os
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from detection import run_yolo_detection
from merge import generate_bb3d
from metrology import MAKE_CLEAN_DEFAULT, compute_metrology_info
from report import generate_pdf_report
from segmentation import (
    SegmentationConfig,
    SegmentationInference,
    assemble_full_volume,
    segment_bboxes,
)
from utils import PipelineMetrics, create_folder_structure, nii2jpg


def process_metrology(input_folder, output_folder, clean_out_path, clean_mask, bb_3d_list):
    """
    Process 3D segmented files and compute metrology measurements.

    Iterates through all .nii.gz segmentation files, computes metrology
    measurements (BLT, void ratio, pad misalignment, etc.).

    Args:
        input_folder: Path to folder containing segmented .nii.gz files
        output_folder: Path to save metrology CSV reports
        clean_out_path: Path to save cleaned/processed masks
        clean_mask: Whether to apply morphological cleaning to masks
        bb_3d_list: Array of 3D bounding boxes [num_classes, num_samples]

    Returns:
        DataFrame with metrology results for all samples

    Output Files:
        - metrology.csv: Metrology data for all samples (memory and logic dies)
    """
    segmented_files = sorted(glob.glob(os.path.join(input_folder, "**/*.nii.gz"), recursive=True))
    num_files = len(segmented_files)
    print(f"Number of segmented files to process: {num_files}")

    # Create output folders if they don't exist
    os.makedirs(output_folder, exist_ok=True)
    if clean_mask:
        os.makedirs(clean_out_path, exist_ok=True)

    # Initialize DataFrame with all expected columns
    columns = [
        "filename",
        "is_memory",
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

    metrology_df = pd.DataFrame(columns=columns)

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
                "is_memory": measurements["is_memory"],
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

            # Append to DataFrame
            metrology_df = pd.concat([metrology_df, pd.DataFrame([record])], ignore_index=True)

        except Exception as e:
            print(f"Error processing file {segmented_file}")
            print(f"Error details: {str(e)}")
            print("Stack trace:", traceback.format_exc())
            continue

    # Save DataFrame to CSV
    metrology_df.to_csv(os.path.join(output_folder, "metrology.csv"), index=False)

    return metrology_df


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
        os.makedirs(outfolder, exist_ok=True)

        # Single detection model for both views
        detection_model = "models/detection_model.pt"
        segmentation_model = "models/segmentation_model.ckpt"

        views = ["view1", "view2"]

        # Initialize pipeline metrics for this task
        metrics = PipelineMetrics(task_id=inputfile)

        try:
            with metrics.phase("Folder Creation"):
                folder_structure = create_folder_structure(outfolder, views)

            with metrics.phase("NII to JPG Conversion") as p:
                print("Converting NII to JPG...")
                nii2jpg(inputfile, folder_structure["view1"]["input_images"], 0)
                nii2jpg(inputfile, folder_structure["view2"]["input_images"], 1)
                # Count total slices generated across both views
                num_slices = len(fnmatch.filter(os.listdir(folder_structure["view1"]["input_images"]), "*.jpg")) + len(
                    fnmatch.filter(os.listdir(folder_structure["view2"]["input_images"]), "*.jpg")
                )
                p.complete(count=num_slices)

            with metrics.phase("2D Detection Inference") as p:
                print("Running object detection inference...")

                for view in views:
                    input_folder = folder_structure[view]["input_images"]
                    output_folder = folder_structure[view]["detections"]

                    print(f"\nRunning detection on {view}...")
                    run_yolo_detection(
                        model_path=detection_model,
                        input_folder=input_folder,
                        output_folder=output_folder,
                    )

                # Count slices processed (same as JPGs generated)
                num_slices = len(fnmatch.filter(os.listdir(folder_structure["view1"]["input_images"]), "*.jpg")) + len(
                    fnmatch.filter(os.listdir(folder_structure["view2"]["input_images"]), "*.jpg")
                )
                p.complete(count=num_slices)

            with metrics.phase("3D Bounding Box Generation") as p:
                print("Generating 3D bounding boxes...")
                view1_num_slices = len(fnmatch.filter(os.listdir(folder_structure["view1"]["input_images"]), "*.jpg"))
                view2_num_slices = len(fnmatch.filter(os.listdir(folder_structure["view2"]["input_images"]), "*.jpg"))
                view1_params = [
                    folder_structure["view1"]["detections"],
                    folder_structure["view1"]["input_images"],
                    0,
                    view1_num_slices,
                ]
                view2_params = [
                    folder_structure["view2"]["detections"],
                    folder_structure["view2"]["input_images"],
                    0,
                    view2_num_slices,
                ]
                bb_3d = generate_bb3d(
                    view1_params, view2_params, os.path.join(outfolder, "3d_bounding_boxes.npy"), is_normalized=False
                )
                # Save as single array (no longer list of arrays per class)
                np.save(os.path.join(outfolder, "bb3d.npy"), bb_3d)
                # Count effective slices used (slices that contributed to 3D bbox)
                effective_slices = view1_num_slices + view2_num_slices
                p.complete(count=effective_slices)

            with metrics.phase("3D Segmentation") as p:
                bb_3d = np.load(os.path.join(outfolder, "bb3d.npy"))
                num_bboxes = len(bb_3d) if bb_3d.ndim == 1 else bb_3d.shape[0]
                print(f"3D bounding boxes shape: {bb_3d.shape}")

                print("Running 3D segmentation...")
                img = nib.load(inputfile)
                data3d = img.get_fdata()

                # Create output directories for per-bbox results (metrology compatibility)
                seg_output_dir = Path(outfolder) / "mmt"
                (seg_output_dir / "img").mkdir(parents=True, exist_ok=True)
                (seg_output_dir / "pred").mkdir(parents=True, exist_ok=True)

                # Initialize segmentation engine
                engine = SegmentationInference(segmentation_model, SegmentationConfig())

                # Segment all bounding boxes
                results = segment_bboxes(engine, data3d, bb_3d, save_dir=seg_output_dir)

                # Assemble into full volume and save
                full_segmentation = assemble_full_volume(results, data3d.shape)
                nib.save(
                    nib.Nifti1Image(full_segmentation, np.eye(4)),
                    os.path.join(outfolder, "segmentation.nii.gz"),
                )
                print(f"Segmentation completed, saved to {outfolder}")
                # Count 3D bboxes provided to segmentation
                p.complete(count=num_bboxes)

            with metrics.phase("Metrology") as p:
                metrology_input_path = os.path.join(outfolder, "mmt", "pred")
                print(f"Checking metrology input folder: {metrology_input_path}")
                mask_files = []
                if os.path.exists(metrology_input_path):
                    mask_files = glob.glob(os.path.join(metrology_input_path, "**/*.nii.gz"), recursive=True)
                    print(f"Found {len(mask_files)} .nii.gz files to process")
                    if len(mask_files) == 0:
                        print("No files found! Check if previous steps generated the required files.")
                else:
                    print(f"Error: Metrology input folder does not exist: {metrology_input_path}")

                bb_3d = np.load(os.path.join(outfolder, "bb3d.npy"))
                # Reshape to match expected format [1, N] for single model
                bb_3d_list = bb_3d.reshape(1, -1) if bb_3d.ndim == 1 else np.expand_dims(bb_3d, axis=0)

                metrology_df = process_metrology(
                    input_folder=metrology_input_path,
                    output_folder=os.path.join(outfolder, "metrology"),
                    clean_out_path=os.path.join(outfolder, "cleaned_metrology"),
                    clean_mask=MAKE_CLEAN_DEFAULT,
                    bb_3d_list=bb_3d_list,
                )
                # Count 3D masks loaded to metrology
                p.complete(count=len(mask_files))

            outf = os.path.join(outfolder, "metrology", "metrology_report.pdf")
            with metrics.phase("Report Generation"):
                generate_pdf_report(
                    os.path.join(outfolder, "metrology", "metrology.csv"),
                    outf,
                    input_filename=os.path.basename(inputfile),
                )

            # Save metrics for this task
            metrics.write_log(os.path.join(outfolder, "timing.log"))
            metrics.save(os.path.join(outfolder, "metrics.json"))

            print(f"Successfully processed {inputfile}")
            outf2 = os.path.dirname(outf)
            print(f"Report generated at {outf2}")

        except Exception as e:
            print(f"Error processing {inputfile}: {str(e)}")
            # Still save partial metrics on error
            metrics.write_log(os.path.join(outfolder, "timing.log"))
            metrics.save(os.path.join(outfolder, "metrics.json"))
            continue


if __name__ == "__main__":
    main()
