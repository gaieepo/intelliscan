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

import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from detection import run_yolo_detection, run_yolo_detection_inmemory
from merge import generate_bb3d, generate_bb3d_inmemory
from metrology import MAKE_CLEAN_DEFAULT, compute_metrology_from_array, compute_metrology_info
from report import generate_pdf_report
from segmentation import (
    SegmentationConfig,
    SegmentationInference,
    assemble_full_volume,
    expand_bbox,
    segment_bboxes,
)
from utils import PipelineLogger, PipelineMetrics, create_folder_structure, log, nii2jpg

# Pipeline configuration flags
USE_COMBINED_SEG_METROLOGY = True  # Combines segmentation and metrology into single phase
USE_INMEMORY_DETECTION = True  # Use in-memory detection/merge (faster, no intermediate TXT files)
VERBOSE = True  # Print detailed logs to console (always writes to log file)


def process_segmentation_and_metrology_combined(
    engine: SegmentationInference,
    volume: np.ndarray,
    bboxes: np.ndarray,
    seg_output_dir: Path,
    metrology_output_dir: Path,
    clean_mask: bool = MAKE_CLEAN_DEFAULT,
) -> tuple[list, pd.DataFrame]:
    """
    Combined segmentation and metrology processing per bbox.

    Processes each bbox: segment -> compute metrology -> save prediction.
    This reduces I/O by computing metrology directly from in-memory predictions
    instead of saving then reloading each prediction file.

    Args:
        engine: Initialized SegmentationInference instance
        volume: Full 3D volume array
        bboxes: Array of bboxes [N, 6] format [x_min, x_max, y_min, y_max, z_min, z_max]
        seg_output_dir: Directory to save per-bbox predictions
        metrology_output_dir: Directory to save metrology CSV
        clean_mask: Whether to apply morphological cleaning to masks

    Returns:
        Tuple of (segmentation_results, metrology_dataframe)
    """
    from segmentation import BBoxSegmentationResult

    results = []
    cfg = engine.config

    # Ensure model is loaded
    engine.load_model()

    # Ensure output directories exist
    (seg_output_dir / "img").mkdir(parents=True, exist_ok=True)
    (seg_output_dir / "pred").mkdir(parents=True, exist_ok=True)
    metrology_output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize metrology DataFrame
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

    for idx, bbox in enumerate(bboxes):
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

        # Run segmentation inference
        prediction = engine.infer_crop(crop)

        # Compute metrology directly from in-memory prediction
        try:
            measurements = compute_metrology_from_array(
                prediction.copy(),  # copy to avoid mutation
                clean_mask=clean_mask,
            )

            # Create metrology record
            record = {
                "filename": f"pred_{idx}.nii.gz",
                "is_memory": measurements["is_memory"],
                "void_ratio_defect": measurements["void_ratio_defect"],
                "solder_extrusion_defect": measurements["solder_extrusion_defect"],
                "pad_misalignment_defect": measurements["pad_misalignment_defect"],
                "bb": bbox.tolist() if hasattr(bbox, "tolist") else list(bbox),
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
            metrology_df = pd.concat([metrology_df, pd.DataFrame([record])], ignore_index=True)

        except Exception as e:
            print(f"Error computing metrology for bbox {idx}: {e}")

        # Save crop and prediction to disk (for debugging/compatibility)
        img_path = seg_output_dir / "img" / f"img_{idx}.nii.gz"
        pred_path = seg_output_dir / "pred" / f"pred_{idx}.nii.gz"
        nib.save(nib.Nifti1Image(crop.astype(np.float32), np.eye(4)), str(img_path))
        nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), str(pred_path))

        # Store result
        results.append(
            BBoxSegmentationResult(
                bbox_index=idx,
                prediction=prediction,
                original_bbox=bbox,
                expanded_bbox=expanded,
                crop_shape=crop.shape,
            )
        )

    # Save metrology CSV
    metrology_df.to_csv(metrology_output_dir / "metrology.csv", index=False)

    return results, metrology_df


def process_metrology(
    input_folder: Path,
    output_folder: Path,
    clean_out_path: Path,
    clean_mask: bool,
    bb_3d_list: np.ndarray,
) -> pd.DataFrame:
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
    segmented_files = sorted(input_folder.rglob("*.nii.gz"))
    num_files = len(segmented_files)
    print(f"Number of segmented files to process: {num_files}")

    # Create output folders if they don't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    if clean_mask:
        clean_out_path.mkdir(parents=True, exist_ok=True)

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
        rel_path = segmented_file.relative_to(input_folder)
        output_path = clean_out_path / rel_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            measurements = compute_metrology_info(
                nii_file=str(segmented_file),
                output_path=str(output_path) if clean_mask else None,
                clean_mask=clean_mask,
            )

            # Create record with all measurements (values already in micrometers)
            record = {
                "filename": str(rel_path),
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
    metrology_df.to_csv(output_folder / "metrology.csv", index=False)

    return metrology_df


def main():
    # Read input files from files.txt
    with open("files.txt") as f:
        input_files = [line.strip() for line in f.readlines()]

    log(f"Found {len(input_files)} files to process")

    # Each line in files.txt is a task
    for inputfile in input_files:
        log(f"Processing file: {inputfile}")

        # Original path
        p = Path(inputfile)

        # Remove the extension and build output folder path
        p_no_ext = p.with_suffix("")  # yields folder1/folder2/filename
        input_base_name = Path(*p_no_ext.parts[-2:])  # yields folder2/filename
        outfolder = Path("output") / input_base_name
        outfolder.mkdir(parents=True, exist_ok=True)

        # Single detection model for both views
        detection_model = "models/detection_model.pt"
        segmentation_model = "models/segmentation_model.ckpt"

        views = ["view1", "view2"]

        # Initialize logger and metrics for this task
        logger = PipelineLogger(log_file=outfolder / "execution.log", verbose=VERBOSE)
        metrics = PipelineMetrics(task_id=inputfile)

        try:
            with metrics.phase("Folder Creation"):
                folder_structure = create_folder_structure(str(outfolder), views)

            # Load volume data once for reuse across pipeline stages
            log(f"Loading volume: {inputfile}")
            img = nib.load(inputfile)
            data3d = img.get_fdata()
            if data3d.ndim == 4:
                data3d = data3d[..., 3]
            data_max = data3d.max()

            with metrics.phase("NII to JPG Conversion") as phase:
                log("Converting NII to JPG...")
                view1_slices = nii2jpg(data3d, folder_structure["view1"]["input_images"], 0, data_max)
                view2_slices = nii2jpg(data3d, folder_structure["view2"]["input_images"], 1, data_max)
                num_slices = view1_slices + view2_slices
                phase.complete(count=num_slices)

            if USE_INMEMORY_DETECTION:
                # In-memory detection pipeline (faster, no intermediate files)
                with metrics.phase("2D Detection Inference") as phase:
                    log("Running object detection inference (in-memory)...")

                    log("Running detection on view1...")
                    view1_detections = run_yolo_detection_inmemory(
                        model_path=detection_model,
                        input_folder=folder_structure["view1"]["input_images"],
                    )

                    log("Running detection on view2...")
                    view2_detections = run_yolo_detection_inmemory(
                        model_path=detection_model,
                        input_folder=folder_structure["view2"]["input_images"],
                    )

                    num_slices = len(view1_detections) + len(view2_detections)
                    phase.complete(count=num_slices)

                with metrics.phase("3D Bounding Box Generation") as phase:
                    log("Generating 3D bounding boxes (in-memory)...")

                    # Get image dimensions from first image
                    from PIL import Image

                    sample_image_path = Path(folder_structure["view1"]["input_images"]) / "image0.jpg"
                    with Image.open(sample_image_path) as img:
                        image_dimensions = img.size  # (width, height)

                    view1_num_slices = len(view1_detections)
                    view2_num_slices = len(view2_detections)

                    bb_3d = generate_bb3d_inmemory(
                        view1_detections=view1_detections,
                        view2_detections=view2_detections,
                        view1_num_slices=view1_num_slices,
                        view2_num_slices=view2_num_slices,
                        image_dimensions=image_dimensions,
                        output_file=str(outfolder / "3d_bounding_boxes.npy"),
                        is_normalized=False,
                    )

                    # Save as single array
                    np.save(outfolder / "bb3d.npy", bb_3d)

                    # Free detection memory after merging
                    del view1_detections, view2_detections

                    effective_slices = view1_num_slices + view2_num_slices
                    phase.complete(count=effective_slices)

                # Explicit cleanup before loading segmentation model
                import gc

                gc.collect()

            else:
                # File-based detection pipeline (original behavior)
                with metrics.phase("2D Detection Inference") as phase:
                    log("Running object detection inference...")

                    for view in views:
                        input_folder = folder_structure[view]["input_images"]
                        output_folder = folder_structure[view]["detections"]

                        log(f"Running detection on {view}...")
                        run_yolo_detection(
                            model_path=detection_model,
                            input_folder=input_folder,
                            output_folder=output_folder,
                            batch_size=64,
                        )

                    view1_dir = Path(folder_structure["view1"]["input_images"])
                    view2_dir = Path(folder_structure["view2"]["input_images"])
                    num_slices = len(list(view1_dir.glob("*.jpg"))) + len(list(view2_dir.glob("*.jpg")))
                    phase.complete(count=num_slices)

                with metrics.phase("3D Bounding Box Generation") as phase:
                    log("Generating 3D bounding boxes...")
                    view1_dir = Path(folder_structure["view1"]["input_images"])
                    view2_dir = Path(folder_structure["view2"]["input_images"])
                    view1_num_slices = len(list(view1_dir.glob("*.jpg")))
                    view2_num_slices = len(list(view2_dir.glob("*.jpg")))
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
                        view1_params, view2_params, str(outfolder / "3d_bounding_boxes.npy"), is_normalized=False
                    )
                    np.save(outfolder / "bb3d.npy", bb_3d)
                    effective_slices = view1_num_slices + view2_num_slices
                    phase.complete(count=effective_slices)

            num_bboxes = len(bb_3d) if bb_3d.ndim == 1 else bb_3d.shape[0]
            log(f"3D bounding boxes shape: {bb_3d.shape}")

            seg_output_dir = outfolder / "mmt"
            engine = SegmentationInference(segmentation_model, SegmentationConfig())

            if USE_COMBINED_SEG_METROLOGY:
                # Combined segmentation + metrology in single phase
                with metrics.phase("3D Segmentation + Metrology") as phase:
                    log("Running combined 3D segmentation and metrology...")

                    results, metrology_df = process_segmentation_and_metrology_combined(
                        engine=engine,
                        volume=data3d,
                        bboxes=bb_3d,
                        seg_output_dir=seg_output_dir,
                        metrology_output_dir=outfolder / "metrology",
                        clean_mask=MAKE_CLEAN_DEFAULT,
                    )

                    # Assemble into full volume and save
                    full_segmentation = assemble_full_volume(results, data3d.shape)
                    nib.save(
                        nib.Nifti1Image(full_segmentation, np.eye(4)),
                        str(outfolder / "segmentation.nii.gz"),
                    )
                    log(f"Combined segmentation + metrology completed, saved to {outfolder}")
                    phase.complete(count=num_bboxes)

            else:
                # Separate segmentation and metrology phases (original behavior)
                with metrics.phase("3D Segmentation") as phase:
                    log("Running 3D segmentation...")

                    # Create output directories for per-bbox results
                    (seg_output_dir / "img").mkdir(parents=True, exist_ok=True)
                    (seg_output_dir / "pred").mkdir(parents=True, exist_ok=True)

                    # Segment all bounding boxes
                    results = segment_bboxes(engine, data3d, bb_3d, save_dir=seg_output_dir)

                    # Assemble into full volume and save
                    full_segmentation = assemble_full_volume(results, data3d.shape)
                    nib.save(
                        nib.Nifti1Image(full_segmentation, np.eye(4)),
                        str(outfolder / "segmentation.nii.gz"),
                    )
                    log(f"Segmentation completed, saved to {outfolder}")
                    phase.complete(count=num_bboxes)

                with metrics.phase("Metrology") as phase:
                    metrology_input_path = outfolder / "mmt" / "pred"
                    log(f"Checking metrology input folder: {metrology_input_path}")
                    mask_files = []
                    if metrology_input_path.exists():
                        mask_files = list(metrology_input_path.rglob("*.nii.gz"))
                        log(f"Found {len(mask_files)} .nii.gz files to process")
                        if len(mask_files) == 0:
                            log("No files found! Check if previous steps generated files.", level="warning")
                    else:
                        log(f"Metrology input folder does not exist: {metrology_input_path}", level="error")

                    # Reshape to match expected format [1, N] for single model
                    bb_3d_list = bb_3d.reshape(1, -1) if bb_3d.ndim == 1 else np.expand_dims(bb_3d, axis=0)

                    metrology_df = process_metrology(
                        input_folder=metrology_input_path,
                        output_folder=outfolder / "metrology",
                        clean_out_path=outfolder / "cleaned_metrology",
                        clean_mask=MAKE_CLEAN_DEFAULT,
                        bb_3d_list=bb_3d_list,
                    )
                    phase.complete(count=len(mask_files))

            report_path = outfolder / "metrology" / "metrology_report.pdf"
            with metrics.phase("Report Generation"):
                generate_pdf_report(
                    str(outfolder / "metrology" / "metrology.csv"),
                    str(report_path),
                    input_filename=p.name,
                )

            # Save metrics for this task
            metrics.write_log(str(outfolder / "timing.log"))
            metrics.save(str(outfolder / "metrics.json"))

            log(f"Successfully processed {inputfile}")
            log(f"Report generated at {report_path.parent}")

        except Exception as e:
            log(f"Error processing {inputfile}: {str(e)}", level="error")
            # Still save partial metrics on error
            metrics.write_log(str(outfolder / "timing.log"))
            metrics.save(str(outfolder / "metrics.json"))
            continue

        finally:
            logger.close()


if __name__ == "__main__":
    main()
