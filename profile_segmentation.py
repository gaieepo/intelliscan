#!/usr/bin/env python3
"""Profile segmentation pipeline to identify bottlenecks."""
from __future__ import annotations

import time
import numpy as np
import nibabel as nib
import torch
from monai.inferers import sliding_window_inference

from segmentation import (
    SegmentationConfig,
    SegmentationInference,
    TensorRTPredictor,
    expand_bbox,
)
from utils import log


def profile_single_bbox(engine, volume, bbox, cfg):
    """Profile each step of single bbox processing."""
    timings = {}

    # 1. Expand bbox
    t0 = time.perf_counter()
    expanded = expand_bbox(bbox, volume.shape, cfg.margin)
    timings["expand_bbox"] = time.perf_counter() - t0

    region_size = (
        expanded[1] - expanded[0],
        expanded[3] - expanded[2],
        expanded[5] - expanded[4],
    )

    # 2. Extract crop
    t0 = time.perf_counter()
    crop = volume[
        expanded[0] : expanded[1],
        expanded[2] : expanded[3],
        expanded[4] : expanded[5],
    ]
    timings["extract_crop"] = time.perf_counter() - t0

    # 3. Normalize
    t0 = time.perf_counter()
    if cfg.apply_normalization:
        crop_norm = engine.normalize(crop)
    else:
        crop_norm = crop.astype(np.float32)
    timings["normalize"] = time.perf_counter() - t0

    # 4. To tensor
    t0 = time.perf_counter()
    tensor = torch.from_numpy(crop_norm).unsqueeze(0).unsqueeze(0).to(engine.device)
    timings["to_tensor"] = time.perf_counter() - t0

    # 5. Sliding window inference
    t0 = time.perf_counter()
    with torch.no_grad():
        logits = sliding_window_inference(
            inputs=tensor,
            roi_size=cfg.roi_size,
            sw_batch_size=cfg.sw_batch_size,
            predictor=engine._predictor,
            overlap=cfg.overlap,
            mode="gaussian",
        )
    torch.cuda.synchronize()
    timings["sliding_window"] = time.perf_counter() - t0

    # 6. Argmax + to numpy
    t0 = time.perf_counter()
    prediction = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    timings["argmax_to_numpy"] = time.perf_counter() - t0

    # 7. NII save (simulate what segment_bboxes does)
    t0 = time.perf_counter()
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        tmppath = f.name
    nib.save(nib.Nifti1Image(crop.astype(np.float32), np.eye(4)), tmppath)
    nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), tmppath)
    os.unlink(tmppath)
    timings["nii_save_x2"] = time.perf_counter() - t0

    timings["crop_shape"] = region_size
    return timings


def main():
    # Load volume
    nii_path = "/home/kaixin/yj/wys/data/SN002_3D_Feb24/4_die_stack_0.75um_3Drecon_txm.nii"
    log(f"Loading volume: {nii_path}")
    img = nib.load(nii_path)
    data3d = img.get_fdata()
    if data3d.ndim == 4:
        data3d = data3d[..., 3]

    # Load bboxes
    bb_path = "output/SN002/bb3d.npy"
    bboxes = np.load(bb_path)
    log(f"Loaded {len(bboxes)} bboxes")

    configs = {
        "Original PyTorch": {
            "model_path": "models/segmentation_model.ckpt",
            "use_trt": False,
        },
        "Pruned PyTorch": {
            "model_path": "models/segmentation_model_pruned.ckpt",
            "use_trt": False,
        },
        "Pruned TRT FP16": {
            "model_path": "models/segmentation_model_pruned.ckpt",
            "use_trt": True,
            "trt_engine_path": "models/segmentation_pruned_fp16.engine",
        },
    }

    for name, cfg_dict in configs.items():
        log(f"\n{'='*60}")
        log(f"Profiling: {name}")
        log(f"{'='*60}")

        seg_cfg = SegmentationConfig(
            use_trt=cfg_dict.get("use_trt", False),
            trt_engine_path=cfg_dict.get("trt_engine_path"),
        )
        engine = SegmentationInference(cfg_dict["model_path"], seg_cfg)
        engine.load_model()

        # Warmup with first bbox
        log("Warmup...")
        _ = profile_single_bbox(engine, data3d, bboxes[0], seg_cfg)

        # Profile first 10 bboxes
        num_profile = min(10, len(bboxes))
        all_timings = []
        for i in range(num_profile):
            t = profile_single_bbox(engine, data3d, bboxes[i], seg_cfg)
            all_timings.append(t)

        # Aggregate
        keys = [k for k in all_timings[0] if k != "crop_shape"]
        log(f"\nPer-step average over {num_profile} bboxes:")
        log(f"  {'Step':<25} {'Avg (ms)':>10} {'% Total':>10}")
        log(f"  {'-'*45}")

        total_avg = sum(
            np.mean([t[k] for t in all_timings]) for k in keys
        )

        for k in keys:
            vals = [t[k] * 1000 for t in all_timings]
            avg = np.mean(vals)
            pct = (avg / (total_avg * 1000)) * 100
            log(f"  {k:<25} {avg:>10.2f} {pct:>9.1f}%")

        log(f"  {'-'*45}")
        log(f"  {'TOTAL':<25} {total_avg*1000:>10.2f}")
        log(f"  Projected 95 bboxes: {total_avg * 95:.2f}s")

        # Show crop shapes
        shapes = [t["crop_shape"] for t in all_timings]
        log(f"\n  Crop shapes (first {num_profile}):")
        for i, s in enumerate(shapes):
            log(f"    bbox {i}: {s}")

        # Cleanup
        del engine
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
