#!/usr/bin/env python3
"""Evaluate Dice accuracy of the pipeline's inference path (GPU normalize + batch).

Loads test set crops, runs them through segmentation.py's infer_crop()
(which uses GPU normalize), and computes Dice per class vs ground truth.

This verifies that GPU normalize + batch inference doesn't degrade accuracy
compared to the original CPU normalize + sliding_window path.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / ".." / "wp5-seg"))

from segmentation import SegmentationConfig, SegmentationInference


def dice_score(pred: np.ndarray, label: np.ndarray, num_classes: int = 5) -> list[float]:
    """Compute per-class Dice."""
    scores = []
    for c in range(num_classes):
        p = (pred == c).astype(np.float32)
        g = (label == c).astype(np.float32)
        intersection = (p * g).sum()
        union = p.sum() + g.sum()
        if union == 0:
            scores.append(1.0 if intersection == 0 else 0.0)
        else:
            scores.append(2.0 * intersection / union)
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="/home/kaixin/yj/wys/3ddl-dataset/data")
    parser.add_argument("--use_trt", action="store_true")
    parser.add_argument("--trt_engine", type=str, default=None)
    parser.add_argument("--batch", action="store_true", help="Test batch inference path")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_cases", type=int, default=-1)
    args = parser.parse_args()

    import train
    _, test_list = train.build_datalists(args.data_dir)
    if args.max_cases > 0:
        test_list = test_list[:args.max_cases]

    print(f"Test samples: {len(test_list)}")

    seg_cfg = SegmentationConfig(
        use_trt=args.use_trt,
        trt_engine_path=args.trt_engine,
    )
    engine = SegmentationInference(args.ckpt, seg_cfg)
    engine.load_model()

    # Remap class 6 -> 0 (same as train.py)
    remap = {6: 0}

    all_dice = []

    if args.batch:
        # Test the batch inference path used in segment_bboxes
        from segmentation import BBoxSegmentationResult, expand_bbox
        device = engine.device
        roi = seg_cfg.roi_size

        crops = []
        labels_list = []
        for i, sample in enumerate(test_list):
            img = nib.load(sample["image"]).get_fdata().astype(np.float32)
            label = nib.load(sample["label"]).get_fdata().astype(np.int64)
            for old, new in remap.items():
                label[label == old] = new
            crops.append(img)
            labels_list.append(label)

        # Batch process: normalize on GPU, pad to roi, batch inference
        print(f"Running batch inference (batch_size={args.batch_size})...")
        predictions = []

        # Normalize all crops on GPU
        normalized = []
        for crop in crops:
            t = torch.from_numpy(crop).to(device)
            if seg_cfg.apply_normalization:
                t = engine.normalize_gpu(t)
            normalized.append(t)

        # Classify into batchable (fits roi) and oversized
        from monai.inferers import sliding_window_inference

        batchable_indices = []
        oversized_indices = []
        for i, t in enumerate(normalized):
            fits = all(t.shape[d] <= roi[d] for d in range(3))
            if fits:
                batchable_indices.append(i)
            else:
                oversized_indices.append(i)

        print(f"  Batchable: {len(batchable_indices)}, Oversized: {len(oversized_indices)}")

        # Pre-allocate predictions list
        predictions = [None] * len(normalized)

        # Batch inference for batchable crops
        for start in range(0, len(batchable_indices), args.batch_size):
            batch_idx = batchable_indices[start : start + args.batch_size]
            batch_tensors = [normalized[i] for i in batch_idx]
            batch_shapes = [t.shape for t in batch_tensors]

            padded = []
            for t in batch_tensors:
                pad = []
                for dim in reversed(range(3)):
                    diff = roi[dim] - t.shape[dim]
                    half = diff // 2
                    pad.extend([half, diff - half])
                p = torch.nn.functional.pad(t, pad, mode="constant", value=0.0)
                padded.append(p.unsqueeze(0).unsqueeze(0))

            batch_tensor = torch.cat(padded, dim=0)
            with torch.no_grad():
                batch_logits = engine._predictor(batch_tensor)

            for j, idx in enumerate(batch_idx):
                s = batch_shapes[j]
                slices = []
                for dim in range(3):
                    diff = roi[dim] - s[dim]
                    half = diff // 2
                    slices.append(slice(half, half + s[dim]))
                logits_j = batch_logits[j:j+1, :, slices[0], slices[1], slices[2]]
                pred = torch.argmax(logits_j, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                predictions[idx] = pred

            done = min(start + args.batch_size, len(batchable_indices))
            if done % 40 == 0 or done == len(batchable_indices):
                print(f"  batch: {done}/{len(batchable_indices)}")

        # Sliding window for oversized crops
        for idx in oversized_indices:
            t = normalized[idx].unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits = sliding_window_inference(
                    inputs=t, roi_size=roi, sw_batch_size=1,
                    predictor=engine._predictor, overlap=0.5, mode="gaussian",
                )
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            predictions[idx] = pred
            print(f"  oversized [{idx}]: shape={normalized[idx].shape}")

        for pred, label in zip(predictions, labels_list):
            d = dice_score(pred, label)
            all_dice.append(d)

    else:
        # Test infer_crop path (single crop, GPU normalize)
        for i, sample in enumerate(test_list):
            img = nib.load(sample["image"]).get_fdata().astype(np.float32)
            label = nib.load(sample["label"]).get_fdata().astype(np.int64)
            for old, new in remap.items():
                label[label == old] = new

            pred = engine.infer_crop(img)
            d = dice_score(pred, label)
            all_dice.append(d)

            if (i + 1) % 20 == 0 or (i + 1) == len(test_list):
                avg = np.mean(all_dice, axis=0)
                print(f"  [{i+1}/{len(test_list)}] running avg Dice={avg.mean():.4f}")

    all_dice = np.array(all_dice)
    avg_dice = all_dice.mean(axis=0)
    mean_dice = avg_dice.mean()

    print(f"\nPer-class Dice (0..4):")
    for c in range(5):
        print(f"  class {c}: {avg_dice[c]:.6f}")
    print(f"Average Dice: {mean_dice:.6f}")
    print(f"Average IoU:  {(avg_dice / (2 - avg_dice)).mean():.6f}")


if __name__ == "__main__":
    main()
