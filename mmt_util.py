#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Segmentation Testing Utilities

This module provides comprehensive utility functions for testing and evaluating 
3D segmentation models on semiconductor data. It handles patch-based inference, 
post-processing, and performance evaluation.

Key Features:
- Patch-based 3D inference with overlapping strategy
- Bounding box-guided region processing
- Internal void detection and filling
- Multi-class segmentation evaluation
- Memory-efficient processing for large volumes
- NIfTI format integration

Author: Wang Jie
Date: 1st Aug 2025
Version: 1.0
"""

import math

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label


def fill_internal_voids(prediction, background_class=0, void_class=3):
    """
    Find all connected regions of `background_class` in `prediction`
    that do NOT touch the border of the volume, and relabel them
    as `void_class`.

    Parameters
    ----------
    prediction : np.ndarray, shape (D, H, W)
        Integer-valued segmentation map.
    background_class : int
        The label in `prediction` considered as "true" background.
    void_class : int
        The label to assign to any background-hole inside class 2.

    Returns
    -------
    np.ndarray
        The same array, but with interior holes relabelled.
    """
    # 1) mask and label the background
    bg_mask = (prediction == background_class)
    bg_labels, n_labels = label(bg_mask)

    # 2) find which bg-labels touch the volume boundary
    boundary = np.zeros_like(bg_mask, dtype=bool)
    # six faces of a cuboid
    boundary[0,   :,   :] = True
    boundary[-1,  :,   :] = True
    boundary[:,   0,   :] = True
    boundary[:,  -1,   :] = True
    boundary[:,   :,   0] = True
    boundary[:,   :,  -1] = True

    # get the set of labels on the boundary (excluding 0)
    external_labels = set(np.unique(bg_labels[boundary & bg_mask]).tolist())
    external_labels.discard(0)

    # 3) any other label is an "internal" hole → relabel
    for lab in range(1, n_labels + 1):
        if lab not in external_labels:
            prediction[bg_labels == lab] = void_class

    return prediction


def test_all_case_ectc(
    net,
    img,
    bboxes,
    num_classes,
    patch_size=(112, 112, 80),
    stride_xy=18,
    stride_z=4,
    save_result=True,
    test_save_path=None,
    preproc_fn=None,
    model="vnet",
    single_img="img",
    single_pred="pred",
    seg_class="0",
    margin=0,
):
    """Process all 3D image patches and evaluate segmentation performance.

    @param net (torch.nn.Module): The neural network model for segmentation.
    @param img (numpy.ndarray): The 3D medical image data.
    @param bboxes (list): List of bounding boxes to process in the image.
    @param num_classes (int): Number of segmentation classes.
    @param patch_size (tuple): Size of the 3D patch to process.
    @param stride_xy (int): Stride in the XY plane.
    @param stride_z (int): Stride in the Z direction.
    @param save_result (bool): Flag to save the result.
    @param test_save_path (str): Path to save the test results.
    @param preproc_fn (function): Preprocessing function to apply to the image patches.
    @param model (str): Model type used for segmentation.
    @param output_dir (str): Directory to save the output results.

    @return
        numpy.ndarray: Metrics for each class.
        list: Predictions for each processed patch.
    """
    total_metric = np.zeros((num_classes, 4))
    predictions = []

    print("####### In test_all_case #########")
    for idx, bbox in enumerate(bboxes):
        print(f"Processing bbox {idx}, x:{bbox[1]-bbox[0]}, y:{bbox[3]-bbox[2]}, z: {bbox[5]-bbox[4]}")
        expanded_bbox = [
            max(bbox[0] - margin, 0),
            min(bbox[1] + margin, img.shape[0]),
            max(bbox[2] - margin, 0),
            min(bbox[3] + margin, img.shape[1]),
            max(bbox[4] - margin, 0),
            min(bbox[5] + margin, img.shape[2]),
        ]
        crop = img[
            expanded_bbox[0] : expanded_bbox[1],
            expanded_bbox[2] : expanded_bbox[3],
            expanded_bbox[4] : expanded_bbox[5],
        ]

        crop_transposed = crop.transpose((1, 0, 2))
        # print(crop.shape, crop_transposed.shape)

        prediction, score_map, aux1, aux2, aux3 = test_single_case(
            net, crop_transposed, stride_xy, stride_z, patch_size, num_classes=num_classes, model=model
        )

        # Apply void-filling post-processing before storing the prediction
        prediction = fill_internal_voids(prediction, background_class=0, void_class=3)

        predictions.append(prediction)

        nib.save(
            nib.Nifti1Image(crop_transposed.astype(np.float32), np.eye(4)),
            f"{single_img}/class_{seg_class}/img_{idx}.nii.gz",
        )
        nib.save(
            nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)),
            f"{single_pred}/class_{seg_class}/pred_{idx}.nii.gz",
        )

    return total_metric, predictions


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1, model="vnet"):
    """Segment a single 3D image patch and return the prediction.

    @param net (torch.nn.Module): The neural network model for segmentation.
    @param image (numpy.ndarray): The 3D image patch.
    @param stride_xy (int): Stride in the XY plane.
    @param stride_z (int): Stride in the Z direction.
    @param patch_size (tuple): Size of the 3D patch to process.
    @param num_classes (int): Number of segmentation classes.
    @param model (str): Model type used for segmentation.

    @return (tuple): Prediction, score map, and auxiliary outputs.
    """
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(
            image, [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)], mode="constant", constant_values=0
        )
    ww, hh, dd = image.shape
    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[xs : xs + patch_size[0], ys : ys + patch_size[1], zs : zs + patch_size[2]]
                test_patch = np.expand_dims(np.expand_dims(test_patch, axis=0), axis=0).astype(np.float32)
                test_patch = torch.from_numpy(test_patch).cuda()

                with torch.no_grad():
                    y1, outputs_aux1, outputs_aux2, outputs_aux3 = net(test_patch)

                outputs_aux1 = F.softmax(outputs_aux1, dim=1).cpu().data.numpy()[0, ...]
                outputs_aux2 = F.softmax(outputs_aux2, dim=1).cpu().data.numpy()[0, ...]
                outputs_aux3 = F.softmax(outputs_aux3, dim=1).cpu().data.numpy()[0, ...]

                y2 = F.softmax(y1, dim=1)
                y2 = y2.cpu().data.numpy()

                y2 = y2[0, :, :, :, :]

                score_map[:, xs : xs + patch_size[0], ys : ys + patch_size[1], zs : zs + patch_size[2]] = (
                    score_map[:, xs : xs + patch_size[0], ys : ys + patch_size[1], zs : zs + patch_size[2]] + y2
                )
                cnt[xs : xs + patch_size[0], ys : ys + patch_size[1], zs : zs + patch_size[2]] = (
                    cnt[xs : xs + patch_size[0], ys : ys + patch_size[1], zs : zs + patch_size[2]] + 1
                )
    aux1 = np.argmax(outputs_aux1, axis=0)
    aux2 = np.argmax(outputs_aux2, axis=0)
    aux3 = np.argmax(outputs_aux3, axis=0)

    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)

    if add_pad:
        label_map = label_map[wl_pad : wl_pad + w, hl_pad : hl_pad + h, dl_pad : dl_pad + d]
        score_map = score_map[:, wl_pad : wl_pad + w, hl_pad : hl_pad + h, dl_pad : dl_pad + d]
        aux1 = aux1[wl_pad : wl_pad + w, hl_pad : hl_pad + h, dl_pad : dl_pad + d]
        aux2 = aux2[wl_pad : wl_pad + w, hl_pad : hl_pad + h, dl_pad : dl_pad + d]
        aux3 = aux3[wl_pad : wl_pad + w, hl_pad : hl_pad + h, dl_pad : dl_pad + d]
    return label_map, score_map, aux1, aux2, aux3
