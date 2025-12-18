#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Segmentation Inference Engine

This module performs 3D semantic segmentation inference on semiconductor 
data using trained VNet models. It processes 3D volumes within specified 
bounding boxes and generates detailed segmentation masks for different 
material classes (solder, copper, void, etc.).

Key Features:
- VNet-based 3D semantic segmentation
- Bounding box-guided region of interest processing
- Multi-class segmentation (5 classes: background, solder, copper, void, pad)
- Patch-based inference with overlapping strategy
- NIfTI format input/output support
- GPU-accelerated processing

Author: Wang Jie
Date: 1st Aug 2025
Version: 1.0
"""

import argparse
import os

import nibabel as nib
import numpy as np
import torch

from mmt_util import test_all_case_ectc
from networks.vnet_pyramid import VNet as vnet_pyramid

MARGIN = 15


def test_calculate_metric(
    model_weights,
    bboxes3d,
    output_path,
    output_filename,
    seg_class,
    num_classes=5,
    data_3d=None,
    new_data_3d=None,
    input_im=None,
):
    """Perform inference on 3D input data using a trained VNet model and calculate metrics.

    The function loads a pre-trained VNet model and performs inference on the provided 3D input data.
    It uses the provided 3D bounding boxes to segment the input data, calculate metrics, and save the resulting
    segmentation as a NIfTI file.

    @return (float): The average metric calculated from the predictions.
    """
    # Load the VNet model with specified parameters and weights
    net = vnet_pyramid(
        n_channels=1,
        n_classes=num_classes,
        normalization="batchnorm",
        has_dropout=False,
        pyramid_has_dropout=False,
    ).cuda()
    net.load_state_dict(torch.load(model_weights))
    net.eval()

    # Perform inference and calculate metrics
    single_output_dir = os.path.join(output_path, "mmt")
    os.makedirs(single_output_dir, exist_ok=True)
    single_img = os.path.join(single_output_dir, "img")
    os.makedirs(single_img, exist_ok=True)
    os.makedirs(os.path.join(single_img, f"class_{seg_class}"), exist_ok=True)
    single_pred = os.path.join(single_output_dir, "pred")
    os.makedirs(single_pred, exist_ok=True)
    os.makedirs(os.path.join(single_pred, f"class_{seg_class}"), exist_ok=True)

    avg_metric, predictions = test_all_case_ectc(
        net,
        data_3d,
        bboxes3d,
        num_classes=num_classes,
        patch_size=(112, 112, 80),
        stride_xy=18,
        stride_z=4,
        model="mmt",
        single_img=single_img,
        single_pred=single_pred,
        seg_class=seg_class,
        margin=MARGIN,
    )

    # Create an empty array for the new 3D data with the same shape as the input data
    if new_data_3d is None:
        new_data_3d = np.zeros(data_3d.shape)
    for prediction, bbox in zip(predictions, bboxes3d):
        # Expand the bounding box coordinates to include a margin
        expanded_bbox = [
            max(bbox[0] - MARGIN, 0),
            min(bbox[1] + MARGIN, data_3d.shape[0]),
            max(bbox[2] - MARGIN, 0),
            min(bbox[3] + MARGIN, data_3d.shape[1]),
            max(bbox[4] - MARGIN, 0),
            min(bbox[5] + MARGIN, data_3d.shape[2]),
        ]

        # Transpose the prediction array to match the desired shape
        prediction = prediction.transpose((1, 0, 2))

        if expanded_bbox[1] - expanded_bbox[0]!= prediction.shape[0]:
            expanded_bbox[0]+= round(((expanded_bbox[1] - expanded_bbox[0])-prediction.shape[0])/2)
            expanded_bbox[1]=expanded_bbox[0]+prediction.shape[0]

        if expanded_bbox[3] - expanded_bbox[2]!= prediction.shape[1]:
            expanded_bbox[2]+=round(((expanded_bbox[3] - expanded_bbox[2])-prediction.shape[1])/2)
            expanded_bbox[3]=expanded_bbox[2]+prediction.shape[1]
        if expanded_bbox[5] - expanded_bbox[4]!= prediction.shape[2]:
            expanded_bbox[4]+=round(((expanded_bbox[5] - expanded_bbox[4])-prediction.shape[2])/2)
            expanded_bbox[5]=expanded_bbox[4]+prediction.shape[2]

        if expanded_bbox[1] - expanded_bbox[0] < 5 or \
            expanded_bbox[3] - expanded_bbox[2] < 5 or \
            expanded_bbox[5] - expanded_bbox[4] < 5:
                continue



        # Update the new 3D data array with the prediction in the expanded bounding box region
        new_data_3d[
            expanded_bbox[0] : expanded_bbox[1],
            expanded_bbox[2] : expanded_bbox[3],
            expanded_bbox[4] : expanded_bbox[5],
        ] = prediction

    # Create a NIfTI image from the new 3D data array
    new_image = nib.Nifti1Image(new_data_3d, affine=np.eye(4))

    # Save the NIfTI image to the specified output folder
    save_file = os.path.join(output_path, output_filename)
    nib.save(new_image, save_file)

    print(f"Resulting nii saved at {save_file}")

    return avg_metric, new_data_3d


if __name__ == "__main__":
    from scipy.ndimage import zoom

    from mmt_util import test_single_case

    def resize_3d(input_data, target_shape):
        """
        Resize a 3D numpy array to the target shape.

        Parameters:
            input_data (numpy.ndarray): The input 3D array.
            target_shape (tuple): The desired output shape (x, y, z).

        Returns:
            numpy.ndarray: The resized 3D array.
        """
        input_shape = input_data.shape
        zoom_factors = (
            target_shape[0] / input_shape[0],
            target_shape[1] / input_shape[1],
            target_shape[2] / input_shape[2],
        )
        resized_data = zoom(input_data, zoom_factors, order=3)  # Use spline interpolation of order 3
        return resized_data

    model_path = "models/mmt.pth"
    test_data = "data/sigray_recon.nii"
    # test_data = "data/img_100.nii.gz"
    patch_size = (112, 112, 80)

    net = vnet_pyramid(
        n_channels=1,
        n_classes=5,
        normalization="batchnorm",
        has_dropout=False,
        pyramid_has_dropout=False,
    ).cuda()
    net.load_state_dict(torch.load(model_path))
    net.eval()
    print(f"Model {model_path} loaded.")

    img = nib.load(test_data).get_fdata()
    img = img.transpose((1, 0, 2))
    img = resize_3d(img, patch_size)
    print(f"Test data {test_data} loaded with shape {img.shape}.")

    prediction, *_ = test_single_case(net, img, stride_xy=18, stride_z=4, patch_size=patch_size, num_classes=5)
    print(f"Predicted with array shape {prediction.shape} ({np.unique(prediction)})")

    nib.save(nib.Nifti1Image(img.astype(np.float32), np.eye(4)), "img_sigray.nii.gz")
    nib.save(nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)), "pred_sigray.nii.gz")
    print("Predicted saved.")
