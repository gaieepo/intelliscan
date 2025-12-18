#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D NII to 2D Image Converter

This module converts 3D NIfTI medical imaging files into 2D JPEG slices
along specified viewing directions (axial or coronal). It supports both
3D and 4D volumes and applies proper normalization and image transformation
for visualization.

Key Features:
- Supports axial (XY) and coronal (XZ) slice extraction
- Automatic normalization and 8-bit conversion
- Proper image rotation and flipping for standard viewing
- Handles both 3D and 4D NIfTI volumes

Author: Wang Jie
Date: 1st Aug 2025
Version: 1.0
Original Implementation: Richard Chang (2020)
"""

import os
import sys

import nibabel as nib
from PIL import Image


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


def conv(inputfile, outputdir, view):
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


if __name__ == "__main__":
    if len(sys.argv) - 1 < 2:
        print("Missing Arguments")
        sys.exit(5)

    inputfile, outputdir, view = sys.argv[1], sys.argv[2], sys.argv[3]

    if not os.path.isfile(inputfile) or not inputfile.endswith((".nii", ".nii.gz")):
        print(inputfile, "File not correct")
        sys.exit(2)

    if not os.path.isdir(outputdir + "/" + view):
        print(outputdir, "No such directory, creating...")
        os.makedirs(outputdir + "/" + view)

    conv(inputfile, outputdir + "/" + view, view)
