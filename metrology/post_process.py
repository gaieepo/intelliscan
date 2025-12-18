#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D Mask Post-Processing and Metrology Analysis

This module provides comprehensive 3D mask processing and metrology measurement 
functions for semiconductor inspection. It handles segmented volume analysis, 
defect quantification, and precise dimensional measurements.

Key Features:
- 3D mask cleaning and morphological operations
- Bond line thickness (BLT) measurement
- Pad misalignment detection and quantification
- Void-to-solder ratio analysis
- Solder extrusion measurement
- Pillar dimension extraction
- Defect classification and flagging
- NIfTI format handling and visualization

Author: Wang Jie
Date: 1st Aug 2025
Version: 1.0
"""

import os

import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

THRESHOLD = 40  # 20 for memory


def nnz(mask):
    """
    Calculate the number of non-zero elements in a 3D mask.

    This function takes a 3D numpy array (mask) and returns the count of
    non-zero elements within the array.

    Parameters:
    mask (numpy.ndarray): A 3D numpy array where non-zero elements are to be counted.

    Returns:
    int: The number of non-zero elements in the input mask.
    """
    x, y, z = np.nonzero(mask)
    return len(x)


def get_extreme_points(mask_3D):
    """
    Calculate the extreme points in a 3D mask.
    This function takes a 3D numpy array (mask_3D) and finds the non-zero elements.
    It then calculates the minimum and maximum coordinates along each axis (x, y, z)
    where the non-zero elements are located.
    Args:
        mask_3D (numpy.ndarray): A 3D numpy array representing the mask.
    Returns:
        tuple: A tuple containing:
            - int: The number of non-zero elements in the mask.
            - list: A list of six integers representing the minimum and maximum
              coordinates along each axis in the order [min_x, max_x, min_y, max_y, min_z, max_z].
              If there are no non-zero elements, returns -1 and a list of six -1s.
    """

    x, y, z = np.nonzero(mask_3D)
    num_nnz = len(x)
    if len(x) > 0:
        # print(len(x), [min(x), max(x), min(y), max(y), min(z), max(z)])
        return len(x), [min(x), max(x), min(y), max(y), min(z), max(z)]
    else:
        return -1, [-1, -1, -1, -1, -1, -1]


def make_vertical(mask_3D):
    """
    Rotates a 3D mask array to make the longest dimension vertical (aligned with the y-axis).
    This function analyzes the dimensions of the input 3D mask and rotates it such that the
    longest dimension becomes vertical. It checks the dimensions along the x, y, and z axes
    and performs the necessary transpositions to align the longest dimension with the y-axis.
    Parameters:
    mask_3D (numpy.ndarray): A 3D numpy array representing the mask.
    Returns:
    numpy.ndarray: The rotated 3D mask array with the longest dimension aligned with the y-axis.
    """

    num1, [xmin1, xmax1, ymin1, ymax1, zmin1, zmax1] = get_extreme_points(mask_3D == 1)  # CuPillar
    del_x, del_y, del_z = xmax1 - xmin1, ymax1 - ymin1, zmax1 - zmin1
    if del_y < min(del_x, del_z):
        ...
    elif del_x < min(del_y, del_z):
        # rotate to make it y
        print("x is vertical")
        mask_3D = np.transpose(mask_3D, (1, 0, 2))
    else:
        # rotate to make it y
        print("z is vertical")
        mask_3D = np.transpose(mask_3D, (2, 0, 1))
    # num1, [xmin1, xmax1, ymin1, ymax1, zmin1, zmax1] = get_extreme_points(mask_3D == 1) # CuPillar
    # del_x, del_y, del_z = xmax1-xmin1, ymax1-ymin1, zmax1-zmin1

    # print([del_x, del_y,del_z], [xmin1, xmax1, ymin1, ymax1, zmin1, zmax1])
    return mask_3D


def centre_of_mass(mask_3D):
    """
    Calculate the center of mass for a 3D mask.
    This function computes the median coordinates of the non-zero elements
    in a 3D numpy array, which represents the center of mass.
    Parameters:
    mask_3D (numpy.ndarray): A 3D numpy array where non-zero elements represent
                             the mask for which the center of mass is to be calculated.
    Returns:
    list: A list containing the median x, y, and z coordinates of the non-zero elements.
    """

    x, y, z = np.nonzero(mask_3D)
    # print('median = ', [np.median(x), np.median(y), np.median(z)])
    # print('mean = ', [np.mean(x), np.mean(y), np.mean(z)])

    return [np.median(x), np.median(y), np.median(z)]


def clean_up_mask(mask):
    """
    Cleans up a 3D binary mask by performing morphological operations and
    removing elements based on their distance from the center of mass.
    Parameters:
    mask (numpy.ndarray): A 3D binary mask (numpy array) with shape (nr, nc, nz).
    Returns:
    numpy.ndarray: The cleaned-up 3D binary mask.
    Notes:
    - The function performs binary closing on the mask with 2 iterations.
    - Elements in the mask that are farther than a specified threshold (THRESHOLD)
      from the center of mass are set to 0.
    - The center of mass is computed using `scipy.ndimage.center_of_mass`.
    """

    thres = THRESHOLD
    import matplotlib.pyplot as plt
    from scipy import ndimage  # morphology

    nr, nc, nz = mask.shape
    # print(nr,nc,nz)
    tt = sum(np.nonzero(mask[:]))
    if tt.any():
        # Perform closing
        mask = ndimage.binary_closing(mask, iterations=2)
        # mask = ndimage.morphology.binary_dilation(mask, iterations=2)
        # mask = ndimage.morphology.binary_erosion(mask, iterations=2)
        com = ndimage.center_of_mass(mask)
        # print(com)
        for i in range(nr):
            for j in range(nc):
                for k in range(nz):
                    diff = [i - com[0], j - com[1], k - com[2]]
                    # print(diff)
                    if np.linalg.norm(diff, 2) > thres:
                        mask[i, j, k] = 0

    return mask


def clean_up_mask_slice(mask):
    """
    Cleans up a 3D mask slice by slice using morphological operations.

    This function processes each slice of the input 3D mask along the second dimension (columns).
    It performs erosion followed by dilation (opening) to remove small objects and noise.
    If any non-zero values are found in the processed slice, it further refines the mask by
    finding the largest contour and applying it to the original mask slice.

    Parameters:
    mask (numpy.ndarray): A 3D numpy array representing the mask to be cleaned.
                          The shape of the array should be (nr, nc, nz).

    Returns:
    numpy.ndarray: The cleaned 3D mask with the same shape as the input.
    """
    import matplotlib.pyplot as plt
    from skimage import morphology

    nr, nc, nz = mask.shape
    sq = morphology.square(width=3)
    dia = morphology.diamond(radius=1)
    for i in range(nc):
        img = mask[:i, :]
        tt = sum(np.nonzero(img))
        # print(tt)
        # x,y = np.nonzero(img)
        if tt.any():
            mask_out = morphology.erosion(mask[:, i, :], sq)
            mask_out = morphology.dilation(mask_out, sq)
            # # Perform closing
            # data = morphology.binary_dilation(data, iterations=1)
            # data = morphology.binary_erosion(data, iterations=1)

            val = np.max(mask_out[:])
            if val:
                plt.imshow(mask_out, cmap="gray")
                plt.show()
                img_u8 = mask_out.astype(np.uint8)
                contours, hierarchy = cv2.findContours(img_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                c = max(contours, key=cv2.contourArea)
                # mask2 = cv2.drawContours(np.zeros( (mask.shape)), contours, (255), cv2.FILLED, 8)
                mask2 = cv2.drawContours(np.zeros(img_u8.shape), [c], 0, 1, thickness=cv2.FILLED)
                # plt.imshow(mask2, cmap='gray')
                # plt.show()
                # apply the mask to the original image
                # print(img_u8.shape, mask2.shape)
                result = np.multiply(mask_out, mask2)
                # plt.imshow(result, cmap='gray')
                # plt.show()
                mask[:, i, :] = result
            else:
                mask[:, i, :] = mask_out
    return mask


def clean_up_mask3D(mask_3D):
    """
    Cleans up a 3D mask by processing different mask values separately and then combining them.
    Parameters:
    mask_3D (numpy.ndarray): The input 3D mask array.
    Returns:
    numpy.ndarray: The cleaned-up 3D mask array.
    The function performs the following steps:
    1. Rounds the input mask values.
    2. Saves the original mask as a NIfTI file.
    3. Processes mask values of 1, 2, 3, and 4 separately using the `clean_up_mask` function.
    4. Combines the processed masks back into a single 3D mask.
    5. Returns the cleaned-up 3D mask.
    """
    print('mask3d shape = ', mask_3D.shape)
    mask_3D = np.round(mask_3D)
    save_nifti(mask_3D, "./out_ori_mask", pixel_size=0.7)

    mask1 = mask_3D == 1
    print('mask1 shape, nnz  = ', mask1.shape, nnz(mask1))
    if nnz(mask1) > 0:
        mask1 = clean_up_mask(mask1)
        # save_nifti(mask1*1, './out_mask1', pixel_size = 0.7)
    mask2 = mask_3D == 2
    if nnz(mask2) > 0:
        mask2 = clean_up_mask(mask2)
        # save_nifti(mask1*1+mask2*2, './out_mask2', pixel_size = 0.7)
    mask4 = mask_3D == 4
    if nnz(mask4) > 0:
        mask4 = clean_up_mask(mask4)
        # save_nifti(mask1*1+mask2*2+mask4*4, './out_mask4', pixel_size = 0.7)
    mask3 = mask_3D == 3
    # save_nifti(mask3*3, './out_mask3', pixel_size = 0.7)
    mask_3D = mask2 * 2
    mask_3D[np.where(mask1 == 1)] = 1
    mask_3D[np.where(mask4 == 1)] = 4
    mask_3D[np.where(mask3 == 1)] = 3
    # print('max vals = ', np.max(mask_3D[:]))

    # save_nifti(mask_3D, './out_before_mask', pixel_size = 0.7)

    # print('saving mask')
    # save_nifti(mask_3D, './out_mask', pixel_size = 0.7)

    return mask_3D


def compute_metrology_info(nii_file, clean_mask, output_path):
    """
    Computes metrology information from a NIfTI file.

    Parameters:
    nii_file (str): Path to the NIfTI file.
    clean_mask (bool): Flag indicating whether to clean the mask.
    output_path (str): Path to save the cleaned NIfTI file if clean_mask is True.

    Returns:
    dict: A dictionary containing all metrology measurements in CSV-friendly format:
        - is_memory (bool): Whether memory die is present
        - blt (float): Bond line thickness
        - pad_misalignment (float): Pad misalignment measurement
        - void_solder_ratio (float): Ratio of void to solder
        - solder_extrusion_left (float): Left solder extrusion
        - solder_extrusion_right (float): Right solder extrusion
        - solder_extrusion_front (float): Front solder extrusion
        - solder_extrusion_back (float): Back solder extrusion
        - empty_connection (int): Empty connection measurement
        - pillar_width (float): Width of the pillar
        - pillar_height (float): Height of the pillar
        - void_ratio_defect (bool): Whether void ratio exceeds threshold
        - solder_extrusion_defect (bool): Whether solder extrusion exceeds threshold
        - pad_misalignment_defect (bool): Whether pad misalignment exceeds threshold
    """
    data = nib.load(nii_file)
    data = data.get_fdata()
    if clean_mask:
        data = clean_up_mask3D(data)
        save_nifti(data, output_path)
    data = make_vertical(data)

    # Get raw measurements
    is_memory, blt, pad_misalignment, void_solder_ratio, solder_extrusion, empty_connection, pillar_size, defects = compute_key_measurements(data)

    # Convert to CSV-friendly format
    measurements = {
        'is_memory': bool(is_memory),
        'blt': float(blt),
        'pad_misalignment': float(pad_misalignment),
        'void_solder_ratio': float(void_solder_ratio),
        'empty_connection': int(empty_connection),
        'pillar_width': float(pillar_size['width']),
        'pillar_height': float(pillar_size['height'])
    }

    # Handle solder extrusion measurements
    if len(solder_extrusion) > 0:
        extrusion = solder_extrusion[0]  # Get first set of measurements
        measurements.update({
            'solder_extrusion_left': float(extrusion[0]),
            'solder_extrusion_right': float(extrusion[1]),
            'solder_extrusion_front': float(extrusion[2]),
            'solder_extrusion_back': float(extrusion[3])
        })
    else:
        measurements.update({
            'solder_extrusion_left': 0.0,
            'solder_extrusion_right': 0.0,
            'solder_extrusion_front': 0.0,
            'solder_extrusion_back': 0.0
        })

    # Add defect flags
    measurements.update({
        'void_ratio_defect': bool(defects['void_ratio_defect']),
        'solder_extrusion_defect': bool(defects['solder_extrusion_defect']),
        'pad_misalignment_defect': bool(defects['pad_misalignment_defect'])
    })

    return measurements


def compute_key_measurements(mask_3D):
    """
    Compute key measurements from a 3D mask.
    Parameters:
    mask_3D (numpy.ndarray): A 3D array representing the mask with different classes.
    Returns:
    tuple: A tuple containing:
        - is_memory (bool): Indicates if the memory die is present
        - blt (int): Bond line thickness
        - pad_misalignment (float): Pad misalignment
        - void_solder_ratio (float): Ratio of void to solder
        - solder_extrusion (list): List of solder extrusion measurements [left, right, front, back]
        - empty_connection (int): Placeholder for empty connection measurement
        - pillar_size (dict): Dictionary containing pillar dimensions {'width': float, 'height': float}
        - defects (dict): Dictionary containing defect flags {
            'void_ratio_defect': bool,
            'solder_extrusion_defect': bool,
            'pad_misalignment_defect': bool
          }
    """
    # Key measurements: blt, pad misalignment, void to solder ratio, solder extrusion, empty connection
    is_memory = False
    blt, pad_misalignment, void_solder_ratio, empty_connection = -1, -1, 0, -1
    solder_extrusion = []
    pillar_size = {'width': 0, 'height': 0}
    defects = {
        'void_ratio_defect': False,
        'solder_extrusion_defect': False,
        'pad_misalignment_defect': False
    }

    num_classes = int(0.2 + np.max(mask_3D[:]))
    print("num_classes found = ", num_classes)

    if num_classes < 2 or num_classes > 4:
        return -1, blt, pad_misalignment, void_solder_ratio, solder_extrusion, empty_connection, pillar_size, defects

    num1, [xmin1, xmax1, ymin1, ymax1, zmin1, zmax1] = get_extreme_points(mask_3D == 1)  # CuPillar
    com1 = centre_of_mass(mask_3D == 1)
    num2, [xmin2, xmax2, ymin2, ymax2, zmin2, zmax2] = get_extreme_points(mask_3D == 2)  # solder

    if num_classes >= 3:
        num3, [xmin3, xmax3, ymin3, ymax3, zmin4, zmax4] = get_extreme_points(mask_3D == 3)  # void
        if num3 > 0:
            void_solder_ratio = num3 / num2
            if abs(void_solder_ratio) > 0.50:
                print("in v2s:            ", num3, num2)

    if num_classes == 4:
        is_memory = True
        num4, [xmin4, xmax4, ymin4, ymax4, zmin4, zmax4] = get_extreme_points(mask_3D == 4)  # CuPad
        com4 = centre_of_mass(mask_3D == 4)

    # Calculate solder extrusion measurements
    extrusion = [xmin2 - xmin1, -(xmax2 - xmax1), zmin2 - zmin1, -(zmax2 - zmax1)]
    solder_extrusion.append(extrusion)

    # Bond line thickness
    if num_classes == 4:  # memory die
        blt = (ymax4 - ymin1) + 1
        pad_misalignment = (abs(com4[0] - com1[0]) ** 2 + abs(com4[2] - com1[2]) ** 2) ** 0.5
        extrusion2 = [xmin2 - xmin4, -(xmax2 - xmax4), zmin2 - zmin4, -(zmax2 - zmax4)]
        solder_extrusion.append(extrusion2)
    else:
        blt = ymax2 - ymin1 + 1

    # Get pillar dimensions
    pillar_width = max(xmax1 - xmin1, zmax1 - zmin1)
    pillar_height = ymax1 - ymin1
    pillar_size = {'width': pillar_width, 'height': pillar_height}

    # Check for defects
    if void_solder_ratio > 0.30:  # 30% threshold for void ratio
        defects['void_ratio_defect'] = True

    # Check solder extrusion
    if len(solder_extrusion) > 0:
        max_extrusion = max(abs(float(val)) for val in solder_extrusion[0])  # Get maximum extrusion
        if max_extrusion > (0.10 * pillar_width):  # 10% of pillar width threshold
            defects['solder_extrusion_defect'] = True

    # Check pad misalignment
    if pad_misalignment > (0.10 * pillar_width):  # 10% of pillar width threshold
        defects['pad_misalignment_defect'] = True

    return (is_memory, blt, pad_misalignment, void_solder_ratio,
            solder_extrusion, empty_connection, pillar_size, defects)


def plot_img_and_mask(img, mask):
    """
    Plots an input image alongside its corresponding mask(s).

    Parameters:
    img (numpy.ndarray): The input image to be displayed.
    mask (numpy.ndarray): The mask(s) to be displayed. If the mask has more than two dimensions,
                          it is assumed to contain multiple classes, and each class will be displayed
                          in a separate subplot.

    Returns:
    None
    """
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title("Input image")
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f"Output mask (class {i + 1})")
            ax[i + 1].imshow(mask[:, :, i])
    else:
        ax[1].set_title("Output mask")
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()


def compile_data_3D_incomplete_masks_old(image_files):
    """
    Compiles a 3D array from a list of 2D image files, handling incomplete masks.
    Args:
        image_files (list of str): List of file paths to the 2D image files.
    Returns:
        numpy.ndarray: A 3D numpy array containing the compiled image data, or None if no images are provided.
    Notes:
        - The function assumes that the image file names follow a specific naming convention, such as
          's8_mem_seg43_view1_slice_00018_pred.png', where the slice index is embedded in the file name.
        - If the images are in color (3 channels), they are converted to grayscale.
        - The function pads the beginning of the 3D array if the starting index of the image files is greater than 1.
    """
    import cv2

    image_num = len(image_files)
    if image_num == 0:
        print("image num == 0")
        return None
    I = np.asarray(PIL.Image.open(image_files[0]))
    dim_len = len(I.shape)
    if dim_len == 2:
        nr, nc = I.shape
        nchannels = 1
    else:
        nr, nc, nchannels = I.shape

    # now check if we need to pad beginning of the masks
    # assuming name like this s8_mem_seg43_view1_slice_00018_pred.png
    name1 = image_files[0]
    idx = int(name1[-14:-9])
    # print(idx)
    data3D = np.zeros((nr, nc, image_num + idx - 1))
    for i in range(idx - 1, image_num):
        if nchannels == 3:
            I = PIL.Image.open(image_files[i])
            I = np.asarray(ImageOps.grayscale(I))
        else:
            I = np.asarray(PIL.Image.open(image_files[i]))

        # cv2.imwrite('image_%d.jpg'.format(i),I)
        data3D[:, :, i] = I
    # print(data3D.shape)
    return data3D


def compile_data_3D_incomplete_masks(image_files, start_idx, end_idx):
    """
    Compiles a 3D array from a list of 2D image files, handling incomplete masks.
    Args:
        image_files (list of str): List of file paths to the 2D image files.
        start_idx (int): The starting index for the 3D array.
        end_idx (int): The ending index for the 3D array.
    Returns:
        numpy.ndarray: A 3D array compiled from the 2D image files, or None if the image_files list is empty.
    Notes:
        - The function assumes that the image files are named in a specific format.
        - The function handles both grayscale and RGB images, converting RGB images to grayscale.
        - The resulting 3D array has dimensions (nr, nc, end_idx), where nr and nc are the number of rows and columns of the images.
    """
    import cv2

    image_num = len(image_files)
    if image_num == 0:
        print("image num == 0")
        return None
    I = np.asarray(PIL.Image.open(image_files[0]))
    dim_len = len(I.shape)
    if dim_len == 2:
        nr, nc = I.shape
        nchannels = 1
    else:
        nr, nc, nchannels = I.shape

    # now check if we need to pad beginning of the masks
    # assuming name like this s8_mem_seg43_view1_slice_00018_pred.png
    # name1 = image_files[0]
    # idx = int(name1[-14:-9])
    # print(idx)
    # data3D = np.zeros((nr,nc,image_num+idx-1))
    data3D = np.zeros((nr, nc, end_idx))
    for i in range(image_num):
        if nchannels == 3:
            I = PIL.Image.open(image_files[i])
            I = np.asarray(ImageOps.grayscale(I))
        else:
            I = np.asarray(PIL.Image.open(image_files[i]))

        # cv2.imwrite('image_%d.jpg'.format(i),I)
        data3D[
            :,
            :,
            i + start_idx - 1,
        ] = I
    # print(data3D.shape)
    return data3D


def compile_data_3D(image_files):
    import cv2

    image_num = len(image_files)
    if image_num == 0:
        print("image num == 0")
        return None
    I = np.asarray(PIL.Image.open(image_files[0]))
    dim_len = len(I.shape)
    if dim_len == 2:
        nr, nc = I.shape
        nchannels = 1
    else:
        nr, nc, nchannels = I.shape
    data3D = np.zeros((nr, nc, image_num))
    for i in range(image_num):
        if nchannels == 3:
            I = PIL.Image.open(image_files[i])
            I = np.asarray(ImageOps.grayscale(I))
        else:
            I = np.asarray(PIL.Image.open(image_files[i]))

        # cv2.imwrite('image_%d.jpg'.format(i),I)
        data3D[:, :, i] = I

    return data3D


def convert_colored_mask(image_files):
    """
    Converts a list of image files into a 3D numpy array based on specific color codes.

    Args:
        image_files (list of str): List of file paths to the images to be processed.

    Returns:
        numpy.ndarray: A 3D numpy array where each slice along the third dimension corresponds to an image.
                       The array values are determined by the color codes in the images:
                       - 1 for [255, 0, 0] (CuPillar)
                       - 2 for [0, 0, 255] (solder)
                       - 3 for [255, 0, 255] (void)
                       - 4 for [0, 255, 0] (CuPad)
                       Returns None if the input list is empty.

    Notes:
        - The function assumes that all images have the same dimensions and number of channels.
        - The function prints the file path of each image being processed.
    """
    image_num = len(image_files)
    if image_num == 0:
        print("image num == 0")
        return None
    I = np.asarray(PIL.Image.open(image_files[0]))
    nr, nc, nchannels = I.shape
    data3D = np.zeros((nr, nc, image_num))
    for i in range(image_num):
        print(image_files[i])
        I = np.asarray(PIL.Image.open(image_files[i]))
        for j in range(nr):
            for k in range(nc):
                val = I[j, k, :]
                # print('val: ', val, j, k)
                # if np.allclose(val,[0,0,0]):
                #    print('val is zero: ', j, k)
                if np.allclose(val, [255, 0, 0]):  # CuPillar
                    data3D[j, k, i] = 1
                elif np.allclose(val, [0, 0, 255]):  # solder
                    data3D[j, k, i] = 2
                elif np.allclose(val, [255, 0, 255]):  # void
                    data3D[j, k, i] = 3
                elif np.allclose(val, [0, 255, 0]):  # CuPad
                    data3D[j, k, i] = 4
    return data3D


def save_nifti(data, save_path, pixel_size=0.7):
    """
    Save a 3D numpy array as a NIfTI file.

    Parameters:
    data (numpy.ndarray): The 3D array containing the image data.
    save_path (str): The path where the NIfTI file will be saved, without the file extension.
    pixel_size (float, optional): The size of each pixel in the image. Default is 0.7.

    Returns:
    None
    """
    # Convert to Nifti
    # print("Converting to NifTI format..")  # ~ 90 seconds
    # Create header
    header = get_sample_header("metrology")
    nifti_image = nib.Nifti1Image(data, None, header)  # need to confirm affine matrix
    # print("Saving nii file to ", save_path)
    nib.save(nifti_image, save_path + ".nii.gz")
    # print("Saved nii finished ")


def get_sample_header(path, filename="sample_file.nii.gz"):
    """
    Load a NIfTI file and return its header information.

    Parameters:
    filename (str): The path to the NIfTI file. Default is 'sample_file.nii.gz'.

    Returns:
    nibabel.nifti1.Nifti1Header: The header information of the NIfTI file.
    """
    # Convert to Nifti
    data = nib.load(os.path.join(path, filename))
    header = data.header
    return header


def test_save_nifti(data, save_path):
    """
    Converts the given data to NIfTI format and saves it to the specified path.

    Parameters:
    data (nibabel.Nifti1Image): The input data to be converted and saved.
    save_path (str): The path where the NIfTI file will be saved, without the file extension.

    Returns:
    None
    """
    # Convert to Nifti
    print("Converting to NifTI format..")  # ~ 90 seconds
    # Create header
    header = data.get_header()
    # print(header)
    # print(type(header))
    image_data = data.get_fdata()
    nifti_image = nib.Nifti1Image(image_data, None, header)  # need to confirm affine matrix
    print("Saving nii file to ", save_path)
    nib.save(nifti_image, save_path + ".nii.gz")
    # print("Saved nii finished ")
