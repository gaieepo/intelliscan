#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Object Detection Inference

This module performs batch inference on 2D images using YOLO object detection
models. It processes directories of images and generates bounding box coordinates
along with visualization outputs for semiconductor component detection.

Features:
- YOLO model integration for fast inference
- Batch processing of image directories
- Automatic bounding box coordinate extraction
- Visualization generation and saving
- Support for multi-class detection with confidence scoring

Author: Wang Jie
Date: 1st Aug 2025
Version: 1.0
"""

import asyncio
import os

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO


def infer_batch_yolo(model_path, input_folder, output_folder, batch_size=16):
    """
    Perform inference on a batch of images using Detectron2.

    Parameters:
    - cfg_path (str): Path to the configuration file (.yaml).
    - model_path (str): Path to the trained model file (.pth).
    - input_folder (str): Path to the folder containing input images.
    - output_folder (str): Path to the folder where outputs will be saved.

    Description:
    This function reads images from the specified folder, runs inference on each
    image using a Detectron2 model, and saves the resulting bounding box
    coordinates to text files in the output folder.

    Raises:
    - ValueError: If any of the provided paths are invalid.
    """
    # Validate inputs
    # if not os.path.isfile(cfg_path):
    #     raise ValueError(f"Configuration file not found: {cfg_path}")
    if not os.path.isfile(model_path):
        raise ValueError(f"Model file not found: {model_path}")
    if not os.path.isdir(input_folder):
        raise ValueError(f"Input folder not found: {input_folder}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)

    # Initialize the predictor based on the trainer type
    # Load configurations and model weights
    model = YOLO(model_path)

    # Create visualization folder
    vis_folder = os.path.join(os.path.dirname(output_folder), "visualize")
    os.makedirs(vis_folder, exist_ok=True)

    images = []
    filenames = []

    results = model.predict(source=input_folder)

    for res in results:
        if len(res) > 0:
            # Save bounding box coordinates
            filename = os.path.basename(res.path)
            output_path = os.path.join(output_folder, filename[:-4] + ".txt")
            class_id = res.boxes.cls.cpu().numpy()
            bb = res.boxes.xyxy.cpu().numpy()
            np.savetxt(output_path, np.column_stack((class_id.reshape(-1, 1), bb)))

            # Save visualization
            vis_path = os.path.join(vis_folder, filename)
            res.save(vis_path)


def infer_batch_l(in_list):
    """
    Wrapper for infer_batch to process a list of parameters.

    Parameters:
    - in_list (list): A list of 3 parameters in order:
        - [0]: Path to the model file (.pth).
        - [1]: Path to the folder containing input images.
        - [2]: Path to the output folder.

    Raises:
    - ValueError: If the input list does not contain exactly 4 elements.
    """
    if len(in_list) != 3:
        raise ValueError("Expected a list of 3 parameters: [model_path, input_folder, output_folder]")

    model_path, input_folder, output_folder = in_list
    infer_batch_yolo(model_path[0], input_folder, output_folder)
