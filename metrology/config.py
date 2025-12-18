#!/usr/bin/env python3
"""
Metrology Configuration Settings

This module defines configuration parameters for the metrology analysis
subsystem, including pixel dimensions, thresholds, output settings, and
column specifications for different die types.

Key Configuration Groups:
- METROLOGY: Main metrology analysis parameters
- Column definitions for memory and logic dies
- Pixel scaling and precision settings

Author: Wang Jie
Date: 1st Aug 2025
Version: 1.0
"""

from easydict import EasyDict as edict

__C = edict()
cfg = __C
# Consumers can get config by: from metrology.config import cfg

# Main script
# fmt:off
__C.METROLOGY = edict()
__C.METROLOGY.INPUT_FOLDER = "./input"
__C.METROLOGY.OUTPUT_FOLDER = "./output"

# fmt:on
__C.METROLOGY.PIXEL_SIZE_UM = 0.7
__C.METROLOGY.NUM_OUTS = 1
__C.METROLOGY.NUM_DECIMALS = 2
__C.METROLOGY.MAKE_CLEAN = True
__C.METROLOGY.CLEAN_OUT_PATH = "./cleaned_output/"
# Column specifications for different die types
__C.METROLOGY.METROLOGY_COLUMNS_MEMORY_DIE = [
    "filename",
    "BLT",
    "Pad_misalignment",
    "Void_to_solder_ratio",
    "solder_extrusion_copper_pillar",
    "solder_extrusion_copper_pad",
]
__C.METROLOGY.METROLOGY_COLUMNS_LOGIC_DIE = [
    "filename",
    "BLT",
    "Pad_misalignment",
    "Void_to_solder_ratio",
    "solder_extrusion_copper_pillar",
]
