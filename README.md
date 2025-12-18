# 3D-IntelliScan SWIP

**AI Solution for 3D Metrology and Defect Detection for Semiconductor Manufacturing**

## Overview

3D-IntelliScan SWIP is a comprehensive AI-powered solution designed for 3D metrology and defect detection in semiconductor manufacturing. The system processes 3D NIfTI medical imaging data to identify and analyze defects such as bond line thickness issues, void ratios, solder extrusion, and pad misalignment.

## Features

- **3D NIfTI Processing**: Convert 3D medical imaging data to 2D slices for analysis
- **Object Detection**: AI-powered detection of defects using YOLO models
- **3D Bounding Box Generation**: Convert 2D detections to 3D bounding boxes
- **3D Segmentation**: Deep learning-based segmentation using VNet models
- **Metrology Analysis**: Automated measurement and defect quantification
- **Report Generation**: PDF reports with visualizations and analysis
- **Web Interface**: Streamlit-based web application for easy interaction
- **Batch Processing**: Process multiple files efficiently

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for deep learning models)
- 8GB+ RAM

### Dependencies

The code requires the same external dependencies as the original codebase:
- nibabel
- numpy
- pandas
- torch
- ultralytics
- opencv-python
- PIL
- sklearn
- matplotlib
- seaborn
- absl-py
- easydict
- anthropic

### Installation

```bash
# Clone the repository
git clone ssh://git@gitlab.i2r.a-star.edu.sg:10022/wangj2/swip-isds-2025-12.git
cd swip-isds-2025-12

# Install dependencies
pip install -r requirements.txt
```

## Demo Data and Models

⚠️ **Important**: The demo data and pre-trained models are not included in this repository due to file size constraints. You must download them separately before running the code.

### Download Required Files

Download the demo data and models from Google Drive:

📦 **[Download Demo Data & Models](https://drive.google.com/drive/folders/1it_f_k5B5sPC40TUz6lsJ8r27LRypFta?usp=sharing)**

The download includes:
- `code/data/` - Sample NIfTI files for testing and demonstration
- `code/models/` - Pre-trained detection and segmentation models
  - `detection_model.pt` - YOLO detection model
  - `segmentation_model.pth` - VNet segmentation model

### Setup Instructions

1. **Download** the files from the Google Drive link above
2. **Extract** the downloaded archive to the project root directory
3. **Verify** the folder structure matches:

```
swip-isds-2025-12/
├── code/
│   ├── data/                    # ← Downloaded demo data
│   │   ├── sample1.nii.gz
│   │   ├── sample2.nii.gz
│   │   └── ...
│   ├── models/                  # ← Downloaded pre-trained models
│   │   ├── detection_model.pt
│   │   ├── segmentation_model.pth
│   │   └── ...
│   ├── main.py
│   └── ...
└── ...
```

4. **Update** `code/files.txt` to point to your sample data files:

```bash
# Example content for files.txt
data/sample1.nii.gz
data/sample2.nii.gz
```

### File Size Information

- **Demo Data**: ~500MB (3-5 sample NIfTI files)
- **Models**: ~200MB (detection + segmentation models)
- **Total Download**: ~700MB

⚠️ **Note**: Make sure you have sufficient disk space and a stable internet connection for the download.

## Quick Start

### Command Line Interface

```bash
# Process NIfTI files under "code/files.txt"
cd code
python main.py
```

### Web-GUI Interface

```bash
# Launch the web interface
cd code
streamlit run demo.py
```

## Project Structure

```
SWIP/
├── code/                          # Main source code
├── docs/                          # Documentation
├── tests/                         # Test suite
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

Module overview:
- `main.py` - Main entry point
- `demo.py` - Streamlit Web-GUI
- `conv_nii_jpg.py` - NII to JPG conversion utilities
- `generate_bb3d_2.py` - 3D bounding box generation
- `generate_report.py` - PDF report generation
- `infer_batch_new.py` - Batch inference functionality
- `mmt.py` - Multi-modal transformer model
- `mmt_util.py` - MMT utilities
- `metrology/` - Metrology processing modules
- `networks/` - Neural network architectures
- `configs/` - Configuration files
- `models/` - Pre-trained model files
- `files.txt` - Input file list

## Pipeline Overview

The 3D-IntelliScan pipeline consists of several key stages:

1. **NIfTI to JPG Conversion**: Convert 3D NIfTI files to 2D JPG slices
2. **Object Detection**: Detect defects using trained YOLO models
3. **3D Bounding Box Generation**: Convert 2D detections to 3D coordinates
4. **3D Segmentation**: Perform detailed segmentation using VNet models
5. **Metrology Analysis**: Measure and quantify defects
6. **Report Generation**: Create comprehensive PDF reports

## Configuration

### Detection Configuration

Create a YAML configuration file for detection settings under `code/configs`

### Metrology Configuration

Configure metrology parameters in `code/metrology/config.py`:

```python
class MetrologyConfig:
    PIXEL_SIZE_UM = 1.0  # Pixel size in micrometers
    NUM_DECIMALS = 3     # Number of decimal places for measurements
    MAKE_CLEAN = True    # Generate cleaned segmentation masks
```

## Model Requirements

### Detection Models

- **Format**: YOLO format (.pt files)
- **Input**: 2D JPG images
- **Output**: Bounding box coordinates

### Segmentation Models

- **Format**: PyTorch (.pth files)
- **Architecture**: VNet with pyramid structure
- **Input**: 3D NIfTI data with bounding boxes
- **Output**: Segmented 3D volumes

## Output Structure

```
output/
├── view1/                          # First view results
│   ├── input_images/              # Converted JPG images
│   ├── detections/                # Detection results
│   └── visualize/                 # Visualization images
├── view2/                          # Second view results
│   ├── input_images/              # Converted JPG images
│   ├── detections/                # Detection results
│   └── visualize/                 # Visualization images
├── bb3d.npy                       # 3D bounding boxes
├── class_0_segmentation.nii.gz    # Segmentation results
├── mmt/                           # Segmentation details
│   ├── img/                       # Input images for segmentation
│   └── pred/                      # Segmentation predictions
├── metrology/                     # Metrology analysis
│   ├── memory.csv                 # Measurement results
│   └── memory_report.pdf          # Generated report
└── timing.log                     # Processing timestamps
```

## Testing

The SWIP package includes a comprehensive test suite with detailed documentation.

### Quick Test Execution

```bash
# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/test_pipeline.py -v

# Run with coverage
pytest tests/ --cov=code --cov-report=html
```

### Advanced Test Options

Use the test runner script for more options:

```bash
cd tests/

# Check test environment
python run_tests.py --check-env

# Run all tests with coverage
python run_tests.py --coverage

# Run specific test suite
python run_tests.py --suite pipeline

# Run specific test case
python run_tests.py --test STC-1.1

# Generate comprehensive test report
python run_tests.py --report
```

### Test Data Generation

Generate realistic test data for testing:

```bash
cd tests/

# Generate all test data
python generate_test_data.py

# Generate with custom output directory
python generate_test_data.py --output my_test_data
```

### Test Documentation

Comprehensive test documentation is available in `docs/test_documentation.md`, including:

- **Test Case ID Scheme**: STC-{Suite}.{Case} numbering system
- **Detailed Test Cases**: Pre-conditions, test data, steps, and expected results
- **Test Categories**: Import tests, functionality tests, error handling, integration tests
- **Test Environment Requirements**: Dependencies, system requirements
- **Quality Metrics**: Coverage targets, execution time, reliability standards

### Test Structure

```
tests/
├── test_pipeline.py          # Main test suite
├── run_tests.py             # Test execution script
├── generate_test_data.py    # Test data generator
└── __init__.py
```

**Test Suites:**
- **STC-1**: Pipeline Core Functionality (4 test cases)
- **STC-2**: Utility Functions (3 test cases)
- **STC-3**: Visualization Functions (3 test cases)
- **STC-4**: Metrology Analysis (2 test cases)

**Total: 12 test cases** covering all major functionality.

## Citation

If you use this software in your research, please cite:

```bibtex
@INPROCEEDINGS{10909770,
  author={Wang, Jie and Chang, Richard and Lim, Meng Keong and Chong, Ser Choong and Yang, Xulei and Pahwa, Ramanpreet Singh},
    booktitle={2024 IEEE 26th Electronics Packaging Technology Conference (EPTC)},
      title={End-to-End Fast Segmentation Framework for 3D Visual Inspection of HBMs},
        year={2024},
          volume={},
            number={},
              pages={1102-1107},
                keywords={Visualization;Analytical models;Three-dimensional displays;Manuals;Inspection;Probabilistic logic;Software;Data models;Load modeling;Defect detection},
                  doi={10.1109/EPTC62800.2024.10909770}}

```
