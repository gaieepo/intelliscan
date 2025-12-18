#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D-IntelliScan Version Information

This module contains centralized version information for the 3D-IntelliScan
semiconductor metrology and defect detection pipeline.

Author: Wang Jie
Project: SWIP-ISDS-2025-12
"""

from datetime import datetime
from typing import Dict, Any

# Version information
__version__ = "2.1.0"
__version_info__ = (2, 1, 0)

# Build information
__build_date__ = "2025-09-19"
__build_time__ = "20:00:00"
__commit_hash__ = "latest"  # Will be updated by build process

# Project information
__project__ = "3D-IntelliScan"
__project_code__ = "SWIP-ISDS-2025-12"
__author__ = "Wang Jie"
__organization__ = "A*STAR I2R"
__license__ = "Proprietary"

# Component versions
__component_versions__ = {
    "core_pipeline": "2.1.0",
    "web_interface": "2.1.0",
    "metrology": "2.1.0",
    "detection": "2.1.0",
    "segmentation": "2.1.0",
    "reporting": "2.1.0"
}

# Python version requirements
__python_min_version__ = (3, 8)
__python_recommended_version__ = (3, 10)

# Release information
__release_notes__ = {
    "2.1.0": {
        "date": "2025-09-19",
        "type": "minor",
        "features": [
            "Comprehensive test suite implementation",
            "Fixed validation test issues",
            "Updated documentation",
            "Centralized version management"
        ],
        "fixes": [
            "BLT measurement validation",
            "NaN data handling in tests",
            "File I/O error handling"
        ],
        "breaking_changes": []
    },
    "2.0.0": {
        "date": "2025-08-01",
        "type": "major",
        "features": [
            "Complete pipeline implementation",
            "Streamlit web interface",
            "3D visualization with PyVista",
            "PDF report generation"
        ],
        "fixes": [],
        "breaking_changes": [
            "New API structure",
            "Updated configuration format"
        ]
    }
}


def get_version() -> str:
    """Get the current version string."""
    return __version__


def get_version_info() -> tuple:
    """Get the version as a tuple of integers."""
    return __version_info__


def get_build_info() -> Dict[str, str]:
    """Get build information."""
    return {
        "version": __version__,
        "build_date": __build_date__,
        "build_time": __build_time__,
        "commit_hash": __commit_hash__,
        "python_version": f"{__python_min_version__[0]}.{__python_min_version__[1]}+"
    }


def get_full_info() -> Dict[str, Any]:
    """Get complete version and project information."""
    return {
        "project": __project__,
        "project_code": __project_code__,
        "version": __version__,
        "version_info": __version_info__,
        "build_date": __build_date__,
        "build_time": __build_time__,
        "commit_hash": __commit_hash__,
        "author": __author__,
        "organization": __organization__,
        "license": __license__,
        "components": __component_versions__,
        "python_requirements": {
            "minimum": f"{__python_min_version__[0]}.{__python_min_version__[1]}",
            "recommended": f"{__python_recommended_version__[0]}.{__python_recommended_version__[1]}"
        }
    }


def print_version() -> None:
    """Print formatted version information."""
    info = get_full_info()
    print(f"""
╔═══════════════════════════════════════════════════════════════╗
║                      {info['project']}                      ║
║                   Semiconductor Metrology                    ║
╠═══════════════════════════════════════════════════════════════╣
║ Version:      {info['version']:<20} Build: {info['build_date']}     ║
║ Project:      {info['project_code']:<45} ║
║ Author:       {info['author']:<45} ║
║ Organization: {info['organization']:<45} ║
╚═══════════════════════════════════════════════════════════════╝
    """.strip())


def print_component_versions() -> None:
    """Print component version information."""
    print(f"\n🔧 Component Versions (v{__version__}):")
    print("─" * 40)
    for component, version in __component_versions__.items():
        print(f"  {component:<15}: {version}")


def check_python_version() -> tuple[bool, str]:
    """Check if current Python version meets requirements."""
    import sys
    current = sys.version_info[:2]

    if current < __python_min_version__:
        return False, f"Python {current[0]}.{current[1]} is too old. Minimum required: {__python_min_version__[0]}.{__python_min_version__[1]}"
    elif current < __python_recommended_version__:
        return True, f"Python {current[0]}.{current[1]} works but {__python_recommended_version__[0]}.{__python_recommended_version__[1]}+ is recommended"
    else:
        return True, f"Python {current[0]}.{current[1]} ✓"


def get_release_notes(version: str = None) -> Dict[str, Any]:
    """Get release notes for a specific version or latest."""
    if version is None:
        version = __version__

    return __release_notes__.get(version, {
        "date": "Unknown",
        "type": "unknown",
        "features": [],
        "fixes": [],
        "breaking_changes": []
    })


if __name__ == "__main__":
    # Command line version display
    print_version()
    print_component_versions()

    # Python version check
    valid, message = check_python_version()
    print(f"\n🐍 Python Version: {message}")

    # Recent release notes
    latest_notes = get_release_notes()
    print(f"\n📋 Release Notes (v{__version__}):")
    print(f"   Release Date: {latest_notes['date']}")
    if latest_notes.get('features'):
        print("   New Features:")
        for feature in latest_notes['features']:
            print(f"     • {feature}")