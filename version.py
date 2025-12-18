#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3D-IntelliScan Version Utility

Command-line utility for displaying version information, checking dependencies,
and system compatibility for the 3D-IntelliScan pipeline.

Usage:
    python version.py                    # Display basic version
    python version.py --full            # Display full information
    python version.py --check           # System compatibility check
    python version.py --components      # Component versions
    python version.py --release-notes   # Release notes

Author: Wang Jie
Project: SWIP-ISDS-2025-12
"""

import argparse
import sys
from pathlib import Path

# Import version information
from _version import (
    get_version, get_version_info, get_full_info, get_build_info,
    print_version, print_component_versions, check_python_version,
    get_release_notes, __version__, __project__, __author__
)


def check_dependencies():
    """Check if required dependencies are available."""
    required_packages = [
        ('numpy', 'numpy'),
        ('nibabel', 'nibabel'),
        ('pandas', 'pandas'),
        ('streamlit', 'streamlit'),
        ('pyvista', 'pyvista'),
        ('matplotlib', 'matplotlib'),
        ('PIL', 'Pillow'),
        ('yaml', 'PyYAML'),
        ('cv2', 'opencv-python')
    ]

    missing = []
    available = []

    for module_name, package_name in required_packages:
        try:
            __import__(module_name)
            available.append(f"✅ {package_name}")
        except ImportError:
            missing.append(f"❌ {package_name}")

    print("📦 Dependency Status:")
    print("─" * 30)
    for pkg in available:
        print(f"  {pkg}")
    for pkg in missing:
        print(f"  {pkg}")

    if missing:
        print(f"\n⚠️  Missing {len(missing)} required packages")
        print("Install with: conda activate cdfdemo")
    else:
        print(f"\n✅ All {len(available)} dependencies available")


def check_system_compatibility():
    """Perform comprehensive system compatibility check."""
    print(f"🔍 System Compatibility Check for {__project__} v{__version__}")
    print("=" * 60)

    # Python version
    valid_python, py_message = check_python_version()
    print(f"🐍 {py_message}")

    # Dependencies
    print()
    check_dependencies()

    # File structure check
    print(f"\n📁 File Structure:")
    current_dir = Path(__file__).parent
    required_files = [
        "_version.py",
        "main.py",
        "demo.py",
        "conv_nii_jpg.py",
        "generate_report.py",
        "metrology/__init__.py",
        "networks/__init__.py"
    ]

    missing_files = []
    for file_path in required_files:
        full_path = current_dir / file_path
        if full_path.exists():
            print(f"  ✅ {file_path}")
        else:
            print(f"  ❌ {file_path}")
            missing_files.append(file_path)

    # Summary
    print("\n" + "=" * 60)
    if valid_python and not missing_files:
        print("🎉 System is compatible and ready to run!")
    else:
        print("⚠️  Issues found. Please address the above problems.")


def show_release_notes(version=None):
    """Display release notes for specified version."""
    notes = get_release_notes(version)
    display_version = version or __version__

    print(f"📋 Release Notes - v{display_version}")
    print("=" * 50)
    print(f"Release Date: {notes['date']}")
    print(f"Release Type: {notes['type']}")

    if notes.get('features'):
        print(f"\n🆕 New Features ({len(notes['features'])}):")
        for i, feature in enumerate(notes['features'], 1):
            print(f"  {i}. {feature}")

    if notes.get('fixes'):
        print(f"\n🔧 Bug Fixes ({len(notes['fixes'])}):")
        for i, fix in enumerate(notes['fixes'], 1):
            print(f"  {i}. {fix}")

    if notes.get('breaking_changes'):
        print(f"\n⚠️  Breaking Changes ({len(notes['breaking_changes'])}):")
        for i, change in enumerate(notes['breaking_changes'], 1):
            print(f"  {i}. {change}")


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description=f"{__project__} Version Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python version.py                 # Basic version info
  python version.py --full          # Complete information
  python version.py --check         # Compatibility check
  python version.py --components    # Component versions
  python version.py --release-notes # Release notes

Project: SWIP-ISDS-2025-12
Author: {__author__}
        """
    )

    parser.add_argument(
        '--full', '-f',
        action='store_true',
        help='Display full version and build information'
    )

    parser.add_argument(
        '--check', '-c',
        action='store_true',
        help='Perform system compatibility check'
    )

    parser.add_argument(
        '--components', '--comp',
        action='store_true',
        help='Display component version information'
    )

    parser.add_argument(
        '--release-notes', '--notes',
        nargs='?',
        const=__version__,
        help='Show release notes (optionally for specific version)'
    )

    parser.add_argument(
        '--json',
        action='store_true',
        help='Output version information in JSON format'
    )

    args = parser.parse_args()

    # Handle different options
    if args.check:
        check_system_compatibility()
    elif args.release_notes:
        show_release_notes(args.release_notes)
    elif args.components:
        print_version()
        print_component_versions()
    elif args.full:
        info = get_full_info()
        if args.json:
            import json
            print(json.dumps(info, indent=2))
        else:
            print_version()
            print_component_versions()
            print(f"\n🔧 Build Information:")
            build_info = get_build_info()
            for key, value in build_info.items():
                print(f"  {key}: {value}")
    elif args.json:
        import json
        info = get_full_info()
        print(json.dumps(info, indent=2))
    else:
        # Default: basic version information
        print(f"{__project__} v{get_version()}")

        # Show Python version check
        valid, message = check_python_version()
        print(message)


if __name__ == "__main__":
    main()