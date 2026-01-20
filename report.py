#!/usr/bin/env python3
"""
Automated PDF Report Generator

This module generates comprehensive PDF reports for semiconductor metrology
analysis results. It creates multi-page reports with statistical visualizations,
defect analysis, and detailed cross-sectional views of critical defects.

Features:
- Automated statistical analysis and visualization
- Defect identification and ranking
- Cross-sectional image generation with segmentation overlays
- AI-powered analysis summaries using Claude API
- Multi-page PDF generation with professional formatting
"""

import ast
import os
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PyPDF2 import PdfReader, PdfWriter

# A4 size in inches
A4_WIDTH = 8.27
A4_HEIGHT = 11.69


def add_header_footer(fig, page_num, total_pages, filename, swip_code):
    """Add header and footer to the figure."""
    # Add header
    fig.text(0.5, 0.95, f"Report of sample {filename}", ha="center", fontsize=12, weight="bold")
    fig.text(0.5, 0.92, f"Generated with {swip_code}", ha="center", fontsize=10)

    # Add footer with page number
    fig.text(0.5, 0.02, f"Page {page_num} of {total_pages}", ha="center", fontsize=8)
    fig.text(0.02, 0.02, f"Generated on {datetime.now().strftime('%Y-%m-%d')}", ha="left", fontsize=8)
    fig.text(0.98, 0.02, "Confidential", ha="right", fontsize=8)


def get_slice(img_data, pred_data):
    small_axis = img_data.shape.index(min(img_data.shape))
    mid_idx = img_data.shape[small_axis] // 2

    raw_slice0 = img_data[:, mid_idx, :]
    seg_slice0 = pred_data[:, mid_idx, :]

    raw_slice1 = img_data[:, :, mid_idx]
    seg_slice1 = pred_data[:, :, mid_idx]

    # normalized for display
    raw_normalized0 = (raw_slice0 - raw_slice0.min()) / (raw_slice0.max() - raw_slice0.min())
    masked_seg0 = np.ma.masked_where(seg_slice0 == 0, seg_slice0)

    raw_normalized1 = (raw_slice1 - raw_slice1.min()) / (raw_slice1.max() - raw_slice1.min())
    masked_seg1 = np.ma.masked_where(seg_slice1 == 0, seg_slice1)

    # Calculate padding needed to make both views square
    max_dim = max(max(raw_slice0.shape), max(raw_slice1.shape))
    pad_h0 = (max_dim - raw_slice0.shape[0]) // 2
    pad_w0 = (max_dim - raw_slice0.shape[1]) // 2
    pad_h1 = (max_dim - raw_slice1.shape[0]) // 2
    pad_w1 = (max_dim - raw_slice1.shape[1]) // 2

    # Pad both views to make them square
    raw_normalized0 = np.pad(raw_normalized0, ((pad_h0, pad_h0), (pad_w0, pad_w0)), mode="constant", constant_values=0)
    masked_seg0 = np.pad(masked_seg0, ((pad_h0, pad_h0), (pad_w0, pad_w0)), mode="constant", constant_values=0)
    raw_normalized1 = np.pad(raw_normalized1, ((pad_h1, pad_h1), (pad_w1, pad_w1)), mode="constant", constant_values=0)
    masked_seg1 = np.pad(masked_seg1, ((pad_h1, pad_h1), (pad_w1, pad_w1)), mode="constant", constant_values=0)

    return masked_seg0, raw_normalized0, masked_seg1, raw_normalized1


def generate_pdf_report(
    csv_path,
    # chroma_client,
    output_path="report.pdf",
    swip_code="SWIP-2025-011",
    input_filename=None,
):
    """
    Generate PDF report with analysis of semiconductor measurements.

    Parameters:
    - csv_path: path to input CSV file
    - output_path: path for output PDF
    - input_filename: name of the input file to use in report title
    """
    # Read CSV
    df = pd.read_csv(csv_path)

    # Convert string representation of lists to actual lists
    df["solder_extrusion_copper_pillar"] = df["solder_extrusion_copper_pillar"].apply(ast.literal_eval)

    # Get filename from path if not provided
    if input_filename is None:
        filename = os.path.basename(csv_path).replace("_memory.csv", "")
    else:
        filename = os.path.basename(input_filename)

    # Create PDF
    with PdfPages(output_path) as pdf:
        # 1. Cover Page
        fig = plt.figure(figsize=(A4_WIDTH, A4_HEIGHT))
        plt.axis("off")

        # Add title and information
        plt.text(0.5, 0.7, f"Report of sample {filename}", ha="center", fontsize=16, weight="bold")
        plt.text(0.5, 0.6, f"Generated with {swip_code}", ha="center", fontsize=12)
        plt.text(0.5, 0.5, f"Created on {datetime.now().strftime('%Y-%m-%d')}", ha="center", fontsize=12)

        pdf.savefig(fig)
        plt.close(fig)

        # 2. Current plot page
        fig = plt.figure(figsize=(A4_WIDTH, A4_HEIGHT))
        n_rows, n_cols = 2, 3
        gs = fig.add_gridspec(n_rows, n_cols, top=0.85, bottom=0.1, left=0.1, right=0.9)
        axes = [fig.add_subplot(gs[i, j]) for i in range(n_rows) for j in range(n_cols)]

        # Add header and footer
        add_header_footer(fig, 2, 4, filename, swip_code)

        # 1. BLT Distribution
        """
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df["BLT"], fill=True)
        plt.title("BLT Distribution")
        plt.xlabel("BLT Value (μm)")
        plt.ylabel("Density")
        pdf.savefig()
        plt.close()
        """
        ax = axes[0]
        sns.histplot(data=df["BLT"], ax=ax)
        ax.set_title("BLT Distribution")
        ax.set_xlabel("BLT Value (μm)")
        ax.set_ylabel("Count")

        # 2. Pillar Dimensions
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        sns.boxplot(y=df["pillar_width"], ax=ax1)
        ax1.set_title("Pillar Width Distribution")
        ax1.set_ylabel("Width (μm)")

        sns.boxplot(y=df["pillar_height"], ax=ax2)
        ax2.set_title("Pillar Height Distribution")
        ax2.set_ylabel("Height (μm)")

        plt.tight_layout()
        pdf.savefig()
        plt.close()
        """
        ax = axes[1]
        sns.boxplot(y=df["pillar_width"], ax=ax)
        ax.set_title("Pillar Width Distribution")
        ax.set_ylabel("Width (μm)")

        ax = axes[2]
        sns.boxplot(y=df["pillar_height"], ax=ax)
        ax.set_title("Pillar Height Distribution")
        ax.set_ylabel("Height (μm)")

        # 3. Defect Analysis
        """
        plt.figure(figsize=(10, 6))
        defect_counts = {
            'Pad Misalignment': df['pad_misalignment_defect'].sum(),
            'Void Ratio': df['void_ratio_defect'].sum(),
            'Solder Extrusion': df['solder_extrusion_defect'].sum()
        }

        plt.bar(defect_counts.keys(), defect_counts.values())
        plt.title("Defect Count Analysis")
        plt.ylabel("Number of Defects")
        plt.xticks(rotation=45)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
        """
        defect_counts = {
            "Pad Misalignment": df["pad_misalignment_defect"].sum(),
            "Void Ratio": df["void_ratio_defect"].sum(),
            "Solder Extrusion": df["solder_extrusion_defect"].sum(),
        }
        ax = axes[3]
        bars = ax.bar(range(len(defect_counts)), defect_counts.values())
        ax.set_title("Defect Count Analysis")
        ax.set_ylabel("Number of Defects")
        ax.set_xticks(range(len(defect_counts)))
        ax.set_xticklabels(["Pad\nMisalignment", "Void\nRatio", "Solder\nExtrusion"], rotation=0, fontsize=8)

        # 4. Pad Misalignment Distribution
        """
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df["Pad_misalignment"], fill=True)
        plt.title("Pad Misalignment Distribution")
        plt.xlabel("Misalignment (μm)")
        plt.ylabel("Density")
        pdf.savefig()
        plt.close()
        """
        ax = axes[4]
        sns.histplot(data=df["Pad_misalignment"], ax=ax)
        ax.set_title("Pad Misalignment")
        ax.set_xlabel("Misalignment (μm)")
        ax.set_ylabel("Count")

        # 5. Void to Solder Ratio Distribution
        """
        plt.figure(figsize=(10, 6))
        sns.kdeplot(data=df["Void_to_solder_ratio"], fill=True)
        plt.title("Void to Solder Ratio Distribution")
        plt.xlabel("Ratio")
        plt.ylabel("Density")
        pdf.savefig()
        plt.close()
        """
        ax = axes[5]
        sns.histplot(data=df["Void_to_solder_ratio"], bins=10, ax=ax)
        ax.set_title("Void to Solder Ratio")
        ax.set_xlabel("Ratio")
        ax.set_ylabel("Count")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 3. Biggest defects page
        fig = plt.figure(figsize=(A4_WIDTH, A4_HEIGHT))
        gs = fig.add_gridspec(
            3, 2, top=0.85, bottom=0.1, left=0.1, right=0.9, height_ratios=[1, 1, 1], width_ratios=[1.2, 1]
        )
        axes = [fig.add_subplot(gs[i, 0]) for i in range(3)]
        text_axes = [fig.add_subplot(gs[i, 1]) for i in range(3)]
        ind_gr = 0
        colors = ["black", "red", "green", "blue", "yellow"]
        cmap = ListedColormap(colors)

        # Add header and footer
        add_header_footer(fig, 3, 4, filename, swip_code)

        # Define thresholds
        thresholds = {
            "Pad Misalignment": 4.0,  # μm
            "Void Ratio": 0.1,  # ratio
            "Solder Extrusion": 0.5,  # ratio
        }

        # Find and display biggest pad misalignment
        pm = np.array(df["pad_misalignment_defect"])
        if np.any(pm):
            pm = np.array(df["Pad_misalignment"])
            ind = np.nanargmax(pm)
            path = Path(csv_path)
            path = path.parent.parent

            filename_img = os.path.join(path, "mmt/img/" + df["filename"][ind])
            img = nib.load(filename_img.replace("pred", "img"))
            pred = nib.load(filename_img.replace("img", "pred"))

            img_data = img.get_fdata()
            pred_data = pred.get_fdata()

            masked_seg0, raw_normalized0, masked_seg1, raw_normalized1 = get_slice(img_data, pred_data)

            ax = axes[ind_gr]
            inset1 = inset_axes(ax, width="45%", height="80%", loc="upper left")
            inset1.imshow(raw_normalized0, cmap="gray", aspect="equal")
            inset1.imshow(masked_seg0, cmap=cmap, alpha=0.5, vmin=0, vmax=4, aspect="equal")
            inset1.axis("off")
            inset2 = inset_axes(ax, width="45%", height="80%", loc="upper right")
            inset2.imshow(raw_normalized1, cmap="gray", aspect="equal")
            inset2.imshow(masked_seg1, cmap=cmap, alpha=0.5, vmin=0, vmax=4, aspect="equal")
            inset2.axis("off")
            inset1.set_title("view0", fontsize=8)
            inset2.set_title("view1", fontsize=8)

            ax.set_title(r"$\mathbf{Pad\ misalignment}$")
            ax.axis("off")

            # Add text in separate axis
            text_ax = text_axes[ind_gr]
            text_ax.axis("off")
            max_val = np.nanmax(pm)
            defect_text = f"""
            • Description: Pad Misalignment
            • Filename: {df["filename"][ind]}
            • Value: {max_val:.2f} μm
            • Threshold: {thresholds["Pad Misalignment"]} μm
            """
            text_ax.text(0, 0.5, defect_text, verticalalignment="center", horizontalalignment="left", fontsize=10)
            ind_gr += 1

        # Biggest Solder extrusion
        se = np.array(df["solder_extrusion_defect"])
        if np.any(se):
            se = np.array(df["solder_extrusion_copper_pillar"].tolist())
            se_sum = np.sum(se, axis=1)
            ind = np.argmax(se_sum)

            path = Path(csv_path)
            path = path.parent.parent
            filename_img = os.path.join(path, "mmt/img/" + df["filename"][ind])
            img = nib.load(filename_img.replace("pred", "img"))
            pred = nib.load(filename_img.replace("img", "pred"))

            img_data = img.get_fdata()
            pred_data = pred.get_fdata()

            masked_seg0, raw_normalized0, masked_seg1, raw_normalized1 = get_slice(img_data, pred_data)

            ax = axes[ind_gr]
            inset1 = inset_axes(ax, width="45%", height="80%", loc="upper left")
            inset1.imshow(raw_normalized0, cmap="gray", aspect="equal")
            inset1.imshow(masked_seg0, cmap=cmap, alpha=0.5, vmin=0, vmax=4, aspect="equal")
            inset1.axis("off")
            inset2 = inset_axes(ax, width="45%", height="80%", loc="upper right")
            inset2.imshow(raw_normalized1, cmap="gray", aspect="equal")
            inset2.imshow(masked_seg1, cmap=cmap, alpha=0.5, vmin=0, vmax=4, aspect="equal")
            inset2.axis("off")
            inset1.set_title("view0", fontsize=8)
            inset2.set_title("view1", fontsize=8)

            ax.set_title(r"$\mathbf{Solder\ Extrusion}$")
            ax.axis("off")

            # Add text in separate axis
            text_ax = text_axes[ind_gr]
            text_ax.axis("off")
            max_val = np.nanmax(se_sum)
            defect_text = f"""
            • Description: Solder Extrusion
            • Filename: {df["filename"][ind]}
            • Value: {max_val:.2f}
            • Threshold: {thresholds["Solder Extrusion"]}
            """
            text_ax.text(0, 0.5, defect_text, verticalalignment="center", horizontalalignment="left", fontsize=10)
            ind_gr += 1

        # Biggest void
        vd = np.array(df["void_ratio_defect"])
        if np.any(vd) or True:  # Temporarily force display for debugging
            vd = np.array(df["Void_to_solder_ratio"])
            ind = np.nanargmax(vd)

            path = Path(csv_path)
            path = path.parent.parent
            filename_img = os.path.join(path, "mmt/img/" + df["filename"][ind])
            img = nib.load(filename_img.replace("pred", "img"))
            pred = nib.load(filename_img.replace("img", "pred"))

            img_data = img.get_fdata()
            pred_data = pred.get_fdata()

            masked_seg0, raw_normalized0, masked_seg1, raw_normalized1 = get_slice(img_data, pred_data)

            ax = axes[ind_gr]
            inset1 = inset_axes(ax, width="45%", height="80%", loc="upper left")
            inset1.imshow(raw_normalized0, cmap="gray", aspect="equal")
            inset1.imshow(masked_seg0, cmap=cmap, alpha=0.5, vmin=0, vmax=4, aspect="equal")
            inset1.axis("off")
            inset2 = inset_axes(ax, width="45%", height="80%", loc="upper right")
            inset2.imshow(raw_normalized1, cmap="gray", aspect="equal")
            inset2.imshow(masked_seg1, cmap=cmap, alpha=0.5, vmin=0, vmax=4, aspect="equal")
            inset2.axis("off")
            inset1.set_title("view0", fontsize=8)
            inset2.set_title("view1", fontsize=8)

            ax.set_title(r"$\mathbf{Void}$")
            ax.axis("off")

            # Add text in separate axis
            text_ax = text_axes[ind_gr]
            text_ax.axis("off")
            max_val = np.nanmax(vd)
            defect_text = f"""
            • Description: Void Ratio
            • Filename: {df["filename"][ind]}
            • Value: {max_val:.2f}
            • Threshold: {thresholds["Void Ratio"]}
            """
            text_ax.text(0, 0.5, defect_text, verticalalignment="center", horizontalalignment="left", fontsize=10)
            ind_gr += 1

        # Fill remaining axes if needed
        for i in range(ind_gr, 3):
            axes[i].axis("off")
            text_axes[i].axis("off")

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)

        # 4. Summary Statistics Table
        fig = plt.figure(figsize=(A4_WIDTH, A4_HEIGHT))
        plt.axis("off")

        # Add header and footer
        add_header_footer(fig, 4, 4, filename, swip_code)

        # Create a table-like display for BLT and Pillar Dimensions
        table_data = [
            ["Measurement", "Mean", "Std", "Min", "Max"],
            [
                "BLT (μm)",
                f"{df['BLT'].mean():.2f}",
                f"{df['BLT'].std():.2f}",
                f"{df['BLT'].min():.2f}",
                f"{df['BLT'].max():.2f}",
            ],
            [
                "Pillar Width (μm)",
                f"{df['pillar_width'].mean():.2f}",
                f"{df['pillar_width'].std():.2f}",
                f"{df['pillar_width'].min():.2f}",
                f"{df['pillar_width'].max():.2f}",
            ],
            [
                "Pillar Height (μm)",
                f"{df['pillar_height'].mean():.2f}",
                f"{df['pillar_height'].std():.2f}",
                f"{df['pillar_height'].min():.2f}",
                f"{df['pillar_height'].max():.2f}",
            ],
        ]

        # Create the measurement table
        measurement_table = plt.table(
            cellText=table_data,
            loc="center",
            cellLoc="center",
            colWidths=[0.3, 0.15, 0.15, 0.15, 0.15],
            bbox=[0.1, 0.5, 0.8, 0.3],
        )
        measurement_table.auto_set_font_size(False)
        measurement_table.set_fontsize(10)
        measurement_table.scale(1, 1.5)

        # Create defect counts table
        defect_table_data = [
            ["Defect Type", "Count", "Samples Analyzed", "Defective %"],
            [
                "Pad Misalignment",
                f"{defect_counts['Pad Misalignment']}",
                f"{len(df)}",
                f"{defect_counts['Pad Misalignment'] / len(df) * 100:.1f}%",
            ],
            [
                "Void Ratio",
                f"{defect_counts['Void Ratio']}",
                f"{len(df)}",
                f"{defect_counts['Void Ratio'] / len(df) * 100:.1f}%",
            ],
            [
                "Solder Extrusion",
                f"{defect_counts['Solder Extrusion']}",
                f"{len(df)}",
                f"{defect_counts['Solder Extrusion'] / len(df) * 100:.1f}%",
            ],
        ]

        # Create the defect table
        defect_table = plt.table(
            cellText=defect_table_data,
            loc="center",
            cellLoc="center",
            colWidths=[0.3, 0.2, 0.2, 0.2],
            bbox=[0.1, 0.2, 0.8, 0.2],
        )
        defect_table.auto_set_font_size(False)
        defect_table.set_fontsize(10)
        defect_table.scale(1, 1.5)

        plt.title("Summary Statistics", fontsize=14, pad=20)
        pdf.savefig(fig)
        plt.close(fig)

    # --- 2. open both files and merge ---
    writer = PdfWriter()
    fold = os.path.dirname(csv_path)

    # add pages created by Matplotlib
    print(fold)
    for page in PdfReader(output_path).pages:
        writer.add_page(page)

    print("read")

    # append an external PDF (single-page or multi-page)
    # gen_report = os.path.join(fold, "gen_report.pdf")
    # ext_reader = PdfReader(gen_report)
    # writer.add_page(ext_reader.pages[0])  # insert a specific page

    with open(os.path.join(fold, "final_report.pdf"), "wb") as fh:
        writer.write(fh)


if __name__ == "__main__":
    generate_pdf_report("output/S01_recon/metrology/memory.csv", "semiconductor_report.pdf")
