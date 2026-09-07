#!/usr/bin/env python3
"""Combine the single-metric classification ladders into one PDF."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures" / "classification_mean_ladders"
OUTPUT = HERE.parents[7] / "output" / "pdf" / "classification_mean_ladders_absolute_ylim_05_10.pdf"

PAGES = [
    ("Matched 40k - Order A", "matched40k_orderA"),
    ("Sequential - Order A", "sequential_orderA"),
    ("Split-aware - Order A", "split_orderA"),
    ("Fixed 10k/source - Order A", "fixed10k_orderA"),
    ("Fixed 10k/source - Order C", "fixed10k_orderC"),
]
METRICS = ["accuracy", "f1", "roc_auc"]


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    page_width, page_height = 1872, 1584
    pdf = canvas.Canvas(str(OUTPUT), pagesize=(page_width, page_height))
    pdf.setTitle("Classification performance across mixture size")

    left_margin = 215
    right_margin = 24
    top_margin = 70
    bottom_margin = 20
    column_gap = 12
    row_gap = 8
    cell_width = (page_width - left_margin - right_margin - 2 * column_gap) / 3
    cell_height = (page_height - top_margin - bottom_margin - 4 * row_gap) / 5

    for column, label in enumerate(("Accuracy (0.5-1.0)", "F1 (0.5-1.0)", "AUC (0.5-1.0)")):
        x = left_margin + column * (cell_width + column_gap) + cell_width / 2
        pdf.setFillColor(HexColor("#161616"))
        pdf.setFont("Helvetica-Bold", 18)
        pdf.drawCentredString(x, page_height - 30, label)

    for row, (page_title, stem) in enumerate(PAGES):
        y = page_height - top_margin - (row + 1) * cell_height - row * row_gap
        pdf.setFillColor(HexColor("#161616"))
        pdf.setFont("Helvetica-Bold", 14)
        pdf.drawRightString(left_margin - 12, y + cell_height / 2 + 5, page_title)
        pdf.setFillColor(HexColor("#77746e"))
        pdf.setFont("Helvetica", 10)
        pdf.drawRightString(left_margin - 12, y + cell_height / 2 - 12, "mean over 4 graphs")

        for column, metric in enumerate(METRICS):
            image_path = FIGURES / f"{stem}_{metric}.png"
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            image = ImageReader(str(image_path))
            image_width, image_height = image.getSize()
            scale = min(cell_width / image_width, cell_height / image_height)
            draw_width = image_width * scale
            draw_height = image_height * scale
            x = left_margin + column * (cell_width + column_gap) + (cell_width - draw_width) / 2
            image_y = y + (cell_height - draw_height) / 2
            pdf.drawImage(
                image,
                x,
                image_y,
                width=draw_width,
                height=draw_height,
                mask="auto",
            )

    pdf.showPage()

    pdf.save()
    print(OUTPUT)


if __name__ == "__main__":
    main()
