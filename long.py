#!/usr/bin/env python3
import os
import sys
import shutil
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
from PIL import Image, ImageChops

# A4 page at 200 DPI
PAGE_WIDTH  = 1654
PAGE_HEIGHT = 2339
DPI = 200

# Margins
LEFT_MARGIN   = 40
RIGHT_MARGIN  = 40
TOP_MARGIN    = 100
BOTTOM_MARGIN = 100

# Spacing between page1 and page2
GAP = 10


# ============================================================
# Whitespace trimming
# ============================================================

def trim_top_whitespace(img):
    bg = Image.new(img.mode, img.size, "white")
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if not bbox:
        return img
    top = bbox[1]
    return img.crop((0, top, img.width, img.height))


def trim_bottom_whitespace(img):
    bg = Image.new(img.mode, img.size, "white")
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if not bbox:
        return img
    bottom = bbox[3]
    return img.crop((0, 0, img.width, bottom))


# ============================================================
# Phase 1: collect multipage PDFs into long/
# ============================================================

def collect_long_pdfs(src_folder, long_folder="long"):
    os.makedirs(long_folder, exist_ok=True)
    for fname in os.listdir(src_folder):
        if not fname.lower().endswith(".pdf"):
            continue
        fpath = os.path.join(src_folder, fname)
        try:
            reader = PdfReader(fpath)
            if len(reader.pages) > 1:
                shutil.copy(fpath, os.path.join(long_folder, fname))
        except:
            pass


# ============================================================
# Phase 2: process long PDFs
# ============================================================

def process_pdf(path, outpath):
    pages = convert_from_path(path, dpi=DPI)
    if len(pages) != 2:
        return

    page1, page2 = pages

    # ----------------------------------------------------------
    # 1️⃣ Page 1 → whitespace trimming only
    # ----------------------------------------------------------
    page1 = trim_top_whitespace(page1)
    page1 = trim_bottom_whitespace(page1)

    # ----------------------------------------------------------
    # 2️⃣ Page 2 → whitespace trimming only
    # ----------------------------------------------------------
    page2 = trim_top_whitespace(page2)
    page2 = trim_bottom_whitespace(page2)

    # ----------------------------------------------------------
    # 3️⃣ Normalize widths
    # ----------------------------------------------------------
    target_w = min(page1.width, page2.width)
    page1 = page1.resize((target_w, int(page1.height * target_w / page1.width)))
    page2 = page2.resize((target_w, int(page2.height * target_w / page2.width)))

    # ----------------------------------------------------------
    # 4️⃣ Combine vertically with GAP
    # ----------------------------------------------------------
    combined_h = page1.height + GAP + page2.height
    combined = Image.new("RGB", (target_w, combined_h), "white")

    combined.paste(page1, (0, 0))
    combined.paste(page2, (0, page1.height + GAP))

    # ----------------------------------------------------------
    # 5️⃣ Auto scale to fit margins
    # ----------------------------------------------------------
    avail_w = PAGE_WIDTH  - LEFT_MARGIN - RIGHT_MARGIN
    avail_h = PAGE_HEIGHT - TOP_MARGIN  - BOTTOM_MARGIN

    scale_w = avail_w / combined.width
    scale_h = avail_h / combined.height

    scale = min(scale_w, scale_h)

    scaled_w = int(combined.width  * scale)
    scaled_h = int(combined.height * scale)

    resized = combined.resize((scaled_w, scaled_h), Image.LANCZOS)

    # ----------------------------------------------------------
    # 6️⃣ Place inside A4 page (top aligned)
    # ----------------------------------------------------------
    final_img = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
    offset_x = LEFT_MARGIN + (avail_w - scaled_w) // 2
    offset_y = TOP_MARGIN

    final_img.paste(resized, (offset_x, offset_y))

    # ----------------------------------------------------------
    # 7️⃣ Save PDF
    # ----------------------------------------------------------
    final_img.save(outpath, "PDF", resolution=DPI)


# ============================================================
# Phase 3: run on long folder
# ============================================================

def process_long_folder(long_folder="long", output_folder="long_fixed"):
    os.makedirs(output_folder, exist_ok=True)
    for fname in os.listdir(long_folder):
        if fname.lower().endswith(".pdf"):
            process_pdf(
                os.path.join(long_folder, fname),
                os.path.join(output_folder, fname)
            )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py /path/to/source_pdfs")
        sys.exit(1)

    src_folder = sys.argv[1]

    collect_long_pdfs(src_folder, "long")
    process_long_folder("long", "long_fixed")

    print("Done.")