#!/usr/bin/env python3
import os
import sys
from PyPDF2 import PdfMerger

def merge_pdfs_in_folder(folder_path):
    if not os.path.isdir(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        sys.exit(1)

    pdf_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(".pdf")
    ]
    if not pdf_files:
        print(f"⚠️ No PDF files found in {folder_path}")
        sys.exit(0)

    # Sort files alphabetically for predictable order
    pdf_files.sort()

    folder_name = os.path.basename(os.path.abspath(folder_path))
    output_path = os.path.join(folder_path, f"{folder_name}.pdf")

    merger = PdfMerger()
    for pdf_file in pdf_files:
        pdf_path = os.path.join(folder_path, pdf_file)
        print(f"➕ Adding: {pdf_file}")
        try:
            merger.append(pdf_path)
        except Exception as e:
            print(f"⚠️ Skipping {pdf_file}: {e}")

    merger.write(output_path)
    merger.close()
    print(f"✅ Merged PDF saved as: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python merge_pdfs.py <folder_path>")
        sys.exit(1)
    merge_pdfs_in_folder(sys.argv[1])