#!/usr/bin/env python3
import os
import sys
from PyPDF2 import PdfMerger

def merge_matching_pdfs(folder1, folder2, output_folder="merged"):
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Collect filenames in both folders
    files1 = {f for f in os.listdir(folder1) if f.lower().endswith(".pdf")}
    files2 = {f for f in os.listdir(folder2) if f.lower().endswith(".pdf")}

    # Find matching filenames
    matches = files1.intersection(files2)

    if not matches:
        print("No matching PDF filenames found.")
        return

    for filename in sorted(matches):
        pdf1 = os.path.join(folder1, filename)
        pdf2 = os.path.join(folder2, filename)

        print(f"Merging: {filename}")

        merger = PdfMerger()
        merger.append(pdf1)
        merger.append(pdf2)

        output_path = os.path.join(output_folder, filename)
        merger.write(output_path)
        merger.close()

    print(f"\nDone! Merged PDFs saved to: {output_folder}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: merge_pdfs.py <folder1> <folder2> [output_folder]")
        sys.exit(1)

    folder1 = sys.argv[1]
    folder2 = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) > 3 else "merged"

    merge_matching_pdfs(folder1, folder2, output)