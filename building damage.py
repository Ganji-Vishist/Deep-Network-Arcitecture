#!/usr/bin/env python3
"""
extract_caption.py

Performs OCR (Optical Character Recognition) on an image that contains text,
such as a screenshot or photo caption. This script reads the image,
extracts any text it finds, prints the text to the console, and saves
it to a text file called 'caption.txt'.

Requirements:
-------------
1. Install Tesseract OCR engine:
   - Ubuntu/Debian:   sudo apt-get install tesseract-ocr
   - macOS (Homebrew): brew install tesseract
   - Windows:          Download from https://github.com/tesseract-ocr/tesseract

2. Install Python dependencies:
   pip install pillow pytesseract opencv-python

Usage:
------
   python extract_caption.py                       # uses default image path
   python extract_caption.py /path/to/your_image.png

Author: OpenAI GPT-5
Date: 2025-10-16
"""

import sys
import os
from pathlib import Path
from PIL import Image, ImageOps
import pytesseract
import cv2
import numpy as np

# Default image path (change this if your image is elsewhere)
# Use a Windows-friendly default under the user's Desktop DNA folder if available
DEFAULT_IMAGE = str(Path("c:/Users/pasal/OneDrive/Desktop/DNA/Damages building.png"))
OUTPUT_TEXT_FILE = "caption.txt"

def preprocess_for_ocr(pil_img):
    """
    Enhances the image for OCR by converting to grayscale,
    increasing contrast, and resizing if needed.
    """
    # Convert to grayscale
    img = pil_img.convert("L")

    # Auto contrast to improve text visibility
    img = ImageOps.autocontrast(img, cutoff=2)

    # Resize if image is small
    w, h = img.size
    if max(w, h) < 1200:
        scale = int(1200 / max(w, h))
        img = img.resize((w * scale, h * scale), Image.LANCZOS)

    return img

def ocr_image(image_path):
    """
    Performs OCR on the given image and returns detected text.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    pil_image = Image.open(image_path)
    preprocessed = preprocess_for_ocr(pil_image)

    # Tesseract configuration: OEM 3 = Default, PSM 6 = Assume block of text
    config = r'--oem 3 --psm 6'
    extracted_text = pytesseract.image_to_string(preprocessed, config=config)
    extracted_text = extracted_text.strip()

    return extracted_text

def main():
    """
    Main entry point for the program.
    """
    # Configure tesseract path on Windows if available
    if os.name == "nt":
        default_tesseract = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
        if os.path.isfile(default_tesseract):
            pytesseract.pytesseract.tesseract_cmd = default_tesseract

    # Use CLI argument if given, else default image path
    image_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE

    try:
        print(f"Reading image: {image_path}")
        text = ocr_image(image_path)

        if text:
            print("\nDetected text:")
            print("------------------------")
            print(text)
            print("------------------------")

            # Save to file
            with open(OUTPUT_TEXT_FILE, "w", encoding="utf-8") as f:
                f.write(text + "\n")

            print(f"\nSaved extracted text to: {OUTPUT_TEXT_FILE}")
        else:
            print("\nNo text detected in image.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()