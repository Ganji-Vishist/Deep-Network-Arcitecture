

# --- Import libraries ---
from PIL import Image, ImageOps
import pytesseract
import matplotlib.pyplot as plt
from pathlib import Path

# --- Default image path in the current directory ---
DEFAULT_IMAGE = Path("bf021d8aa.png")  # <-- change name if needed
OUTPUT_TEXT_FILE = Path("caption.txt")

def preprocess_for_ocr(pil_img):
    """
    Enhances the image for OCR by converting to grayscale,
    increasing contrast, and resizing if needed.
    """
    img = pil_img.convert("L")
    img = ImageOps.autocontrast(img, cutoff=2)

    w, h = img.size
    if max(w, h) < 1200:
        scale = 1200 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
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

    # ✅ Display image inline in Colab
    plt.imshow(preprocessed, cmap="gray")
    plt.title(f"Processing: {image_path.name}")
    plt.axis("off")
    plt.show()

    config = r'--oem 3 --psm 6'
    extracted_text = pytesseract.image_to_string(preprocessed, config=config)
    return extracted_text.strip()


# --- Main execution ---
image_path = DEFAULT_IMAGE
print(f"📷 Reading image: {image_path}")

try:
    text = ocr_image(image_path)

    if text:
        print("\n📝 Detected text:")
        print("------------------------")
        print(text)
        print("------------------------")

        # Save extracted text
        with open(OUTPUT_TEXT_FILE, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"\n✅ Saved extracted text to: {OUTPUT_TEXT_FILE.resolve()}")
    else:
        print("\n⚠️ No text detected in the image.")
except Exception as e:
    print(f"❌ Error: {e}")
