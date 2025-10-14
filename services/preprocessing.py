import cv2
import numpy as np
from pdf2image import convert_from_path
import tempfile
import os

def enhance_handwriting_visibility(image_path: str, output_path: str):
    """
    Enhanced version with better pencil detection and preservation.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: More aggressive denoising for pencil marks
    gray = cv2.fastNlMeansDenoising(gray, h=12, templateWindowSize=7, searchWindowSize=21)

    # Step 2: Adaptive contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(12,12))
    contrast = clahe.apply(gray)

    # Step 3: Multi-stage sharpening for pencil strokes
    kernel_sharpen = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharp = cv2.filter2D(contrast, -1, kernel_sharpen)
    
    # Additional edge enhancement
    kernel_edge = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
    sharp = cv2.filter2D(sharp, -1, kernel_edge)

    # Step 4: Selective dark area enhancement
    # Create mask for dark areas (pencil marks)
    _, dark_mask = cv2.threshold(sharp, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Enhance only dark areas
    enhanced_dark = cv2.addWeighted(sharp, 1.8, sharp, 0, -30)
    
    # Combine enhanced dark areas with original
    result = np.where(dark_mask == 255, enhanced_dark, sharp)

    # Step 5: Adaptive gamma correction
    mean_brightness = np.mean(result)
    gamma = 1.2 if mean_brightness < 128 else 0.9
    gamma_corrected = np.power(result / 255.0, gamma) * 255
    gamma_corrected = np.uint8(gamma_corrected)

    # Step 6: Final noise reduction
    final = cv2.medianBlur(gamma_corrected, 3)

    cv2.imwrite(output_path, final)
    print(f"✅ Enhanced image saved: {output_path}")

def preprocess_pdf_for_handwriting(pdf_path: str, output_dir: str):
    """
    Enhanced PDF processing with error handling.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"📄 Processing PDF: {pdf_path}")

    try:
        # Convert all pages to images
        with tempfile.TemporaryDirectory() as temp_dir:
            pages = convert_from_path(
                pdf_path, 
                dpi=400,
                poppler_path=r"C:\Program Files\Poppler\poppler-24.08.0\Library\bin",
                output_folder=temp_dir
            )
            
            print(f"🖼️ Total pages found: {len(pages)}")

            processed_images = []
            for i, page in enumerate(pages):
                img_path = os.path.join(temp_dir, f"page_{i+1}.png")
                output_path = os.path.join(output_dir, f"page_{i+1}_enhanced.png")

                # Save page as image
                page.save(img_path, "PNG", quality=95)

                # Enhance handwriting visibility
                enhance_handwriting_visibility(img_path, output_path)
                processed_images.append(output_path)

            print(f"\n✅ All pages processed and saved in: {output_dir}")
            return processed_images
            
    except Exception as e:
        print(f"❌ Error processing PDF: {e}")
        return []

def batch_process_pdfs(pdf_folder: str, output_base_dir: str):
    """
    Process multiple PDFs in a folder.
    """
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(pdf_folder, pdf_file)
        output_dir = os.path.join(output_base_dir, os.path.splitext(pdf_file)[0])
        
        print(f"\n🔧 Processing: {pdf_file}")
        processed_files = preprocess_pdf_for_handwriting(pdf_path, output_dir)

if __name__ == "__main__":
    # Single file processing
    pdf_file = "Afbeelding van WhatsApp op 2025-09-02 om 17.50.13_f2889388.pdf"
    output_folder = "enhanced_output"

    processed_files = preprocess_pdf_for_handwriting(pdf_file, output_folder)
    
    if processed_files:
        print("\n📁 Processed image files:")
        for f in processed_files:
            print(f"  - {f}")
    else:
        print("❌ No files were processed.")