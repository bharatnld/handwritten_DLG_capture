import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from app.core.config import TESSERACT_PATH 

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

def extract_text(file_path: bytes) -> tuple[str, int]:
    file_ext = file_path.split('.')[-1].lower()
    if file_ext == 'pdf':
        images = convert_from_path(file_path,poppler_path=r"C:\Program Files\Poppler\poppler-24.08.0\Library\bin")
        text = "\n".join(pytesseract.image_to_string(img) for img in images)
        return text, len(images)
    else:
        img = Image.open(file_path)
        return pytesseract.image_to_string(img), 1


        
