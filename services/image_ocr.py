from google import genai
from google.genai import types
import pathlib
from dotenv import load_dotenv
import os
from pdf2image import convert_from_path
from PIL import Image

# -------------------- CONFIG --------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


def extract_text_llms(file_path: str) -> tuple[str, int]:
    """
    Extract text from a PDF or image using Gemini model.
    Approach: PDF -> images -> Gemini per page.
    Returns: (extracted_text, num_pages)
    """
    filepath = pathlib.Path(file_path)

    # If input is PDF -> convert to images
    if file_path.lower().endswith(".pdf"):
        images = convert_from_path(file_path,poppler_path=r"C:\Program Files\Poppler\poppler-24.08.0\Library\bin")
        num_pages = len(images)
    else:
        # Single image file
        images = [Image.open(filepath).convert("RGB")]
        num_pages = 1

    all_text = []

    for img in images:
        prompt = "Extract all visible text from this image as plain text. Return only the text."
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, img]
        )
        if response.text:
            all_text.append(response.text.strip())

    extracted_text = "\n\n".join(all_text)

    return extracted_text, num_pages
