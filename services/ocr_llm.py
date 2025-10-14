from google import genai
from google.genai import types
import pathlib
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Initialize Gemini client
client = genai.Client(api_key=api_key)

def extract_text_llm(file_path: str) -> tuple[str, int]:
    """
    Extract text (including handwritten) from PDF or image using Gemini 2.5 model.
    Returns: (extracted_text, num_pages)
    """

    filepath = pathlib.Path(file_path)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Determine mime type automatically
    if file_path.lower().endswith(".pdf"):
        mime_type = "application/pdf"
    elif file_path.lower().endswith((".jpg", ".jpeg")):
        mime_type = "image/jpeg"
    elif file_path.lower().endswith(".png"):
        mime_type = "image/png"
    else:
        raise ValueError("Unsupported file type. Must be PDF, PNG, or JPG.")

    # Schema for structured output
    schema = """{
        "shipment_document": {
            "document_type": "string (e.g., 'CMR', 'Delivery Note')",
            "document_number": "string",
            "date_of_issue": "string (YYYY-MM-DD)",
            "consignor_sender": {
                "name": "string",
                "address": "string",
                "city_region": "string (optional)",
                "postcode": "string (optional)",
                "country": "string",
                "contact_info": {
                    "telephone": "string (optional)",
                    "email": "string (optional)"
                }
            },
            "consignee_recipient": {
                "name": "string",
                "address": "string",
                "city_region": "string (optional)",
                "postcode": "string (optional)",
                "country": "string",
                "place_of_delivery": "string (optional)"
            },
            "carrier": {
                "name": "string (optional)",
                "address": "string (optional)",
                "city_region": "string (optional)",
                "postcode": "string (optional)",
                "country": "string (optional)"
            },
            "delivery_information": {
                "place_of_taking_over_goods": "string",
                "date_of_taking_over_goods": "string (YYYY-MM-DD)",
                "expected_delivery_date": "string (YYYY-MM-DD, optional)",
                "order_number": "string (optional)",
                "customer_reference": "string (optional)"
            },
            "goods_description": {
                "items": [
                    {
                        "quantity": "number or string",
                        "unit": "string",
                        "size": "string (optional)",
                        "mark_or_product_identifier": "string (optional)",
                        "description": "string",
                        "product_code": "string (optional)",
                        "origin_country_code": "string (optional)",
                        "gross_weight_kg": "number (optional)",
                        "statistical_number": "string (optional)",
                        "product_dimensions_or_count_per_unit": "string (optional)"
                    }
                ],
                "total_gross_weight_kg": "number (optional)",
                "total_pallets_stated": "number (optional)",
                "total_crates_stated": "string (optional)"
            },
            "transport_details": {
                "trailer_wagon_number": "string (optional)",
                "vehicle_registration_number": "string (optional)",
                "pallets_delivered_count": "number (optional)"
            },
            "payment_instructions": {
                "terms": "string (optional)",
                "location": "string (optional)",
                "date": "string (YYYY-MM-DD, optional)"
            },
            "remarks_observations": {
                "general_remarks": "string (optional)",
                "special_agreements_or_notes": {
                    "reference": "string (optional)",
                    "status_changed": "string (optional)",
                    "currency": "string (optional)",
                    "document_type_code": "string (optional)",
                    "agreement_date": "string (YYYY-MM-DD, optional)",
                    "bol_number": "string (optional)",
                    "quality_and_quantity_correct_by": "string (optional)",
                    "damaged_status": "boolean (optional)",
                    "goods_received_under_discrepancy": "boolean (optional)"
                }
            },
            "reception_confirmation": {
                "date_received": "string (YYYY-MM-DD, optional)",
                "temperature_celsius": "number (optional)",
                "total_cases_accepted": "number (optional)",
                "pallets_in": "number (optional)",
                "pallets_out": "number (optional)",
                "over_short_rejected_status": "string (optional)",
                "scanned_status": "string (YES/NO, optional)",
                "received_by_signature_name": "string (optional)",
                "received_by_print_name": "string (optional)",
                "receiving_signature_present": "boolean (optional)"
            },
            "issuing_party_details": {
                "issued_by_name": "string (optional)",
                "issued_by_address": "string (optional)",
                "issued_by_city_region": "string (optional)",
                "issued_by_country": "string (optional)"
            }
        }
    }"""

    # Build prompt
    prompt = f"""
You are an OCR extraction assistant.

Task:
- Extract only **handwritten** text from this document.
- Fill in the schema strictly as JSON (use null for missing values).
- Do not change key names or structure.

Schema:
{schema}
"""

    # Send file + prompt to Gemini
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            types.Part.from_bytes(
                data=filepath.read_bytes(),
                mime_type=mime_type
            ),
            prompt
        ]
    )

    # Extract text safely
    extracted_text = ""
    try:
        if hasattr(response, "text") and response.text:
            extracted_text = response.text.strip()
        elif hasattr(response, "candidates") and response.candidates:
            extracted_text = response.candidates[0].content.parts[0].text.strip()
    except Exception as e:
        print("⚠️ Error extracting text:", e)

    # Default page count (Gemini doesn’t expose PDF page info)
    num_pages = 1

    return extracted_text, num_pages


