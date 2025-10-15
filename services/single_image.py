from google import genai
from google.genai import types
from dotenv import load_dotenv
from PIL import Image
import os
import json
import re

# -------------------- CONFIG --------------------
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


# -------------------- CLEAN FUNCTION --------------------
def clean_llm_json(raw_text: str):
    """Clean Gemini or LLM output to extract valid JSON text."""
    cleaned = re.sub(r"^```(?:json)?", "", raw_text.strip(), flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE).strip()
    # Trim to first and last curly brace pair
    if "{" in cleaned and "}" in cleaned:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        cleaned = cleaned[start:end]
    return cleaned


# -------------------- MAIN FUNCTION --------------------
def extract_text_and_schema_from_image(image_path: str):
    """
    Extracts both printed and handwritten text from an image using Gemini.
    Returns:
        (raw_extracted_text, structured_json_dict)
    """
    try:
        # Ensure image readable
        img = Image.open(image_path).convert("RGB")

        # -------------------- SCHEMA --------------------
        generic_schema =""""{
  "document_type": "CMR Consignment Note",
  "source_file_name": "Name of the source image or PDF file.",

  "metadata": {
    "country_code": "Country code printed on form (e.g., NL, UK, DE).",
    "form_number": "Unique CMR form or serial number printed on the document.",
    "cmr_version": "CMR or AVC version printed (e.g., 'AVC-2009').",
    "document_date": "Official document issue or shipment date in DD-MM-YYYY format."
  },

  "shipper": {
    "name": "Name of the sender company (usually printed in Box 1).",
    "address": "Street address of sender.",
    "city": "City or town of sender.",
    "postal_code": "Postal or ZIP code of sender.",
    "country": "Country of sender."
  },

  "consignee": {
    "name": "Name of consignee (recipient company).",
    "address": "Street address of consignee.",
    "city": "City or town of consignee.",
    "postal_code": "Postal or ZIP code of consignee.",
    "country": "Country of consignee."
  },

  "carrier": {
    "name": "Name of carrier or transport company.",
    "address": "Carrier's address.",
    "city": "Carrier city.",
    "postal_code": "Carrier postal code.",
    "country": "Carrier country."
  },

  "transport_details": {
    "place_of_loading": "Location where goods are loaded (printed or handwritten).",
    "place_of_delivery": "Location where goods will be delivered.",
    "loading_date": "Date of loading (DD-MM-YYYY).",
    "delivery_date": "Expected or actual delivery date (DD-MM-YYYY).",
    "po_number": "Purchase order or reference number.",
    "delivery_on": "Delivery date/time if different from main delivery_date.",
    "transport_condition": "Terms like DDP, DAP, FOB, etc., or delivery condition."
  },

  "goods_details": [
    {
      "article": "The product's article number, SKU, or model code.",
      "descriptionItem": "A brief, human-readable description of the article.",
      "quantity": "The number of units of the article being purchased. This is an integer.",
      "statistical_number":"`Appears in tariff/customs reference column",
      "unitPrice": "The price for a single unit of the article. This is a floating-point number.",
      "gross_weight": "Gross weight of goods in kg"
    }
  ],

  "totals": {
    "pallets": "Total number of pallets.",
    "cases": "Total number of cases or packages.",
    "total_value": "Total declared value (if provided)."
  },

  "stamps_and_signatures": {
    "receiver_section": {
      "company_name": "Company name printed or stamped in receiver confirmation area.",
      "confirmation_text": "Printed text like 'CONFIRMATION OF RECEIPT' or equivalent.",
      "pallets_in": "Printed or handwritten pallets received.",
      "pallets_out": "Printed or handwritten pallets sent.",
      "temperature": "Printed or handwritten temperature at delivery.",
      "shortage_or_damage": "Handwritten or printed shortage/damage notes.",
      "checked": "Whether 'checked' or 'yes/no' is ticked or marked.",
      "signature_present": "Boolean (true/false) indicating presence of signature.",
      "received_by": "Printed or handwritten name of person who received goods.",
      "date_signed": "Date of signature (handwritten or stamped).",
      "stamp_description": "Text or logo visible in the receiver’s stamp or seal."
    },
    "sender_section": {
      "company_name": "Printed or stamped name of sender company.",
      "city": "City printed near sender stamp.",
      "date": "Date printed or handwritten near sender signature.",
      "signature_present": "Boolean (true/false) indicating if sender signature exists.",
      "stamp_description": "Any visible logo or company seal text for sender."
    },
    "carrier_section": {
      "company_name": "Carrier company name if stamped or written.",
      "signature_present": "Boolean (true/false) if carrier signature/stamp present.",
      "stamp_description": "Text or logo visible in the carrier’s stamp."
    }
  },

  "handwritten_extra": []
}
"""
        schema = """
{
  "shipment_document": {
    "document_type": "string (e.g., 'CMR', 'Delivery Note')",
    "document_number": "string (e.g., CMR: #237029-237215, Delivery Note: No. 1006)",
    "date_of_issue": "string (YYYY-MM-DD, e.g., 2025-09-02)",

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
      "place_of_delivery": "string (optional, e.g., Goldthorpe South Yorkshire)"
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
      "order_number": "string (optional, e.g., PO number: 6503996754, Order Number: 5201019)",
      "customer_reference": "string (optional, e.g., REF: RVS-064, #2050829-1259)"
    },

    "goods_description": {
      "items": [
        {
          "quantity": "number or string (if unit is embedded, e.g., '1,00 Europallet')",
          "unit": "string (e.g., 'stuks', 'crates', 'pallets', 'Europallet - R')",
          "size": "string (optional, e.g., 'x EPS')",
          "mark_or_product_identifier": "string (optional, e.g., '246.03 CRATES', '31S096-')",
          "description": "string (e.g., 'Eggplants 15 stuks', 'BOX BANANA SHALLOTS 14 + 300GR')",
          "product_code": "string (optional, e.g., 'PLU EPS-136', 'R7121226857')",
          "origin_country_code": "string (optional, e.g., 'UK NL', 'NL HA')",
          "gross_weight_kg": "number (optional)",
          "statistical_number": "string (optional)",
          "product_dimensions_or_count_per_unit": "string (optional, e.g., '(24 x 180)', '(28 x 180)')"
        }
      ],
      "total_gross_weight_kg": "number (optional)",
      "total_pallets_stated": "number (optional, e.g., 30.84, 52, 130)",
      "total_crates_stated": "string (optional, e.g., 'VGS CRATES', 'TOTAL GKN PALLETS', 'TOTAL 4 WAY WHITE PALLETS')"
    },

    "transport_details": {
      "trailer_wagon_number": "string (optional, e.g., Ribnummer: 3815803)",
      "vehicle_registration_number": "string (optional, e.g., A1211ZA, SP24235)",
      "pallets_delivered_count": "number (optional, specific to some CMR sections, e.g., 130)"
    },

    "payment_instructions": {
      "terms": "string (optional, e.g., 'DDP', 'Franco/Frei')",
      "location": "string (optional, e.g., Goldthorpe South Yorkshire, Kruiningen)",
      "date": "string (YYYY-MM-DD, optional)"
    },

    "remarks_observations": {
      "general_remarks": "string (optional, e.g., 'Shortages and damages mentioned on the CMR are reported to the supplier by the receiver within 12 hours.')",
      "special_agreements_or_notes": {
        "reference": "string (optional, e.g., 'Lidl GB - Exeter ROC')",
        "status_changed": "string (optional, e.g., 'Reg')",
        "currency": "string (optional, e.g., 'EURO')",
        "document_type_code": "string (optional, e.g., 'DD')",
        "agreement_date": "string (YYYY-MM-DD, optional, e.g., '2025-09-01')",
        "bol_number": "string (optional, e.g., '1927096')",
        "quality_and_quantity_correct_by": "string (optional, e.g., 'DRIVER')",
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
      "over_short_rejected_status": "string (optional, e.g., '-1 CASE')",
      "scanned_status": "string (YES/NO, optional)",
      "received_by_signature_name": "string (optional, e.g., 'Witczak')",
      "received_by_print_name": "string (optional, e.g., 'WITCZAK')",
      "receiving_signature_present": "boolean (optional)"
    },

    "issuing_party_details": {
      "issued_by_name": "string (optional)",
      "issued_by_address": "string (optional)",
      "issued_by_city_region": "string (optional)",
      "issued_by_country": "string (optional)"
    }
    "handwrittem_extra":[]
  }
}
"""

        # -------------------- PROMPT --------------------
        prompt = f"""
You are an advanced document analysis assistant specializing in transport and delivery documents, which often contain both machine-printed and handwritten information.

The image provided contains:
- Machine-printed (computerized) text.
- Handwritten annotations, corrections, or additional notes.

Your primary task is to accurately extract information and populate the provided JSON schema.

**Key Extraction Rules & Priorities:**

1.  **Extract All Visible Text:** Identify and transcribe every piece of text, whether printed or handwritten.
2.  **Handwritten Corrections Take Precedence (Critical):**
    *   If a handwritten value *corrects, overrides, or explicitly amends* a printed value (e.g., by being written over, next to, or clearly replacing a crossed-out printed number, date, or description), the **handwritten value is the definitive, final value** for that field in the JSON output.
    *   **Always prioritize handwritten corrections** over the original printed text when a clear intent to correct is present.
3.  **Schema Adherence:**
    *   Fill out the provided `generic_schema` **exactly as defined**. Do not add, remove, or rename any keys.
    *   Map extracted values to the most appropriate key in the schema based on context and common document formats (e.g., item quantities, descriptions, totals, dates).
4.  **Handling Missing or Unclear Values:** If a specific value required by the schema is not found in the image or is ambiguous, set its corresponding value to `null`.
5.  **Extra Handwritten Notes:** Use the `"extra_handwritten_notes"` field exclusively for handwritten text that does *not* directly correct or fit into any other specific field within the schema (e.g., general comments, unusual observations). **Do not use this field for values that should be placed in other specific schema fields after a handwritten correction.**

**Special Notes for Accurate Extraction:**

*   **Contextual Interpretation:** Use surrounding text, units (e.g., "stuks", "KG"), and common document layouts to correctly interpret and categorize information.
*   **Formatting:** Ensure numbers and values are correctly formatted (e.g., decimal points, commas, dates).
*   **Product Line Structure:** Maintain the structure of item lists, correctly associating quantities, descriptions, and any relevant item-specific codes or details.

Schema:
{generic_schema}
"""


        # -------------------- GEMINI REQUEST --------------------
        mime = "image/jpeg" if image_path.lower().endswith((".jpg", ".jpeg")) else "image/png"
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime),
                prompt
            ]
        )

        # -------------------- RESPONSE HANDLING --------------------
        raw_text = response.text.strip() if getattr(response, "text", None) else ""
        cleaned = clean_llm_json(raw_text)

        try:
            structured_json = json.loads(cleaned)
        except Exception as e:
            structured_json = {"_raw_text": raw_text, "_parse_error": str(e)}

        return raw_text, structured_json

    except Exception as e:
        print(f"❌ Error processing image {image_path}: {e}")
        return "", {}


# -------------------- USAGE EXAMPLE --------------------
if __name__ == "__main__":
    image_path = "Afbeelding van WhatsApp op 2025-09-02 om 17.50.13_f2889388.jpg"
    raw_text, structured = extract_text_and_schema_from_image(image_path)

    print("\n📦 STRUCTURED JSON RESULT:\n", json.dumps(structured, indent=2))
