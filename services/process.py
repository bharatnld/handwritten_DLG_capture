from app.core.db import get_db_connection
from app.services.ocr import extract_text
from app.services.ocr_llm import extract_text_llm
from app.services.image_ocr import extract_text_llms
from app.services.azure_ocr import extract_text_azure
from app.services.gpt_extraction import extract_with_gemini
from psycopg2.extras import Json
from fastapi import HTTPException
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import time

def fetch_configuration():
    with get_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT config_data FROM configurations WHERE id = %s", ('configuration',))
            result = cur.fetchone()
            return result[0] if result else {"id": "configuration"}
            

def clean_llm_json(raw_text: str):
    # Remove markdown fences and language hints
    cleaned = re.sub(r"^```(?:json)?", "", raw_text.strip(), flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"```$", "", cleaned, flags=re.MULTILINE).strip()
    # Trim to first and last curly brace pair (handles extra commentary)
    if cleaned.count("{") > 0 and cleaned.count("}") > 0:
        start = cleaned.find("{")
        end = cleaned.rfind("}") + 1
        cleaned = cleaned[start:end]
    return cleaned

async def process_file(file_path, dataset_name, original_filename: str):
    config = fetch_configuration()
    #prompt_template = config.get(dataset_name, {}).get("model_prompt", "Extract all data.")
    #example_schema = config.get(dataset_name, {}).get("example_schema", {})

  
    loop = asyncio.get_event_loop()
    executor = ThreadPoolExecutor(max_workers=2)  # 2 threads for parallel execution

    # Schedule both functions to run in parallel
    handwritten_future = loop.run_in_executor(executor, extract_text_llm, file_path)
    computerized_future = loop.run_in_executor(executor, extract_text_llms, file_path)

    # ✅ Run both in true parallel
    handwritten_result, computerized_result = await asyncio.gather(
        handwritten_future, computerized_future
    )

    handwritten_text, num_pages_handwritten = handwritten_result
    computerized_text, num_pages_computerized = computerized_result
    num_pages = max(num_pages_handwritten, num_pages_computerized)

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
  }
}
"""
    prompts = f"""
You are an OCR document parser specialized for invoices and packing lists.

Step 1: Extract all possible text directly from the provided PDF file (computerized/digital text layer only).  
       Fill the schema below to create the **initial_schema**. Use null for any missing fields.

Step 2: Use the additional OCR output provided below (from Gemini) which contains only handwritten text.  
       Update only those fields in the initial_schema that are clearly corrected or overwritten by handwritten text, producing the **corrected_schema**.

Step 3: Any handwritten text that cannot be confidently mapped to a schema field should be added under a special key `"handwritten_extras"` inside corrected_schema.

Step 4: Output the result as valid JSON containing two main objects:
       1. "initial_schema" — filled using only PDF/computerized text
       2. "corrected_schema" — strictly updated using handwritten text from Gemini, with unstructured data in "handwritten_extras"

Rules:
- Do not add, remove, or rename any keys in the original schema.
- Preserve the same structure and field order.
- Replace "string"/"number" placeholders with extracted values.
- If a field is missing or cannot be determined, write null.
- Only update fields in corrected_schema if the Gemini OCR shows a clear, unambiguous correction.
- Always return valid JSON only (no explanations, no extra text).

Schema:
{schema}

Additional OCR Output (from Gemini, contains only handwritten text):
{handwritten_text}
"""
    prompt = f"""
You are an OCR document parser specialized in structured extraction and correction for shipment documents (e.g., CMR, Delivery Notes).

### TASK:
You are given two text sources:
1. **Computerized Text** — extracted from the digital PDF.
2. **Handwritten OCR Text** — recognized from annotations or handwriting.

### OBJECTIVE:
1. From the computerized text, fill all possible fields in the schema to create **initial_schema**.
2. Compare with handwritten text and apply any clear corrections, updates, or additions to form **corrected_schema**.
3. Any handwritten text that doesn’t match a field should go under `"handwritten_extras"` in corrected_schema.

### OUTPUT FORMAT:
Return **only valid JSON**:
{{
  "corrected_schema": {{ ... }}
}}

### RULES:
- Maintain schema structure and field order.
- Replace placeholders ("string", "number") with extracted values.
- Use `null` for missing values.
- Update only when handwritten text clearly corrects computerized data.
- If no handwritten correction applies, corrected_schema = initial_schema.
- No explanations or additional text — only valid JSON output.

---

### SCHEMA:
{schema}

---

### COMPUTERIZED TEXT (from PDF):
{computerized_text}

---

### HANDWRITTEN TEXT (from Gemini OCR):
{handwritten_text}
"""



    start_time = time.time()
    
    gpt_output_raw = await extract_with_gemini(prompt)
    
    cleaned = clean_llm_json(gpt_output_raw)
    try:
        gpt_output = json.loads(cleaned)
        parse_error = None
    except Exception as e:
        gpt_output = {"raw": gpt_output_raw}
        parse_error = str(e)
    
    end_time = time.time()
    total_time = round(end_time - start_time, 2)

    data = {
        'id': f"{dataset_name}/{original_filename}",
        'properties': {
            'blob_name': f"{dataset_name}/{os.path.basename(file_path)}",
            'request_timestamp': datetime.utcnow().isoformat(),
            'blob_size': os.path.getsize(file_path),
            'num_pages': num_pages,
            'total_time_seconds': total_time

        },
        'state': {
            'file_landed': True,
            'ocr_completed': bool(handwritten_text),
            'gpt_extraction_completed': bool(gpt_output),
            'processing_completed': bool(handwritten_text and gpt_output)
        },
        'extracted_data': {
            'ocr_output': handwritten_text,
            'gpt_extraction_output': gpt_output,
            'error': parse_error
        }
    }

    with get_db_connection() as conn:
        with conn.cursor() as cur:
            # ✅ Ensure table exists
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    data JSONB NOT NULL
                )
            """)
            conn.commit()

            # ✅ Insert or update row
            cur.execute(
                """
                INSERT INTO documents (id, data) VALUES (%s, %s)
                ON CONFLICT (id) DO UPDATE SET data = EXCLUDED.data
                """,
                (data['id'], Json(data))
            )
            conn.commit()


    return data
