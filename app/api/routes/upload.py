from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import json
import tempfile
import os

from app.services.process import process_file
from app.core.db import get_db_connection

router = APIRouter()

@router.post("/")
async def upload_file(
    file: UploadFile = File(...),
    dataset_name: str = Form(...),
    model_prompt: str = Form("Extract all data."),
    example_schema: str = Form("{}")
):
    # ✅ Validate schema
    try:
        example_schema_dict = json.loads(example_schema)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in example_schema.")

    # ✅ Ensure configurations table exists, then update config if needed
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                # Create table if not exists
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS configurations (
                        id TEXT PRIMARY KEY,
                        config_data JSONB NOT NULL
                    )
                """)
                conn.commit()

                # Get existing config
                cur.execute("SELECT config_data FROM configurations WHERE id = %s", ('configuration',))
                result = cur.fetchone()
                config_data = result[0] if result else {"id": "configuration"}

                # Add dataset-specific config if missing
                if dataset_name not in config_data:
                    config_data[dataset_name] = {
                        "model_prompt": model_prompt,
                        "example_schema": example_schema_dict
                    }

                    cur.execute(
                        """
                        INSERT INTO configurations (id, config_data) VALUES (%s, %s)
                        ON CONFLICT (id) DO UPDATE SET config_data = EXCLUDED.config_data
                        """,
                        ('configuration', json.dumps(config_data))
                    )
                    conn.commit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update configuration: {str(e)}")

    # ✅ Use a temporary file (not stored in UPLOAD_DIR)
    tmp_path = None
    try:
        file_bytes = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        original_filename = file.filename
        # process_file still gets a path (like before)
        result = await process_file(tmp_path, dataset_name,original_filename)

        return JSONResponse(
            content={"message": f"File {file.filename} processed", "data": result}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
