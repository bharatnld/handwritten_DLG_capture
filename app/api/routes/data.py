from fastapi import APIRouter, HTTPException
from app.core.db import get_db_connection

router = APIRouter()

@router.get("/")
async def fetch_data():
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT * FROM documents")
                rows = cur.fetchall()
                columns = [desc[0] for desc in cur.description]
                return {"data": [dict(zip(columns, row)) for row in rows]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


