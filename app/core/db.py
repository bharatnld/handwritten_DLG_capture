import psycopg2
from fastapi import HTTPException
import os

def get_db_connection():
    try:
        return psycopg2.connect(
            host=os.getenv("PG_HOST", "localhost"),
            port=os.getenv("PG_PORT", "5432"),
            database=os.getenv("PG_DB", "argus_db"),
            user=os.getenv("PG_USER", "postgres"),
            password=os.getenv("PG_PASSWORD", "")
        )
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")
