from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from psycopg import connect
from psycopg.errors import Error as PsycopgError

from app.config import DATABASE_URL

router = APIRouter()


@router.get("/health")
def health():
    """Liveness check: confirms the API process can reach Postgres."""
    try:
        with connect(DATABASE_URL, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        return {"status": "ok", "database": "up"}
    except (PsycopgError, OSError):
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "degraded", "database": "down"},
        )
