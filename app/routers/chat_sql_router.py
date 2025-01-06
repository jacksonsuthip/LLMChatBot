from typing import List
from fastapi import APIRouter, HTTPException, status
from app.services.chat_sql_service import query_sql


router = APIRouter(prefix="/chat-sql", tags=["Chat-sql"])

@router.get("/query_sql/{query}")
async def Query_sql_endpoint(query):
    try:
        result = await query_sql(query)
        return result
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

