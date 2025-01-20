from typing import List
from fastapi import APIRouter, HTTPException, status
from app.schemas.chat_sql_schema import SaveQuerySchema
from app.services.chat_sql_service import query_sql, save_query_sql


router = APIRouter(prefix="/chat-sql", tags=["Chat-sql"])

@router.get("/query_sql/{query}")
async def Query_sql_endpoint(query):
    try:
        result = await query_sql(query)
        return result
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/save_query_sql")
async def Save_query_sql_endpoint(saveQuery: List[SaveQuerySchema]):
    try:
        result = await save_query_sql(saveQuery)
        return result
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

