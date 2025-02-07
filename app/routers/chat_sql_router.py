from typing import List
from fastapi import APIRouter, HTTPException, status
from app.schemas.chat_sql_schema import SaveQuerySchema
from app.services.chat_sql_service import delete_query_sql, get_db_query_sql_endpoint, query_sql, save_query_sql


router = APIRouter(prefix="/chat-sql", tags=["Chat-sql"])

@router.get("/query_sql/{query}/{model}/{temperature}")
async def Query_sql_endpoint(query, model, temperature):
    try:
        result = await query_sql(query, model, temperature)
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
    
@router.delete("/delete_query_sql")
async def Delete_query_sql_endpoint():
    try:
        result = await delete_query_sql()
        return result
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    
@router.get("/get_db_query_sql/{count}")
async def Get_db_query_sql_endpoint(count):
    try:
        result = await get_db_query_sql_endpoint(count)
        return result
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

