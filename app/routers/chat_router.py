from typing import List
from fastapi import APIRouter, HTTPException, status
from app.schemas.chat_schema import DeleteUrlResponseSchema, DeleteUrlSchema, IngestUrlResponseSchema, IngestUrlSchema, QueryUrlResponseSchema, QueryUrlSchema
from app.services.chat_service import delete_url, ingest_url, query_url

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("/ingest_url")
async def Ingest_url_endpoint(ingest:IngestUrlSchema) -> List[IngestUrlResponseSchema]:
    try:
        result = await ingest_url(ingest)
        return result
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@router.post("/query_url")
async def Query_url_endpoint(query:QueryUrlSchema) -> QueryUrlResponseSchema:
    try:
        result = await query_url(query)
        return result
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))

@router.post("/delete_url")
async def Delete_url_endpoint(delete:DeleteUrlSchema):
    try:
        result = await delete_url(delete)
        return result
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))