from typing import List
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from app.schemas.chat_schema import DeleteUrlResponseSchema, DeleteUrlSchema, GetUrlsResponse, IngestUrlResponseSchema, IngestUrlSchema, QueryUrlResponseSchema, QueryUrlSchema
from app.services.chat_service import delete_url, get_all_urls, ingest_url, query_url
from app.utils.chat import chatUrlStream

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
async def Delete_url_endpoint(delete:DeleteUrlSchema) -> DeleteUrlResponseSchema:
    try:
        result = await delete_url(delete)
        return result
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    
@router.get("/get_urls")
async def Get_urls_endpoint() -> List[GetUrlsResponse]:
    try:
        result = await get_all_urls()
        return result
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    
@router.get("/query_url_stream/{query}/{model}/{temperature}")
async def Query_url_stream_endpoint(query, model, temperature):
    try:
        return StreamingResponse(chatUrlStream(query, model, temperature), media_type="text/event-stream")
         
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
