
from typing import List
from app.schemas.chat_schema import DeleteUrlSchema, IngestUrlResponseSchema, IngestUrlSchema, QueryUrlSchema
from app.utils.chat import chatUrl, chatUrlStream
from app.utils.embedding_store import delete_url_db, get_all_urls_db, get_sub_urls, process_url

async def ingest_url(ingestUrl: IngestUrlSchema) -> List[IngestUrlResponseSchema]:
    response = []

    if(ingestUrl.subUrl):
        for url in ingestUrl.urls:
            sub_urls = get_sub_urls(url)
            print(f"Found {len(sub_urls)} sub-URLs to process.")
            for url in sub_urls:
                out = process_url(url)
                response.append({"source": url, "documents": out["documents"], "isNewUrl": out["isNewUrl"]})
    else:
        for url in ingestUrl.urls:
            print(f"url {url}")
            out = process_url(url)
            response.append({"source": url, "documents": out["documents"], "isNewUrl": out["isNewUrl"]})

    return response

async def query_url(query:QueryUrlSchema):
    return chatUrl(query)

async def delete_url(delete: DeleteUrlSchema):
    return delete_url_db(delete.url)
 
async def get_all_urls():
    return get_all_urls_db()
