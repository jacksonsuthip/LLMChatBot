from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, model_validator

class LlmModelEnum(str, Enum):
    llama3_2_1b = "llama3.2:1b"
    llama3_8b = "llama3:8b"
    mistral_7b = "mistral:7b"
    gemma_7b = "gemma:7b"
    phi3_3_8b = "phi3:3.8b"
    phi4_14b = "phi4:14b"

class IngestUrlSchema(BaseModel):
    urls: list[str]
    subUrl: bool

class IngestUrlResponseSchema(BaseModel):
    source: str
    documents: int
    isNewUrl: bool

class ChatHistorySchema(BaseModel):
    User: str
    Assistant: str

class QueryUrlSchema(BaseModel):
    history: Optional[List[ChatHistorySchema]] = None
    query: str
    model: LlmModelEnum
    temperature: float

    @model_validator(mode='before')
    def check_query(cls, values):
        query_value = values.get('query', '').strip()
        if not query_value:
            raise ValueError("Input must not be an empty or whitespace-only string.")
        values['query'] = query_value
        return values

# Define the metadata structure as a Pydantic model
class Metadata(BaseModel):
    language: str
    source: str
    title: str

# Define the Document structure as a Pydantic model
class Document(BaseModel):
    metadata: Metadata
    page_content: str

# Define the main structure that represents the entire input/output
class QueryUrlResponseSchema(BaseModel):
    input: str
    context: List[Document]
    answer: str
    history: str
    combined_input: str

class DeleteUrlSchema(BaseModel):
    url: str

class DeleteUrlResponseSchema(BaseModel):
    message: str
    unique_id: str
    source: str

class GetUrlsResponse(BaseModel):
    id: str
    language: str
    source: str
    title: str
    unique_id: str

# class QueryUrlSchema(BaseModel):
#     history: Optional[List[ChatHistorySchema]] = None
#     query: str
#     model: LlmModelEnum
#     temperature: float