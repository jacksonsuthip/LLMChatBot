from enum import Enum
from pydantic import BaseModel

class SaveQuerySchema(BaseModel):
    Question: str
    SQLQuery: str
    SQLResult: str
    Answer: str

class LlmModelGroqEnum(str, Enum):
    gemma2_9b_it = "gemma2-9b-it"
    llama_3_3_70b_versatile = "llama-3.3-70b-versatile"
