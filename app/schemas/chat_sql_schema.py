from pydantic import BaseModel

class SaveQuerySchema(BaseModel):
    Question: str
    SQLQuery: str
    SQLResult: str
    Answer: str