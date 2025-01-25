from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from app.schemas.chat_schema import LlmModelEnum
from app.schemas.chat_sql_schema import LlmModelGroqEnum

class LlmModels:
    def __init__(self, model: LlmModelEnum, temperature: float):

        # ollama pull llm
        self.model_ollama = ChatOllama(
            model=model,
            temperature=temperature,
        )

class LlmModelsGroq:
    def __init__(self, model: LlmModelGroqEnum, temperature: float, key:str):
        self.model_Groq = ChatGroq(
            model_name=model,
            temperature=temperature,
            api_key=key
        )