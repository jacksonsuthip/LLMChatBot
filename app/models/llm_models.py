from langchain_ollama import ChatOllama
from app.schemas.chat_schema import LlmModelEnum

class LlmModels:
    def __init__(self, model: LlmModelEnum, temperature: float):

        # ollama pull llm
        self.model_ollama = ChatOllama(
            model=model,
            temperature=temperature,
        )