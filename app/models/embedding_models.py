from langchain_ollama import OllamaEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

class EmbeddingModels:
    def __init__(self):
        # ollama pull mxbai-embed-large
        self.mxbai_embeddings_ollama = OllamaEmbeddings(
            model="mxbai-embed-large"
        )

        self.nomic_embeddings_ollama = OllamaEmbeddings(
            model="nomic-embed-text"
        )

        # self.HuggingFaceEmbeddings_sentence_transformers = HuggingFaceEmbeddings(
        #     model_name='sentence-transformers/all-MiniLM-L6-v2'
        # )