from langchain_ollama import OllamaEmbeddings

class EmbeddingModels:
    def __init__(self):
        # ollama pull mxbai-embed-large
        self.mxbai_embeddings_ollama = OllamaEmbeddings(
            model="mxbai-embed-large"
        )

        self.nomic_embeddings_ollama = OllamaEmbeddings(
            model="nomic-embed-text"
        )