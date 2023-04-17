from sentence_transformers import SentenceTransformer

from .interfaces import EmbeddingInterface


class SentenceBertEmbedding(EmbeddingInterface):
    def __init__(self, model_name="paraphrase-MiniLM-L3-v2"):
        self.model = SentenceTransformer(model_name)

    def embed(self, sentences: list[str]):
        return self.model.encode(sentences)
