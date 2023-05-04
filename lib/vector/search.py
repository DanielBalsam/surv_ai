import numpy as np
from enum import StrEnum
import faiss


class VectorSearchType(StrEnum):
    L2 = "l2"
    SCALED_DOT_PRODUCT = "scaled_dot_product"
    COSINE = "cosine"
    FAISS = "faiss"


class VectorSearch:
    @staticmethod
    def l2_distance(a, b):
        a = np.array(a)
        b = np.array(b)
        return np.linalg.norm(a - b, axis=1)

    @staticmethod
    def scaled_dot_product(a, b):
        a = np.array(a)
        b = np.array(b)
        return np.sum(a * b, axis=1) / (
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
        )

    @staticmethod
    def cosine_similarity(a, b):
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b))

    @staticmethod
    def faiss_search(embeddings, query, k=10):
        embeddings = np.array(embeddings)
        index = faiss.IndexIDMap(faiss.IndexFlatIP(embeddings.shape[1]))
        index.add_with_ids(embeddings, np.array(range(0, len(embeddings))))
        search_results = index.search(
            x=np.array([query]),
            k=len(embeddings) if len(embeddings) < k else k,
        )

        return search_results

    @classmethod
    def sort_by_similarity(
        cls,
        embeddings,
        query,
        type=VectorSearchType.FAISS,
        similarity_threshold=0.0,
    ):
        has_embeddings = (
            len(embeddings) > 0
            if isinstance(embeddings, list)
            else embeddings.any()
        )
        has_query = len(query) > 0 if isinstance(query, list) else query.any()

        if not has_embeddings or not has_query:
            return [], []

        if type == VectorSearchType.L2:
            similarities = cls.l2_distance(embeddings, query)
            indices = np.argsort(similarities)
        elif type == VectorSearchType.SCALED_DOT_PRODUCT:
            similarities = cls.scaled_dot_product(embeddings, query)
            indices = np.argsort(similarities)[::-1]
        elif type == VectorSearchType.COSINE:
            similarities = cls.cosine_similarity(embeddings, query)
            indices = np.argsort(similarities)[::-1]
        elif type == VectorSearchType.FAISS:
            similarities, rank = cls.faiss_search(embeddings, query)
            indices = np.argsort(rank[0])[::-1]

        if similarity_threshold:
            normalized_similarities = similarities / np.max(similarities)
            indices = indices[normalized_similarities > similarity_threshold]

        return similarities, indices
