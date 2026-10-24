from functools import lru_cache

from secondbrain.config import settings

# bge models are trained to expect this prefix on *queries* (not documents).
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer  # lazy: heavy import

    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    if not texts:
        return []
    vectors = _model().encode(texts, batch_size=batch_size, normalize_embeddings=True)
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    return embed_texts([QUERY_PREFIX + query])[0]
