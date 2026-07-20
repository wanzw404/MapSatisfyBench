import logging

import json
import httpx
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """计算两个向量之间的余弦相似度。"""
    vec_a = np.array(a)
    vec_b = np.array(b)
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot_product / (norm_a * norm_b))


class EmbeddingModelProvider():
    def __init__(self) -> None:
        self.base_url = settings.EMBEDDING_MODEL_URL
        self.model = settings.EMBEDDING_MODEL_NAME
        self.dimensions = settings.EMBEDDING_DIMENSIONS
        self.encoding_format = settings.EMBEDDING_ENCODING_FORMAT
        self.api_key = settings.AI_STUDIO_TOKEN

    def text_embedding(self, text: str) -> list[float]:
        """调用 embedding 接口，返回向量列表。"""
        if not self.base_url:
            raise ValueError("EMBEDDING_MODEL_URL is not configured")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "encoding_format": self.encoding_format,
            "input": text,
            "dimensions": self.dimensions,
            "model": self.model,
        }

        response = httpx.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=30.0,
        )
        response.raise_for_status()

        data = response.json()
        # 返回第一个 embedding 的向量
        embedding = data["data"][0]["embedding"]
        logger.debug("[Embedding] model=%s dimensions=%s vector_len=%d", self.model, self.dimensions, len(embedding))
        return embedding
