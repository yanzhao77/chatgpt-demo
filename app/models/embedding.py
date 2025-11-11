import os
# 使用Hugging Face镜像站点加速模型下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Tuple
import torch
from pathlib import Path
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import threading
models_path = Path("C:/AppData/ai/ai_model/BAAI/bge-small-zh-v1.5")
class EmbeddingModel:
    def __init__(self):
        cache_dir = models_path
        os.makedirs(cache_dir, exist_ok=True)
        self.model = SentenceTransformer('BAAI/bge-small-zh-v1.5', cache_folder=cache_dir)
    
    def encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, convert_to_numpy=True)
    
    def encode_queries(self, queries: List[str]) -> np.ndarray:
        """
        对查询文本进行编码
        
        Args:
            queries: 查询文本列表
            
        Returns:
            编码后的向量数组
        """
        return self.model.encode(queries, convert_to_numpy=True)
