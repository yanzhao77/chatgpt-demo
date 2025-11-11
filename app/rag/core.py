import faiss
import numpy as np
from typing import List, Tuple
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import threading
import os

# 设置Hugging Face镜像以加速模型下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

class Reranker:
    _instance = None
    _model = None
    _tokenizer = None
    _lock = threading.Lock()
    
    def __new__(cls, model_name: str = "BAAI/bge-reranker-base"):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(Reranker, cls).__new__(cls)
                    # 设置模型缓存目录
                    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
                    os.makedirs(cache_dir, exist_ok=True)
                    cls._tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
                    cls._model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=cache_dir)
                    cls._model.eval()  # 设置为评估模式
        return cls._instance
    
    def rerank(self, query: str, texts: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
        with torch.no_grad():  # 禁用梯度计算
            pairs = [[query, text] for text in texts]
            inputs = self._tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = self._model(**inputs).logits.view(-1).float()
            results = [(texts[i], float(scores[i])) for i in range(len(texts))]
            # 按分数降序排序
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

class RAGCore:
    def __init__(self):
        self.embedding_model = None
        self.index = None
        self.texts = []
    
    def set_embedding_model(self, model):
        self.embedding_model = model
        
    def create_index(self, dimension: int):
        self.index = faiss.IndexFlatL2(dimension)
        
    def add_texts(self, texts: List[str]):
        if self.embedding_model is None:
            raise ValueError("Embedding model not set")
            
        if self.index is None:
            # 获取嵌入维度
            sample_emb = self.embedding_model.encode(["sample"])
            self.create_index(sample_emb.shape[1])
        
        self.texts.extend(texts)
        embeddings = self.embedding_model.encode(texts)
        self.index.add(embeddings.astype(np.float32))
    
    def search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        
        # 首先使用FAISS获取初步结果（取更多结果用于重排序）
        # 根据规范，应该先检索9个候选文本块
        initial_k = min(k * 3, len(self.texts))  # 获取3倍的结果用于重排序
        query_emb = self.embedding_model.encode([query])
        distances, indices = self.index.search(query_emb.astype(np.float32), initial_k)
        
        # 收集初步检索到的文本
        candidate_texts = []
        for idx in indices[0]:
            if idx < len(self.texts):
                candidate_texts.append(self.texts[idx])
        
        # 如果没有候选文本，返回空
        if not candidate_texts:
            return []
        
        # 如果只需要很少的结果且候选文本不多，可以直接返回
        if len(candidate_texts) <= k:
            results = []
            for i, idx in enumerate(indices[0]):
                if idx < len(self.texts) and len(results) < k:
                    results.append((self.texts[idx], float(distances[0][i])))
            return results
            
        # 使用reranker进行二次排序
        try:
            reranker = Reranker()
            final_results = reranker.rerank(query, candidate_texts, top_k=k)
            return final_results
        except Exception as e:
            # 如果reranker失败，回退到原始FAISS结果
            print(f"Reranker failed: {e}, falling back to FAISS results")
            results = []
            for idx, dist in zip(indices[0][:k], distances[0][:k]):
                if idx < len(self.texts):
                    results.append((self.texts[idx], float(dist)))
            return results