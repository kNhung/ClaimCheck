import os
from functools import lru_cache

from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import nltk
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import requests
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

_EMBED_DEVICE = os.getenv("FACTCHECKER_EMBED_DEVICE")
_BI_MODEL_NAME = os.getenv("FACTCHECKER_BI_ENCODER", "paraphrase-multilingual-MiniLM-L12-v2")
_CROSS_MODEL_NAME = os.getenv("FACTCHECKER_CROSS_ENCODER", "cross-encoder/ms-marco-MiniLM-L-6-v2")

def scrape_text(url):
    try:
        resp = requests.get(url, timeout=5)
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = soup.find_all('p')
        text = " ".join([p.get_text() for p in paragraphs])
        # Clear soup object to free memory
        del soup
        return text
    except:
        return ""

def chunk_text(text, chunk_size=50):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sent in sentences:
        current_chunk += " " + sent
        if len(current_chunk.split()) >= chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = ""
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

@lru_cache(maxsize=1)
def _get_bi_model(model_name=_BI_MODEL_NAME):
    kwargs = {}
    if _EMBED_DEVICE:
        kwargs["device"] = _EMBED_DEVICE
    # Prevent meta device usage in underlying transformers model
    kwargs["model_kwargs"] = {"device_map": None}
    return SentenceTransformer(model_name, **kwargs)


@lru_cache(maxsize=1)
def _get_cross_model(model_name=_CROSS_MODEL_NAME):
    kwargs = {}
    if _EMBED_DEVICE:
        kwargs["device"] = _EMBED_DEVICE
    # Prevent meta device usage in underlying transformers model
    kwargs["model_kwargs"] = {"device_map": None}
    return CrossEncoder(model_name, **kwargs)


def get_top_evidence(claim, text, top_k_chunks=None, p=10, q=3):
    """
    RAV (Retrieval-Augmented Verification) để lấy top evidence từ text.
    
    Args:
        claim: Câu claim cần fact-check
        text: Text cần tìm evidence
        top_k_chunks: (Deprecated) Giữ để backward compatibility. Nếu được set, dùng cho cả p và q.
        p: Số lượng top candidates từ bi-encoder (default: 10)
        q: Số lượng top candidates từ cross-encoder sau khi re-rank (default: 1)
    
    Returns:
        Nếu q=1: str - best chunk
        Nếu q>1: str - các chunks được join lại
    """
    all_chunks = chunk_text(text)

    if not all_chunks:
        return "No evidence found."
    
    # Backward compatibility: nếu top_k_chunks được set, dùng cho cả p và q
    if top_k_chunks is not None:
        p = top_k_chunks
        q = 1
    
    # Đảm bảo p >= q và p không vượt quá số chunks
    p = min(p, len(all_chunks))
    q = min(q, p)
    
    # Step 1: Bi-encoder - lấy top p candidates
    bi_model = _get_bi_model()
    claim_emb = bi_model.encode(claim)
    chunk_embs = bi_model.encode(all_chunks)
    claim_emb /= np.linalg.norm(claim_emb)
    chunk_embs /= np.linalg.norm(chunk_embs, axis=1, keepdims=True)
    cos_sims = np.dot(chunk_embs, claim_emb)
    
    # Lấy top p indices
    top_p_indices = np.argsort(-cos_sims)[:p]
    top_p_chunks = [all_chunks[i] for i in top_p_indices]
    
    # Step 2: Cross-encoder re-rank - lấy top q từ p candidates
    cross_model = _get_cross_model()
    pairs = [[claim, ch] for ch in top_p_chunks]
    cross_scores = cross_model.predict(pairs)
    
    # Lấy top q từ cross-encoder scores
    top_q_indices = np.argsort(-cross_scores)[:q]
    top_q_chunks = [top_p_chunks[i] for i in top_q_indices]
    
    # Trả về kết quả
    if q == 1:
        return top_q_chunks[0]
    else:
        # Join các chunks lại với nhau
        return " ".join(top_q_chunks)
