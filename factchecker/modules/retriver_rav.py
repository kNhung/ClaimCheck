import os
from functools import lru_cache
from threading import Lock

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

# Global model cache with thread-safe initialization
_bi_model_cache = None
_cross_model_cache = None
_model_lock = Lock()

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

def _get_bi_model(model_name=_BI_MODEL_NAME):
    """Get bi-encoder model with thread-safe caching."""
    global _bi_model_cache
    if _bi_model_cache is None:
        with _model_lock:
            # Double-check pattern to avoid race condition
            if _bi_model_cache is None:
                kwargs = {}
                if _EMBED_DEVICE:
                    kwargs["device"] = _EMBED_DEVICE
                _bi_model_cache = SentenceTransformer(model_name, **kwargs)
    return _bi_model_cache


def _get_cross_model(model_name=_CROSS_MODEL_NAME):
    """Get cross-encoder model with thread-safe caching."""
    global _cross_model_cache
    if _cross_model_cache is None:
        with _model_lock:
            # Double-check pattern to avoid race condition
            if _cross_model_cache is None:
                kwargs = {}
                if _EMBED_DEVICE:
                    kwargs["device"] = _EMBED_DEVICE
                _cross_model_cache = CrossEncoder(model_name, **kwargs)
    return _cross_model_cache


def preload_models():
    """
    Pre-load models to avoid loading them multiple times in multi-threaded scenarios.
    This should be called once before starting parallel processing.
    """
    print("Pre-loading models to avoid multiple loads in threads...")
    try:
        _get_bi_model()
        print("✓ Bi-encoder model pre-loaded")
    except Exception as e:
        print(f"Warning: Failed to pre-load bi-encoder model: {e}")
    
    try:
        _get_cross_model()
        print("✓ Cross-encoder model pre-loaded")
    except Exception as e:
        print(f"Warning: Failed to pre-load cross-encoder model: {e}")


def get_top_evidence(claim, text, top_k_chunks=None, p=10, q=3, log_callback=None):
    """
    RAV (Retrieval-Augmented Verification) để lấy top evidence từ text.
    
    Args:
        claim: Câu claim cần fact-check
        text: Text cần tìm evidence
        top_k_chunks: (Deprecated) Giữ để backward compatibility. Nếu được set, dùng cho cả p và q.
        p: Số lượng top candidates từ bi-encoder (default: 10)
        q: Số lượng top candidates từ cross-encoder sau khi re-rank (default: 1)
        log_callback: Hàm callback để log các bước (optional)
    
    Returns:
        Nếu q=1: str - best chunk
        Nếu q>1: str - các chunks được join lại
    """
    if log_callback:
        log_callback("🔍 BƯỚC 1: Chunking text thành các đoạn nhỏ")
    
    all_chunks = chunk_text(text)
    
    if log_callback:
        log_callback(f"   → Tổng số chunks được tạo: {len(all_chunks)}")
        if len(all_chunks) > 0:
            log_callback(f"   → Độ dài trung bình mỗi chunk: {sum(len(c.split()) for c in all_chunks) / len(all_chunks):.1f} từ")

    if not all_chunks:
        if log_callback:
            log_callback("   ⚠️ Không tìm thấy chunks nào!")
        return "No evidence found."
    
    # Backward compatibility: nếu top_k_chunks được set, dùng cho cả p và q
    if top_k_chunks is not None:
        p = top_k_chunks
        q = 1
    
    # Đảm bảo p >= q và p không vượt quá số chunks
    p = min(p, len(all_chunks))
    q = min(q, p)
    
    if log_callback:
        log_callback(f"\n🔍 BƯỚC 2: Bi-encoder - Lấy top {p} candidates")
        log_callback(f"   → Sử dụng model: {_BI_MODEL_NAME}")
    
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
    top_p_scores = [float(cos_sims[i]) for i in top_p_indices]
    
    if log_callback:
        log_callback(f"   → Top {p} chunks từ bi-encoder (cosine similarity):")
        for idx, (chunk_idx, score) in enumerate(zip(top_p_indices, top_p_scores)):
            # Ghi đầy đủ chunk, không truncate
            log_callback(f"      [{idx+1}] Chunk #{chunk_idx} (score: {score:.4f}): {all_chunks[chunk_idx]}")
    
    if log_callback:
        log_callback(f"\n🔍 BƯỚC 3: Cross-encoder re-rank - Lấy top {q} từ {p} candidates")
        log_callback(f"   → Sử dụng model: {_CROSS_MODEL_NAME}")
    
    # Step 2: Cross-encoder re-rank - lấy top q từ p candidates
    cross_model = _get_cross_model()
    pairs = [[claim, ch] for ch in top_p_chunks]
    cross_scores = cross_model.predict(pairs)
    
    # Lấy top q từ cross-encoder scores
    top_q_indices = np.argsort(-cross_scores)[:q]
    top_q_chunks = [top_p_chunks[i] for i in top_q_indices]
    top_q_scores = [float(cross_scores[i]) for i in top_q_indices]
    
    if log_callback:
        log_callback(f"   → Top {q} chunks sau khi re-rank (cross-encoder scores):")
        for idx, (orig_idx, score) in enumerate(zip(top_q_indices, top_q_scores)):
            # Ghi đầy đủ chunk, không truncate
            log_callback(f"      [{idx+1}] Chunk #{top_p_indices[orig_idx]} (score: {score:.4f}): {top_p_chunks[orig_idx]}")
    
    if log_callback:
        log_callback(f"\n✅ KẾT QUẢ: Đã chọn {q} chunk(s) từ {len(all_chunks)} chunks ban đầu")
    
    # Trả về kết quả
    if q == 1:
        return top_q_chunks[0]
    else:
        # Join các chunks lại với nhau
        return " ".join(top_q_chunks)
