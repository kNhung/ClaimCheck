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
    return SentenceTransformer(model_name, **kwargs)


@lru_cache(maxsize=1)
def _get_cross_model(model_name=_CROSS_MODEL_NAME):
    kwargs = {}
    if _EMBED_DEVICE:
        kwargs["device"] = _EMBED_DEVICE
    return CrossEncoder(model_name, **kwargs)


def get_top_evidence(claim, text, top_k_chunks=5):
    all_chunks = chunk_text(text)

    if not all_chunks:
        return "No evidence found."
    
    # 3. Bi-encoder
    bi_model = _get_bi_model()
    claim_emb = bi_model.encode(claim)
    chunk_embs = bi_model.encode(all_chunks)
    claim_emb /= np.linalg.norm(claim_emb)
    chunk_embs /= np.linalg.norm(chunk_embs, axis=1, keepdims=True)
    cos_sims = np.dot(chunk_embs, claim_emb)
    
    top_indices = np.argsort(-cos_sims)[:top_k_chunks]
    top_chunks = [all_chunks[i] for i in top_indices]
    
    # 4. Cross-encoder re-rank
    cross_model = _get_cross_model()
    pairs = [[claim, ch] for ch in top_chunks]
    scores = cross_model.predict(pairs)
    best_idx = np.argmax(scores)
    best_chunk = top_chunks[best_idx]
    
    # Clear intermediate arrays to free memory
    del claim_emb, chunk_embs, cos_sims, pairs, scores
    
    return best_chunk
