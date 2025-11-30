from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import nltk
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize
import requests
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import threading

# Global model cache to avoid loading models multiple times (saves RAM)
_MODEL_CACHE = {
    'bi_model': None,
    'cross_model': None
}

# Thread locks to ensure thread-safe model loading
_MODEL_LOCKS = {
    'bi_model': threading.Lock(),
    'cross_model': threading.Lock()
}

def _get_bi_model():
    """Get or create bi-encoder model (singleton pattern, thread-safe)"""
    # Double-check locking pattern for thread safety
    if _MODEL_CACHE['bi_model'] is None:
        with _MODEL_LOCKS['bi_model']:
            # Check again after acquiring lock (another thread might have loaded it)
            if _MODEL_CACHE['bi_model'] is None:
                print("Loading SentenceTransformer model (first time, may take a moment)...")
                _MODEL_CACHE['bi_model'] = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                print("✓ SentenceTransformer model loaded")
    return _MODEL_CACHE['bi_model']

def _get_cross_model():
    """Get or create cross-encoder model (singleton pattern, thread-safe)"""
    # Double-check locking pattern for thread safety
    #if _MODEL_CACHE['cross_model'] is None:
        #with _MODEL_LOCKS['cross_model']:
            # Check again after acquiring lock (another thread might have loaded it)
            #if _MODEL_CACHE['cross_model'] is None:
            #    print("Loading CrossEncoder model (first time, may take a moment)...")
            #    _MODEL_CACHE['cross_model'] = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            #    print("✓ CrossEncoder model loaded")
    return _MODEL_CACHE['cross_model']

def clear_model_cache():
    """Clear model cache to free memory (useful when processing many claims) - thread-safe"""
    with _MODEL_LOCKS['bi_model']:
        if _MODEL_CACHE['bi_model'] is not None:
            del _MODEL_CACHE['bi_model']
            _MODEL_CACHE['bi_model'] = None
    with _MODEL_LOCKS['cross_model']:
        if _MODEL_CACHE['cross_model'] is not None:
            del _MODEL_CACHE['cross_model']
            _MODEL_CACHE['cross_model'] = None
    import gc
    gc.collect()
    print("Model cache cleared")

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

def get_top_evidence(claim, text, top_k_chunks=5):
    all_chunks = chunk_text(text)

    if not all_chunks:
        return "No evidence found."
    
    # Use cached models instead of loading each time
    bi_model = _get_bi_model()
    claim_emb = bi_model.encode(claim)
    chunk_embs = bi_model.encode(all_chunks)
    claim_emb /= np.linalg.norm(claim_emb)
    chunk_embs /= np.linalg.norm(chunk_embs, axis=1, keepdims=True)
    cos_sims = np.dot(chunk_embs, claim_emb)
    
    top_indices = np.argsort(-cos_sims)[:top_k_chunks]
    top_chunks = [all_chunks[i] for i in top_indices]
    
    # Use cached cross-encoder model
    cross_model = _get_cross_model()
    pairs = [[claim, ch] for ch in top_chunks]
    scores = cross_model.predict(pairs)
    best_idx = np.argmax(scores)
    best_chunk = top_chunks[best_idx]
    
    # Clear intermediate arrays to free memory
    del claim_emb, chunk_embs, cos_sims, pairs, scores
    
    return best_chunk
