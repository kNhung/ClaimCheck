# pip install requests sentence-transformers cross-encoder nltk beautifulsoup4

import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import nltk
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
from time import perf_counter
from datetime import datetime
import dotenv
import os

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    # It's okay if python-dotenv isn't installed; we'll rely on real env vars.
    pass

api_key = os.getenv('SERPER_API_KEY')

import warnings
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# -------------------------------
# 1️⃣ Search web bằng Serper
# -------------------------------
def search_web_serper(query, num_results=3, api_key="YOUR_SERPER_API_KEY"):
    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": api_key}
    data = {"q": query, "num": num_results}
    resp = requests.post(url, json=data, headers=headers, timeout=10)
    resp_json = resp.json()
    # Lấy URL từ kết quả
    urls = []
    for r in resp_json.get("organic", []):
        link = r.get("link")
        if link:
            urls.append(link)
    return urls[:num_results]

# -------------------------------
# 2️⃣ Scrape text từ URL
# -------------------------------
def scrape_text(url):
    try:
        resp = requests.get(url, timeout=5)
        soup = BeautifulSoup(resp.text, "html.parser")
        paragraphs = soup.find_all('p')
        text = " ".join([p.get_text() for p in paragraphs])
        return text
    except:
        return ""

# -------------------------------
# 3️⃣ Chunk bài báo
# -------------------------------
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

# -------------------------------
# 4️⃣ Pipeline: claim -> evidence
# -------------------------------
def get_top_evidence_with_url(claim, num_articles=3, top_k_chunks=5, api_key="YOUR_SERPER_API_KEY"):
    # 1. Search web
    urls = search_web_serper(claim, num_results=num_articles, api_key=api_key)
    
    # 2. Scrape text & chunk, lưu kèm URL
    all_chunks = []
    chunk_urls = []
    for url in urls:
        text = scrape_text(url)
        chunks = chunk_text(text)
        all_chunks.extend(chunks)
        chunk_urls.extend([url]*len(chunks))  # lưu URL cho từng chunk
    
    if not all_chunks:
        return "No evidence found.", None
    
    # 3. Bi-encoder
    bi_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    claim_emb = bi_model.encode(claim)
    chunk_embs = bi_model.encode(all_chunks)
    claim_emb /= np.linalg.norm(claim_emb)
    chunk_embs /= np.linalg.norm(chunk_embs, axis=1, keepdims=True)
    cos_sims = np.dot(chunk_embs, claim_emb)
    
    top_indices = np.argsort(-cos_sims)[:top_k_chunks]
    top_chunks = [all_chunks[i] for i in top_indices]
    top_chunk_urls = [chunk_urls[i] for i in top_indices]
    
    # 4. Cross-encoder re-rank
    cross_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    pairs = [[claim, ch] for ch in top_chunks]
    scores = cross_model.predict(pairs)
    best_idx = np.argmax(scores)
    best_chunk = top_chunks[best_idx]
    best_url = top_chunk_urls[best_idx]
    
    return best_chunk, best_url


# -------------------------------
# 5️⃣ Test pipeline
# -------------------------------
claim = "Ngân 98 bị bắt vì bán hàng giả"

start_ts = datetime.now()
start = perf_counter()
print(f"[START] {start_ts.isoformat(timespec='seconds')} | Claim: {claim}")

evidence, url = get_top_evidence_with_url(claim, api_key=api_key)
print("Candidate evidence:\n", evidence)
print("Source URL:", url)

end = perf_counter()
end_ts = datetime.now()
elapsed_s = end - start
print(f"[END]   {end_ts.isoformat(timespec='seconds')} | Elapsed: {elapsed_s:.2f}s (~{elapsed_s/60:.2f} min)")

