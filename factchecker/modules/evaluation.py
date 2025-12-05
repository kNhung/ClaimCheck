"""
Graph-based evaluation module for fact verification.
Based on "Evidence Retrieval is almost All You Need for Fact Verification" 
and graph-based evidence aggregation approaches like GEAR.
"""

import re
import json
import numpy as np
import networkx as nx
from functools import lru_cache
from threading import Lock
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
import torch
from . import llm
import dotenv
dotenv.load_dotenv()

# Use same models as retriver_rav for consistency
_EMBED_DEVICE = os.getenv("FACTCHECKER_EMBED_DEVICE")
_BI_MODEL_NAME = os.getenv("FACTCHECKER_BI_ENCODER", "paraphrase-multilingual-MiniLM-L12-v2")
_CROSS_MODEL_NAME = os.getenv("FACTCHECKER_CROSS_ENCODER", "cross-encoder/ms-marco-MiniLM-L-6-v2")
_BERT_MODEL_NAME = os.getenv("FACTCHECKER_BERT_MODEL", "distilbert-base-multilingual-cased")

# Global model cache with thread-safe initialization
_bi_model_cache = None
_cross_model_cache = None
_model_lock = Lock()


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
    print("Pre-loading evaluation models to avoid multiple loads in threads...")
    try:
        _get_bi_model()
        print("✓ Evaluation bi-encoder model pre-loaded")
    except Exception as e:
        print(f"Warning: Failed to pre-load bi-encoder model: {e}")
    
    try:
        _get_cross_model()
        print("✓ Evaluation cross-encoder model pre-loaded")
    except Exception as e:
        print(f"Warning: Failed to pre-load cross-encoder model: {e}")


@lru_cache(maxsize=1)
def _get_bert_model_and_tokenizer(model_name=_BERT_MODEL_NAME, allow_fallback=True):
    """
    Get BERT/DistilBERT model and tokenizer for GEAR-style embeddings.
    
    Default model: distilbert-base-multilingual-cased (smaller, faster than BERT)
    Can be overridden via FACTCHECKER_BERT_MODEL environment variable.
    
    Args:
        model_name: Name of the BERT model to load
        allow_fallback: If True, return None instead of raising error (for fallback to bi-encoder)
    
    Returns:
        Tuple of (model, tokenizer, device) or (None, None, device) if fallback allowed
    """
    from transformers import AutoTokenizer, AutoModel
    from transformers.utils import is_offline_mode
    import time
    
    device = _EMBED_DEVICE if _EMBED_DEVICE else "cpu"
    
    # First, try to load from local cache only (fast, no download)
    try:
        print(f"Checking for cached BERT model: {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModel.from_pretrained(
            model_name, 
            local_files_only=True,
            device_map=None  # Prevent meta device usage
        )
        model.to(device)
        model.eval()
        print(f"BERT model loaded from cache successfully")
        return model, tokenizer, device
    except Exception:
        # Model not in cache, need to download
        pass
    
    # If not in cache and network is slow, use fallback
    if allow_fallback:
        print(f"Warning: BERT model '{model_name}' not found in cache.")
        print("Downloading would take a long time with slow network.")
        print("Falling back to bi-encoder for evidence embeddings (use_claim_context will be ignored).")
        print("To use BERT, pre-download the model first or wait for download to complete.")
        return None, None, device
    
    # Try to download (only if fallback not allowed)
    try:
        print(f"Loading BERT model: {model_name}...")
        print("This may take a while if model needs to be downloaded...")
        start_time = time.time()
        
        # Load tokenizer and model with resume_download for interrupted downloads
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            local_files_only=False,
            resume_download=True  # Resume if download was interrupted
        )
        
        model = AutoModel.from_pretrained(
            model_name,
            local_files_only=False,
            resume_download=True,
            device_map=None  # Prevent meta device usage
        )
        
        model.to(device)
        model.eval()  # Set to evaluation mode
        
        elapsed = time.time() - start_time
        print(f"BERT model loaded successfully in {elapsed:.2f}s")
        
        return model, tokenizer, device
        
    except Exception as e:
        if allow_fallback:
            print(f"Error loading BERT model '{model_name}': {e}")
            print("Falling back to bi-encoder for evidence embeddings.")
            return None, None, device
        else:
            print(f"Error loading BERT model '{model_name}': {e}")
            print("This may be due to network issues or insufficient disk space.")
            raise


def get_bert_embeddings(text_pairs: List[Tuple[str, str]], max_length: int = 512, batch_size: int = 8) -> np.ndarray:
    """
    Get BERT [CLS] embeddings for text pairs (GEAR style).
    Falls back to bi-encoder if BERT model is not available.
    
    Args:
        text_pairs: List of (text1, text2) tuples. For evidence-claim pairs: (evidence, claim)
        max_length: Maximum sequence length for tokenization
        batch_size: Batch size for processing (for efficiency)
    
    Returns:
        Array of [CLS] token embeddings (shape: [num_pairs, hidden_dim])
    """
    if not text_pairs:
        return np.array([])
    
    model, tokenizer, device = _get_bert_model_and_tokenizer(allow_fallback=True)
    
    # Fallback to bi-encoder if BERT is not available
    if model is None or tokenizer is None:
        print("Using bi-encoder fallback for evidence embeddings (BERT not available)...")
        bi_model = _get_bi_model()
        embeddings = []
        for ev, claim in text_pairs:
            # Encode concatenated pair with bi-encoder
            # This is not as good as BERT but works without downloading large models
            pair_text = f"{ev} [SEP] {claim}"
            emb = bi_model.encode([pair_text], normalize_embeddings=True)[0]
            embeddings.append(emb)
        return np.array(embeddings)
    
    # Use BERT (original implementation)
    embeddings = []
    
    # Process in batches for efficiency
    with torch.no_grad():
        for i in range(0, len(text_pairs), batch_size):
            batch_pairs = text_pairs[i:i + batch_size]
            
            # Tokenize batch
            texts1 = [pair[0] for pair in batch_pairs]
            texts2 = [pair[1] for pair in batch_pairs]
            
            # Format: [CLS] text1 [SEP] text2 [SEP] (GEAR style)
            encoded = tokenizer(
                texts1, texts2,
                return_tensors='pt',
                truncation=True,
                max_length=max_length,
                padding='max_length'
            )
            
            # Move to device
            encoded = {k: v.to(device) for k, v in encoded.items()}
            
            # Get BERT output
            output = model(**encoded)
            
            # Get [CLS] token embeddings (first token of each sequence)
            # Shape: [batch_size, hidden_dim]
            cls_embs = output.last_hidden_state[:, 0, :].cpu().numpy()
            
            # Normalize embeddings (like GEAR)
            norms = np.linalg.norm(cls_embs, axis=1, keepdims=True)
            cls_embs = cls_embs / (norms + 1e-8)
            
            embeddings.append(cls_embs)
    
    # Concatenate all batches
    if embeddings:
        return np.vstack(embeddings)
    return np.array([])


def extract_claim_from_record(record: str) -> str:
    """
    Extract the claim from the record.
    Only matches claim at the beginning of the record (first 100 lines) to avoid
    matching with "Claim:" that might appear in Action Needed or other sections.
    """
    # Only search in the first 100 lines to avoid matching with "Claim:" in Action Needed
    lines = record.strip().split('\n')
    first_part = '\n'.join(lines[:100])
    
    # Look for "# Claim: ..." pattern in the first part only
    match = re.search(r'^#\s*Claim:\s*(.+?)(?:\n##|\n###|$)', first_part, re.IGNORECASE | re.MULTILINE | re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Fallback: first line if no pattern found
    if lines:
        first_line = lines[0].strip()
        # Remove markdown formatting
        first_line = first_line.replace('#', '').replace('Claim:', '').strip()
        if first_line:
            return first_line
    return ""


def extract_evidence_pieces(record: str) -> List[str]:
    evidence_pieces = []
    
    # Pattern 1: web_search('...'), Summary: ... (format hiện tại)
    pattern1 = re.compile(r"web_search\([^)]+\)\s*,\s*Summary:\s*(.+?)(?=\n\n|\n###|$)", re.DOTALL | re.IGNORECASE)
    matches1 = pattern1.findall(record)
    evidence_pieces.extend([m.strip() for m in matches1 if m.strip()])
    
    # Pattern 2: web_search(...) summary: ... (format cũ, không có dấu phẩy)
    pattern2 = re.compile(r"web_search\([^)]+\)\s+summary:\s*(.+?)(?=\n\n|\n###|$)", re.DOTALL | re.IGNORECASE)
    matches2 = pattern2.findall(record)
    evidence_pieces.extend([m.strip() for m in matches2 if m.strip()])
    
    # Pattern 3: Look for evidence section với cả 2 formats
    # CHỈ lấy actual evidence summaries, BỎ QUA log text và metadata
    evidence_section_match = re.search(r'###\s*Evidence\s*\n\n(.+?)(?=\n###|$)', record, re.DOTALL | re.IGNORECASE)
    if evidence_section_match:
        evidence_text = evidence_section_match.group(1)
        lines = [line.strip() for line in evidence_text.split('\n') if line.strip()]
        for line in lines:
            # Bỏ qua log text và metadata
            if any(skip in line for skip in ['📋', '🔍', '✅', '→', '•', 'BƯỚC', 'WEB SEARCH', 'WEB SCRAPING', 'RAV', 'Chunk #', 'score:', 'Content preview:', 'Snippets preview:', 'URLs:', 'Query:', 'Domain:', 'Content length:', 'Reason:', 'Failed:', 'Output:', 'Input:']):
                continue
            # Xử lý cả Summary: và summary:
            if 'summary:' in line.lower() or 'Summary:' in line:
                parts = re.split(r'summary:\s*', line, 1, flags=re.IGNORECASE)
                if len(parts) > 1:
                    summary_text = parts[1].strip()
                    # Chỉ thêm nếu không phải là log text
                    if not any(skip in summary_text for skip in ['📋', '🔍', '✅', '→', '•', 'BƯỚC']):
                        evidence_pieces.append(summary_text)
            elif not re.match(r'web_search\([^)]+\)', line, re.IGNORECASE):
                # Chỉ thêm nếu là actual evidence content (không phải log)
                if len(line) > 20 and not any(skip in line for skip in ['📋', '🔍', '✅', '→', '•', 'BƯỚC']):
                    evidence_pieces.append(line)
    
    # Remove duplicates
    seen = set()
    unique_evidence = []
    for ev in evidence_pieces:
        ev_lower = ev.lower()
        if ev_lower not in seen and len(ev) > 10:
            seen.add(ev_lower)
            unique_evidence.append(ev)
    
    return unique_evidence


def build_evidence_graph(claim: str, evidence_pieces: List[str], similarity_threshold: float = 0.0, 
                         use_claim_context: bool = True) -> nx.Graph:
    """
    Build a fully-connected evidence graph (GEAR style).
    All evidence nodes are connected to claim and to each other.
    Edges represent semantic similarity between nodes.
    
    Args:
        claim: The claim to verify
        evidence_pieces: List of evidence text pieces
        similarity_threshold: Minimum similarity (for compatibility, but now always connect)
        use_claim_context: If True, evidence embeddings include claim context (GEAR style)
    
    Returns:
        NetworkX graph with nodes and weighted edges (fully-connected)
    """
    G = nx.Graph()
    
    if not evidence_pieces:
        return G
    
    # Get embeddings based on use_claim_context flag
    if use_claim_context:
        # GEAR style: Use BERT for both claim and evidence embeddings
        # This ensures dimension consistency and proper cross-attention
        
        # Get BERT model for claim embedding (GEAR uses BERT(c))
        model, tokenizer, device = _get_bert_model_and_tokenizer(allow_fallback=True)
        
        if model is not None and tokenizer is not None:
            # Use BERT for claim embedding (GEAR style: BERT(c))
            import torch
            with torch.no_grad():
                # Tokenize claim
                encoded_claim = tokenizer(
                    claim,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                    padding='max_length'
                )
                encoded_claim = {k: v.to(device) for k, v in encoded_claim.items()}
                
                # Get [CLS] token embedding
                output = model(**encoded_claim)
                claim_emb = output.last_hidden_state[0, 0, :].cpu().numpy()
                claim_emb = claim_emb / (np.linalg.norm(claim_emb) + 1e-8)
            
            # Get evidence embeddings with claim context (GEAR style: BERT([ei, c]))
            text_pairs = [(ev, claim) for ev in evidence_pieces]
            evidence_embs = get_bert_embeddings(text_pairs)
        else:
            # Fallback: use bi-encoder for both (dimension consistent)
            print("BERT not available, using bi-encoder for both claim and evidence...")
            bi_model = _get_bi_model()
            claim_emb = bi_model.encode([claim], normalize_embeddings=True)[0]
            evidence_embs = bi_model.encode(evidence_pieces, normalize_embeddings=True)
    else:
        # Original: use bi-encoder for both (no claim context)
        bi_model = _get_bi_model()
        claim_emb = bi_model.encode([claim], normalize_embeddings=True)[0]
        evidence_embs = bi_model.encode(evidence_pieces, normalize_embeddings=True)
    
    # Add claim node
    G.add_node('claim', text=claim, embedding=claim_emb, node_type='claim')
    
    # Add evidence nodes
    for idx, (evidence, emb) in enumerate(zip(evidence_pieces, evidence_embs)):
        node_id = f'evidence_{idx}'
        G.add_node(node_id, text=evidence, embedding=emb, node_type='evidence')
        
        # Always connect claim to evidence (fully-connected, GEAR style)
        claim_evidence_sim = float(np.dot(claim_emb, emb))
        # Clamp similarity to [0, 1] for edge weight
        claim_evidence_sim = max(0.0, min(1.0, (claim_evidence_sim + 1) / 2))
        G.add_edge('claim', node_id, weight=claim_evidence_sim, edge_type='claim_evidence')
    
    # FULLY CONNECT evidence nodes (GEAR style)
    for i in range(len(evidence_pieces)):
        for j in range(i + 1, len(evidence_pieces)):
            node_i = f'evidence_{i}'
            node_j = f'evidence_{j}'
            evidence_sim = float(np.dot(evidence_embs[i], evidence_embs[j]))
            # Clamp similarity to [0, 1] for edge weight
            evidence_sim = max(0.0, min(1.0, (evidence_sim + 1) / 2))
            
            # Always add edge (fully-connected)
            G.add_edge(node_i, node_j, weight=evidence_sim, edge_type='evidence_evidence')
    
    # Add self-loops (GEAR style: each node needs information from itself)
    for i in range(len(evidence_pieces)):
        node_id = f'evidence_{i}'
        G.add_edge(node_id, node_id, weight=1.0, edge_type='self_loop')
    G.add_edge('claim', 'claim', weight=1.0, edge_type='self_loop')
    
    return G


def compute_evidence_scores(claim: str, evidence_pieces: List[str]) -> np.ndarray:
    """
    Compute fine-grained claim-evidence alignment scores using cross-encoder.
    
    Returns:
        Array of scores for each evidence piece
    """
    if not evidence_pieces:
        return np.array([])
    
    cross_model = _get_cross_model()
    pairs = [[claim, ev] for ev in evidence_pieces]
    scores = cross_model.predict(pairs)
    return scores


def attention_aggregate(query: np.ndarray, keys: np.ndarray, values: np.ndarray, 
                        edge_weights: np.ndarray = None, temperature: float = 1.0) -> np.ndarray:
    """
    Scaled dot-product attention aggregation (GEAR style).
    
    Args:
        query: Query vector (shape: [dim])
        keys: Key vectors (shape: [num_keys, dim])
        values: Value vectors (shape: [num_keys, dim])
        edge_weights: Optional edge weights to incorporate graph structure (shape: [num_keys])
        temperature: Temperature for softmax (default: 1.0)
    
    Returns:
        Aggregated output vector (shape: [dim])
    """
    if len(keys) == 0:
        return query
    
    # Compute attention scores: (query @ keys.T) / sqrt(dim)
    dim = query.shape[0]
    scores = np.dot(keys, query) / np.sqrt(dim)
    
    # Incorporate edge weights if provided (graph structure bias)
    if edge_weights is not None and len(edge_weights) == len(keys):
        scores = scores + edge_weights * 0.5  # Scale edge weights contribution
    
    # Apply temperature scaling
    scores = scores / temperature
    
    # Softmax to get attention weights
    # Numerical stability: subtract max
    scores_max = np.max(scores)
    exp_scores = np.exp(scores - scores_max)
    attention_weights = exp_scores / (np.sum(exp_scores) + 1e-8)
    
    # Weighted aggregation
    output = np.sum(values * attention_weights.reshape(-1, 1), axis=0)
    
    return output


def detect_contradiction(claim: str, evidence: str) -> float:
    """
    Detect if evidence contradicts claim (not just irrelevant).
    Returns contradiction score [0, 1]
    
    Based on keyword patterns and semantic analysis.
    Cải thiện: Phát hiện khi claim nói "không thể X" nhưng evidence nói "được X" hoặc "có thể X".
    """
    evidence_lower = evidence.lower()
    claim_lower = claim.lower()
    
    # Pattern 1: Explicit negation keywords in Vietnamese (trong evidence)
    negation_patterns = [
        r'\b(không|chưa|không có|không phải|sai|bác bỏ|phủ nhận)\b',
        r'\b(mâu thuẫn|trái ngược|khác với|không đúng|không chính xác)\b',
        r'\b(tuy nhiên|nhưng|mặt khác|ngược lại)\b'
    ]
    
    # Count negation patterns trong evidence
    negation_count = 0
    for pattern in negation_patterns:
        negation_count += len(re.findall(pattern, evidence_lower))
    
    # Pattern 2: Claim nói "không thể X" nhưng evidence nói "được X" hoặc "có thể X"
    # Tìm các pattern phủ định trong claim
    claim_negation_patterns = [
        r'\b(không thể|không được|không cho phép|không có thể|cấm|không được phép)\b',
        r'\b(không có|không tồn tại|không xảy ra)\b'
    ]
    
    # Tìm các pattern khẳng định trong evidence (đối lập với claim)
    evidence_positive_patterns = [
        r'\b(được|được phép|được cho phép|có thể|cho phép|được sử dụng|được kết hợp)\b',
        r'\b(có|có thể|có khả năng|tồn tại|xảy ra)\b'
    ]
    
    # Kiểm tra contradiction: claim nói "không thể X" nhưng evidence nói "được X"
    claim_has_negation = any(re.search(pattern, claim_lower) for pattern in claim_negation_patterns)
    evidence_has_positive = any(re.search(pattern, evidence_lower) for pattern in evidence_positive_patterns)
    
    # Pattern 3: Check for entity mismatch (same entities but different facts)
    claim_keywords = set(re.findall(r'\b\w{4,}\b', claim_lower))  # Words with 4+ chars
    evidence_keywords = set(re.findall(r'\b\w{4,}\b', evidence_lower))
    
    common_keywords = claim_keywords.intersection(evidence_keywords)
    has_common_entities = len(common_keywords) > 2  # At least 3 common keywords
    
    # Tính contradiction score
    contradiction_score = 0.0
    
    # Case 1: Evidence có từ phủ định + cùng entities → Strong contradiction
    if negation_count > 0 and has_common_entities:
        contradiction_score = max(contradiction_score, min(1.0, 0.4 + negation_count * 0.2))
    
    # Case 2: Claim nói "không thể X" nhưng evidence nói "được X" → Strong contradiction
    # Đây là contradiction rõ ràng nhất: claim phủ định nhưng evidence khẳng định
    if claim_has_negation and evidence_has_positive and has_common_entities:
        contradiction_score = max(contradiction_score, 0.75)  # Very high contradiction score
    
    # Case 3: Claim nói "không thể X" nhưng evidence nói "được X" (không cần common entities)
    # Vẫn là contradiction nếu có cùng chủ đề
    if claim_has_negation and evidence_has_positive and len(common_keywords) > 1:
        contradiction_score = max(contradiction_score, 0.65)
    
    # Case 4: Evidence có từ phủ định nhưng không có common entities → Weak contradiction
    if negation_count > 0 and not has_common_entities:
        contradiction_score = max(contradiction_score, min(0.5, negation_count * 0.15))
    
    return min(1.0, contradiction_score)


def evidence_action_needed_network(G: nx.Graph, claim_emb: np.ndarray, 
                                evidence_embs: List[np.ndarray], 
                                num_layers: int = 2, use_attention: bool = True) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    ERNet-like message passing for evidence action_needed (GEAR style).
    Multiple layers of information propagation between claim and evidence nodes.
    
    Args:
        G: Graph with claim and evidence nodes
        claim_emb: Initial claim embedding
        evidence_embs: List of initial evidence embeddings
        num_layers: Number of message passing layers
    
    Returns:
        Tuple of (refined_claim_emb, refined_evidence_embs) after message passing
    """
    if not evidence_embs:
        return claim_emb, []
    
    # Initialize node representations
    node_reps = {
        'claim': claim_emb.copy()
    }
    for i, emb in enumerate(evidence_embs):
        node_reps[f'evidence_{i}'] = emb.copy()
    
    # Multiple layers of message passing
    for layer in range(num_layers):
        new_reps = {}
        
        # Update claim representation from evidence neighbors
        evidence_neighbors = [n for n in G.neighbors('claim') if n.startswith('evidence_')]
        if evidence_neighbors:
            neighbor_reps = np.array([node_reps[n] for n in evidence_neighbors])
            edge_weights = np.array([G['claim'][n].get('weight', 0.0) for n in evidence_neighbors])
            
            if use_attention:
                # Attention-based aggregation (GEAR style)
                if len(neighbor_reps) > 0:
                    # Query: claim, Keys/Values: evidence neighbors
                    aggregated = attention_aggregate(
                        query=node_reps['claim'],
                        keys=neighbor_reps,
                        values=neighbor_reps,
                        edge_weights=edge_weights,
                        temperature=1.0
                    )
                    # Residual connection
                    new_reps['claim'] = 0.7 * node_reps['claim'] + 0.3 * aggregated
                else:
                    new_reps['claim'] = node_reps['claim']
            else:
                # Original weighted aggregation (backward compatibility)
                if len(neighbor_reps) > 0 and np.sum(edge_weights) > 0:
                    edge_weights_norm = edge_weights / (np.sum(edge_weights) + 1e-8)
                    weighted_neighbors = np.sum(neighbor_reps * edge_weights_norm.reshape(-1, 1), axis=0)
                    new_reps['claim'] = 0.7 * node_reps['claim'] + 0.3 * weighted_neighbors
                else:
                    new_reps['claim'] = node_reps['claim']
            
            # Normalize claim embedding
            claim_norm = np.linalg.norm(new_reps['claim'])
            if claim_norm > 0:
                new_reps['claim'] = new_reps['claim'] / claim_norm
        else:
            new_reps['claim'] = node_reps['claim']
        
        # Update evidence representations from claim and other evidence
        for i in range(len(evidence_embs)):
            node_id = f'evidence_{i}'
            if node_id not in G:
                new_reps[node_id] = node_reps[node_id]
                continue
            
            neighbors = list(G.neighbors(node_id))
            
            # Aggregate from claim
            if G.has_edge(node_id, 'claim'):
                claim_weight = G[node_id]['claim'].get('weight', 0.0)
                if use_attention:
                    # Use attention for claim contribution
                    claim_contribution = attention_aggregate(
                        query=node_reps[node_id],
                        keys=node_reps['claim'].reshape(1, -1),
                        values=node_reps['claim'].reshape(1, -1),
                        edge_weights=np.array([claim_weight]),
                        temperature=1.0
                    )
                else:
                    claim_contribution = node_reps['claim'] * claim_weight
            else:
                claim_contribution = np.zeros_like(node_reps[node_id])
            
            # Aggregate from other evidence neighbors (including self-loop)
            # In GEAR, self-loops are included in neighbors
            evidence_neighbors = [n for n in neighbors if n.startswith('evidence_')]
            # Note: self-loop (node_id -> node_id) is already included in neighbors if it exists
            if evidence_neighbors:
                neighbor_reps = np.array([node_reps[n] for n in evidence_neighbors])
                edge_weights = np.array([G[node_id][n].get('weight', 0.0) for n in evidence_neighbors])
                
                if use_attention:
                    # Attention-based aggregation for evidence neighbors
                    if len(neighbor_reps) > 0:
                        evidence_contribution = attention_aggregate(
                            query=node_reps[node_id],
                            keys=neighbor_reps,
                            values=neighbor_reps,
                            edge_weights=edge_weights,
                            temperature=1.0
                        )
                    else:
                        evidence_contribution = np.zeros_like(node_reps[node_id])
                else:
                    # Original weighted aggregation (backward compatibility)
                    if len(neighbor_reps) > 0 and np.sum(edge_weights) > 0:
                        edge_weights_norm = edge_weights / (np.sum(edge_weights) + 1e-8)
                        evidence_contribution = np.sum(neighbor_reps * edge_weights_norm.reshape(-1, 1), axis=0)
                    else:
                        evidence_contribution = np.zeros_like(node_reps[node_id])
            else:
                evidence_contribution = np.zeros_like(node_reps[node_id])
            
            # Combine: self + claim + evidence neighbors
            new_reps[node_id] = (
                0.5 * node_reps[node_id] + 
                0.3 * claim_contribution + 
                0.2 * evidence_contribution
            )
            
            # Normalize to keep embedding norm stable
            norm = np.linalg.norm(new_reps[node_id])
            if norm > 0:
                new_reps[node_id] = new_reps[node_id] / norm
        
        node_reps = new_reps
    
    return node_reps['claim'], [node_reps[f'evidence_{i}'] for i in range(len(evidence_embs))]


def compute_support_refute_scores_neural(claim: str, evidence_pieces: List[str],
                                         claim_emb: np.ndarray,
                                         refined_evidence_embs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute support/refute scores using CrossEncoder (neural approach, GEAR style).
    
    Args:
        claim: The claim text
        evidence_pieces: List of evidence text pieces
        claim_emb: Claim embedding (after message passing)
        refined_evidence_embs: List of refined evidence embeddings (after message passing)
    
    Returns:
        Tuple of (support_scores, refute_scores) arrays
    """
    if not evidence_pieces:
        return np.array([]), np.array([])
    
    support_scores = []
    refute_scores = []
    
    # Use CrossEncoder to get relevance scores
    cross_model = _get_cross_model()
    pairs = [[claim, ev] for ev in evidence_pieces]
    relevance_scores = cross_model.predict(pairs)
    
    # Also compute semantic similarity from embeddings
    for i, ev in enumerate(evidence_pieces):
        if i >= len(refined_evidence_embs):
            support_scores.append(0.0)
            refute_scores.append(0.0)
            continue
        
        ev_emb = refined_evidence_embs[i]
        
        # Semantic similarity from embeddings
        embedding_sim = float(np.dot(claim_emb, ev_emb))
        embedding_sim = max(0.0, (embedding_sim + 1.0) / 2.0)  # Normalize to [0, 1]
        
        # Combine CrossEncoder score and embedding similarity
        # CrossEncoder scores are typically in range [-1, 1] or [0, 1], normalize to [0, 1]
        cross_score = float(relevance_scores[i])
        if cross_score < 0:
            cross_score = (cross_score + 1.0) / 2.0  # Normalize from [-1, 1] to [0, 1]
        else:
            cross_score = min(1.0, max(0.0, cross_score))  # Clamp to [0, 1]
        
        # Combined relevance: weighted average
        relevance = 0.6 * cross_score + 0.4 * embedding_sim
        
        # For neural approach, we use CrossEncoder to detect contradiction
        # Rule-based contradiction detection (keep full weight for better detection)
        rule_contradiction = detect_contradiction(claim, ev)
        
        # Neural-based contradiction signals:
        # 1. If cross_score is negative or very low but embedding similarity is high
        neural_contradiction_1 = 0.0
        if cross_score < 0.4 and embedding_sim > 0.5:
            neural_contradiction_1 = 0.5 * (0.5 - cross_score)  # Stronger signal if cross_score is lower
        
        # 2. If relevance is high but cross_score is low (mismatch indicates contradiction)
        neural_contradiction_2 = 0.0
        if relevance > 0.6 and cross_score < 0.4:
            neural_contradiction_2 = 0.3 * (relevance - cross_score)
        
        # Combine contradiction signals (take maximum to be more sensitive)
        contradiction = max(rule_contradiction, neural_contradiction_1, neural_contradiction_2)
        # Boost contradiction if multiple signals agree
        if rule_contradiction > 0.3 and (neural_contradiction_1 > 0.2 or neural_contradiction_2 > 0.2):
            contradiction = min(1.0, contradiction * 1.3)  # Boost by 30%
        
        # Support score: high relevance AND low contradiction
        support = relevance * (1.0 - contradiction * 0.9)  # Increase penalty for contradiction
        
        # Refute score: high relevance AND high contradiction
        # Boost refute score more aggressively when contradiction is detected
        refute = relevance * contradiction
        if contradiction > 0.5 and relevance > 0.5:
            refute = min(1.0, refute * 1.2)  # Boost refute score for strong contradictions
        
        support_scores.append(max(0.0, min(1.0, support)))
        refute_scores.append(max(0.0, min(1.0, refute)))
    
    return np.array(support_scores), np.array(refute_scores)


def compute_support_refute_scores(claim: str, evidence_pieces: List[str],
                                   claim_emb: np.ndarray,
                                   refined_evidence_embs: List[np.ndarray],
                                   use_neural_classifier: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute separate support and refute scores (GEAR style).
    Supports both neural (CrossEncoder) and rule-based methods.
    
    Args:
        claim: The claim text
        evidence_pieces: List of evidence text pieces
        claim_emb: Claim embedding (after message passing)
        refined_evidence_embs: List of refined evidence embeddings (after message passing)
        use_neural_classifier: If True, use CrossEncoder-based neural classification (default: True)
    
    Returns:
        Tuple of (support_scores, refute_scores) arrays
    """
    if use_neural_classifier:
        return compute_support_refute_scores_neural(claim, evidence_pieces, claim_emb, refined_evidence_embs)
    
    # Rule-based method (backward compatibility)
    support_scores = []
    refute_scores = []
    
    for i, ev in enumerate(evidence_pieces):
        if i >= len(refined_evidence_embs):
            support_scores.append(0.0)
            refute_scores.append(0.0)
            continue
            
        ev_emb = refined_evidence_embs[i]
        
        # Relevance score: how related is evidence to claim (after message passing)
        relevance = float(np.dot(claim_emb, ev_emb))
        # Normalize to [0, 1] (cosine similarity after normalization is already in [-1, 1])
        relevance = max(0.0, (relevance + 1.0) / 2.0)
        
        # Contradiction detection
        contradiction = detect_contradiction(claim, ev)
        
        # Support score: high relevance AND low contradiction
        support = relevance * (1.0 - contradiction * 0.8)  # Reduce support if contradiction
        
        # Refute score: high relevance AND high contradiction
        # Only count as refute if evidence is relevant AND contradicts
        refute = relevance * contradiction
        
        support_scores.append(max(0.0, min(1.0, support)))
        refute_scores.append(max(0.0, min(1.0, refute)))
    
    return np.array(support_scores), np.array(refute_scores)


def compute_attention_weights(claim_emb: np.ndarray, evidence_embs: List[np.ndarray], 
                              G: nx.Graph) -> np.ndarray:
    """
    Compute attention weights for evidence aggregation (GEAR style).
    Claim embedding (query) attends to evidence embeddings (keys/values).
    
    Args:
        claim_emb: Claim embedding (query)
        evidence_embs: List of evidence embeddings (keys/values)
        G: Evidence graph for incorporating graph structure
    
    Returns:
        Attention weights array (normalized, sums to 1)
    """
    if not evidence_embs or len(evidence_embs) == 0:
        return np.array([])
    
    evidence_embs_array = np.array(evidence_embs)
    dim = claim_emb.shape[0]
    
    # Compute attention scores: (claim_emb @ evidence_embs.T) / sqrt(dim)
    scores = np.dot(evidence_embs_array, claim_emb) / np.sqrt(dim)
    
    # Incorporate graph structure: add edge weights as bias
    graph_bias = np.zeros(len(evidence_embs))
    for i in range(len(evidence_embs)):
        node_id = f'evidence_{i}'
        if node_id in G and G.has_edge('claim', node_id):
            edge_weight = G['claim'][node_id].get('weight', 0.0)
            graph_bias[i] = edge_weight * 0.3  # Scale graph bias contribution
    
    # Also consider evidence-evidence connections (consensus)
    for i in range(len(evidence_embs)):
        node_id = f'evidence_{i}'
        if node_id in G:
            evidence_neighbors = [n for n in G.neighbors(node_id) 
                                if n.startswith('evidence_') and n != node_id]
            if evidence_neighbors:
                # Average edge weights to other evidence (consensus signal)
                neighbor_weights = [G[node_id][nbr].get('weight', 0.0) for nbr in evidence_neighbors]
                if neighbor_weights:
                    consensus_score = np.mean(neighbor_weights)
                    graph_bias[i] += consensus_score * 0.2  # Add consensus contribution
    
    # Combine semantic scores with graph structure
    scores = scores + graph_bias
    
    # Softmax to get attention weights
    scores_max = np.max(scores)
    exp_scores = np.exp(scores - scores_max)
    attention_weights = exp_scores / (np.sum(exp_scores) + 1e-8)
    
    return attention_weights


def max_aggregate(evidence_embs: List[np.ndarray]) -> np.ndarray:
    """
    Max aggregator (GEAR style).
    Performs element-wise Max operation among evidence embeddings.
    
    Args:
        evidence_embs: List of evidence embeddings
    
    Returns:
        Aggregated embedding (element-wise max)
    """
    if not evidence_embs:
        return np.array([])
    evidence_array = np.array(evidence_embs)
    return np.max(evidence_array, axis=0)


def mean_aggregate(evidence_embs: List[np.ndarray]) -> np.ndarray:
    """
    Mean aggregator (GEAR style).
    Performs element-wise Mean operation among evidence embeddings.
    
    Args:
        evidence_embs: List of evidence embeddings
    
    Returns:
        Aggregated embedding (element-wise mean)
    """
    if not evidence_embs:
        return np.array([])
    evidence_array = np.array(evidence_embs)
    return np.mean(evidence_array, axis=0)


def compute_evidence_importance_weights(G: nx.Graph, num_evidence: int, 
                                         claim_emb: np.ndarray = None,
                                         evidence_embs: List[np.ndarray] = None,
                                         use_attention: bool = True) -> np.ndarray:
    """
    Compute importance weights for evidence (GEAR style).
    Supports both attention-based and connectivity-based methods.
    
    Args:
        G: Evidence graph
        num_evidence: Number of evidence pieces
        claim_emb: Claim embedding (for attention-based method)
        evidence_embs: List of evidence embeddings (for attention-based method)
        use_attention: If True, use attention mechanism (default: True)
    
    Returns:
        Normalized importance weights array
    """
    if use_attention and claim_emb is not None and evidence_embs is not None and len(evidence_embs) == num_evidence:
        # Attention-based weights (GEAR style)
        return compute_attention_weights(claim_emb, evidence_embs, G)
    
    # Connectivity-based weights (backward compatibility)
    weights = np.ones(num_evidence)
    
    for i in range(num_evidence):
        node_id = f'evidence_{i}'
        if node_id in G:
            neighbors = list(G.neighbors(node_id))
            
            # Weight based on degree and edge weights
            if neighbors:
                edge_weights = [G[node_id][nbr].get('weight', 0.0) for nbr in neighbors]
                if edge_weights:
                    weights[i] = 1.0 + np.mean(edge_weights)
    
    # Normalize to sum to 1 (attention weights)
    weights = weights / (weights.sum() + 1e-8)
    return weights


def aggregate_evidence_with_graph(G: nx.Graph, claim: str, evidence_pieces: List[str],
                                   use_attention: bool = True, use_neural_classifier: bool = True,
                                   aggregator_type: str = "attention") -> Dict[str, float]:
    """
    GEAR-style evidence aggregation with message passing.
    FIXED: Low alignment != Refutation (low alignment = irrelevant/neutral)
    
    Uses:
    - Fully-connected graph (all evidence connected)
    - Message passing (ERNet-like) for multi-step action_needed
    - Separate support/refute detection (not based on alignment alone)
    - Attention-based aggregation
    
    Returns:
        Dictionary with aggregated scores: support_score, refute_score, neutral_score
    """
    if not evidence_pieces or len(G.nodes()) == 0:
        return {
            'support_score': 0.0, 
            'refute_score': 0.0, 
            'neutral_score': 1.0,
            'mean_alignment': 0.0,
            'num_evidence': len(evidence_pieces)
        }
    
    # Get initial embeddings from graph nodes
    claim_emb = G.nodes['claim'].get('embedding', None)
    evidence_embs = []
    
    for i in range(len(evidence_pieces)):
        node_id = f'evidence_{i}'
        if node_id in G:
            evidence_embs.append(G.nodes[node_id].get('embedding', None))
        else:
            evidence_embs.append(None)
    
    # If embeddings not in graph, compute them
    if claim_emb is None or any(emb is None for emb in evidence_embs):
        bi_model = _get_bi_model()
        all_texts = [claim] + evidence_pieces
        embeddings = bi_model.encode(all_texts, normalize_embeddings=True)
        claim_emb = embeddings[0]
        evidence_embs = embeddings[1:]
    
    # STEP 1: Message passing (ERNet-like, GEAR style)
    refined_claim_emb, refined_evidence_embs = evidence_action_needed_network(
        G, claim_emb, evidence_embs, num_layers=2, use_attention=use_attention
    )
    
    # STEP 2: Compute support/refute scores (separate, not from alignment)
    support_scores, refute_scores = compute_support_refute_scores(
        claim, evidence_pieces, refined_claim_emb, refined_evidence_embs,
        use_neural_classifier=use_neural_classifier
    )
    
    # STEP 3: Aggregate evidence using different strategies (GEAR style)
    if aggregator_type == "max":
        # Max aggregator: element-wise max of evidence embeddings
        aggregated_emb = max_aggregate(refined_evidence_embs)
        # For scores, use max of support/refute scores
        support_score = float(np.max(support_scores)) if len(support_scores) > 0 else 0.0
        refute_score = float(np.max(refute_scores)) if len(refute_scores) > 0 else 0.0
    elif aggregator_type == "mean":
        # Mean aggregator: element-wise mean of evidence embeddings
        aggregated_emb = mean_aggregate(refined_evidence_embs)
        # For scores, use mean of support/refute scores
        support_score = float(np.mean(support_scores)) if len(support_scores) > 0 else 0.0
        refute_score = float(np.mean(refute_scores)) if len(refute_scores) > 0 else 0.0
    else:
        # Attention aggregator (default): weighted aggregation using importance weights
        evidence_weights = compute_evidence_importance_weights(
            G, len(evidence_pieces), 
            claim_emb=refined_claim_emb,
            evidence_embs=refined_evidence_embs,
            use_attention=use_attention
        )
        # Weighted aggregation using importance weights
        support_score = float(np.sum(support_scores * evidence_weights))
        refute_score = float(np.sum(refute_scores * evidence_weights))
        # For neural classification: compute aggregated embedding using weighted mean
        if refined_evidence_embs:
            evidence_embs_array = np.array(refined_evidence_embs)
            # Weighted mean: sum(weight_i * emb_i) / sum(weights)
            aggregated_emb = np.average(evidence_embs_array, axis=0, weights=evidence_weights)
        else:
            aggregated_emb = None
    
    # Mean alignment: average relevance score (not support + refute)
    # This should represent how relevant evidence is to claim, not the sum of support+refute
    # Use the maximum of support or refute as relevance indicator
    relevance_scores = np.maximum(support_scores, refute_scores)  # Max of support/refute per evidence
    relevant_mask = relevance_scores > 0.1  # Evidence that is relevant
    if np.any(relevant_mask):
        mean_alignment = float(np.mean(relevance_scores[relevant_mask]))
    else:
        # No relevant evidence - all irrelevant
        mean_alignment = float(np.mean(relevance_scores))
    
    # Clamp mean_alignment to [0, 1] to avoid values > 1.0
    mean_alignment = max(0.0, min(1.0, mean_alignment))
    
    # STEP 5: Normalize to probabilities
    total = support_score + refute_score + 0.1  # Add small neutral component
    if total > 0:
        support_score = support_score / total
        refute_score = refute_score / total
    else:
        support_score = 0.0
        refute_score = 0.0
    
    neutral_score = 1.0 - support_score - refute_score
    
    # For neural classification: also return aggregated embedding and refined claim embedding
    # If aggregated_emb is None (attention aggregator), compute it from refined evidence embeddings
    if aggregated_emb is None:
        if refined_evidence_embs:
            # Use mean aggregation as default for neural classifier input
            aggregated_emb = mean_aggregate(refined_evidence_embs)
        else:
            aggregated_emb = None
    
    return {
        'support_score': float(support_score),
        'refute_score': float(refute_score),
        'neutral_score': float(neutral_score),
        'mean_alignment': float(mean_alignment),
        'num_evidence': len(evidence_pieces),
        # Embeddings for neural classification (GEAR style)
        'aggregated_emb': aggregated_emb,
        'refined_claim_emb': refined_claim_emb
    }


def _initialize_neural_classifier(input_dim: int, hidden_dim: int = 128, output_dim: int = 3, seed: int = 42):
    """
    Initialize neural classifier weights (MLP) with Xavier initialization.
    GEAR-style: 2-layer MLP for final classification.
    
    Args:
        input_dim: Dimension of input features (aggregated_emb + claim_emb)
        hidden_dim: Hidden layer dimension
        output_dim: Output dimension (3 classes: Supported, Refuted, Not Enough Evidence)
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary with weights and biases: W1, b1, W2, b2
    """
    np.random.seed(seed)
    
    # Xavier/Glorot initialization
    limit1 = np.sqrt(6.0 / (input_dim + hidden_dim))
    W1 = np.random.uniform(-limit1, limit1, (input_dim, hidden_dim))
    b1 = np.zeros(hidden_dim)
    
    limit2 = np.sqrt(6.0 / (hidden_dim + output_dim))
    W2 = np.random.uniform(-limit2, limit2, (hidden_dim, output_dim))
    b2 = np.zeros(output_dim)
    
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


@lru_cache(maxsize=1)
def _get_neural_classifier_weights():
    """
    Get or initialize neural classifier weights.
    Cached to avoid re-initialization on every call.
    """
    # Default dimensions: assume 768-dim embeddings (BERT/DistilBERT)
    # If using bi-encoder, will be 384-dim, but we'll handle that dynamically
    return None  # Will be initialized on first use with correct dimensions


def classify_verdict_neural(aggregated_scores: Dict[str, float], 
                           aggregated_emb: np.ndarray = None,
                           refined_claim_emb: np.ndarray = None,
                           hidden_dim: int = 128) -> Tuple[str, str]:
    """
    Neural classification layer (GEAR style).
    Uses MLP to classify based on aggregated evidence and claim embeddings.
    
    Args:
        aggregated_scores: Dictionary with support_score, refute_score, etc.
        aggregated_emb: Aggregated evidence embedding (from aggregation step)
        refined_claim_emb: Refined claim embedding (after message passing)
        hidden_dim: Hidden layer dimension for MLP
    
    Returns:
        Tuple of (verdict, justification)
    """
    num_evidence = aggregated_scores.get('num_evidence', 0)
    
    # Check if we have enough evidence
    if num_evidence < 1:
        justification = f"Không có đủ bằng chứng để xác nhận hoặc bác bỏ. Chỉ tìm thấy {num_evidence} bằng chứng."
        return "Not Enough Evidence", justification
    
    # Prepare input features
    if aggregated_emb is not None and refined_claim_emb is not None:
        # Use embeddings (GEAR style): concatenate aggregated evidence + claim embeddings
        if len(aggregated_emb.shape) == 0:
            aggregated_emb = np.array([aggregated_emb])
        if len(refined_claim_emb.shape) == 0:
            refined_claim_emb = np.array([refined_claim_emb])
        
        # Ensure same dimension (handle dimension mismatch)
        min_dim = min(len(aggregated_emb), len(refined_claim_emb))
        aggregated_emb = aggregated_emb[:min_dim]
        refined_claim_emb = refined_claim_emb[:min_dim]
        
        combined_emb = np.concatenate([aggregated_emb, refined_claim_emb])
    else:
        # Fallback: use scores as features (when embeddings not available)
        support_score = aggregated_scores.get('support_score', 0.0)
        refute_score = aggregated_scores.get('refute_score', 0.0)
        mean_alignment = aggregated_scores.get('mean_alignment', 0.0)
        neutral_score = aggregated_scores.get('neutral_score', 0.0)
        
        # Normalize num_evidence
        num_evidence_norm = min(num_evidence / 10.0, 1.0)
        
        combined_emb = np.array([
            support_score,
            refute_score,
            neutral_score,
            mean_alignment,
            num_evidence_norm
        ])
    
    input_dim = len(combined_emb)
    output_dim = 3  # Supported, Refuted, Not Enough Evidence
    
    # Initialize weights (with correct input dimension)
    weights = _initialize_neural_classifier(input_dim, hidden_dim, output_dim)
    
    # Forward pass through MLP
    # Layer 1: hidden = tanh(W1 @ x + b1)
    hidden = np.tanh(np.dot(combined_emb, weights['W1']) + weights['b1'])
    
    # Layer 2: logits = W2 @ hidden + b2
    logits = np.dot(hidden, weights['W2']) + weights['b2']
    
    # Softmax to get probabilities
    exp_logits = np.exp(logits - np.max(logits))  # Numerical stability
    probs = exp_logits / (np.sum(exp_logits) + 1e-8)
    
    # Get predicted class
    class_idx = int(np.argmax(probs))
    verdicts = ["Supported", "Refuted", "Not Enough Evidence"]
    verdict = verdicts[class_idx]
    
    # Create justification
    justification = (
        f"Neural classification (GEAR style): "
        f"Supported: {probs[0]:.2f}, Refuted: {probs[1]:.2f}, "
        f"Not Enough Evidence: {probs[2]:.2f}. "
        f"Predicted: {verdict} (confidence: {probs[class_idx]:.2f})"
    )
    
    return verdict, justification


def classify_verdict(aggregated_scores: Dict[str, float], 
                     min_evidence_threshold: int = 1,
                     use_neural_classifier: bool = False) -> Tuple[str, str]:
    """
    Classify verdict based on aggregated scores.
    
    Args:
        aggregated_scores: Dictionary with scores and embeddings
        min_evidence_threshold: Minimum number of evidence pieces required
        use_neural_classifier: If True, use neural MLP classifier (GEAR style)
                              If False, use rule-based classification (default)
    
    Returns:
        Tuple of (verdict, justification)
    """
    # If neural classifier requested, try to use it
    if use_neural_classifier:
        aggregated_emb = aggregated_scores.get('aggregated_emb', None)
        refined_claim_emb = aggregated_scores.get('refined_claim_emb', None)
        
        # Use neural classifier if embeddings available
        if aggregated_emb is not None or refined_claim_emb is not None:
            return classify_verdict_neural(
                aggregated_scores,
                aggregated_emb=aggregated_emb,
                refined_claim_emb=refined_claim_emb
            )
        # Fallback to rule-based if embeddings not available
        # (will continue to rule-based classification below)
    
    # Rule-based classification (original method)
    support_score = aggregated_scores['support_score']
    refute_score = aggregated_scores['refute_score']
    neutral_score = aggregated_scores['neutral_score']
    num_evidence = aggregated_scores.get('num_evidence', 0)
    mean_alignment = aggregated_scores.get('mean_alignment', 0.0)
    
    # Check if we have enough evidence
    if num_evidence < min_evidence_threshold:
        justification = f"Không có đủ bằng chứng để xác nhận hoặc bác bỏ. Chỉ tìm thấy {num_evidence} bằng chứng."
        return "Not Enough Evidence", justification
    
    # Decision logic with adaptive thresholds (cải thiện)
    score_diff = support_score - refute_score
    
    # Adaptive thresholds dựa trên alignment (cải thiện để phát hiện tốt hơn)
    if mean_alignment > 0.9:  # Evidence rất liên quan
        support_threshold = 0.50
        refute_threshold = 0.40  # Giảm threshold cho refute để dễ phát hiện hơn
        diff_threshold = 0.10
    elif mean_alignment > 0.7:  # Evidence khá liên quan
        support_threshold = 0.52  # Giảm từ 0.55 xuống 0.52
        refute_threshold = 0.45  # Giảm từ 0.50 xuống 0.45
        diff_threshold = 0.12
    elif mean_alignment > 0.5:  # Evidence trung bình
        support_threshold = 0.55
        refute_threshold = 0.48
        diff_threshold = 0.15
    else:  # Evidence ít liên quan
        support_threshold = 0.58  # Giảm từ 0.60 xuống 0.58
        refute_threshold = 0.50  # Giảm từ 0.55 xuống 0.50
        diff_threshold = 0.18
    
    # ƯU TIÊN 1: Refuted - kiểm tra refute trước (vì refutation thường rõ ràng hơn)
    # Refuted: clear refutation với threshold thấp hơn
    if refute_score > refute_threshold and score_diff < -diff_threshold:
        justification = f"Bằng chứng cho thấy yêu cầu bị bác bỏ (điểm hỗ trợ: {support_score:.2f}, điểm bác bỏ: {refute_score:.2f}, điểm căn chỉnh trung bình: {mean_alignment:.2f})."
        return "Refuted", justification
    
    # Refuted với threshold thấp hơn nếu alignment cao (evidence rất liên quan)
    # Cải thiện: giảm threshold và điều kiện để phát hiện refuted tốt hơn
    if mean_alignment > 0.7 and refute_score > 0.35 and refute_score > support_score and score_diff < -0.08:
        justification = f"Bằng chứng cho thấy yêu cầu bị bác bỏ (điểm hỗ trợ: {support_score:.2f}, điểm bác bỏ: {refute_score:.2f}, điểm căn chỉnh trung bình: {mean_alignment:.2f})."
        return "Refuted", justification
    
    # Thêm điều kiện: nếu refute_score cao hơn support_score đáng kể, có thể là refuted
    if refute_score > 0.45 and refute_score > support_score * 1.3 and mean_alignment > 0.6:
        justification = f"Bằng chứng cho thấy yêu cầu bị bác bỏ (điểm hỗ trợ: {support_score:.2f}, điểm bác bỏ: {refute_score:.2f}, điểm căn chỉnh trung bình: {mean_alignment:.2f})."
        return "Refuted", justification
    
    # ƯU TIÊN 2: Supported - clear support with significant margin
    if support_score > support_threshold and score_diff > diff_threshold:
        justification = f"Bằng chứng cho thấy yêu cầu được hỗ trợ (điểm hỗ trợ: {support_score:.2f}, điểm bác bỏ: {refute_score:.2f}, điểm căn chỉnh trung bình: {mean_alignment:.2f})."
        return "Supported", justification
    
    # Low alignment: evidence doesn't clearly relate to claim
    # Cải thiện: chỉ áp dụng nếu không có support hoặc refute signal mạnh
    if mean_alignment < 0.35 and support_score < 0.5 and refute_score < 0.4:
        justification = f"Bằng chứng không đủ rõ ràng hoặc không liên quan trực tiếp đến yêu cầu (điểm căn chỉnh trung bình: {mean_alignment:.2f}, số bằng chứng: {num_evidence})."
        return "Not Enough Evidence", justification
    
    # Mixed or unclear evidence
    if abs(score_diff) < diff_threshold:
        justification = f"Bằng chứng hỗn hợp hoặc không rõ ràng (điểm hỗ trợ: {support_score:.2f}, điểm bác bỏ: {refute_score:.2f}, điểm căn chỉnh trung bình: {mean_alignment:.2f})."
        return "Not Enough Evidence", justification
    
    # Weak support (not strong enough)
    # Cải thiện: nếu support_score gần threshold và alignment cao, có thể là Supported
    if support_score > refute_score and support_score <= support_threshold:
        # Nếu support_score gần threshold (trong vòng 0.05) và alignment cao, có thể là Supported
        if support_score > support_threshold - 0.05 and mean_alignment > 0.65 and num_evidence >= 2:
            justification = f"Bằng chứng cho thấy yêu cầu được hỗ trợ (điểm hỗ trợ: {support_score:.2f}, điểm bác bỏ: {refute_score:.2f}, điểm căn chỉnh trung bình: {mean_alignment:.2f})."
            return "Supported", justification
        justification = f"Bằng chứng có xu hướng hỗ trợ nhưng chưa đủ mạnh (điểm hỗ trợ: {support_score:.2f}, điểm bác bỏ: {refute_score:.2f}, điểm căn chỉnh trung bình: {mean_alignment:.2f})."
        return "Not Enough Evidence", justification
    
    # Weak refutation (not strong enough) - nhưng vẫn có thể là refuted nếu alignment cao
    if refute_score > support_score:
        # Cải thiện: giảm threshold và điều kiện để phát hiện refuted tốt hơn
        if refute_score > 0.35 and mean_alignment > 0.7:
            # Nếu refute score > 0.35 và alignment cao, có thể là refuted
            justification = f"Bằng chứng cho thấy yêu cầu bị bác bỏ (điểm hỗ trợ: {support_score:.2f}, điểm bác bỏ: {refute_score:.2f}, điểm căn chỉnh trung bình: {mean_alignment:.2f})."
            return "Refuted", justification
        elif refute_score > 0.30 and mean_alignment > 0.8 and refute_score > support_score * 1.2:
            # Nếu refute score > 0.30, alignment rất cao, và refute > support đáng kể
            justification = f"Bằng chứng cho thấy yêu cầu bị bác bỏ (điểm hỗ trợ: {support_score:.2f}, điểm bác bỏ: {refute_score:.2f}, điểm căn chỉnh trung bình: {mean_alignment:.2f})."
            return "Refuted", justification
        else:
            justification = f"Bằng chứng có xu hướng bác bỏ nhưng chưa đủ mạnh (điểm hỗ trợ: {support_score:.2f}, điểm bác bỏ: {refute_score:.2f}, điểm căn chỉnh trung bình: {mean_alignment:.2f})."
            return "Not Enough Evidence", justification
    
    # Default fallback - insufficient evidence
    justification = f"Bằng chứng không đủ để xác nhận hoặc bác bỏ rõ ràng (điểm hỗ trợ: {support_score:.2f}, điểm bác bỏ: {refute_score:.2f}, điểm căn chỉnh trung bình: {mean_alignment:.2f})."
    return "Not Enough Evidence", justification


def judge(record, decision_options, rules="", think=True, 
          use_attention: bool = True, use_neural_classifier: bool = True,
          use_claim_context: bool = True, aggregator_type: str = "attention"):
    """
    Graph-based fact verification without using LLM.
    
    Args:
        record: The record containing evidence for the judgement.
        decision_options: The available decision options (for compatibility, not used).
        rules: Additional rules (for compatibility, not used).
        think: For compatibility with old interface (not used).
        use_attention: If True, use attention mechanism for message passing and aggregation (default: True).
        use_neural_classifier: If True, use CrossEncoder-based neural classification (default: True).
        use_claim_context: If True, evidence embeddings include claim context (GEAR style, default: True).
        aggregator_type: Type of aggregator to use: "attention", "max", or "mean" (default: "attention").
    
    Returns:
        str: The judgement in the same format as the LLM version.
    """
    # Extract claim and evidence
    claim = extract_claim_from_record(record)
    evidence_pieces = extract_evidence_pieces(record)
    
    if not claim:
        return "### Justification:\nKhông thể xác định yêu cầu từ bản ghi.\n\n### Verdict:\n`Not Enough Evidence`"
    
    if not evidence_pieces:
        return "### Justification:\nKhông tìm thấy bằng chứng nào trong bản ghi.\n\n### Verdict:\n`Not Enough Evidence`"
    
    # Build evidence graph (with claim context if enabled)
    G = build_evidence_graph(claim, evidence_pieces, use_claim_context=use_claim_context)
    
    # Aggregate evidence using graph-based mechanism with attention and neural classifier
    aggregated_scores = aggregate_evidence_with_graph(
        G, claim, evidence_pieces,
        use_attention=use_attention,
        use_neural_classifier=use_neural_classifier,
        aggregator_type=aggregator_type
    )
    
    # Classify verdict (dùng rule-based mặc định để tránh MLP ngẫu nhiên chưa được huấn luyện).
    # Nếu sau này có huấn luyện classifier riêng thì có thể bật lại.
    use_neural_final_classifier = False
    verdict, justification = classify_verdict(
        aggregated_scores,
        use_neural_classifier=use_neural_final_classifier
    )
    
    # Format output to match LLM format
    output = f"### Justification:\n{justification}\n\n### Verdict:\n`{verdict}`"
    
    return output


def extract_verdict(conclusion, decision_options, rules=""):
    """
    Extract verdict from conclusion text (for compatibility).
    Uses simple regex matching instead of LLM.
    """
    # Look for verdict in backticks
    match = re.search(r'`([^`]+)`', conclusion)
    if match:
        verdict = match.group(1).strip()
        # Normalize to standard labels
        verdict_lower = verdict.lower()
        if 'support' in verdict_lower:
            return '`Supported`'
        elif 'refut' in verdict_lower or 'bác bỏ' in verdict_lower:
            return '`Refuted`'
        elif 'not enough' in verdict_lower or 'không đủ' in verdict_lower:
            return '`Not Enough Evidence`'
        return f'`{verdict}`'
    
    # Look for verdict in markdown bold
    match = re.search(r'\*\*([^*]+)\*\*', conclusion)
    if match:
        return f'`{match.group(1).strip()}`'
    
    # Default
    return '`Not Enough Evidence`'


def _normalize_llm_verdict(raw: str) -> str:
    """
    Chuẩn hóa verdict từ LLM về 3 lớp chuẩn.
    """
    if not raw:
        return "Not Enough Evidence"
    s = str(raw).strip().lower()
    if "support" in s or "đúng" in s or "có căn cứ" in s or "được hỗ trợ" in s:
        return "Supported"
    if "refut" in s or "sai" in s or "bác bỏ" in s or "trái sự thật" in s:
        return "Refuted"
    if "not enough" in s or "không đủ" in s or "chưa đủ" in s or "không có đủ" in s:
        return "Not Enough Evidence"
    # Fallback: nếu không khớp rõ, coi là Not Enough Evidence để an toàn
    return "Not Enough Evidence"


def filter_evidence_by_relevance(claim: str, evidence_pieces: List[str], 
                                  relevance_threshold: float = 0.3,
                                  min_keep: int = 3,
                                  log_callback=None) -> Tuple[List[str], List[float]]:
    """
    Lọc evidence dựa trên relevance score với claim.
    Chỉ giữ lại evidence có relevance score > threshold.
    NHƯNG luôn đảm bảo giữ lại ít nhất min_keep evidence (top evidence).
    
    Args:
        claim: Claim cần fact-check
        evidence_pieces: Danh sách evidence pieces
        relevance_threshold: Ngưỡng relevance tối thiểu (default: 0.3)
        min_keep: Số lượng evidence tối thiểu cần giữ lại (default: 3)
        log_callback: Hàm callback để log các bước (optional)
    
    Returns:
        Tuple[List[str], List[float]]: (filtered_evidence, relevance_scores) - chỉ giữ evidence liên quan
    """
    if not evidence_pieces:
        if log_callback:
            log_callback("⚠️ Không có evidence pieces để filter!")
        return [], []
    
    if log_callback:
        log_callback(f"\n🔍 BƯỚC 1: Tính relevance scores cho {len(evidence_pieces)} evidence pieces")
        log_callback(f"   → Sử dụng CrossEncoder model")
        log_callback(f"   → Claim: {claim}")  # Ghi đầy đủ claim, không truncate
    
    try:
        # Tính relevance scores bằng CrossEncoder
        scores = compute_evidence_scores(claim, evidence_pieces)
        
        if log_callback:
            log_callback(f"   → Raw scores range: [{scores.min():.4f}, {scores.max():.4f}]")
        
        # Normalize scores về [0, 1] nếu cần
        if scores.size > 0:
            # CrossEncoder scores có thể âm hoặc dương
            # Strategy: normalize về [0, 1] nhưng giữ nguyên ranking
            min_score = scores.min()
            max_score = scores.max()
            if max_score > min_score:
                scores_normalized = (scores - min_score) / (max_score - min_score)
            else:
                # Tất cả scores bằng nhau, set về 0.5
                scores_normalized = np.full(len(scores), 0.5)
        else:
            scores_normalized = np.zeros(len(evidence_pieces))
        
        if log_callback:
            log_callback(f"   → Normalized scores range: [{scores_normalized.min():.4f}, {scores_normalized.max():.4f}]")
            log_callback(f"   → Top 5 evidence scores:")
            top_5_indices = np.argsort(-scores_normalized)[:5]
            for idx, ev_idx in enumerate(top_5_indices):
                # Ghi đầy đủ evidence, không truncate
                log_callback(f"      [{idx+1}] Score: {scores_normalized[ev_idx]:.4f} - {evidence_pieces[ev_idx]}")
        
        # Lọc evidence có relevance > threshold
        # NHƯNG: luôn giữ lại ít nhất top 1 evidence nếu có
        filtered_evidence = []
        filtered_scores = []
        
        # Tìm top evidence indices và điều chỉnh threshold
        adjusted_threshold = relevance_threshold
        top_indices = []
        
        if len(scores_normalized) > 0:
            top_indices = list(np.argsort(-scores_normalized))  # Descending order
            if len(top_indices) > 0:
                top_score = scores_normalized[top_indices[0]]
                # Nếu top score > 0.5 nhưng dưới threshold, giảm threshold một chút
                if top_score > 0.5 and top_score < relevance_threshold:
                    adjusted_threshold = min(relevance_threshold, top_score * 0.8)
                    if log_callback:
                        log_callback(f"\n🔍 BƯỚC 2: Điều chỉnh threshold")
                        log_callback(f"   → Threshold ban đầu: {relevance_threshold}")
                        log_callback(f"   → Top score: {top_score:.4f}")
                        log_callback(f"   → Threshold sau điều chỉnh: {adjusted_threshold:.4f}")
        
        if log_callback:
            log_callback(f"\n🔍 BƯỚC 3: Lọc evidence theo threshold ({adjusted_threshold:.4f})")
        
        # Bước 1: Lọc evidence theo threshold
        filtered_indices = set()  # Track indices đã được thêm vào filtered_evidence
        for i, (ev, score) in enumerate(zip(evidence_pieces, scores_normalized)):
            if score >= adjusted_threshold:
                filtered_evidence.append(ev)
                filtered_scores.append(float(score))
                filtered_indices.add(i)
        
        if log_callback:
            log_callback(f"   → Số evidence sau khi lọc: {len(filtered_evidence)}/{len(evidence_pieces)}")
            if len(filtered_evidence) > 0:
                log_callback(f"   → Evidence được giữ lại:")
                for idx, (ev, score) in enumerate(zip(filtered_evidence, filtered_scores)):
                    # Ghi đầy đủ evidence, không truncate
                    log_callback(f"      [{idx+1}] Score: {score:.4f} - {ev}")
        
        # Bước 2: Nếu số lượng evidence sau khi lọc < min_keep, bổ sung top evidence
        # Đảm bảo luôn có ít nhất min_keep evidence (hoặc tất cả nếu ít hơn min_keep)
        if len(filtered_evidence) < min_keep and len(top_indices) > 0:
            if log_callback:
                log_callback(f"\n🔍 BƯỚC 4: Bổ sung evidence để đạt min_keep={min_keep}")
                log_callback(f"   → Hiện tại có {len(filtered_evidence)} evidence, cần thêm {min_keep - len(filtered_evidence)}")
            
            # Thêm top evidence chưa có trong filtered_evidence
            added_count = 0
            for idx in top_indices:
                if len(filtered_evidence) >= min_keep:
                    break
                if idx not in filtered_indices:
                    # Chỉ thêm nếu score > 0.2 (ngưỡng tối thiểu)
                    if scores_normalized[idx] > 0.2:
                        filtered_evidence.append(evidence_pieces[idx])
                        filtered_scores.append(float(scores_normalized[idx]))
                        filtered_indices.add(idx)
                        added_count += 1
                        if log_callback:
                            # Ghi đầy đủ evidence, không truncate
                            log_callback(f"      [+] Thêm evidence #{idx} (score: {scores_normalized[idx]:.4f}) - {evidence_pieces[idx]}")
            
            if log_callback:
                log_callback(f"   → Đã thêm {added_count} evidence")
        
        # Bước 3: Sắp xếp lại theo score (descending) để đảm bảo top evidence ở đầu
        if filtered_evidence and len(filtered_evidence) > 1:
            if log_callback:
                log_callback(f"\n🔍 BƯỚC 5: Sắp xếp lại evidence theo score (descending)")
            
            # Tạo list of tuples (score, evidence) để sort
            evidence_score_pairs = list(zip(filtered_scores, filtered_evidence))
            evidence_score_pairs.sort(reverse=True, key=lambda x: x[0])
            filtered_evidence = [ev for _, ev in evidence_score_pairs]
            filtered_scores = [score for score, _ in evidence_score_pairs]
        
        if log_callback:
            log_callback(f"\n✅ KẾT QUẢ: Đã chọn {len(filtered_evidence)} evidence từ {len(evidence_pieces)} evidence ban đầu")
            if len(filtered_evidence) > 0:
                log_callback(f"   → Score range: [{min(filtered_scores):.4f}, {max(filtered_scores):.4f}]")
        
        return filtered_evidence, filtered_scores
    except Exception as e:
        if log_callback:
            log_callback(f"❌ LỖI khi filter evidence: {e}")
        print(f"Error filtering evidence by relevance: {e}")
        # Nếu lỗi, trả về toàn bộ evidence (không filter)
        return evidence_pieces, [0.5] * len(evidence_pieces)


def _llm_judge_with_evidence(claim: str, evidence_pieces: List[str], top_k: int = 5, log_callback=None) -> tuple:
    """
    Dùng LLM (Ollama) để ra phán quyết dựa trên claim + các bằng chứng web.
    Giảm tối đa rule-based; LLM chịu trách nhiệm phân loại NLI.
    
    BƯỚC 1: Lọc evidence không liên quan trước khi judge.
    BƯỚC 2: Chọn top_k evidence liên quan nhất.
    BƯỚC 3: LLM judge với prompt yêu cầu kiểm tra relevance.
    
    Args:
        claim: Claim cần fact-check
        evidence_pieces: Danh sách evidence pieces
        top_k: Số lượng evidence tối đa để đưa vào judge
        log_callback: Hàm callback để log các bước (optional)
    
    Returns:
        tuple: (verdict_string, evidence_info_dict)
        - verdict_string: String chứa verdict và justification
        - evidence_info_dict: Dict chứa thông tin về evidence (selected_evidence, selected_scores, stats)
    """
    if log_callback:
        log_callback(f"\n{'='*80}")
        log_callback(f"🔍 QUÁ TRÌNH LỌC VÀ CHỌN EVIDENCE CHO JUDGE")
        log_callback(f"{'='*80}")
        log_callback(f"Claim: {claim}")
        log_callback(f"Tổng số evidence ban đầu: {len(evidence_pieces)}")
        log_callback(f"Top_k: {top_k}")
    
    # BƯỚC 1: Lọc evidence không liên quan (relevance threshold = 0.15, giảm để giữ nhiều evidence hơn)
    # Đảm bảo giữ lại ít nhất top_k evidence (hoặc tất cả nếu ít hơn top_k)
    filtered_evidence, relevance_scores = filter_evidence_by_relevance(
        claim, evidence_pieces, relevance_threshold=0.15, min_keep=top_k, log_callback=log_callback
    )
    
    # Nếu không có evidence nào liên quan, trả về Not Enough Evidence ngay
    if not filtered_evidence:
        justification = (
            f"Không tìm thấy bằng chứng nào liên quan đến claim. "
            f"Đã kiểm tra {len(evidence_pieces)} bằng chứng nhưng tất cả đều có độ liên quan thấp."
        )
        verdict_string = f"### Justification:\n{justification}\n\n### Verdict:\n`Not Enough Evidence`"
        evidence_info = {
            "claim": claim,
            "total_evidence": len(evidence_pieces),
            "filtered_evidence_count": 0,
            "selected_evidence_count": 0,
            "top_k": top_k,
            "selected_evidence": [],
            "selected_scores": []
        }
        return verdict_string, evidence_info
    
    # BƯỚC 2: Chọn top_k evidence liên quan nhất từ filtered_evidence
    if log_callback:
        log_callback(f"\n🔍 BƯỚC 6: Chọn top_{top_k} evidence từ {len(filtered_evidence)} evidence đã lọc")
    
    try:
        # Tính lại scores cho filtered evidence để rank chính xác
        scores = compute_evidence_scores(claim, filtered_evidence)
        if scores.size == 0:
            ranked_indices = list(range(len(filtered_evidence)))
        else:
            ranked_indices = list(np.argsort(-scores))
    except Exception:
        ranked_indices = list(range(len(filtered_evidence)))

    top_k = min(top_k, len(ranked_indices))
    selected_idx = ranked_indices[:top_k]
    selected_evidence = [filtered_evidence[i] for i in selected_idx]
    selected_scores = [relevance_scores[i] for i in selected_idx]
    
    if log_callback:
        log_callback(f"   → Đã chọn {len(selected_evidence)} evidence:")
        for i, (ev, score) in enumerate(zip(selected_evidence, selected_scores)):
            # Ghi đầy đủ evidence, không truncate
            log_callback(f"      [E{i}] Score: {score:.4f} - {ev}")

    # Tạo evidence_info dict để trả về
    evidence_info = {
        "claim": claim,
        "total_evidence": len(evidence_pieces),
        "filtered_evidence_count": len(filtered_evidence),
        "selected_evidence_count": len(selected_evidence),
        "top_k": top_k,
        "selected_evidence": selected_evidence,
        "selected_scores": selected_scores
    }
    
    # In ra danh sách bằng chứng trước khi đưa vào judge
    print("\n" + "=" * 80)
    print("📋 DANH SÁCH BẰNG CHỨNG ĐƯỢC CHỌN CHO JUDGE:")
    print("=" * 80)
    print(f"Claim: {claim}")
    print(f"\nTổng số bằng chứng ban đầu: {len(evidence_pieces)}")
    print(f"Số bằng chứng sau khi lọc (relevance > 0.15): {len(filtered_evidence)}")
    print(f"Số bằng chứng được chọn (top_k={top_k}): {len(selected_evidence)}")
    print("\n" + "-" * 80)
    for i, (ev, score) in enumerate(zip(selected_evidence, selected_scores)):
        print(f"\n[E{i}] (Relevance score: {score:.4f})")
        ev_preview = ev
        print(f"{ev_preview}")
    print("\n" + "=" * 80 + "\n")

    # Xây prompt cho LLM (tiếng Việt, output JSON)
    evidence_block_lines = []
    for i, ev in enumerate(selected_evidence):
        evidence_block_lines.append(f"- [E{i}] {ev}")
    evidence_block = "\n".join(evidence_block_lines)

    prompt = f"""Phân loại YÊU CẦU dựa trên BẰNG CHỨNG thành 1 trong NHÃN. Trả về JSON.
NHÃN:
Supported
- Dùng khi có bằng chứng E[i] rõ ràng, trực tiếp ỦNG HỘ yêu cầu.
- Nếu yêu cầu có nhiều khía cạnh, TẤT CẢ các khía cạnh phải được ỦNG HỘ để chọn phán quyết này.

Refuted
- Dùng khi có bằng chứng E[i] rõ ràng BÁC BỎ hoặc MÂU THUẪN trực tiếp với yêu cầu.
- Nếu yêu cầu có nhiều khía cạnh, dù chỉ 1 khía cạnh bị BÁC BỎ trong khi các khía cạnh còn lại được ủng hộ cũng đủ để chọn phán quyết này.

Not Enough Evidence
- Dùng khi tất cả E[i] KHÔNG ĐỦ thông tin để xác nhận hoặc bác bỏ yêu cầu.
- Cũng dùng nếu yêu cầu quá MƠ HỒ hoặc không thể kiểm chứng bằng dữ liệu hiện có.
- Nếu yêu cầu có nhiều khía cạnh, chỉ cần 1 khía cạnh không đủ bằng chứng để chọn phán quyết này.

ĐỊNH DẠNG (bắt buộc JSON, không có text khác):
{{
  "verdict": "Supported|Refuted|Not Enough Evidence",
  "justification": "Giải thích ngắn gọn (1-2 câu), nêu rõ [Ei] nào được dùng và lý do chọn nhãn này."
}}

YÊU CẦU:
{claim}

BẰNG CHỨNG:
{evidence_block}

JSON:
"""

    try:
        # Kiểm tra judge provider từ env var
        judge_provider = os.getenv("FACTCHECKER_JUDGE_PROVIDER", "ollama").lower()
        
        if judge_provider == "gemini":
            # Dùng Gemini API
            gemini_model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
            raw = llm.prompt_gemini(prompt, model=gemini_model)
        else:
            # Mặc định dùng Ollama
            raw = llm.prompt_ollama(prompt, think=False, use_judge_model=True)
    except Exception as e:
        # Nếu LLM lỗi, fallback an toàn
        justification = f"Lỗi khi gọi LLM judge: {e}. Mặc định Not Enough Evidence."
        verdict_string = f"### Justification:\n{justification}\n\n### Verdict:\n`Not Enough Evidence`"
        return verdict_string, evidence_info

    # Cải thiện JSON parsing với nhiều strategies
    if not raw or not raw.strip():
        justification = "LLM judge không trả về kết quả. Mặc định Not Enough Evidence."
        verdict_string = f"### Justification:\n{justification}\n\n### Verdict:\n`Not Enough Evidence`"
        return verdict_string, evidence_info
    
    # Strategy 1: Tìm JSON block trong markdown code block
    import re
    json_in_code_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
    if json_in_code_block:
        json_candidate = json_in_code_block.group(1).strip()
        try:
            obj = json.loads(json_candidate)
            verdict_raw = obj.get("verdict", "")
            justification = obj.get("justification", "").strip()
            verdict = _normalize_llm_verdict(verdict_raw)
            if justification:
                verdict_string = f"### Justification:\n{justification}\n\n### Verdict:\n`{verdict}`"
                return verdict_string, evidence_info
        except Exception:
            pass
    
    # Strategy 2: Tìm JSON object trong text (từ '{' đầu tiên đến '}' cuối cùng matching)
    # Tìm tất cả các cặp {} và thử parse
    brace_pattern = re.search(r'\{[^{}]*"verdict"[^{}]*\}', raw, re.DOTALL)
    if brace_pattern:
        json_candidate = brace_pattern.group(0)
        try:
            obj = json.loads(json_candidate)
            verdict_raw = obj.get("verdict", "")
            justification = obj.get("justification", "").strip()
            verdict = _normalize_llm_verdict(verdict_raw)
            if justification:
                verdict_string = f"### Justification:\n{justification}\n\n### Verdict:\n`{verdict}`"
                return verdict_string, evidence_info
        except Exception:
            pass
    
    # Strategy 3: Tìm từ '{' đầu tiên đến '}' cuối cùng (original method)
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        json_candidate = raw[start:end + 1]
        try:
            obj = json.loads(json_candidate)
            verdict_raw = obj.get("verdict", "")
            justification = obj.get("justification", "").strip()
            verdict = _normalize_llm_verdict(verdict_raw)
            if justification:
                verdict_string = f"### Justification:\n{justification}\n\n### Verdict:\n`{verdict}`"
                return verdict_string, evidence_info
        except Exception as e:
            pass
    
    # Strategy 4: Extract verdict từ text nếu không parse được JSON
    # Tìm các từ khóa verdict trong text
    raw_lower = raw.lower()
    if '"supported"' in raw_lower or "supported" in raw_lower:
        verdict = "Supported"
    elif '"refuted"' in raw_lower or "refuted" in raw_lower:
        verdict = "Refuted"
    else:
        verdict = "Not Enough Evidence"
    
    # Fallback: normalize từ raw text
    verdict = _normalize_llm_verdict(raw)
    justification = f"Không parse được JSON từ output LLM. Dựa trên nội dung: {raw[:200]}... Chọn nhãn {verdict}."
    verdict_string = f"### Justification:\n{justification}\n\n### Verdict:\n`{verdict}`"
    return verdict_string, evidence_info


def judge(record, decision_options, rules="", think=True,
          use_attention: bool = True, use_neural_classifier: bool = True,
          use_claim_context: bool = True, aggregator_type: str = "attention"):
    """
    Phiên bản judge mới:
    - Vẫn dùng retrieval từ web (evidence đã được thu thập ở bước trước).
    - Bỏ phần rule-based phức tạp cho verdict cuối cùng.
    - Dùng LLM (Ollama) để phân loại dựa trên claim + các evidence quan trọng nhất.
    """
    # Extract claim and evidence từ record (report.md)
    claim = extract_claim_from_record(record)
    evidence_pieces = extract_evidence_pieces(record)

    if not claim:
        verdict_string = "### Justification:\nKhông thể xác định yêu cầu từ bản ghi.\n\n### Verdict:\n`Not Enough Evidence`"
        evidence_info = {
            "claim": "",
            "total_evidence": 0,
            "filtered_evidence_count": 0,
            "selected_evidence_count": 0,
            "top_k": 6,
            "selected_evidence": [],
            "selected_scores": []
        }
        return verdict_string, evidence_info

    if not evidence_pieces:
        verdict_string = "### Justification:\nKhông tìm thấy bằng chứng nào trong bản ghi.\n\n### Verdict:\n`Not Enough Evidence`"
        evidence_info = {
            "claim": claim,
            "total_evidence": 0,
            "filtered_evidence_count": 0,
            "selected_evidence_count": 0,
            "top_k": 6,
            "selected_evidence": [],
            "selected_scores": []
        }
        return verdict_string, evidence_info

    # Gọi LLM để judge dựa trên claim + evidence (đã chọn top bằng CrossEncoder)
    # Tạo log callback để ghi lại quá trình filter và select evidence
    filter_log_lines = []
    def filter_log_callback(msg):
        filter_log_lines.append(msg)
        print(f"[EVIDENCE_FILTER] {msg}")
    
    # Dùng top_k=6 để có đủ bằng chứng để đánh giá, nhưng vẫn tập trung vào bằng chứng liên quan nhất
    verdict_string, evidence_info = _llm_judge_with_evidence(claim, evidence_pieces, top_k=3, log_callback=filter_log_callback)
    
    # Ghi log vào evidence_info để có thể append vào report sau
    if filter_log_lines:
        evidence_info['filter_log'] = filter_log_lines
    
    return verdict_string, evidence_info
