"""
Graph-based evaluation module for fact verification.
Based on "Evidence Retrieval is almost All You Need for Fact Verification" 
and graph-based evidence aggregation approaches like GEAR.
"""

import re
import numpy as np
import networkx as nx
from functools import lru_cache
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer, CrossEncoder
import os

# Use same models as retriver_rav for consistency
_EMBED_DEVICE = os.getenv("FACTCHECKER_EMBED_DEVICE")
_BI_MODEL_NAME = os.getenv("FACTCHECKER_BI_ENCODER", "paraphrase-multilingual-MiniLM-L12-v2")
_CROSS_MODEL_NAME = os.getenv("FACTCHECKER_CROSS_ENCODER", "cross-encoder/ms-marco-MiniLM-L-6-v2")


@lru_cache(maxsize=1)
def _get_bi_model(model_name=_BI_MODEL_NAME):
    """Get bi-encoder model for embeddings."""
    kwargs = {}
    if _EMBED_DEVICE:
        kwargs["device"] = _EMBED_DEVICE
    return SentenceTransformer(model_name, **kwargs)


@lru_cache(maxsize=1)
def _get_cross_model(model_name=_CROSS_MODEL_NAME):
    """Get cross-encoder model for fine-grained scoring."""
    kwargs = {}
    if _EMBED_DEVICE:
        kwargs["device"] = _EMBED_DEVICE
    return CrossEncoder(model_name, **kwargs)


def extract_claim_from_record(record: str) -> str:
    """Extract the claim from the record."""
    # Look for "# Claim: ..." pattern
    match = re.search(r'#\s*Claim:\s*(.+?)(?:\n|$)', record, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: first line if no pattern found
    lines = record.strip().split('\n')
    if lines:
        return lines[0].replace('#', '').replace('Claim:', '').strip()
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
    evidence_section_match = re.search(r'###\s*Evidence\s*\n\n(.+?)(?=\n###|$)', record, re.DOTALL | re.IGNORECASE)
    if evidence_section_match:
        evidence_text = evidence_section_match.group(1)
        lines = [line.strip() for line in evidence_text.split('\n') if line.strip()]
        for line in lines:
            # Xử lý cả Summary: và summary:
            if 'summary:' in line.lower() or 'Summary:' in line:
                parts = re.split(r'summary:\s*', line, 1, flags=re.IGNORECASE)
                if len(parts) > 1:
                    evidence_pieces.append(parts[1].strip())
            elif not re.match(r'web_search\([^)]+\)', line, re.IGNORECASE):
                if len(line) > 20:
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


def build_evidence_graph(claim: str, evidence_pieces: List[str], similarity_threshold: float = 0.0) -> nx.Graph:
    """
    Build a fully-connected evidence graph (GEAR style).
    All evidence nodes are connected to claim and to each other.
    Edges represent semantic similarity between nodes.
    
    Args:
        claim: The claim to verify
        evidence_pieces: List of evidence text pieces
        similarity_threshold: Minimum similarity (for compatibility, but now always connect)
    
    Returns:
        NetworkX graph with nodes and weighted edges (fully-connected)
    """
    G = nx.Graph()
    
    if not evidence_pieces:
        return G
    
    # Get embeddings for all nodes
    bi_model = _get_bi_model()
    all_texts = [claim] + evidence_pieces
    embeddings = bi_model.encode(all_texts, normalize_embeddings=True)
    
    claim_emb = embeddings[0]
    evidence_embs = embeddings[1:]
    
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


def detect_contradiction(claim: str, evidence: str) -> float:
    """
    Detect if evidence contradicts claim (not just irrelevant).
    Returns contradiction score [0, 1]
    
    Based on keyword patterns and semantic analysis.
    """
    evidence_lower = evidence.lower()
    claim_lower = claim.lower()
    
    # Pattern 1: Explicit negation keywords in Vietnamese
    negation_patterns = [
        r'\b(không|chưa|không có|không phải|sai|bác bỏ|phủ nhận)\b',
        r'\b(mâu thuẫn|trái ngược|khác với|không đúng|không chính xác)\b',
        r'\b(tuy nhiên|nhưng|mặt khác|ngược lại)\b'
    ]
    
    # Count negation patterns
    negation_count = 0
    for pattern in negation_patterns:
        negation_count += len(re.findall(pattern, evidence_lower))
    
    # Pattern 2: Check for entity mismatch (same entities but different facts)
    # Extract entities from both claim and evidence
    # If same entities mentioned but different values/actions, might be contradiction
    
    # Simple heuristic: if evidence contains negation AND mentions claim keywords
    claim_keywords = set(re.findall(r'\b\w{4,}\b', claim_lower))  # Words with 4+ chars
    evidence_keywords = set(re.findall(r'\b\w{4,}\b', evidence_lower))
    
    common_keywords = claim_keywords.intersection(evidence_keywords)
    has_common_entities = len(common_keywords) > 2  # At least 3 common keywords
    
    # Contradiction score
    if negation_count > 0 and has_common_entities:
        # Strong contradiction: negation + same entities
        contradiction_score = min(1.0, 0.3 + negation_count * 0.2)
    elif negation_count > 0:
        # Weak contradiction: just negation
        contradiction_score = min(0.6, negation_count * 0.15)
    else:
        contradiction_score = 0.0
    
    return contradiction_score


def evidence_reasoning_network(G: nx.Graph, claim_emb: np.ndarray, 
                                evidence_embs: List[np.ndarray], 
                                num_layers: int = 2) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    ERNet-like message passing for evidence reasoning (GEAR style).
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
        return []
    
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
            
            # Weighted aggregation of evidence into claim
            if len(neighbor_reps) > 0 and np.sum(edge_weights) > 0:
                edge_weights_norm = edge_weights / (np.sum(edge_weights) + 1e-8)
                weighted_neighbors = np.sum(neighbor_reps * edge_weights_norm.reshape(-1, 1), axis=0)
                new_reps['claim'] = 0.7 * node_reps['claim'] + 0.3 * weighted_neighbors
                # Normalize claim embedding
                claim_norm = np.linalg.norm(new_reps['claim'])
                if claim_norm > 0:
                    new_reps['claim'] = new_reps['claim'] / claim_norm
            else:
                new_reps['claim'] = node_reps['claim']
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
                claim_contribution = node_reps['claim'] * claim_weight
            else:
                claim_contribution = np.zeros_like(node_reps[node_id])
            
            # Aggregate from other evidence neighbors
            evidence_neighbors = [n for n in neighbors if n.startswith('evidence_') and n != node_id]
            if evidence_neighbors:
                neighbor_reps = np.array([node_reps[n] for n in evidence_neighbors])
                edge_weights = np.array([G[node_id][n].get('weight', 0.0) for n in evidence_neighbors])
                
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


def compute_support_refute_scores(claim: str, evidence_pieces: List[str],
                                   claim_emb: np.ndarray,
                                   refined_evidence_embs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute separate support and refute scores (GEAR style).
    Not just based on alignment, but explicit contradiction detection.
    
    Args:
        claim: The claim text
        evidence_pieces: List of evidence text pieces
        claim_emb: Claim embedding (after message passing)
        refined_evidence_embs: List of refined evidence embeddings (after message passing)
    
    Returns:
        Tuple of (support_scores, refute_scores) arrays
    """
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


def compute_evidence_importance_weights(G: nx.Graph, num_evidence: int) -> np.ndarray:
    """
    Compute importance weights for evidence based on graph connectivity (GEAR style).
    Well-connected evidence (consensus with other evidence) gets higher weight.
    
    Args:
        G: Evidence graph
        num_evidence: Number of evidence pieces
    
    Returns:
        Normalized importance weights array
    """
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


def aggregate_evidence_with_graph(G: nx.Graph, claim: str, evidence_pieces: List[str]) -> Dict[str, float]:
    """
    GEAR-style evidence aggregation with message passing.
    FIXED: Low alignment != Refutation (low alignment = irrelevant/neutral)
    
    Uses:
    - Fully-connected graph (all evidence connected)
    - Message passing (ERNet-like) for multi-step reasoning
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
    refined_claim_emb, refined_evidence_embs = evidence_reasoning_network(G, claim_emb, evidence_embs, num_layers=2)
    
    # STEP 2: Compute support/refute scores (separate, not from alignment)
    support_scores, refute_scores = compute_support_refute_scores(
        claim, evidence_pieces, refined_claim_emb, refined_evidence_embs
    )
    
    # STEP 3: Compute evidence importance weights based on connectivity
    evidence_weights = compute_evidence_importance_weights(G, len(evidence_pieces))
    
    # STEP 4: Aggregate with attention mechanism
    # Weighted aggregation using importance weights
    support_score = float(np.sum(support_scores * evidence_weights))
    refute_score = float(np.sum(refute_scores * evidence_weights))
    
    # Mean alignment: average of (support + refute) for relevant evidence
    relevant_mask = (support_scores + refute_scores) > 0.1  # Evidence that is relevant
    if np.any(relevant_mask):
        mean_alignment = float(np.mean((support_scores + refute_scores)[relevant_mask]))
    else:
        # No relevant evidence - all irrelevant
        mean_alignment = float(np.mean(support_scores + refute_scores))
    
    # STEP 5: Normalize to probabilities
    total = support_score + refute_score + 0.1  # Add small neutral component
    if total > 0:
        support_score = support_score / total
        refute_score = refute_score / total
    else:
        support_score = 0.0
        refute_score = 0.0
    
    neutral_score = 1.0 - support_score - refute_score
    
    return {
        'support_score': float(support_score),
        'refute_score': float(refute_score),
        'neutral_score': float(neutral_score),
        'mean_alignment': float(mean_alignment),
        'num_evidence': len(evidence_pieces)
    }


def classify_verdict(aggregated_scores: Dict[str, float], min_evidence_threshold: int = 1) -> Tuple[str, str]:
    """
    Classify verdict based on aggregated scores.
    
    Returns:
        Tuple of (verdict, justification)
    """
    support_score = aggregated_scores['support_score']
    refute_score = aggregated_scores['refute_score']
    neutral_score = aggregated_scores['neutral_score']
    num_evidence = aggregated_scores.get('num_evidence', 0)
    mean_alignment = aggregated_scores.get('mean_alignment', 0.0)
    
    # Check if we have enough evidence
    if num_evidence < min_evidence_threshold:
        justification = f"Không có đủ bằng chứng để xác nhận hoặc bác bỏ. Chỉ tìm thấy {num_evidence} bằng chứng."
        return "Not Enough Evidence", justification
    
    # Decision logic with clear thresholds
    score_diff = support_score - refute_score
    
    # Supported: clear support with significant margin
    if support_score > 0.55 and score_diff > 0.15:
        justification = f"Bằng chứng cho thấy yêu cầu được hỗ trợ (điểm hỗ trợ: {support_score:.2f}, điểm bác bỏ: {refute_score:.2f}, điểm căn chỉnh trung bình: {mean_alignment:.2f})."
        return "Supported", justification
    
    # Refuted: clear refutation with significant margin
    if refute_score > 0.55 and score_diff < -0.15:
        justification = f"Bằng chứng cho thấy yêu cầu bị bác bỏ (điểm hỗ trợ: {support_score:.2f}, điểm bác bỏ: {refute_score:.2f}, điểm căn chỉnh trung bình: {mean_alignment:.2f})."
        return "Refuted", justification
    
    # Low alignment: evidence doesn't clearly relate to claim
    if mean_alignment < 0.35:
        justification = f"Bằng chứng không đủ rõ ràng hoặc không liên quan trực tiếp đến yêu cầu (điểm căn chỉnh trung bình: {mean_alignment:.2f}, số bằng chứng: {num_evidence})."
        return "Not Enough Evidence", justification
    
    # Mixed or unclear evidence
    if abs(score_diff) < 0.15:
        justification = f"Bằng chứng hỗn hợp hoặc không rõ ràng (điểm hỗ trợ: {support_score:.2f}, điểm bác bỏ: {refute_score:.2f}, điểm căn chỉnh trung bình: {mean_alignment:.2f})."
        return "Not Enough Evidence", justification
    
    # Weak support (not strong enough)
    if support_score > refute_score and support_score <= 0.55:
        justification = f"Bằng chứng có xu hướng hỗ trợ nhưng chưa đủ mạnh (điểm hỗ trợ: {support_score:.2f}, điểm bác bỏ: {refute_score:.2f}, điểm căn chỉnh trung bình: {mean_alignment:.2f})."
        return "Not Enough Evidence", justification
    
    # Weak refutation (not strong enough)
    if refute_score > support_score and refute_score <= 0.55:
        justification = f"Bằng chứng có xu hướng bác bỏ nhưng chưa đủ mạnh (điểm hỗ trợ: {support_score:.2f}, điểm bác bỏ: {refute_score:.2f}, điểm căn chỉnh trung bình: {mean_alignment:.2f})."
        return "Not Enough Evidence", justification
    
    # Default fallback - insufficient evidence
    justification = f"Bằng chứng không đủ để xác nhận hoặc bác bỏ rõ ràng (điểm hỗ trợ: {support_score:.2f}, điểm bác bỏ: {refute_score:.2f}, điểm căn chỉnh trung bình: {mean_alignment:.2f})."
    return "Not Enough Evidence", justification


def judge(record, decision_options, rules="", think=True):
    """
    Graph-based fact verification without using LLM.
    
    Args:
        record: The record containing evidence for the judgement.
        decision_options: The available decision options (for compatibility, not used).
        rules: Additional rules (for compatibility, not used).
        think: For compatibility with old interface (not used).
    
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
    
    # Build evidence graph
    G = build_evidence_graph(claim, evidence_pieces)
    
    # Aggregate evidence using graph-based mechanism
    aggregated_scores = aggregate_evidence_with_graph(G, claim, evidence_pieces)
    
    # Classify verdict
    verdict, justification = classify_verdict(aggregated_scores)
    
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
