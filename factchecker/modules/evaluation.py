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
    """
    Extract evidence pieces from the record.
    Evidence is typically stored as: "web_search('query') summary: {evidence_text}"
    """
    evidence_pieces = []
    
    # Pattern 1: web_search('...') summary: ...
    pattern1 = re.compile(r"web_search\([^)]+\)\s+summary:\s*(.+?)(?=\n\n|\n###|$)", re.DOTALL | re.IGNORECASE)
    matches1 = pattern1.findall(record)
    evidence_pieces.extend([m.strip() for m in matches1 if m.strip()])
    
    # Pattern 2: Look for evidence section
    evidence_section_match = re.search(r'###\s*Evidence\s*\n\n(.+?)(?=\n###|$)', record, re.DOTALL | re.IGNORECASE)
    if evidence_section_match:
        evidence_text = evidence_section_match.group(1)
        # Split by lines and extract non-empty lines that look like evidence
        lines = [line.strip() for line in evidence_text.split('\n') if line.strip()]
        for line in lines:
            # Skip action lines, extract actual evidence
            if 'summary:' in line.lower():
                parts = line.split('summary:', 1)
                if len(parts) > 1:
                    evidence_pieces.append(parts[1].strip())
            elif not re.match(r'web_search\([^)]+\)', line, re.IGNORECASE):
                # If it's not an action line and looks like evidence
                if len(line) > 20:  # Reasonable length for evidence
                    evidence_pieces.append(line)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_evidence = []
    for ev in evidence_pieces:
        ev_lower = ev.lower()
        if ev_lower not in seen and len(ev) > 10:  # Minimum length check
            seen.add(ev_lower)
            unique_evidence.append(ev)
    
    return unique_evidence


def build_evidence_graph(claim: str, evidence_pieces: List[str], similarity_threshold: float = 0.3) -> nx.Graph:
    """
    Build a graph where nodes are the claim and evidence pieces.
    Edges represent semantic similarity between nodes.
    
    Args:
        claim: The claim to verify
        evidence_pieces: List of evidence text pieces
        similarity_threshold: Minimum similarity to create an edge
    
    Returns:
        NetworkX graph with nodes and weighted edges
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
        
        # Add edge between claim and evidence
        claim_evidence_sim = float(np.dot(claim_emb, emb))
        if claim_evidence_sim > similarity_threshold:
            G.add_edge('claim', node_id, weight=claim_evidence_sim, edge_type='claim_evidence')
    
    # Add edges between evidence pieces
    for i in range(len(evidence_pieces)):
        for j in range(i + 1, len(evidence_pieces)):
            node_i = f'evidence_{i}'
            node_j = f'evidence_{j}'
            evidence_sim = float(np.dot(evidence_embs[i], evidence_embs[j]))
            
            if evidence_sim > similarity_threshold:
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


def aggregate_evidence_with_graph(G: nx.Graph, claim: str, evidence_pieces: List[str]) -> Dict[str, float]:
    """
    Aggregate evidence using graph-based attention mechanism.
    Similar to GEAR (Graph-based Evidence Aggregating and Reasoning).
    
    Returns:
        Dictionary with aggregated scores: support_score, refute_score, neutral_score
    """
    if not evidence_pieces or len(G.nodes()) == 0:
        return {'support_score': 0.0, 'refute_score': 0.0, 'neutral_score': 1.0}
    
    # Compute fine-grained claim-evidence alignment scores
    alignment_scores = compute_evidence_scores(claim, evidence_pieces)
    
    if len(alignment_scores) == 0:
        return {'support_score': 0.0, 'refute_score': 0.0, 'neutral_score': 1.0}
    
    # Get claim-evidence edge weights from graph (semantic similarity)
    claim_evidence_weights = []
    for i, ev in enumerate(evidence_pieces):
        node_id = f'evidence_{i}'
        if 'claim' in G and node_id in G and G.has_edge('claim', node_id):
            weight = G['claim'][node_id].get('weight', 0.0)
            claim_evidence_weights.append(weight)
        else:
            claim_evidence_weights.append(0.0)
    
    claim_evidence_weights = np.array(claim_evidence_weights)
    
    # Combine alignment scores (relevance) with semantic similarity weights
    # Alignment scores are from cross-encoder (more accurate for relevance)
    # Graph weights are semantic similarity (helps with relatedness)
    combined_scores = 0.7 * alignment_scores + 0.3 * claim_evidence_weights
    
    # Normalize scores to [0, 1]
    score_min = combined_scores.min()
    score_max = combined_scores.max()
    if score_max - score_min > 1e-8:
        combined_scores = (combined_scores - score_min) / (score_max - score_min)
    else:
        combined_scores = np.ones_like(combined_scores) * 0.5  # All same, neutral
    
    # Graph-based aggregation: evidence that are well-connected get higher weights
    # Well-connected evidence pieces are more reliable (consensus)
    evidence_weights = np.ones(len(evidence_pieces))
    for i, ev in enumerate(evidence_pieces):
        node_id = f'evidence_{i}'
        if node_id in G:
            # Compute node importance based on degree and edge weights
            neighbors = list(G.neighbors(node_id))
            if neighbors:
                neighbor_weights = [G[node_id][nbr].get('weight', 0.0) for nbr in neighbors]
                # Evidence that aligns with other evidence gets higher weight
                evidence_weights[i] = 1.0 + np.mean(neighbor_weights) if neighbor_weights else 1.0
    
    # Normalize evidence weights
    evidence_weights = evidence_weights / (evidence_weights.sum() + 1e-8)
    
    # Weighted aggregation
    weighted_scores = combined_scores * evidence_weights
    mean_score = np.mean(weighted_scores)
    
    # Classify scores into support/refute/neutral regions
    # High scores (>= 0.6): Strong support
    # Medium scores (0.4-0.6): Weak support or unclear
    # Low scores (< 0.4): Potential refutation or irrelevant
    
    high_scores = weighted_scores[weighted_scores >= 0.6]
    low_scores = weighted_scores[weighted_scores < 0.4]
    medium_scores = weighted_scores[(weighted_scores >= 0.4) & (weighted_scores < 0.6)]
    
    # Compute support and refute scores
    if len(high_scores) > 0:
        support_score = np.mean(high_scores)
        support_weight = np.sum(evidence_weights[weighted_scores >= 0.6])
    else:
        support_score = 0.0
        support_weight = 0.0
    
    if len(low_scores) > 0:
        # For low scores, check if evidence is semantically related but contradicts
        # Low alignment + high semantic similarity to claim = potential contradiction
        refute_score_raw = 1.0 - np.mean(low_scores)
        # Check if low-score evidence is still semantically related (potential contradiction)
        low_indices = np.where(weighted_scores < 0.4)[0]
        low_semantic_similarity = np.mean(claim_evidence_weights[low_indices]) if len(low_indices) > 0 else 0.0
        # If low alignment but high semantic similarity, it's more likely a contradiction
        if low_semantic_similarity > 0.5:
            refute_score = refute_score_raw * 1.2  # Boost refute score for contradictions
        else:
            refute_score = refute_score_raw * 0.5  # Likely just irrelevant
        refute_weight = np.sum(evidence_weights[weighted_scores < 0.4])
    else:
        refute_score = 0.0
        refute_weight = 0.0
    
    # Adjust based on consensus (well-connected evidence)
    # Compute mean edge weights to neighbors for each evidence node
    evidence_connectivity_scores = []
    for i in range(len(evidence_pieces)):
        node_id = f'evidence_{i}'
        if node_id in G:
            neighbors = list(G.neighbors(node_id))
            if neighbors:
                # Get edge weights to neighbors
                edge_weights = [G[node_id][nbr].get('weight', 0.0) for nbr in neighbors]
                evidence_connectivity_scores.append(np.mean(edge_weights))
            else:
                evidence_connectivity_scores.append(0.0)
        else:
            evidence_connectivity_scores.append(0.0)
    
    evidence_connectivity = np.mean(evidence_connectivity_scores) if evidence_connectivity_scores else 0.0
    
    # If evidence pieces agree with each other (high connectivity), trust the scores more
    if evidence_connectivity > 0:
        connectivity_factor = min(1.0, evidence_connectivity)  # Normalize by max possible weight (1.0)
        support_score *= (1.0 + connectivity_factor * 0.2)
        refute_score *= (1.0 + connectivity_factor * 0.2)
    
    # Normalize based on overall mean
    if mean_score > 0.65:
        # Strong overall support
        support_score = mean_score * support_weight
        refute_score = (1 - mean_score) * refute_weight * 0.3
    elif mean_score < 0.35:
        # Strong potential refutation
        refute_score = (1 - mean_score) * refute_weight
        support_score = mean_score * support_weight * 0.3
    else:
        # Mixed or unclear
        support_score = mean_score * support_weight
        refute_score = (1 - mean_score) * refute_weight
    
    # Normalize to probabilities
    total = support_score + refute_score + 0.1  # Add small neutral component
    if total > 0:
        support_score = support_score / total
        refute_score = refute_score / total
    else:
        support_score = 0.33
        refute_score = 0.33
    neutral_score = 1.0 - support_score - refute_score
    
    return {
        'support_score': float(support_score),
        'refute_score': float(refute_score),
        'neutral_score': float(neutral_score),
        'mean_alignment': float(mean_score),
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
