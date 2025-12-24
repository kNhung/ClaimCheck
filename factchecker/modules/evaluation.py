"""
Graph-based evaluation module for fact verification.
Based on "Evidence Retrieval is almost All You Need for Fact Verification" 
and graph-based evidence aggregation approaches like GEAR.
"""

import re
import json
import numpy as np
from functools import lru_cache
from threading import Lock
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer, CrossEncoder
import os
from . import llm

# Use same models as retriver_rav for consistency
_EMBED_DEVICE_ENV = os.getenv("FACTCHECKER_EMBED_DEVICE")
_BI_MODEL_NAME = os.getenv("FACTCHECKER_BI_ENCODER", "paraphrase-multilingual-MiniLM-L12-v2")
_CROSS_MODEL_NAME = os.getenv("FACTCHECKER_CROSS_ENCODER", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Global model cache with thread-safe initialization
_bi_model_cache = None
_cross_model_cache = None
_model_lock = Lock()


def _get_safe_device():
    """
    Get device for model loading with automatic fallback to CPU if GPU is not available.
    
    Returns:
        str: Device string ('cuda', 'cpu', etc.) that is safe to use
    """
    device = _EMBED_DEVICE_ENV or "cpu"
    
    # If device is set to GPU-related values, check availability
    if device.lower() in ("cuda", "gpu"):
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            else:
                print(f"⚠️  GPU requested but not available. Falling back to CPU.")
                return "cpu"
        except ImportError:
            print(f"⚠️  PyTorch not available. Falling back to CPU.")
            return "cpu"
    
    return device


def _get_bi_model(model_name=_BI_MODEL_NAME):
    """Get bi-encoder model with thread-safe caching."""
    global _bi_model_cache
    if _bi_model_cache is None:
        with _model_lock:
            # Double-check pattern to avoid race condition
            if _bi_model_cache is None:
                try:
                    device = _get_safe_device()
                    kwargs = {"device": device}
                    _bi_model_cache = SentenceTransformer(model_name, **kwargs)
                    print(f"✓ Loaded bi-encoder model on device: {device}")
                except Exception as e:
                    # Fallback to CPU if any error occurs
                    print(f"⚠️  Error loading model on {device}, falling back to CPU: {e}")
                    kwargs = {"device": "cpu"}
                    _bi_model_cache = SentenceTransformer(model_name, **kwargs)
                    print(f"✓ Loaded bi-encoder model on device: cpu (fallback)")
    return _bi_model_cache


def _get_cross_model(model_name=_CROSS_MODEL_NAME):
    """Get cross-encoder model with thread-safe caching."""
    global _cross_model_cache
    if _cross_model_cache is None:
        with _model_lock:
            # Double-check pattern to avoid race condition
            if _cross_model_cache is None:
                try:
                    device = _get_safe_device()
                    kwargs = {"device": device}
                    _cross_model_cache = CrossEncoder(model_name, **kwargs)
                    print(f"✓ Loaded cross-encoder model on device: {device}")
                except Exception as e:
                    # Fallback to CPU if any error occurs
                    print(f"⚠️  Error loading model on {device}, falling back to CPU: {e}")
                    kwargs = {"device": "cpu"}
                    _cross_model_cache = CrossEncoder(model_name, **kwargs)
                    print(f"✓ Loaded cross-encoder model on device: cpu (fallback)")
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

def compute_evidence_scores_bi_encoder(claim: str, evidence_pieces: List[str]) -> np.ndarray:
    """
    Compute claim-evidence similarity scores using bi-encoder (fast, approximate).
    This is used for pre-filtering before using the slower cross-encoder.
    
    Returns:
        Array of cosine similarity scores [0, 1] for each evidence piece
    """
    if not evidence_pieces:
        return np.array([])
    
    bi_model = _get_bi_model()
    claim_emb = bi_model.encode(claim, normalize_embeddings=True)
    evidence_embs = bi_model.encode(evidence_pieces, normalize_embeddings=True)
    
    # Compute cosine similarities (already normalized)
    cos_sims = np.dot(evidence_embs, claim_emb)
    
    # Normalize to [0, 1] range (cosine similarity is in [-1, 1])
    cos_sims = (cos_sims + 1.0) / 2.0
    
    return cos_sims

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

def extract_verdict(conclusion):
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
                                  bi_prefilter_top_n: int = 15,
                                  log_callback=None) -> Tuple[List[str], List[float]]:
    """
    Lọc evidence dựa trên relevance score với claim.
    Sử dụng 2-stage filtering:
    1. Pre-filter với Bi-encoder (nhanh) để lấy top N evidence
    2. Fine-grained filtering với CrossEncoder (chậm nhưng chính xác) cho top N
    
    Chỉ giữ lại evidence có relevance score > threshold.
    NHƯNG luôn đảm bảo giữ lại ít nhất min_keep evidence (top evidence).
    
    Args:
        claim: Claim cần fact-check
        evidence_pieces: Danh sách evidence pieces
        relevance_threshold: Ngưỡng relevance tối thiểu (default: 0.3)
        min_keep: Số lượng evidence tối thiểu cần giữ lại (default: 3)
        bi_prefilter_top_n: Số lượng evidence để pre-filter bằng Bi-encoder trước khi dùng CrossEncoder (default: 15)
        log_callback: Hàm callback để log các bước (optional)
    
    Returns:
        Tuple[List[str], List[float]]: (filtered_evidence, relevance_scores) - chỉ giữ evidence liên quan
    """
    if not evidence_pieces:
        if log_callback:
            log_callback("⚠️ Không có evidence pieces để filter!")
        return [], []
    
    if log_callback:
        log_callback(f"\n🔍 BƯỚC 1: Pre-filter với Bi-encoder cho {len(evidence_pieces)} evidence pieces")
        log_callback(f"   → Sử dụng Bi-encoder model (nhanh) để lấy top {bi_prefilter_top_n} candidates")
        log_callback(f"   → Claim: {claim}")  # Ghi đầy đủ claim, không truncate
    
    try:
        # BƯỚC 1: Pre-filter với Bi-encoder để giảm số lượng evidence cần xử lý bằng CrossEncoder
        # Bi-encoder nhanh hơn nhiều vì có thể encode tất cả cùng lúc
        bi_scores = compute_evidence_scores_bi_encoder(claim, evidence_pieces)
        
        # Lấy top N evidence từ Bi-encoder scores
        bi_prefilter_top_n = min(bi_prefilter_top_n, len(evidence_pieces))
        top_bi_indices = np.argsort(-bi_scores)[:bi_prefilter_top_n]
        top_bi_evidence = [evidence_pieces[i] for i in top_bi_indices]
        
        if log_callback:
            log_callback(f"   → Đã chọn top {len(top_bi_evidence)} evidence từ Bi-encoder scores")
            log_callback(f"   → Bi-encoder score range: [{bi_scores.min():.4f}, {bi_scores.max():.4f}]")
            log_callback(f"   → Top {min(5, len(top_bi_evidence))} Bi-encoder scores:")
            for idx, ev_idx in enumerate(top_bi_indices[:5]):
                log_callback(f"      [{idx+1}] Score: {bi_scores[ev_idx]:.4f} - {evidence_pieces[ev_idx][:100]}...")
        
        # BƯỚC 2: Tính relevance scores bằng CrossEncoder chỉ cho top N evidence (chậm nhưng chính xác)
        if log_callback:
            log_callback(f"\n🔍 BƯỚC 2: Fine-grained filtering với CrossEncoder cho {len(top_bi_evidence)} evidence")
            log_callback(f"   → Sử dụng CrossEncoder model (chậm nhưng chính xác)")
        
        scores = compute_evidence_scores(claim, top_bi_evidence)
        
        if log_callback:
            log_callback(f"   → Raw scores range: [{scores.min():.4f}, {scores.max():.4f}]")
        
        # Normalize CrossEncoder scores về [0, 1] nếu cần
        if scores.size > 0:
            # CrossEncoder scores có thể âm hoặc dương
            # Strategy: normalize về [0, 1] nhưng giữ nguyên ranking
            min_score = scores.min()
            max_score = scores.max()
            if max_score > min_score:
                cross_scores_normalized = (scores - min_score) / (max_score - min_score)
            else:
                # Tất cả scores bằng nhau, set về 0.5
                cross_scores_normalized = np.full(len(scores), 0.5)
        else:
            cross_scores_normalized = np.zeros(len(top_bi_evidence))
        
        # Map CrossEncoder scores về original evidence_pieces indices
        # cross_scores_normalized[i] corresponds to top_bi_evidence[i] which is evidence_pieces[top_bi_indices[i]]
        scores_map = {}  # original_idx -> normalized_score
        for i in range(len(top_bi_indices)):
            orig_idx = top_bi_indices[i]
            scores_map[orig_idx] = float(cross_scores_normalized[i])
        
        # Create normalized scores array for all evidence pieces
        # Non-selected evidence (not in top_bi_indices) get score 0.0
        scores_normalized = np.zeros(len(evidence_pieces))
        for orig_idx, score in scores_map.items():
            scores_normalized[orig_idx] = score
        
        if log_callback:
            log_callback(f"   → Normalized scores range: [{cross_scores_normalized.min():.4f}, {cross_scores_normalized.max():.4f}]")
            log_callback(f"   → Top 5 evidence scores (CrossEncoder):")
            top_5_cross_indices = np.argsort(-cross_scores_normalized)[:5]
            for idx, cross_idx in enumerate(top_5_cross_indices):
                orig_idx = top_bi_indices[cross_idx]
                log_callback(f"      [{idx+1}] Score: {cross_scores_normalized[cross_idx]:.4f} - {evidence_pieces[orig_idx]}")
        
        # Lọc evidence có relevance > threshold
        # NHƯNG: luôn giữ lại ít nhất top 1 evidence nếu có
        filtered_evidence = []
        filtered_scores = []
        
        # Tìm top evidence indices và điều chỉnh threshold (dựa trên CrossEncoder scores)
        adjusted_threshold = relevance_threshold
        top_cross_indices = list(np.argsort(-cross_scores_normalized))  # Sorted by CrossEncoder scores
        
        if len(top_cross_indices) > 0:
            top_cross_score = cross_scores_normalized[top_cross_indices[0]]
            # Nếu top score > 0.5 nhưng dưới threshold, giảm threshold một chút
            if top_cross_score > 0.5 and top_cross_score < relevance_threshold:
                adjusted_threshold = min(relevance_threshold, top_cross_score * 0.8)
                if log_callback:
                    log_callback(f"\n🔍 BƯỚC 3: Điều chỉnh threshold")
                    log_callback(f"   → Threshold ban đầu: {relevance_threshold}")
                    log_callback(f"   → Top score: {top_cross_score:.4f}")
                    log_callback(f"   → Threshold sau điều chỉnh: {adjusted_threshold:.4f}")
        
        # Map top_cross_indices về original indices
        top_indices = [top_bi_indices[cross_idx] for cross_idx in top_cross_indices]
        
        if log_callback:
            log_callback(f"\n🔍 BƯỚC 4: Lọc evidence theo threshold ({adjusted_threshold:.4f})")
        
        # Bước 1: Lọc evidence theo threshold (chỉ xét các evidence đã được pre-filter)
        filtered_indices = set()  # Track indices đã được thêm vào filtered_evidence
        for orig_idx in top_bi_indices:
            score = scores_map[orig_idx]
            if score >= adjusted_threshold:
                filtered_evidence.append(evidence_pieces[orig_idx])
                filtered_scores.append(score)
                filtered_indices.add(orig_idx)
        
        if log_callback:
            log_callback(f"   → Số evidence sau khi lọc: {len(filtered_evidence)}/{len(top_bi_evidence)} (từ {len(evidence_pieces)} ban đầu)")
            if len(filtered_evidence) > 0:
                log_callback(f"   → Evidence được giữ lại:")
                for idx, (ev, score) in enumerate(zip(filtered_evidence, filtered_scores)):
                    # Ghi đầy đủ evidence, không truncate
                    log_callback(f"      [{idx+1}] Score: {score:.4f} - {ev}")
        
        # Bước 2: Nếu số lượng evidence sau khi lọc < min_keep, bổ sung top evidence
        # Đảm bảo luôn có ít nhất min_keep evidence (hoặc tất cả nếu ít hơn min_keep)
        if len(filtered_evidence) < min_keep and len(top_indices) > 0:
            if log_callback:
                log_callback(f"\n🔍 BƯỚC 5: Bổ sung evidence để đạt min_keep={min_keep}")
                log_callback(f"   → Hiện tại có {len(filtered_evidence)} evidence, cần thêm {min_keep - len(filtered_evidence)}")
            
            # Thêm top evidence chưa có trong filtered_evidence
            added_count = 0
            for orig_idx in top_indices:
                if len(filtered_evidence) >= min_keep:
                    break
                if orig_idx not in filtered_indices:
                    score = scores_map[orig_idx]
                    # Chỉ thêm nếu score > 0.2 (ngưỡng tối thiểu)
                    if score > 0.2:
                        filtered_evidence.append(evidence_pieces[orig_idx])
                        filtered_scores.append(score)
                        filtered_indices.add(orig_idx)
                        added_count += 1
                        if log_callback:
                            # Ghi đầy đủ evidence, không truncate
                            log_callback(f"      [+] Thêm evidence #{orig_idx} (score: {score:.4f}) - {evidence_pieces[orig_idx]}")
            
            if log_callback:
                log_callback(f"   → Đã thêm {added_count} evidence")
        
        # Bước 3: Sắp xếp lại theo score (descending) để đảm bảo top evidence ở đầu
        if filtered_evidence and len(filtered_evidence) > 1:
            if log_callback:
                log_callback(f"\n🔍 BƯỚC 6: Sắp xếp lại evidence theo score (descending)")
            
            # Tạo list of tuples (score, evidence) để sort
            evidence_score_pairs = list(zip(filtered_scores, filtered_evidence))
            evidence_score_pairs.sort(reverse=True, key=lambda x: x[0])
            filtered_evidence = [ev for _, ev in evidence_score_pairs]
            filtered_scores = [score for score, _ in evidence_score_pairs]
        
        if log_callback:
            log_callback(f"\n✅ KẾT QUẢ: Đã chọn {len(filtered_evidence)} evidence từ {len(evidence_pieces)} evidence ban đầu")
            log_callback(f"   → Đã pre-filter {len(top_bi_evidence)} evidence bằng Bi-encoder, sau đó dùng CrossEncoder")
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
- Nếu yêu cầu có nhiều khía cạnh, TẤT CẢ các khía cạnh CỐT LÕI phải được ỦNG HỘ để chọn phán quyết này.
- LƯU Ý: Chỉ cần bằng chứng hỗ trợ CỐT LÕI của yêu cầu, không cần phải khớp 100% từng từ.
- LƯU Ý: Nếu bằng chứng xác nhận Ý NGHĨA của yêu cầu (dù dùng từ khác) thì vẫn coi là Supported.
- VỀ THỜI GIAN/TIẾN TRÌNH: 
  * Nếu yêu cầu nói "đã được quyết định" nhưng bằng chứng nói "đã đề nghị/đã tờ trình", hãy kiểm tra kỹ:
    - Nếu bằng chứng CŨNG nói về "Quyết định số...", "được bổ sung vào...", "xếp hạng...", "công nhận..." → coi là đã được quyết định → Supported
    - Nếu bằng chứng CHỈ nói "tờ trình đề nghị" mà KHÔNG có thông tin về quyết định sau đó → Not Enough Evidence
  * Quy trình: "đề nghị" → "thông qua" → "quyết định" là bình thường. Nếu bằng chứng nói về quyết định hoặc kết quả cuối cùng, coi là Supported.

Refuted
- Dùng khi có bằng chứng E[i] rõ ràng BÁC BỎ hoặc MÂU THUẪN trực tiếp với yêu cầu.
- Ví dụ: Yêu cầu nói "A là B" nhưng bằng chứng nói "A không phải là B" hoặc "A là C" (không phải B).
- Nếu yêu cầu có nhiều khía cạnh, dù chỉ 1 khía cạnh CỐT LÕI bị BÁC BỎ thì cũng đủ để chọn phán quyết này.
- LƯU Ý: KHÔNG chọn Refuted nếu bằng chứng chỉ KHÔNG NHẮC ĐẾN một số chi tiết phụ trong yêu cầu. Chỉ chọn khi có mâu thuẫn trực tiếp.
- LƯU Ý: KHÔNG chọn Refuted chỉ vì bằng chứng dùng từ khác nhưng ý nghĩa giống nhau.

Not Enough Evidence
- Dùng khi tất cả E[i] KHÔNG ĐỦ thông tin để xác nhận hoặc bác bỏ yêu cầu.
- Dùng khi bằng chứng không liên quan hoặc quá mơ hồ.
- Dùng nếu yêu cầu quá MƠ HỒ hoặc không thể kiểm chứng bằng dữ liệu hiện có.
- LƯU Ý: KHÔNG chọn Not Enough Evidence nếu bằng chứng đã hỗ trợ CỐT LÕI của yêu cầu, chỉ thiếu một số chi tiết phụ.

QUAN TRỌNG:
1. Tập trung vào CỐT LÕI của yêu cầu, không yêu cầu khớp 100% từng từ.
2. Hiểu Ý NGHĨA của yêu cầu, không chỉ tìm cụm từ chính xác.
3. Về THỜI GIAN/TIẾN TRÌNH: 
   - Nếu yêu cầu nói "đã được X" và bằng chứng nói về quyết định/kết quả liên quan đến X → Supported
   - Nếu bằng chứng nói về cả "đề nghị" VÀ "quyết định/kết quả" → Supported (quyết định là kết quả cuối)
   - Nếu bằng chứng CHỈ nói về "đề nghị" mà không có thông tin về kết quả → Not Enough Evidence
4. Nếu bằng chứng hỗ trợ phần lớn yêu cầu, chỉ thiếu một số chi tiết phụ, vẫn chọn Supported.
5. Đọc KỸ toàn bộ bằng chứng: một bằng chứng có thể chứa cả "đề nghị" và "quyết định" ở các phần khác nhau.

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
        justification = f"Lỗi khi gọi LLM judge: {e}.Initial Action Execution:Initial Action Execution: Mặc định Not Enough Evidence."
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


def judge(record):
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
    
    # Dùng top_k=2 để tăng tốc độ judge (giảm từ 3 xuống 2)
    verdict_string, evidence_info = _llm_judge_with_evidence(claim, evidence_pieces, top_k=3, log_callback=filter_log_callback)
    
    # Ghi log vào evidence_info để có thể append vào report sau
    if filter_log_lines:
        evidence_info['filter_log'] = filter_log_lines
    
    return verdict_string, evidence_info
