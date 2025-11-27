import re
from .llm import prompt_ollama

plan_prompt = """HƯỚNG DẪN
Kiến thức hiện tại vẫn chưa đủ để đánh giá tính xác thực của YÊU CẦU.
Hãy đề xuất ngắn gọn các hành động để thu thập bằng chứng mới, theo đúng quy tắc sau:

QUY TẮC:
- Mỗi hành động là 1 cụm từ được liệt kê trong mục HÀNH ĐỘNG HỢP LỆ, đi cùng là mô tả sau dấu ":". Hành động được dùng theo định dạng trong mục ĐỊNH DẠNG ĐẦU RA BẮT BUỘC.
- Đề xuất hành động phải liên quan trực tiếp đến Yêu cầu và chứa ít nhất một từ khóa hoặc thực thể trong Yêu cầu.
- Không đề xuất các hành động tương tự hoặc đã được sử dụng trước đó trong BẢN GHI.
- KHÔNG in bất cứ thứ gì ngoài các Hành động đề xuất.

HÀNH ĐỘNG HỢP LỆ (Hành động: mô tả):
{valid_actions}

ĐỊNH DẠNG ĐẦU RA BẮT BUỘC (thay ... bằng từ, cụm từ, hoặc thực thể từ Yêu cầu):
{examples}

BẢN GHI:
{record}

YÊU CẦU: {claim}

In ngắn gọn đề xuất của bạn, chỉ có hành động theo ĐỊNH DẠNG ĐẦU RA BẮT BUỘC, mỗi hành động nằm trên 1 dòng:
"""

decompose_prompt = """Instructions
Decompose the claim into smaller, manageable sub-claims or questions that can be addressed individually. Each sub-claim should be specific and focused.
There should be no more than 5 sub-claims.

Claim: {claim}
Your Sub-Claims:

"""

def heuristic_plan(claim, actions=None):
    """
    Fast heuristic-based planner that extracts keywords and generates web_search queries.
    Returns a string with web_search actions, or None if heuristics fail.
    """
    if not actions or "web_search" not in actions:
        return None
    
    # Extract key entities and keywords from the claim
    claim_lower = claim.lower()
    
    # Common Vietnamese names/entities (add more as needed)
    entities = []
    name_patterns = [
        r'\b(putin|trump|biden|zelensky|xi jinping|kim jong|modi)\b',
        r'\b(nga|russia|ukraine|trung quốc|china|mỹ|usa|anh|britain)\b',
        r'\b(vietnam|việt nam|hà nội|hồ chí minh|sài gòn)\b'
    ]
    
    for pattern in name_patterns:
        matches = re.findall(pattern, claim_lower, re.IGNORECASE)
        entities.extend(matches)
    
    # Extract important keywords (nouns, verbs related to actions)
    # Remove common stop words
    stop_words = {'nói', 'sẽ', 'nếu', 'bị', 'được', 'có', 'là', 'và', 'của', 'cho', 'với', 'the', 'a', 'an', 'is', 'are', 'was', 'were', 'will', 'would', 'said', 'says'}
    words = re.findall(r'\b\w+\b', claim_lower)
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Build search queries
    queries = []
    
    # Strategy 1: Use the full claim as-is (most direct)
    if len(claim) < 100:  # Only if claim is reasonably short
        queries.append(claim)
    
    # Strategy 2: Combine key entities + important keywords
    if entities:
        # Take first 2-3 most important keywords
        important_keywords = [k for k in keywords[:3] if k not in [e.lower() for e in entities]]
        query_parts = entities[:2] + important_keywords[:2]
        if query_parts:
            queries.append(' '.join(query_parts))
    
    # Strategy 3: Entity-focused query
    if entities:
        queries.append(' '.join(entities[:2]))
    
    # Remove duplicates and limit to 2 queries
    seen = set()
    unique_queries = []
    for q in queries:
        q_normalized = q.lower().strip()
        if q_normalized and q_normalized not in seen and len(q_normalized) > 5:
            seen.add(q_normalized)
            unique_queries.append(q)
            if len(unique_queries) >= 2:
                break
    
    if not unique_queries:
        return None
    
    # Format as web_search actions
    actions_list = [f'web_search("{q}")' for q in unique_queries]
    return '\n'.join(actions_list)


def plan(claim, record="", examples="", actions=None, think=True, use_heuristic_first=True):
    """
    Hybrid planner: tries fast heuristics first, falls back to LLM if needed.
    
    Args:
        use_heuristic_first: If True, try heuristic planner first before LLM
    """
    # Try heuristic planner first (fast, no LLM call)
    if use_heuristic_first:
        heuristic_result = heuristic_plan(claim, actions)
        if heuristic_result:
            print("[PLANNER] Using heuristic planner (fast)")
            return heuristic_result
        print("[PLANNER] Heuristic planner returned empty, falling back to LLM")
    
    # Fallback to LLM planner
    action_definitions = {
        "geolocate": {"desc": "Xác định quốc gia nơi ảnh được chụp bằng cách cung cấp ID ảnh.", "example": "geolocate(<image:k>)"},
        "reverse_search": {"desc": "Thực hiện tìm kiếm ảnh ngược trên web để tìm ảnh tương tự.", "example": "reverse_search(<image:k>)"},
        "web_search": {"desc": "Thực hiện tìm kiếm web mở cho các trang liên quan.", "example": 'web_search("...")'},
        "image_search": {"desc": "Lấy các hình ảnh liên quan cho một truy vấn.", "example": 'image_search("...")'}
    }
    if not actions:
        actions = ["web_search", "image_search"]
    elif actions == "All":
        actions = list(action_definitions.keys())
    valid_actions = "\n".join([f"{a}: {action_definitions[a]['desc']}" for a in actions])
    examples = "\n".join([f"{action_definitions[a]['example']}" for a in actions])
    prompt = plan_prompt.format(valid_actions=valid_actions, examples=examples, record=record, claim=claim)
    print("[PLANNER] Using LLM planner (slower)")
    response = prompt_ollama(prompt, think=think)
    return response
