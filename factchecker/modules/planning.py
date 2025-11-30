import re
from .llm import prompt_ollama

plan_prompt = """HƯỚNG DẪN
Mọi phát ngôn đều phải có thể kiểm chứng và dựa trên dữ liệu đã được khai thác. Nếu không đủ thông tin, phải kết luận `Not Enough Information`. Tuyệt đối không được giả định.
GIAO THỨC KHÔNG ẢO GIÁC: trước khi trả lời, cần kiểm tra lại tính xác thực của thông tin.

Nhiệm vụ: Tìm bằng chứng để kiểm chứng phát biểu (Claim).
Claim: {claim}

Quy tắc:
- THỰC HIỆN TỐI THIỂU 2 HÀNH ĐỘNG.
- Bạn CHỈ ĐƯỢC phép in ra danh sách các hành động theo đúng cú pháp bên dưới.
- Mỗi hành động phải được in trong cùng một khối mã (```) duy nhất.
- Không in ra danh sách các bước.
- Không được giải thích, không được in văn bản tự nhiên, không thêm mô tả.


Hành động hợp lệ:
web_search("{claim}")
{valid_actions}

ĐỊNH DẠNG ĐẦU RA BẮT BUỘC (thay ... bằng từ, cụm từ, hoặc thực thể từ Yêu cầu):
{examples}

BẢN GHI:
{record}

Your Actions:
"""
decompose_prompt = """Hướng dẫn
Phân tách phát biểu dưới đây thành 1-3 **tiểu phát biểu** có thể được kiểm chứng độc lập, giữ ngữ nghĩa gốc, tránh suy diễn hoặc mở rộng nội dung.


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
        "web_search": {
            "desc": "Thực hiện tìm kiếm web mở cho các trang liên quan.", 
            #"example": 'web_search("{claim}")',
            "params": {
                "query": "Từ khóa tìm kiếm"
            }
        }
    }
    
    decompose_text = decompose_prompt.format(claim=claim)
    subclaims_response = prompt_ollama(decompose_text, think=think)
    subclaims = [] 
    
    for line in subclaims_response.splitlines():
        line = line.strip()
        if line:
            subclaims.append(line)
    
    if not subclaims:
        subclaims = [claim]
    
    if not actions:
        actions = ["web_search"]
        
    valid_actions = "\n".join([f"{a}: {action_definitions[a]['desc']}" for a in actions])
    #examples = "\n".join([f"{action_definitions[a]['example']}" for a in actions])
    prompt = plan_prompt.format(valid_actions=valid_actions, examples=examples, record=record, claim=claim)
    print("[PLANNER] Using LLM planner (slower)")
    response = prompt_ollama(prompt, think=think)
    # return response

    all_actions = []

    for sub in subclaims:
        prompt = plan_prompt.format(
            valid_actions=valid_actions, 
            examples=examples,
            record=record,
            claim=sub
        )
        response = prompt_ollama(prompt, think=think)
        all_actions.extend(response.splitlines())
    return "\n".join(all_actions)