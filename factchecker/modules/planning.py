# factchecker/modules/planning.py

import re
from underthesea import word_tokenize, pos_tag
from typing import List, Tuple
import os

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# Import cho NER multilingual (Hugging Face Transformers)
from transformers import pipeline

NER_MODEL = os.getenv("FACTCHECKER_NER_MODEL", "Davlan/xlm-roberta-base-wikiann-ner")

# Load NER pipeline multilingual (XLM-RoBERTa fine-tuned cho NER trên WikiANN)
# Model: Davlan/xlm-roberta-base-wikiann-ner
# Hỗ trợ đa ngôn ngữ thực sự, bao gồm tiếng Việt và tiếng Anh
_ner_pipeline = None

def get_ner_pipeline():
    """Lazy load NER pipeline to avoid loading model at import time."""
    global _ner_pipeline
    if _ner_pipeline is None:
        # Load NER pipeline multilingual (XLM-RoBERTa fine-tuned cho NER trên WikiANN)
        # Model: Davlan/xlm-roberta-base-wikiann-ner
        # Hỗ trợ đa ngôn ngữ thực sự, bao gồm tiếng Việt và tiếng Anh
        _ner_pipeline = pipeline("ner", model=NER_MODEL, aggregation_strategy="simple")
    return _ner_pipeline

# Stopwords tiếng Việt (mở rộng)
VIETNAMESE_STOPWORDS = {
    'là', 'của', 'và', 'có', 'được', 'trong', 'với', 'theo', 
    'này', 'đó', 'khi', 'sẽ', 'đã', 'mà', 'về', 'cho', 'từ',
    'vào', 'trên', 'dưới', 'sau', 'trước', 'trong', 'ngoài',
    'một', 'hai', 'ba', 'các', 'những', 'mỗi', 'mọi',
    'thì', 'nếu', 'nên', 'để', 'như', 'vì', 'do'
}

def extract_entities_and_keywords(claim: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Trích xuất entities, keywords và phrases quan trọng từ claim.
    
    Returns:
        Tuple[List[str], List[str], List[str]]: (entities, keywords, phrases)
    """
    # 1. NER để tìm thực thể (dùng model multilingual)
    # Lazy load pipeline khi cần
    ner_pipeline = get_ner_pipeline()
    entities_result = ner_pipeline(claim)
    entities = []
    for entity in entities_result:
        if entity['entity_group'] in ['PER', 'ORG', 'LOC', 'MISC']:
            entities.append(entity['word'])
    
    # 2. POS tagging để tìm từ khóa quan trọng
    tokens = pos_tag(claim)
    keywords = []
    for word, pos in tokens:
        word_lower = word.lower()
        # Giữ danh từ (N*), động từ (V*), tính từ (A*)
        if (pos.startswith(('N', 'V', 'A')) and 
            word_lower not in VIETNAMESE_STOPWORDS and
            len(word) > 1):
            keywords.append(word)
    
    # 3. Trích xuất cụm từ quan trọng (bigrams, trigrams)
    words = word_tokenize(claim)
    phrases = []
    
    # Bigrams và trigrams không chứa stopwords
    for i in range(len(words) - 1):
        if words[i].lower() not in VIETNAMESE_STOPWORDS:
            # Bigram
            if i + 1 < len(words):
                if words[i+1].lower() not in VIETNAMESE_STOPWORDS:
                    phrases.append(f"{words[i]} {words[i+1]}")
            # Trigram
            if i + 2 < len(words):
                if (words[i+1].lower() not in VIETNAMESE_STOPWORDS and
                    words[i+2].lower() not in VIETNAMESE_STOPWORDS):
                    phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")
    
    return entities, keywords, phrases


def generate_query_from_entities(entities: List[str], keywords: List[str]) -> str:
    """
    Tạo query từ entities và keywords.
    """
    query_parts = []
    
    # Ưu tiên entities
    if entities:
        query_parts.extend(entities[:2])  # Tối đa 2 entities
    
    # Thêm keywords quan trọng
    remaining_keywords = [k for k in keywords if k not in entities][:3]
    query_parts.extend(remaining_keywords)
    
    return ' '.join(query_parts[:5])  # Tối đa 5 từ


def simplify_claim(claim: str, max_words: int = 12) -> str:
    """
    Rút gọn claim bằng cách loại bỏ stopwords và từ không quan trọng.
    """
    words = word_tokenize(claim)
    filtered_words = [
        w for w in words 
        if w.lower() not in VIETNAMESE_STOPWORDS and len(w) > 1
    ]
    
    # Giữ các từ quan trọng nhất
    return ' '.join(filtered_words[:max_words])


def is_claim_length_acceptable(claim: str, max_words: int = 30, max_chars: int = 250) -> bool:
    """
    Kiểm tra xem claim có độ dài hợp lý để search trực tiếp không.
    
    Args:
        claim: Câu claim cần kiểm tra
        max_words: Số từ tối đa cho phép (mặc định 30)
        max_chars: Số ký tự tối đa cho phép (mặc định 250)
    
    Returns:
        bool: True nếu claim có độ dài hợp lý
    """
    word_count = len(claim.split())
    char_count = len(claim)
    
    return word_count <= max_words and char_count <= max_chars


def validate_query(query: str, original_claim: str = None) -> Tuple[bool, str]:
    """
    Validate query để đảm bảo không chứa prompt leak và hợp lệ.
    
    Args:
        query: Query cần validate
        original_claim: Claim gốc để kiểm tra query có chứa từ khóa quan trọng không
    
    Returns:
        Tuple[bool, str]: (is_valid, reason) - True nếu hợp lệ, False nếu không và lý do
    """
    if not query or not query.strip():
        return False, "Query rỗng"
    
    query_lower = query.lower().strip()
    
    # Kiểm tra prompt leak patterns (mở rộng danh sách)
    prompt_leak_patterns = [
        "dưới đây là",
        "câu tìm kiếm đã rút ngắn",
        "câu tìm kiếm:",
        "rất nhiều thông tin",
        "không cần thiết",
        "quy tắc",
        "hướng dẫn",
        "định dạng đầu ra",
        "yêu cầu gốc:",
        "kết quả:",
        "ví dụ:",
        "yêu cầu này có thể",
        "có thể giúp tôi",
        "giúp tôi cải thiện",
        "cải thiện chi tiết",
        "chi tiết hơn",
        "câu hỏi của bạn",
        "yêu cầu của bạn",
        "theo yêu cầu",
        "đáp ứng yêu cầu",
        "bạn muốn",
        "bạn cần",
        "để tìm kiếm",
        "để tra cứu",
    ]
    
    for pattern in prompt_leak_patterns:
        if pattern in query_lower:
            return False, f"Chứa prompt leak: '{pattern}'"
    
    # Kiểm tra query chỉ là hướng dẫn/mô tả (không phải query thực sự)
    instruction_starters = [
        "bạn được",
        "hãy",
        "bạn cần",
        "bạn phải",
        "hãy thử",
        "bạn có thể",
        "để",
    ]
    
    first_few_words = ' '.join(query.split()[:3]).lower()
    for starter in instruction_starters:
        if first_few_words.startswith(starter):
            return False, f"Query bắt đầu bằng hướng dẫn: '{starter}'"
    
    # Kiểm tra độ dài tối thiểu (tăng lên 5 từ)
    words = query.split()
    if len(words) < 5:
        return False, f"Query quá ngắn ({len(words)} từ, tối thiểu 5 từ)"
    
    # Kiểm tra query không chỉ là dấu câu hoặc từ đơn lẻ
    meaningful_words = [w for w in words if len(w) > 1 and w.lower() not in VIETNAMESE_STOPWORDS]
    if len(meaningful_words) < 2:
        return False, "Query không chứa đủ từ có nghĩa"
    
    # Kiểm tra query có chứa từ khóa từ claim không (nếu có claim gốc)
    if original_claim:
        claim_tokens = set([w.lower() for w in word_tokenize(original_claim)
                           if w.lower() not in VIETNAMESE_STOPWORDS and len(w) > 1])
        query_tokens = set([w.lower() for w in word_tokenize(query)
                           if w.lower() not in VIETNAMESE_STOPWORDS and len(w) > 1])
        
        # Kiểm tra có ít nhất 2 từ khóa chung (entities/keywords quan trọng)
        overlap = claim_tokens & query_tokens
        if len(overlap) < 2:
            return False, f"Query chỉ có {len(overlap)} từ khóa chung với claim (cần ít nhất 2 từ khóa)"
    
    return True, ""


def shorten_claim_with_llm(claim: str) -> str:
    """
    Dùng LLM để rút ngắn claim nhưng vẫn giữ thông tin quan trọng (tên riêng, thời gian, địa điểm).
    
    Args:
        claim: Câu claim cần rút ngắn
    
    Returns:
        str: Claim đã được rút ngắn (tối đa 30 từ hoặc 250 ký tự)
    """
    from .llm import prompt_ollama
    
    shorten_prompt = """Bạn được cung cấp một câu YÊU CẦU dài. Hãy rút ngắn câu YÊU CẦU này thành một CÂU TÌM KIẾM ngắn gọn (tối đa 30 từ hoặc 250 ký tự) để tìm kiếm thông tin trên web.

### QUY TẮC RÚT NGẮN:
1. PHẢI GIỮ LẠI:
   - Tất cả tên riêng (người, địa danh, tổ chức)
   - Thời gian cụ thể (ngày, tháng, năm, giờ)
   - Số liệu, con số quan trọng
   - Động từ và danh từ chính thể hiện hành động/sự việc

2. CÓ THỂ LOẠI BỎ:
   - Từ nối không cần thiết (và, nhưng, để, mà, v.v.)
   - Từ mô tả dài dòng
   - Phần giải thích không cần thiết

3. KẾT QUẢ:
   - CÂU TÌM KIẾM phải là câu hoàn chỉnh, có nghĩa
   - CÂU TÌM KIẾM phải chứa đủ thông tin để tìm được bằng chứng liên quan
   - Không thay đổi ý nghĩa của YÊU CẦU gốc

### YÊU CẦU GỐC:
{claim}

### ĐỊNH DẠNG ĐẦU RA BẮT BUỘC:
Chỉ in CÂU TÌM KIẾM đã rút ngắn, KHÔNG in thêm mô tả hay giải thích.

CÂU TÌM KIẾM đã rút ngắn:
"""
    prompt = shorten_prompt.format(claim=claim)
    
    try:
        shortened = prompt_ollama(prompt, think=False)
        
        # Nếu LLM trả về nhiều dòng (bao gồm cả hướng dẫn), giữ lại toàn bộ để xử lý sau
        # nhưng loại bỏ khoảng trắng thừa ở hai đầu.
        shortened = shortened.strip()
        
        # Loại bỏ các prefix/suffix thừa mà LLM có thể thêm vào
        unwanted_prefixes = [
            'dữ liệu chính xác:',
            'quý khách,',
            'phải điền vào:',
            'câu tìm kiếm đã rút ngắn:',
            'câu tìm kiếm:',
            'tìm kiếm:',
            'query:',
            'đây là câu tìm kiếm:',
            'câu hỏi:',
            'câu yêu cầu rút ngắn là:',
            'câu yêu cầu ngắn gọn được cung cấp.',
            'câu yêu cầu ngắn gọn:',
            'sắp xếp theo độ dài của câu:',
            'hãy thay thế',
            'dưới đây là câu tìm kiếm cuối cùng:',
            'dưới đây là câu tìm kiếm:',
            'câu tìm kiếm cuối cùng:',
        ]
        
        # Loại bỏ prefix (kiểm tra startswith)
        shortened_lower = shortened.lower()
        for prefix in unwanted_prefixes:
            if shortened_lower.startswith(prefix):
                shortened = shortened[len(prefix):].strip()
                # Loại bỏ dấu hai chấm hoặc dấu phẩy ở đầu nếu có
                shortened = re.sub(r'^[:,\s]+', '', shortened)
                break
        
        # Pattern 0: Nếu có nhiều dòng, loại bỏ các dòng hướng dẫn dạng bullet
        lines = [ln.strip() for ln in shortened.splitlines() if ln.strip()]
        if len(lines) > 1:
            filtered_lines = []
            instruction_keywords = [
                "tên riêng", "thời gian cụ thể", "số liệu", "con số quan trọng",
                "từ nối", "từ mô tả", "phần giải thích", "từ chỉ", "danh từ", "tính từ"
            ]
            for ln in lines:
                ln_lower = ln.lower()
                # Loại các bullet line bắt đầu bằng dấu '-' hoặc chứa các cụm chỉ dẫn rõ ràng
                if ln_lower.startswith('-'):
                    if any(k in ln_lower for k in instruction_keywords):
                        continue
                if any(k in ln_lower for k in instruction_keywords) and '{claim}' not in ln_lower:
                    continue
                filtered_lines.append(ln)
            if filtered_lines:
                # Ưu tiên dòng dài nhất trong các dòng đã lọc (thường là câu tìm kiếm)
                shortened = max(filtered_lines, key=len)
                shortened_lower = shortened.lower()
        
        # Pattern 1: Tìm text trong ngoặc kép (thường là query thực sự)
        quoted_match = re.search(r'[""]([^""]+)[""]', shortened)
        if quoted_match:
            shortened = quoted_match.group(1).strip()
        
        # Pattern 2: Loại bỏ các câu mô tả/hướng dẫn ở đầu.
        # Ưu tiên câu chứa ít nhất một từ quan trọng từ claim.
        sentences = re.split(r'[.!?]\s+', shortened)
        if len(sentences) > 1:
            claim_tokens = [w.lower() for w in word_tokenize(claim)
                            if w.lower() not in VIETNAMESE_STOPWORDS and len(w) > 1]
            best_sentence = None
            best_overlap = -1
            for s in sentences:
                s_clean = s.strip()
                if not s_clean:
                    continue
                s_tokens = [w.lower() for w in word_tokenize(s_clean)
                            if w.lower() not in VIETNAMESE_STOPWORDS and len(w) > 1]
                overlap = len(set(s_tokens) & set(claim_tokens))
                if overlap > best_overlap or (overlap == best_overlap and len(s_tokens) > len(best_sentence.split()) if best_sentence else True):
                    best_sentence = s_clean
                    best_overlap = overlap
            if best_sentence:
                shortened = best_sentence.strip()
        
        # Pattern 3: Loại bỏ các pattern như "1. Thời gian...", "2. Số liệu..."
        # Nếu bắt đầu bằng số và dấu chấm, loại bỏ
        shortened = re.sub(r'^\d+\.\s*', '', shortened)
        
        # Pattern 4: Loại bỏ các câu mô tả dài dòng ở đầu.
        # Nếu có dấu hai chấm, lấy phần sau dấu hai chấm nếu đủ dài.
        if ':' in shortened:
            parts = shortened.split(':', 1)
            if len(parts) == 2:
                # Nếu phần sau dấu hai chấm dài hơn 20 ký tự, có thể là query
                if len(parts[1].strip()) > 20:
                    shortened = parts[1].strip()
        
        # Loại bỏ dấu ngoặc kép thừa ở đầu/cuối
        shortened = shortened.strip('"\'')
        
        # Loại bỏ các từ/cụm từ không cần thiết ở đầu
        remove_start_patterns = [
            r'^đôi khi,?\s*',
            r'^hãy\s+',
            r'^bắt đầu\s+',
            r'^đầu tiên\s+',
            r'^để\s+',
        ]
        for pattern in remove_start_patterns:
            shortened = re.sub(pattern, '', shortened, flags=re.IGNORECASE)
        
        # Loại bỏ khoảng trắng thừa lại một lần nữa
        shortened = ' '.join(shortened.split()).strip()
        
        # Đảm bảo không quá dài (fallback)
        if len(shortened.split()) > 35 or len(shortened) > 280:
            # Nếu LLM trả về vẫn quá dài, cắt xuống
            words = shortened.split()[:30]
            shortened = ' '.join(words)
        
        # VALIDATION: Kiểm tra query có hợp lệ không
        is_valid, reason = validate_query(shortened, original_claim=claim)
        if not is_valid:
            print(f"Warning: LLM-generated query không hợp lệ: {reason}. Query: '{shortened[:100]}'")
            # Fallback: dùng rule-based simplification
            return simplify_claim(claim, max_words=20)
        
        return shortened if shortened else claim
    except Exception as e:
        print(f"Error shortening claim with LLM: {e}")
        # Fallback: dùng rule-based simplification
        return simplify_claim(claim, max_words=20)


def generate_queries_rule_based(claim: str, use_llm_for_long_claims: bool = True) -> List[str]:
    """
    Sinh nhiều queries từ claim. Query đầu tiên là claim (nguyên vẹn nếu ngắn, hoặc rút ngắn bằng LLM nếu dài).
    
    Args:
        claim: Câu claim cần tạo queries
        use_llm_for_long_claims: Nếu True, dùng LLM để rút ngắn claim khi quá dài
    
    Returns:
        List[str]: Danh sách các queries, query đầu tiên là claim (nguyên vẹn hoặc đã rút ngắn)
    """
    entities, keywords, phrases = extract_entities_and_keywords(claim)
    queries = []
    seen = set()
    
    # Query 0: Claim (nguyên vẹn nếu ngắn, hoặc rút ngắn bằng LLM nếu dài)
    if is_claim_length_acceptable(claim, max_words=30, max_chars=250):
        # Claim ngắn: dùng nguyên vẹn
        clean_claim = ' '.join(claim.split())
        if clean_claim and clean_claim.lower() not in seen:
            # Validate query trước khi thêm
            is_valid, _ = validate_query(clean_claim, original_claim=claim)
            if is_valid:
                queries.append(clean_claim)
                seen.add(clean_claim.lower())
    elif use_llm_for_long_claims:
        # Claim dài: rút ngắn bằng LLM
        shortened_claim = shorten_claim_with_llm(claim)
        if shortened_claim and shortened_claim.lower() not in seen:
            # Validate lại sau khi LLM rút ngắn (đã validate trong shorten_claim_with_llm nhưng kiểm tra lại)
            is_valid, _ = validate_query(shortened_claim, original_claim=claim)
            if is_valid:
                queries.append(shortened_claim)
                seen.add(shortened_claim.lower())
            else:
                # Nếu LLM query không hợp lệ, fallback về rule-based
                simplified = simplify_claim(claim, max_words=20)
                is_valid_fallback, _ = validate_query(simplified, original_claim=claim)
                if is_valid_fallback and simplified.lower() not in seen:
                    queries.append(simplified)
                    seen.add(simplified.lower())
    else:
        # Fallback: dùng rule-based simplification
        simplified = simplify_claim(claim, max_words=20)
        if simplified and simplified.lower() not in seen:
            is_valid, _ = validate_query(simplified, original_claim=claim)
            if is_valid:
                queries.append(simplified)
                seen.add(simplified.lower())
    
    # Query 1: Entity + Keywords (backup)
    if entities and keywords:
        query1 = generate_query_from_entities(entities, keywords)
        if query1 and query1.lower() not in seen:
            is_valid, _ = validate_query(query1, original_claim=claim)
            if is_valid:
                queries.append(query1)
                seen.add(query1.lower())
    
    # Query 2: Chỉ entities (backup)
    if entities:
        entity_query = ' '.join(entities[:3])  # Tăng lên 3 để bao gồm nhiều entities hơn
        if entity_query and entity_query.lower() not in seen:
            is_valid, _ = validate_query(entity_query, original_claim=claim)
            if is_valid:
                queries.append(entity_query)
                seen.add(entity_query.lower())
    
    # Query 3: Claim đã rút gọn bằng rule-based (nếu chưa có)
    if len(claim.split()) > 15:
        simplified = simplify_claim(claim, max_words=15)
        if simplified and simplified.lower() not in seen:
            is_valid, _ = validate_query(simplified, original_claim=claim)
            if is_valid:
                queries.append(simplified)
                seen.add(simplified.lower())
    
    # Query 4: Key phrases (backup, chỉ thêm nếu chưa đủ queries)
    if phrases and len(queries) < 3:
        for phrase in phrases[:2]:
            if phrase.lower() not in seen:
                is_valid, _ = validate_query(phrase, original_claim=claim)
                if is_valid:
                    queries.append(phrase)
                    seen.add(phrase.lower())
                    if len(queries) >= 3:
                        break
    
    # Đảm bảo tối thiểu 1 query
    if not queries:
        # Fallback 1: Claim nguyên vẹn (nếu chưa thêm)
        if claim and claim.lower() not in seen:
            queries.append(claim)
        # Fallback 2: lấy keywords
        elif keywords:
            queries.append(' '.join(keywords[:8]))
        # Fallback 3: lấy 10 từ đầu không có stopwords
        else:
            words = [w for w in word_tokenize(claim) 
                    if w.lower() not in VIETNAMESE_STOPWORDS][:10]
            if words:
                queries.append(' '.join(words))
            else:
                queries.append(claim[:100])  # Last resort
    
    return queries[:3]  # Trả về tối đa 3 queries tốt nhất


def plan(claim: str, use_hybrid: bool = False, use_llm_for_long_claims: bool = True) -> str:
    """
    Tạo queries từ claim. Query đầu tiên là claim (nguyên vẹn nếu ngắn, hoặc rút ngắn bằng LLM nếu dài).
    
    Args:
        claim: Câu claim cần fact-check
        think: (Deprecated) Giữ để tương thích với code cũ
        use_hybrid: Nếu True, sử dụng hybrid approach (LLM fallback - deprecated, giữ để tương thích)
        use_llm_for_long_claims: Nếu True, dùng LLM để rút ngắn claim khi quá dài (mặc định True)
    
    Returns:
        str: Các queries, mỗi query trên một dòng. Query đầu tiên là claim (nguyên vẹn hoặc đã rút ngắn).
    """
    # Sinh queries: query đầu tiên là claim (nguyên vẹn hoặc rút ngắn bằng LLM)
    queries = generate_queries_rule_based(claim, use_llm_for_long_claims=use_llm_for_long_claims)
    
    # Hybrid approach (deprecated - giữ để tương thích với code cũ)
    if use_hybrid:
        # Kiểm tra điều kiện cần LLM fallback
        needs_llm_fallback = (
            len(claim.split()) > 100 or  # Claim quá dài
            len(queries) == 0 or  # Không tạo được query
            any(len(q.split()) < 3 for q in queries)  # Query quá ngắn
        )
        
        if needs_llm_fallback:
            # Fallback to LLM (giữ code cũ)
            from .llm import prompt_ollama
            plan_prompt = """HƯỚNG DẪN
            Kiến thức hiện tại vẫn chưa đủ để đánh giá tính xác thực của YÊU CẦU.
            Hãy in ra CÂU TÌM KIẾM để thu thập bằng chứng mới, theo đúng quy tắc sau:

            ### QUY TẮC:
            - CÂU TÌM KIẾM là một cụm từ hoặc câu ngắn gọn dùng để tìm kiếm thông tin trên web.
            - CÂU TÌM KIẾM liên quan trực tiếp đến YÊU CẦU và chứa ít nhất một từ khóa hoặc thực thể trong YÊU CẦU.
            - QUAN TRỌNG: Phải bao gồm tên riêng, địa danh, và thời gian nếu có trong YÊU CẦU.

            ### YÊU CẦU: 
            {claim}

            ### ĐỊNH DẠNG ĐẦU RA BẮT BUỘC:
            Chỉ in CÂU TÌM KIẾM, KHÔNG in thêm mô tả hay giải thích.

            In ra câu tìm kiếm của bạn:
            """
            prompt = plan_prompt.format(claim=claim)
            llm_response = prompt_ollama(prompt, think=False)
            return llm_response
    
    # Trả về queries
    return '\n'.join(queries)