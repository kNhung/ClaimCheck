# factchecker/modules/planning.py

import re
from underthesea import word_tokenize, pos_tag, ner
from typing import List, Tuple

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
    # 1. NER để tìm thực thể
    entities_result = ner(claim)
    entities = []
    for entity in entities_result:
        if len(entity) == 4:  # Format: (word, pos, chunk, ner_label)
            word, _, _, label = entity
            if label in ['PER', 'ORG', 'LOC', 'MISC']:
                entities.append(word)
    
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


def generate_queries_rule_based(claim: str) -> List[str]:
    """
    Sinh nhiều queries từ claim bằng rule-based (KHÔNG dùng LLM).
    
    Returns:
        List[str]: Danh sách các queries
    """
    entities, keywords, phrases = extract_entities_and_keywords(claim)
    queries = []
    seen = set()
    
    # Query 1: Entity + Keywords (ưu tiên nhất)
    if entities and keywords:
        query1 = generate_query_from_entities(entities, keywords)
        if query1 and len(query1.split()) >= 2:
            queries.append(query1)
            seen.add(query1.lower())
    
    # Query 2: Chỉ entities (nếu có)
    if entities:
        entity_query = ' '.join(entities[:2])
        if entity_query and entity_query.lower() not in seen:
            queries.append(entity_query)
            seen.add(entity_query.lower())
    
    # Query 3: Claim đã rút gọn (nếu claim dài)
    if len(claim.split()) > 10:
        simplified = simplify_claim(claim, max_words=10)
        if simplified and simplified.lower() not in seen and len(simplified.split()) >= 3:
            queries.append(simplified)
            seen.add(simplified.lower())
    
    # Query 4: Key phrases (bigrams/trigrams)
    if phrases:
        for phrase in phrases[:2]:  # Lấy 2 phrases tốt nhất
            if phrase.lower() not in seen and len(phrase.split()) >= 2:
                queries.append(phrase)
                seen.add(phrase.lower())
                if len(queries) >= 4:  # Tối đa 4 queries
                    break
    
    # Query 5: Fallback - claim gốc nếu ngắn
    if not queries and len(claim.split()) <= 12:
        queries.append(claim)
    
    # Đảm bảo tối thiểu 1 query
    if not queries:
        # Fallback: lấy 8 từ đầu không có stopwords
        words = [w for w in word_tokenize(claim) 
                if w.lower() not in VIETNAMESE_STOPWORDS][:8]
        if words:
            queries.append(' '.join(words))
        else:
            queries.append(claim[:50])  # Last resort
    
    return queries[:3]  # Trả về tối đa 3 queries tốt nhất


def plan(claim: str, think: bool = True, use_hybrid: bool = False) -> str:
    """
    Tạo queries từ claim.
    
    Args:
        claim: Câu claim cần fact-check
        think: (Deprecated) Giữ để tương thích với code cũ
        use_hybrid: Nếu True, sử dụng hybrid approach (LLM fallback)
    
    Returns:
        str: Các queries, mỗi query trên một dòng
    """
    # Phương án 1: Rule-based (nhanh, không dùng LLM)
    queries = generate_queries_rule_based(claim)
    
    # Phương án 2: Hybrid (tùy chọn - chỉ dùng LLM khi cần)
    if use_hybrid:
        # Kiểm tra điều kiện cần LLM
        needs_llm = (
            len(claim.split()) > 100 or  # Claim quá dài
            len(queries) == 0 or  # Không tạo được query
            any(len(q.split()) < 3 for q in queries)  # Query quá ngắn
        )
        
        if needs_llm:
            # Fallback to LLM (giữ code cũ)
            from .llm import prompt_ollama
            plan_prompt = """HƯỚNG DẪN
Kiến thức hiện tại vẫn chưa đủ để đánh giá tính xác thực của YÊU CẦU.
Hãy in ra CÂU TÌM KIẾM để thu thập bằng chứng mới, theo đúng quy tắc sau:

### QUY TẮC:
- CÂU TÌM KIẾM là một cụm từ hoặc câu ngắn gọn dùng để tìm kiếm thông tin trên web.
- CÂU TÌM KIẾM liên quan trực tiếp đến YÊU CẦU và chứa ít nhất một từ khóa hoặc thực thể trong YÊU CẦU.

### YÊU CẦU: 
{claim}

### ĐỊNH DẠNG ĐẦU RA BẮT BUỘC:
Chỉ in CÂU TÌM KIẾM, KHÔNG in thêm mô tả hay giải thích.

In ra câu tìm kiếm của bạn:
"""
            prompt = plan_prompt.format(claim=claim)
            llm_response = prompt_ollama(prompt, think=False)
            return llm_response
    
    # Trả về queries rule-based
    return '\n'.join(queries)