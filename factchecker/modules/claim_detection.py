from underthesea import pos_tag, ner, sent_tokenize
import re

def claim_score(text):
    if not text: return -10
    
    # --- 1. Lọc rác sơ bộ (Giữ nguyên dấu câu để NLP chạy đúng) ---
    text_clean = text.strip()
    words = text_clean.split()
    
    # Loại ngay nếu quá ngắn (dưới 4 từ) hoặc là câu hỏi
    if len(words) < 4: return -10
    if "?" in text_clean: return -10
    
    # Loại các từ cảm thán nhảm nhí (nếu câu ngắn)
    chat_words = ["haha", "hihi", "hic", "wow", "alo", "chúc mừng", "cảm ơn"]
    text_lower = text_clean.lower()
    if len(words) < 7 and any(w in text_lower for w in chat_words):
        return -10

    # --- 2. Phân tích NLP ---
    try:
        # pos_tag: tìm loại từ (Danh từ, Động từ...)
        pos = pos_tag(text_clean) 
        # ner: tìm thực thể (Tên người, Tổ chức...)
        entities = ner(text_clean)
    except:
        return 0

    score = 0

    # --- 3. Tính điểm (Cộng điểm cho dấu hiệu của Claim) ---

    # entities thường trả về dạng: [('Hà Nội', 'LOC'), ...] hoặc [('Ông A', 'Np', 'B-PER'), ...]
    # Ta chỉ cần kiểm tra xem trong tuple đó có chứa từ khóa PER/ORG/LOC không.
    has_entity = False
    for item in entities:
        # Kiểm tra từng thành phần trong tuple của entity
        # Cách này bắt được cả 'PER', 'B-PER', 'I-PER' mà không cần quan tâm index
        if any(tag in str(item) for tag in ['PER', 'ORG', 'LOC']):
            has_entity = True
            break
            
    if has_entity:
        score += 3  # Điểm thưởng lớn

    # Kiểm tra Số liệu (Number) hoặc Thời gian (Date/Time)
    # Trong pos_tag: 'M' là số từ. Trong ner: 'B-DATE', 'I-TIME'...
    has_number_pos = any(tag == 'M' for w, tag in pos)
    has_time_ner = False
    for item in entities:
        if any(t in str(item) for t in ['DATE', 'TIME', 'NUMBER', 'PERCENT']):
            has_time_ner = True
            break
            
    if has_number_pos or has_time_ner:
        score += 3 # Điểm thưởng lớn

    # Kiểm tra từ ngữ báo cáo (Reporting Verbs)
    report_verbs = ["cho biết", "tuyên bố", "báo cáo", "khẳng định", "thống kê", "dự báo"]
    if any(v in text_lower for v in report_verbs):
        score += 2

    # Kiểm tra cấu trúc ngữ pháp cơ bản: Phải có Danh từ (N) và Động từ (V)
    has_noun = any(tag.startswith("N") for w, tag in pos)
    has_verb = any(tag == "V" for w, tag in pos)
    
    if has_noun and has_verb:
        score += 1
        # Nếu câu dài (>15 từ) + đủ cấu trúc -> Khả năng cao là câu trần thuật chứa thông tin
        if len(words) > 15:
            score += 2

    # --- 4. Trừ điểm (Chỉ lọc những câu thuần túy cảm xúc) ---
    
    # Chỉ trừ điểm nếu câu KHÔNG CÓ thực thể và KHÔNG CÓ số liệu
    # (Tức là: Nếu có số liệu/tên riêng thì chấp nhận cả tính từ cảm xúc)
    if not has_entity and not (has_number_pos or has_time_ner):
        
        # Ý kiến cá nhân thuần túy
        opinions = ["tôi nghĩ", "mình nghĩ", "theo tôi", "tôi tin", "mong rằng"]
        if any(text_lower.startswith(op) for op in opinions):
            score -= 5

        # Tính từ cảm xúc mạnh (khi không có số liệu chứng minh)
        emotions = ["tuyệt vời", "tồi tệ", "đáng ghét", "hạnh phúc", "vui quá", "buồn quá"]
        if any(adj in text_lower for adj in emotions):
            score -= 3

    return score

def is_claim(text, threshold=1):
    # Threshold = 1: Chỉ cần câu có cấu trúc Noun-Verb là xem xét (High Recall)
    return claim_score(text) >= threshold

def claim_filter(text, threshold=1):
    if not text: return ""
    
    # --- BƯỚC 1: Tiền xử lý văn bản (Safe Preprocessing) ---
    
    # 1. Xử lý dấu ba chấm (...) dính liền với chữ cái viết hoa
    # Ví dụ: "Hết...Rồi" -> "Hết... Rồi"
    # Logic: Tìm ... theo sau là chữ hoa -> Thêm dấu cách
    text = re.sub(r'\.\.\.(?=[A-Z])', '... ', text)
    
    # 2. Xử lý dấu câu (.!?) dính liền, TRÁNH tên viết tắt (U.S.A)
    # Logic: Chỉ thêm dấu cách nếu:
    #   - Trước dấu chấm là chữ thường (?<=[a-z]) -> Tức là hết một từ bình thường
    #   - Sau dấu chấm là chữ Hoa (?=[A-Z]) -> Tức là bắt đầu câu mới
    # Ví dụ: "xong.Tiếp" -> "xong. Tiếp" (Đúng)
    # Ví dụ: "U.S.A" -> Giữ nguyên (Vì trước dấu chấm là chữ Hoa 'U')
    text = re.sub(r'(?<=[a-z])([.!?])(?=[A-Z])', r'\1 ', text)
    
    # --- BƯỚC 2: Tách câu bằng thư viện ---
    sentences = [s for s in sent_tokenize(text) if s.strip()]
    
    # --- BƯỚC 3: Lọc Claim ---
    claims = []
    for s in sentences:
        if is_claim(s, threshold=threshold):
            claims.append(s.strip())
            
    return ' '.join(claims)