#=========================PHÂN CÁCH CÁC TỪ=========================
import re
import unicodedata

class VietnameseSegmenter:
    def __init__(self, dict_file=None):
        self.dictionary = set()
        self.special_tokens = []

        if dict_file:
            self.load_dictionary(dict_file)
        else:
            self.load_default_dict()

    def load_default_dict(self):
        # Từ điển mặc định (phòng hờ)
        self.dictionary = {'ngày', 'tháng', 'năm'}

    def normalize_text(self, text):
        return unicodedata.normalize('NFC', text).lower()

    def load_dictionary(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    word = self.normalize_text(line.strip())
                    if word:
                        self.dictionary.add(word)
            print(f"--> Đã load {len(self.dictionary)} từ vào bộ nhớ.")
        except Exception as e:
            print(f"Lỗi load từ điển: {e}")

    def preprocess(self, text):
        """
        Regex bắt các token đặc biệt để không bị tách rời bởi Max Match:
        1. TP.HCM / TP.XXX
        2. Ngày tháng
        3. Số tự nhiên (Sửa lỗi 2 0 2 3)
        """
        self.special_tokens = []
        result = []
        i = 0
        text = unicodedata.normalize('NFC', text)

        while i < len(text):
            matched = False

            # 1. Bắt TP.HCM hoặc TP viết tắt (ví dụ trong TP Hải Phòng)
            # Regex này bắt "TP." hoặc "TP" đứng trước chữ hoa (nếu muốn chặt chẽ hơn)
            # Ở đây mình ưu tiên bắt chuỗi TP.HCM cụ thể hoặc từ 'TP' dính liền
            city_match = re.match(r'(TP\.?HCM|TP(?=[A-Z]))', text[i:], re.IGNORECASE)

            # 2. Bắt Ngày tháng (dd/mm/yyyy)
            if not matched:
                date_match = re.match(r'\d{1,2}[-/]\d{1,2}[-/]\d{4}', text[i:])
                if date_match:
                    token = date_match.group()
                    idx = len(self.special_tokens)
                    self.special_tokens.append(token)
                    result.append(f'〖TOKEN{idx}〗')
                    i += len(token)
                    matched = True

            # 3. Bắt SỐ (Sửa lỗi quan trọng: 2023 -> 2023 thay vì 2 0 2 3)
            if not matched:
                number_match = re.match(r'\d+', text[i:])
                if number_match:
                    token = number_match.group()
                    idx = len(self.special_tokens)
                    self.special_tokens.append(token)
                    result.append(f'〖TOKEN{idx}〗')
                    i += len(token)
                    matched = True

            if not matched and city_match:
                 token = city_match.group()
                 idx = len(self.special_tokens)
                 self.special_tokens.append(token)
                 result.append(f'〖TOKEN{idx}〗')
                 i += len(token)
                 matched = True

            if not matched:
                result.append(text[i])
                i += 1
        return ''.join(result)

    def max_match(self, text, max_word_len=6):
        result = []
        i = 0
        n = len(text)

        while i < n:
            # Bỏ qua token đặc biệt đã encode
            if text[i] == '〖':
                end = text.find('〗', i)
                if end != -1:
                    result.append(text[i:end+1])
                    i = end + 1
                    continue

            # Xử lý dấu câu: , . ! ? -> tách riêng ra luôn
            if text[i] in ",.!?;:":
                result.append(text[i])
                i += 1
                continue

            if text[i].isspace():
                i += 1
                continue

            matched = False
            # Thuật toán Max Match: tìm từ dài nhất trong từ điển
            for length in range(min(max_word_len, n - i), 0, -1):
                word = text[i:i+length]
                if '〖' in word or word in ",.!?;:": continue

                if self.normalize_text(word) in self.dictionary:
                    result.append(word)
                    i += length
                    matched = True
                    break

            if not matched:
                result.append(text[i])
                i += 1
        return result

    def postprocess(self, tokens):
        result = []
        for token in tokens:
            match = re.match(r'〖TOKEN(\d+)〗', token)
            if match:
                idx = int(match.group(1))
                result.append(self.special_tokens[idx])
            else:
                result.append(token)
        return result

    def segment(self, text):
        processed = self.preprocess(text)
        tokens = self.max_match(processed)
        return ' '.join(self.postprocess(tokens))

#=========================KHÔI PHỤC DẤU TIẾNG VIỆT=========================
import torch
import numpy as np
import re
import os
import urllib.request
from transformers import AutoTokenizer, AutoModelForTokenClassification

TAG_URL = "https://huggingface.co/peterhung/vietnamese-accent-marker-xlm-roberta/resolve/main/selected_tags_names.txt"
TAG_PATH = "selected_tags_names.txt"

if not os.path.exists(TAG_PATH):
    urllib.request.urlretrieve(TAG_URL, TAG_PATH)

# ------------------- Load model & tokenizer -------------------
tokenizer = AutoTokenizer.from_pretrained("peterhung/vietnamese-accent-marker-xlm-roberta", add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained(
    "peterhung/vietnamese-accent-marker-xlm-roberta",
    device_map=None  # Prevent meta device usage
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
# ------------------- Load danh sách nhãn -------------------
with open(TAG_PATH, 'r', encoding='utf-8') as f:
    label_list = [line.strip() for line in f if line.strip()]
# ------------------- Hàm chính: phục hồi dấu tiếng Việt -------------------
def restore_vietnamese_accents(text: str) -> str:
    """
    Input : "Su kien hoat dong Nam Du lich quoc gia 2023"
    Output: "Sự kiện hoạt động Năm Du lịch quốc gia 2023"
    """
    # Bước 1: Tách thành list từ
    words = text.strip().split()
    if not words:
        return text

    # Bước 2: Tokenize + dự đoán
    inputs = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)[0].cpu().numpy()

    # Lấy token và prediction (bỏ <s> và </s>)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])[1:-1]
    predictions = predictions[1:-1]

    # Bước 3: Ghép lại subword + gom prediction
    merged = []
    i = 0
    while i < len(tokens):
        if tokens[i].startswith("▁"):  # bắt đầu từ mới
            word_parts = [tokens[i][1:]]   # bỏ prefix ▁
            pred_set = {predictions[i]}

            j = i + 1
            while j < len(tokens) and not tokens[j].startswith("▁"):
                word_parts.append(tokens[j])
                pred_set.add(predictions[j])
                j += 1

            raw_word = ''.join(word_parts)
            merged.append((raw_word, pred_set))
            i = j
        else:
            i += 1

    # Bước 4: Áp dụng nhãn để thêm dấu
    result = []
    for raw_word, pred_set in merged:
        restored = raw_word
        # Dùng nhãn đầu tiên có thể thay đổi từ
        for pred_idx in pred_set:
            tag = label_list[pred_idx]
            if "-" in tag:
                no_accent, with_accent = tag.split("-", 1)
                if no_accent in raw_word:
                    restored = raw_word.replace(no_accent, with_accent, 1)
                    break  # chỉ thay 1 lần là đủ
        result.append(restored)

    return " ".join(result)

# ====================== XÓA EMOJI & KÝ TỰ LẠ ======================
def remove_noise(text: str) -> str:
    # Xóa emoji
    emoji_pattern = re.compile("[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002702-\U000027B0\U000024C2-\U0001F251]+")
    text = emoji_pattern.sub(' ', text)
    # Xóa ký tự lạ
    text = re.sub(r'[^a-zA-ZÀ-Ỵà-ỵđĐ0-9\s\.,!?;:()%/-]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

  # ====================== HÀM CHÍNH ======================
def preprocess(text: str, use_accent_restorer: bool = True) -> str:
    """
    Pipeline hoàn chỉnh cho ViFact-Checking
    Input: text bẩn từ FB, TikTok, OCR...
    Output: text sạch, có dấu, chuẩn hóa
    """
    if not text or not text.strip():
        return ""

    # Bước 1: Luôn luôn xóa noise + chuẩn hóa
    text = remove_noise(text)
    text = unicodedata.normalize('NFC', text)

    # Bước 2: Tự động phát hiện text bẩn kiểu OCR (dính chữ in hoa)
    is_ocr_dirty = bool(re.search(r'[a-z][A-Z]|[A-Z]{2,}[a-z]', text)) or len(text.split()) < len(text) // 15

    # Bước 3: Chọn 1 trong 2 chiến lược
    if is_ocr_dirty:
        # Cách 1: Dính chữ → tách trước, rồi phục hồi dấu
        segmenter = VietnameseSegmenter('/content/ClaimCheck/factchecker/preprocessing/vn_dict.txt')
        text = segmenter.segment(text)
    else:
        # Cách 2: Đã sạch → không cần tách
        pass  # giữ nguyên

    # Bước 4: LUÔN LUÔN phục hồi dấu (cái này là vũ khí bí mật)
    try:
        text = restore_vietnamese_accents(text.lower())
    except:
        pass  # nếu lỗi thì thôi, nhưng hiếm khi lỗi

    # Bước 5: Dọn dẹp cuối
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    return text.strip().lower()