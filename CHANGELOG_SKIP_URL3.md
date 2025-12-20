# 📝 CHANGELOG: Skip URL #3 Optimization

## Tổng quan
Implement tính năng **Skip URL thứ 3 nếu 2 URLs đầu đã có bằng chứng tốt (score > 0.8)** để tối ưu tốc độ xử lý.

---

## 🔧 Các thay đổi chi tiết

### 1. File: `factchecker/modules/retriver_rav.py`

#### Thay đổi chính:
- **Thêm parameter `return_score=False`** vào hàm `get_top_evidence()`
- **Trả về tuple `(summary, score)`** khi `return_score=True`
- **Tính relevance score** dựa trên bi-encoder (cosine similarity) normalize về [0, 1]

#### Code changes:

```python
# TRƯỚC:
def get_top_evidence(claim, text, top_k_chunks=None, p=6, q=2, log_callback=None):
    # ...
    return summary  # Chỉ trả về string

# SAU:
def get_top_evidence(claim, text, top_k_chunks=None, p=6, q=2, log_callback=None, return_score=False):
    # ...
    if return_score:
        # Tính relevance score từ bi-encoder
        max_bi_score = max(top_p_scores) if top_p_scores else 0.0
        max_relevance_score = (max_bi_score + 1.0) / 2.0  # Normalize từ [-1, 1] về [0, 1]
        return summary, max_relevance_score
    return summary
```

#### Lý do:
- Cần score để đánh giá chất lượng bằng chứng
- Score dựa trên bi-encoder (nhanh, đã có sẵn)
- Normalize về [0, 1] để dễ so sánh với threshold

---

### 2. File: `factchecker/factchecker.py`

#### Thay đổi chính:

**A. Chuyển từ xử lý song song → tuần tự:**
- Trước: URLs được xử lý song song với `ThreadPoolExecutor.map()`
- Sau: URLs được xử lý tuần tự trong vòng lặp `for`

**B. Thêm logic skip URL #3:**
- Track score cho mỗi URL theo index
- Khi đến URL thứ 3 (index 2), kiểm tra scores của URL 1 và 2
- Skip URL 3 nếu cả 2 URLs đầu có score > 0.8

**C. Cập nhật function `process_result()`:**
- Thêm parameter `url_index`
- Gọi RAV với `return_score=True`
- Trả về tuple `(summary, score)` thay vì chỉ `summary`

#### Code changes:

```python
# TRƯỚC:
def process_result(result):
    # ...
    summary = retriver_rav.get_top_evidence(...)
    # Xử lý song song
    list(self._result_executor.map(process_result, urls))

# SAU:
def process_result(result, url_index):
    # ...
    summary, relevance_score = retriver_rav.get_top_evidence(
        ...,
        return_score=True
    )
    return summary, relevance_score

# Xử lý tuần tự với logic skip
url_scores = {}
for url_index, url in enumerate(urls):
    if url_index == 2:  # URL thứ 3
        url1_score = url_scores.get(0, 0.0)
        url2_score = url_scores.get(1, 0.0)
        
        if url1_score > 0.8 and url2_score > 0.8:
            # Skip URL 3
            continue
    
    summary, score = process_result(url, url_index)
    url_scores[url_index] = score
```

#### Chi tiết logic skip:

```python
EVIDENCE_SCORE_THRESHOLD = 0.8  # Threshold để quyết định skip

# Khi đến URL thứ 3 (index 2):
if url_index == 2:
    url1_score = url_scores.get(0, 0.0)  # Score của URL 1
    url2_score = url_scores.get(1, 0.0)  # Score của URL 2
    
    # Chỉ skip nếu cả 2 URLs đầu đều có score > threshold
    if url1_score > EVIDENCE_SCORE_THRESHOLD and url2_score > EVIDENCE_SCORE_THRESHOLD:
        # Log skip reason
        # Skip URL 3
        continue
```

---

## 📊 Cách hoạt động

### Flow xử lý:

1. **URL 1** (index 0):
   - Web scraping
   - RAV → lấy summary + score
   - Lưu score vào `url_scores[0]`

2. **URL 2** (index 1):
   - Web scraping
   - RAV → lấy summary + score
   - Lưu score vào `url_scores[1]`

3. **URL 3** (index 2):
   - **Kiểm tra:**
     - Nếu `url_scores[0] > 0.8` AND `url_scores[1] > 0.8`:
       - ✅ **SKIP** URL 3 (đã có đủ bằng chứng tốt)
       - Log reason vào report
     - Nếu không:
       - ⏭️ **Xử lý** URL 3 như bình thường

---

## ✅ Lợi ích

1. **Tăng tốc độ:**
   - Giảm thời gian xử lý khi đã có đủ bằng chứng tốt
   - Tiết kiệm thời gian scrape + RAV cho URL thứ 3

2. **Giảm thiểu rủi ro:**
   - Chỉ skip khi có bằng chứng tốt từ 2 URLs đầu
   - Vẫn xử lý URL 3 nếu một trong 2 URLs đầu không đủ tốt

3. **Logging rõ ràng:**
   - Ghi lại lý do skip trong report
   - Hiển thị scores của 2 URLs đầu

---

## ⚙️ Thông số

- **Threshold:** `0.8` (có thể điều chỉnh)
- **Score range:** [0, 1] (normalized từ bi-encoder cosine similarity)
- **Score tính từ:** Bi-encoder (cosine similarity) normalize về [0, 1]

---

## 🔍 Ví dụ

### Scenario 1: Skip URL 3
```
URL 1: score = 0.92 ✅ (tốt)
URL 2: score = 0.85 ✅ (tốt)
URL 3: ⏭️ SKIPPED (cả 2 URLs đầu đã có score > 0.8)
```

### Scenario 2: Vẫn xử lý URL 3
```
URL 1: score = 0.92 ✅ (tốt)
URL 2: score = 0.65 ❌ (chưa đủ tốt)
URL 3: ⏭️ Vẫn xử lý (vì URL 2 chưa đủ tốt)
```

### Scenario 3: Vẫn xử lý URL 3 (URL 1 bị skip)
```
URL 1: skipped (unsupported domain) → score = 0.0
URL 2: score = 0.85 ✅ (tốt)
URL 3: ⏭️ Vẫn xử lý (vì chỉ có 1 URL tốt)
```

---

## 📝 Notes

- **Backward compatible:** Nếu không truyền `return_score=True`, RAV vẫn trả về string như cũ
- **Sequential processing:** URLs giờ được xử lý tuần tự thay vì song song để có thể check scores
- **Flexible threshold:** Có thể điều chỉnh `EVIDENCE_SCORE_THRESHOLD` nếu cần


