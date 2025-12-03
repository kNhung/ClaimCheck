# Tóm tắt các cải tiến đã thực hiện

## 📅 Ngày: 02-12-2025

Dựa trên phân tích report `021225-2253`, đã thực hiện các cải tiến sau:

---

## 1. ✅ Cải thiện Query Generation Validation (`planning.py`)

### Vấn đề:
- Query generation vẫn có prompt leak: "Yêu cầu này có thể giúp tôi cải thiện chi tiết hơn."
- Queries quá ngắn: "Sự kiện ,", "tháng 12-2022 ,"
- Validation chưa đủ mạnh

### Giải pháp đã triển khai:

1. **Mở rộng prompt leak patterns**:
   - Thêm nhiều patterns như: "yêu cầu này có thể", "có thể giúp tôi", "giúp tôi cải thiện", "cải thiện chi tiết", "chi tiết hơn", v.v.
   - Tổng cộng thêm ~15 patterns mới

2. **Kiểm tra instruction starters**:
   - Reject queries bắt đầu bằng: "bạn được", "hãy", "bạn cần", "bạn phải", "hãy thử", "bạn có thể", "để"
   - Đảm bảo query không phải là hướng dẫn

3. **Tăng độ dài tối thiểu**:
   - Từ 3 từ → **5 từ** (yêu cầu nghiêm ngặt hơn)

4. **Tăng yêu cầu từ khóa chung**:
   - Từ ít nhất 1 từ khóa → **ít nhất 2 từ khóa** chung với claim gốc

### Kết quả mong đợi:
- Loại bỏ được prompt leak như "Yêu cầu này có thể giúp tôi cải thiện chi tiết hơn."
- Reject queries quá ngắn hoặc không hợp lệ
- Đảm bảo queries có đủ thông tin từ claim gốc

---

## 2. ✅ Cải thiện LLM Judge JSON Parsing (`evaluation.py`)

### Vấn đề:
- LLM judge không parse được JSON trong nhiều cases
- Fallback về text parsing không đáng tin cậy
- Prompt quá dài, có thể gây confusion cho LLM nhỏ

### Giải pháp đã triển khai:

1. **Rút ngắn và tối ưu prompt**:
   - Giảm độ dài prompt từ ~30 dòng → ~15 dòng
   - Loại bỏ các hướng dẫn dài dòng, tập trung vào yêu cầu chính
   - Format rõ ràng hơn, dễ đọc hơn

2. **Nhiều strategies cho JSON parsing**:
   - **Strategy 1**: Tìm JSON trong markdown code block (```json ... ```)
   - **Strategy 2**: Tìm JSON object với pattern matching regex (tìm cặp {} chứa "verdict")
   - **Strategy 3**: Original method (từ '{' đầu tiên đến '}' cuối cùng)
   - **Strategy 4**: Extract verdict từ text nếu không parse được JSON

3. **Cải thiện error handling**:
   - Better error messages
   - Fallback an toàn hơn

### Kết quả mong đợi:
- Tăng tỷ lệ parse thành công JSON từ LLM output
- Giảm số lượng "Không parse được JSON" errors
- Prompt ngắn gọn hơn → LLM nhỏ dễ tuân thủ format hơn

---

## 3. ✅ Điều chỉnh Relevance Threshold và Filtering Logic (`evaluation.py`)

### Vấn đề:
- Relevance threshold 0.3 quá strict, loại bỏ quá nhiều evidence
- Report #17: evidence rất liên quan nhưng vẫn bị filter hoặc verdict sai
- Không có logic đặc biệt cho top evidence

### Giải pháp đã triển khai:

1. **Giảm relevance threshold**:
   - Từ 0.3 → **0.2** (ít strict hơn)

2. **Logic đặc biệt cho top evidence**:
   - Luôn giữ lại ít nhất top 1 evidence nếu score > 0.3
   - Nếu top score > 0.5 nhưng dưới threshold, tự động điều chỉnh threshold xuống
   - Đảm bảo không mất evidence quan trọng nhất

3. **Cải thiện normalization**:
   - Better normalization strategy giữ nguyên ranking
   - Xử lý edge cases (tất cả scores bằng nhau)

### Kết quả mong đợi:
- Giữ lại nhiều evidence liên quan hơn
- Top evidence quan trọng không bị loại bỏ
- Tăng số lượng evidence được judge

---

## 4. ✅ Cải thiện Class Balance trong LLM Judge Prompt (`evaluation.py`)

### Vấn đề:
- Hệ thống bias mạnh về "Not Enough Evidence" (72.97% recall)
- Class 1 (Refuted) chỉ detect được 2.44% recall
- Class 0 (Supported) chỉ detect được 14.29% recall
- LLM judge quá conservative, dễ chọn "Not Enough Evidence"

### Giải pháp đã triển khai:

1. **Cải thiện prompt để giảm bias**:
   - Thêm warning: "Phân tích kỹ lưỡng trước khi chọn 'Not Enough Evidence'"
   - Nhấn mạnh: "Nếu bằng chứng rõ ràng support/refute, hãy chọn nhãn đó"
   - Làm rõ điều kiện cho từng nhãn:
     - Supported: trùng khớp >80%
     - Refuted: mâu thuẫn rõ ràng
     - Not Enough Evidence: chỉ khi thực sự không đủ

2. **Rõ ràng hơn về các nhãn**:
   - Supported: "Xác nhận, trùng khớp"
   - Refuted: "Mâu thuẫn, sai"
   - Not Enough Evidence: "Không liên quan, thiếu, hoặc mâu thuẫn nhẹ"

### Kết quả mong đợi:
- Giảm bias về "Not Enough Evidence"
- Tăng recall cho Supported và Refuted classes
- LLM judge phân tích kỹ hơn trước khi chọn "Not Enough Evidence"

---

## 📊 Tóm tắt thay đổi

### Files đã sửa:
1. `/home/v1nk4n/Working/ClaimCheck/factchecker/modules/planning.py`
   - Cải thiện `validate_query()` function
   - Thêm nhiều prompt leak patterns
   - Tăng yêu cầu độ dài và từ khóa

2. `/home/v1nk4n/Working/ClaimCheck/factchecker/modules/evaluation.py`
   - Cải thiện `_llm_judge_with_evidence()` prompt
   - Multiple JSON parsing strategies
   - Cải thiện `filter_evidence_by_relevance()` logic
   - Giảm relevance threshold và thêm logic đặc biệt

### Metrics mong đợi:
- **Accuracy**: Tăng từ 28.33% → ~35-40%
- **F1-score**: Tăng từ 20.79% → ~28-35%
- **Class balance**: 
  - Refuted recall: Tăng từ 2.44% → ~10-15%
  - Supported recall: Tăng từ 14.29% → ~25-30%
  - Not Enough Evidence recall: Giảm từ 72.97% → ~50-60%

### Next steps:
1. Chạy test lại trên dataset để verify improvements
2. Monitor JSON parsing success rate
3. Track query generation quality (số queries bị reject)
4. Analyze confusion matrix để đánh giá class balance

---

## 🔧 Technical Details

### Query Validation Changes:
- Minimum words: 3 → **5**
- Minimum keyword overlap: 1 → **2**
- New patterns: +15 prompt leak patterns
- New check: Instruction starters detection

### JSON Parsing Improvements:
- Strategies: 1 → **4**
- Error handling: Improved
- Prompt length: ~30 lines → **~15 lines**

### Relevance Filtering:
- Threshold: 0.3 → **0.2**
- Top evidence protection: ✅ Added
- Dynamic threshold adjustment: ✅ Added

### Prompt Improvements:
- Length: Reduced by ~50%
- Clarity: Improved
- Bias reduction: Explicit warnings added


