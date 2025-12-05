#!/usr/bin/env python3
"""
Script để test NER pipeline (extract_entities_and_keywords) từ planning.py.
Test với các claim mẫu để kiểm tra entities, keywords, phrases.
"""

from factchecker.modules.planning import extract_entities_and_keywords

# Danh sách claim để test NER
test_claims = [
    "Thủ tướng chính phủ Việt Nam hiện tại là ông Phạm Minh Chính",
    "Công ty Google có trụ sở tại Mountain View, California",
    "Madonna is dating boxer Josh Popper",
    "Cuộc bầu cử Tổng thống Mỹ 2024 sẽ diễn ra vào tháng 11 năm 2024",
    "Theo báo cáo của Tổ chức Y tế Thế giới, Việt Nam đã tiêm vaccine COVID-19 cho hơn 80 triệu người tính đến tháng 10 năm 2023",
    "Trong cuộc họp báo hôm nay, Bộ trưởng Bộ Y tế Nguyễn Thanh Long đã thông báo rằng Việt Nam sẽ nhận thêm 20 triệu liều vaccine AstraZeneca từ Quỹ COVAX vào quý 3 năm 2021"
]

def test_ner(claim: str):
    """Test NER cho một claim."""
    print(f"\nClaim: {claim}")
    entities, keywords, phrases = extract_entities_and_keywords(claim)
    print(f"Entities: {entities}")
    print(f"Keywords: {keywords}")
    print(f"Phrases: {phrases}")

if __name__ == "__main__":
    print("Testing NER Pipeline (Davlan/xlm-roberta-base-wikiann-ner)")
    print("=" * 60)
    
    for i, claim in enumerate(test_claims, 1):
        print(f"\n=== Test {i} ===")
        test_ner(claim)
    
    print("\n" + "=" * 60)
    print("NER Test Complete.")