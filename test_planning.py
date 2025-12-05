from factchecker.modules import planning

# Danh sách các claim để test
test_claims = [
    "Thủ tướng chính phủ Việt Nam hiện tại là ông Phạm Minh Chính",
    "Công ty Google có trụ sở tại Mountain View, California",
    "Madonna is dating boxer Josh Popper",
    "Cuộc bầu cử Tổng thống Mỹ 2024 sẽ diễn ra vào tháng 11 năm 2024",
    "Theo báo cáo của Tổ chức Y tế Thế giới, Việt Nam đã tiêm vaccine COVID-19 cho hơn 80 triệu người tính đến tháng 10 năm 2023",
    "Trong cuộc họp báo hôm nay, Bộ trưởng Bộ Y tế Nguyễn Thanh Long đã thông báo rằng Việt Nam sẽ nhận thêm 20 triệu liều vaccine AstraZeneca từ Quỹ COVAX vào quý 3 năm 2021"
]

# Test từng claim
for i, claim in enumerate(test_claims, 1):
    print(f"\n=== Test Claim {i} ===")
    print(f"Claim: {claim}")
    plan = planning.plan(claim=claim)
    print(f"Plan:\n{plan}")
    print("=" * 50)