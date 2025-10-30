from factchecker.factchecker import factcheck
import sys
import json
from datetime import datetime, timezone, timedelta
    
if __name__ == "__main__":
    # Kiểm tra tham số
    if len(sys.argv) < 3:
        print("Usage: python factchecker.py <path_to_json> <num_records>")
        sys.exit(1)

    json_path = sys.argv[1]
    num_records = int(sys.argv[2])

    # Đọc file JSON
    with open(json_path, "r") as f:
        data = json.load(f)

    # Giới hạn số bản ghi
    data = data[:num_records]

    # Thời gian Việt Nam
    VN_TIMEZONE = timezone(timedelta(hours=7))
    now_vn = datetime.now(VN_TIMEZONE)
    # Tạo một thư mục chạy duy nhất theo định dạng ddmmyy-hhmm
    run_identifier = now_vn.strftime("%d%m%y-%H%M")

    # Chạy fact-check cho từng claim
    for i, record in enumerate(data):
        claim = record["claim"]
        date = record.get("date", now_vn.strftime("%d-%m-%Y"))
        claim_id = str(record.get("id", i + 1))
        print(f"\n=== [{i+1}/{num_records}] Fact-checking: {claim}")
        # Lưu kết quả vào reports/<ddmmyy-hhmm>/<id>/
        identifier = f"{run_identifier}/{claim_id}"
        verdict, report_path = factcheck(claim, date, identifier=identifier)
        print(f"Verdict: {verdict}")
        print(f"Report saved at: {report_path}")