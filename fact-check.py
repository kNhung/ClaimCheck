from factchecker.factchecker import factcheck
import sys
import json
import time
import os
import pandas as pd
from datetime import datetime, timezone, timedelta
import random

time_sleep = random.randint(70, 90)  # Thời gian chờ giữa các lần gọi API để tránh bị giới hạn
    
def csv_to_json(csv_path):
    # Read CSV file
    df = pd.read_csv(csv_path)
    # Convert DataFrame to list of dictionaries
    return df.to_dict('records')

if __name__ == "__main__":
    # Kiểm tra tham số
    if len(sys.argv) < 3:
        print("Usage: python factchecker.py <path_to_json> <num_records>")
        print("Supported formats: .json, .csv")
        sys.exit(1)

    input_path = sys.argv[1]
    num_records = int(sys.argv[2])

    # Determine file type and load data
    if input_path.endswith('.json'):
        with open(input_path, "r") as f:
            data = json.load(f)
    elif input_path.endswith('.csv'):
        data = csv_to_json(input_path)
    else:
        print("Error: Unsupported file format. Please use .json or .csv")
        sys.exit(1)

    # Giới hạn số bản ghi
    data = data[:num_records]

    # Thời gian Việt Nam
    VN_TIMEZONE = timezone(timedelta(hours=7))
    now_vn = datetime.now(VN_TIMEZONE)
    # Tạo một thư mục chạy duy nhất theo định dạng ddmmyy-hhmm
    run_identifier = now_vn.strftime("%d%m%y-%H%M")

    #key_number = 1
    # key_count = 1
    # claims_per_key = 2

    # Chạy fact-check cho từng claim
    for i, record in enumerate(data):
        claim = record["claim"]
        date = record.get("date", now_vn.strftime("%d-%m-%Y"))
        claim_id = str(record.get("id", i + 1))
        expected_label = record.get("labels", None)  # Get expected label if exists

        # if key_count > claims_per_key:
        #     key_number += 1
        #     key_count = 1
        #     print(f"API key used: GEMINI_API_KEY_{key_number}")

        print(f"\n=== [{i+1}/{num_records}] Fact-checking: {claim}")
        print(f"Expected label: {expected_label}")
        
        
        # Lưu kết quả vào reports/<ddmmyy-hhmm>/<id>/
        identifier = f"{run_identifier}/{claim_id}"
        verdict, report_path = factcheck(
            claim, 
            date, 
            identifier=identifier,
            expected_label=expected_label
        )
        
        print(f"Predicted verdict: {verdict}")
        print(f"Report saved at: {report_path}")

        # key_count += 1
        # print(f"Waiting for {time_sleep} seconds before next claim...")
        # time.sleep(time_sleep)  # Chờ giữa các lần gọi API