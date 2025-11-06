from factchecker.factchecker import factcheck
from factchecker.report import report_writer
import sys
import json
import pandas as pd
import os
from datetime import datetime, timezone, timedelta
    
def csv_to_json(csv_path):
    # Read CSV file
    df = pd.read_csv(csv_path)
    # Convert DataFrame to list of dictionaries
    return df.to_dict('records')

if __name__ == "__main__":
    # Kiểm tra tham số
    if len(sys.argv) < 3:
        print("Usage: python fact-check.py <path_to_file> <num_records>")
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

    # Chạy fact-check cho từng claim
    for i, record in enumerate(data):
        claim = record["claim"]
        date = record.get("date", now_vn.strftime("%d-%m-%Y"))
        claim_id = str(record.get("id", i + 1))  # Get id from input or use index+1
        expected_label = record.get("labels", None)  # Changed from "labels" to "label"

        
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
        

        # Get content from report for CSV
        evidence, reasoning, verdict_text, justification = report_writer.get_report_content()
        
        # Write to CSV with claim_id
        csv_path = os.path.join(os.getcwd(), 'reports', run_identifier, 'detailed_results.csv')
        report_writer.write_detailed_csv(
            claim=claim,
            date=date,
            evidence=evidence,
            reasoning=reasoning,
            verdict=verdict,
            justification=justification,
            report_path=report_path,
            csv_path=csv_path,
            expected_label=expected_label,
            claim_id=claim_id  # Pass the id
        )
    # After all samples processed: compute metrics once for the run
    csv_path = os.path.join(os.getcwd(), 'reports', run_identifier, 'detailed_results.csv')
    if os.path.exists(csv_path):
        metrics = report_writer.calculate_metrics(csv_path)
        if metrics:
            print("\nFinal Evaluation Metrics (all samples):")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-score: {metrics['f1']:.4f}")
            print(f"Metrics written to: {os.path.join('reports', run_identifier, 'metrics.txt')}")
        else:
            print("Metrics could not be calculated (insufficient / invalid label data).")
    else:
        print(f"No detailed_results.csv found at {csv_path} — skip metrics calculation.")