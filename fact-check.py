from factchecker.factchecker import factcheck
from factchecker.report import report_writer
from factchecker.preprocessing.preprocessing import preprocess
import sys
import json
import pandas as pd
import os
import random
from datetime import datetime, timezone, timedelta
    
def csv_to_json(csv_path):
    # Read CSV file
    df = pd.read_csv(csv_path)
    # Convert DataFrame to list of dictionaries
    return df.to_dict('records')

if __name__ == "__main__":
    # Kiểm tra tham số
    if len(sys.argv) < 3:
        print("Usage: python fact-check.py <path_to_file> <num_records> [model_name] [--shuffle] [--seed <seed_value>]")
        print("Supported formats: .json, .csv")
        print("Options:")
        print("  --shuffle: Shuffle dataset before selecting records")
        print("  --seed <value>: Random seed for shuffling (use with --shuffle, default: random)")
        print("\nExamples:")
        print("  python fact-check.py dataset.csv 10 qwen2.5:0.5b")
        print("  python fact-check.py dataset.csv 10 qwen2.5:0.5b --shuffle")
        print("  python fact-check.py dataset.csv 10 qwen2.5:0.5b --shuffle --seed 42")
        sys.exit(1)

    input_path = sys.argv[1]
    num_records = int(sys.argv[2])
    model_name = None
    shuffle = False
    random_seed = None
    
    # Parse additional arguments
    i = 3
    while i < len(sys.argv):
        arg = sys.argv[i].strip()
        if arg == "--shuffle":
            shuffle = True
        elif arg == "--seed":
            # Next argument should be seed value
            if i + 1 < len(sys.argv):
                try:
                    random_seed = int(sys.argv[i + 1])
                    i += 1  # Skip next argument as it's the seed value
                except ValueError:
                    print(f"Warning: Invalid seed value '{sys.argv[i + 1]}', ignoring --seed")
        elif arg.startswith("--"):
            print(f"Warning: Unknown option '{arg}', ignoring")
        else:
            # Assume it's model_name if not a flag and model_name not set yet
            if model_name is None:
                model_name = arg
        i += 1
    
    # Set default model name if not provided
    if not model_name:
        # Unified primary env var: FACTCHECKER_MODEL_NAME (with backward-compatible fallback)
        model_name = (
            os.getenv("FACTCHECKER_MODEL_NAME")
            or os.getenv("FACTCHECK_MODEL_NAME")
            or "qwen3:4b"
        )
    
    # Set random seed if shuffle is enabled
    if shuffle:
        if random_seed is not None:
            random.seed(random_seed)
            try:
                import numpy as np
                np.random.seed(random_seed)
            except ImportError:
                pass
            print(f"Shuffling dataset with seed: {random_seed}")
        else:
            print("Shuffling dataset (no seed specified - random)")

    # Determine file type and load data
    if input_path.endswith('.json'):
        with open(input_path, "r") as f:
            data = json.load(f)
    elif input_path.endswith('.csv'):
        data = csv_to_json(input_path)
    else:
        print("Error: Unsupported file format. Please use .json or .csv")
        sys.exit(1)

    # Shuffle dataset if requested
    if shuffle:
        print(f"Shuffling dataset (total records: {len(data)})...")
        random.shuffle(data)
        print("Dataset shuffled successfully.")
    
    # Giới hạn số bản ghi sau khi shuffle
    # Note: num_records might be larger than dataset size, so take min
    total_available = len(data)
    data = data[:min(num_records, len(data))]
    actual_num_records = len(data)
    
    if shuffle:
        print(f"Selected {actual_num_records} records after shuffling (from {total_available} total).")
    elif actual_num_records < num_records:
        print(f"Warning: Only {actual_num_records} records available (requested {num_records}).")

    # Thời gian Việt Nam
    VN_TIMEZONE = timezone(timedelta(hours=7))
    now_vn = datetime.now(VN_TIMEZONE)
    # Tạo một thư mục chạy duy nhất theo định dạng ddmmyy-hhmm
    run_identifier = now_vn.strftime("%d%m%y-%H%M")

    # Tracking thời gian tổng thể
    total_start_time = datetime.now(VN_TIMEZONE)
    processing_times = []  # Lưu thời gian xử lý của từng sample
    
    # Chạy fact-check cho từng claim
    for i, record in enumerate(data):
        raw_claim = record["claim"]
        date = record.get("date", now_vn.strftime("%d-%m-%Y"))
        claim_id = str(record.get("id", i + 1))  # Get id from input or use index+1
        expected_label = record.get("labels", None)  # Changed from "labels" to "label"
        
        clean_claim = preprocess(raw_claim)

        print(f"\n=== [{i+1}/{num_records}] Fact-checking:")
        print(f"RAW:   {raw_claim}")
        print(f"CLEAN: {clean_claim}")
        print(f"Expected label: {expected_label}")
        
        # Bắt đầu đo thời gian cho sample này
        sample_start_time = datetime.now(VN_TIMEZONE)
        
        # Lưu kết quả vào reports/<ddmmyy-hhmm>/<id>/
        identifier = f"{run_identifier}/{claim_id}"
        verdict, report_path = factcheck(
            clean_claim, 
            date, 
            identifier=identifier,
            expected_label=expected_label,
            model_name=model_name
        )
        
        # Kết thúc đo thời gian cho sample này
        sample_end_time = datetime.now(VN_TIMEZONE)
        processing_time_seconds = (sample_end_time - sample_start_time).total_seconds()
        processing_times.append(processing_time_seconds)
        
        print(f"⏱️  Thời gian xử lý sample này: {processing_time_seconds:.2f} giây")
        
        # Get content from report for CSV
        evidence, reasoning, verdict_text, justification = report_writer.get_report_content()
        
        # Write to CSV with claim_id
        csv_path = os.path.join(os.getcwd(), 'reports', run_identifier, 'detailed_results.csv')
        report_writer.write_detailed_csv(
            claim=raw_claim,   
            clean_claim=clean_claim,
            date=date,
            evidence=evidence,
            reasoning=reasoning,
            verdict=verdict,
            justification=justification,
            report_path=report_path,
            csv_path=csv_path,
            expected_label=expected_label,
            claim_id=claim_id,  # Pass the id
            model_name=model_name,
            processing_time=processing_time_seconds
        )
    # Tính toán thời gian tổng thể
    total_end_time = datetime.now(VN_TIMEZONE)
    total_time_seconds = (total_end_time - total_start_time).total_seconds()
    avg_time_seconds = sum(processing_times) / len(processing_times) if processing_times else 0
    
    # In ra thời gian ở cuối log
    print("\n" + "=" * 80)
    print("⏱️  THỐNG KÊ THỜI GIAN:")
    print("=" * 80)
    print(f"Tổng số samples đã xử lý: {len(processing_times)}")
    print(f"Tổng thời gian: {total_time_seconds:.2f} giây ({total_time_seconds/60:.2f} phút)")
    print(f"Thời gian trung bình mỗi sample: {avg_time_seconds:.2f} giây")
    if processing_times:
        print(f"Thời gian nhanh nhất: {min(processing_times):.2f} giây")
        print(f"Thời gian chậm nhất: {max(processing_times):.2f} giây")
    print("=" * 80 + "\n")
    
    # After all samples processed: compute metrics once for the run
    csv_path = os.path.join(os.getcwd(), 'reports', run_identifier, 'detailed_results.csv')
    if os.path.exists(csv_path):
        metrics = report_writer.calculate_metrics(csv_path, avg_processing_time=avg_time_seconds, total_processing_time=total_time_seconds)
        if metrics:
            print("\nFinal Evaluation Metrics (all samples):")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-score: {metrics['f1']:.4f}")
            if metrics.get('classification_report'):
                print("\nClassification report (per class):")
                print(metrics['classification_report'])
            model_list = metrics.get('model_names')
            if model_list:
                print(f"Models in this run: {', '.join(model_list)}")
            print(f"Metrics written to: {os.path.join('reports', run_identifier, 'metrics.txt')}")
        else:
            print("Metrics could not be calculated (insufficient / invalid label data).")
    else:
        print(f"No detailed_results.csv found at {csv_path} — skip metrics calculation.")