import os
import csv
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import pandas as pd

REPORT_PATH = None
EVIDENCE_PATH = None

def get_report_path(identifier):
    """Returns the report path based on the identifier."""
    base_dir = "../../reports/" + identifier
    filename = "report.md"
    return os.path.join(base_dir, filename)

def init_report(claim, identifier):
    """Initializes the report directory and files for a given identifier."""
    global REPORT_PATH
    global EVIDENCE_PATH
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../reports', identifier))
    os.makedirs(base_dir, exist_ok=True)
    images_dir = os.path.join(base_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    report_md_path = os.path.join(base_dir, 'report.md')
    with open(report_md_path, 'w') as f:
        f.write(f"# Claim: {claim}\n\n") 
    evidence_md_path = os.path.join(base_dir, 'evidence.md')
    with open(evidence_md_path, 'w') as f:
        f.write("### Raw Evidence\n\n") 
    REPORT_PATH = report_md_path
    EVIDENCE_PATH = evidence_md_path

def append_iteration_actions(iteration, actions):
    try:
        with open(REPORT_PATH, "a") as f:
            f.write(f"## Iteration {iteration}: Actions\n\n")
            f.write(actions.strip() + "\n\n")
            f.write("### Evidence\n\n")
    except Exception as e:
        print(f"Error appending actions: {e}")

def append_evidence(evidence):
    try:
        with open(REPORT_PATH, "a") as f:
            f.write(evidence + "\n\n")
    except Exception as e:
        print(f"Error appending evidence: {e}")

def append_raw(evidence):
    try:
        with open(EVIDENCE_PATH, "a") as f:
            f.write(evidence.strip() + "\n\n")
    except Exception as e:
        print(f"Error appending evidence: {e}")

def append_verdict(verdict):
    try:
        with open(REPORT_PATH, "a") as f:
            f.write("### Verdict\n\n")
            f.write(verdict.strip() + "\n\n")
    except Exception as e:
        print(f"Error appending verdict: {e}")

def append_justification(justification):
    try:
        with open(REPORT_PATH, "a") as f:
            f.write("### Justification\n\n")
            f.write(justification.strip() + "\n\n")
    except Exception as e:
        print(f"Error appending justification: {e}")

def write_detailed_csv(claim, evidence, verdict, report_path, csv_path, expected_label=None, numeric_verdict=None, claim_id=None, judge_model_name=None, processing_time=None, clean_claim=None):
    """Writes detailed fact-checking results to a CSV file with fixed columns.
    Ensures a sample is only written once (skip if same report_path or id already present)."""
    try:
        LABEL_TO_NUM = {
            "supported": 0,
            "refuted": 1,
            "not enough evidence": 2
        }

        def normalize_label_to_num(val):
            if val is None:
                return None
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return int(val)
            s = str(val).strip()
            if not s:
                return None
            if s.isdigit():
                return int(s)
            s_low = s.lower()
            for key in LABEL_TO_NUM:
                if key in s_low:
                    return LABEL_TO_NUM[key]
            return None

        pred_num = normalize_label_to_num(numeric_verdict) if numeric_verdict is not None else normalize_label_to_num(verdict)
        label_num = normalize_label_to_num(expected_label)

        # Ensure directory exists
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        # If file exists, avoid duplicates by checking report_path or id
        if os.path.exists(csv_path):
            try:
                df_existing = pd.read_csv(csv_path, dtype=str)
                # check by report_path
                if report_path and 'report_path' in df_existing.columns:
                    if (df_existing['report_path'].astype(str) == str(report_path)).any():
                        print(f"Skipping write: report_path already present in {csv_path}")
                        return
                # check by id
                if claim_id is not None and 'id' in df_existing.columns:
                    if (df_existing['id'].astype(str) == str(claim_id)).any():
                        print(f"Skipping write: id {claim_id} already present in {csv_path}")
                        return
            except Exception:
                # if reading fails, continue to append
                pass

        file_exists = os.path.isfile(csv_path)
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            fieldnames = [
                'id',
                'claim',
                'clean_claim',
                'evidence',
                'verdict',
                'predicted_label',
                'expected_label',
                'compare',
                'timestamp',
                'report_path',
                'judge_model',
                'processing_time'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            evidence_clean = ' '.join(evidence.strip().split()) if evidence else ""

            writer.writerow({
                'id': claim_id if claim_id is not None else "",
                'claim': claim,
                'clean_claim': clean_claim if clean_claim is not None else claim,
                'evidence': evidence_clean,
                'verdict': verdict,
                'predicted_label': pred_num if pred_num is not None else "",
                'expected_label': label_num if label_num is not None else "",
                'compare': 1 if (label_num is not None and pred_num is not None and label_num == pred_num) else 0,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'report_path': report_path or "",
                'judge_model': judge_model_name or "",
                'processing_time': f"{processing_time:.2f}" if processing_time is not None else ""
            })
    except Exception as e:
        print(f"Error writing to CSV: {e}")

def calculate_metrics(csv_path, avg_processing_time=None, total_processing_time=None):
    """Calculate metrics using numeric labels and write them to metrics.txt"""
    try:
        print(f"Reading CSV from: {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"CSV columns found: {df.columns.tolist()}")
        print(f"Number of rows: {len(df)}")
        
        LABEL_TO_NUM = {
            "supported": 0,
            "refuted": 1,
            "not enough evidence": 2
        }

        def normalize_label_to_num(val):
            if pd.isna(val):
                return None
            try:
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    return int(val)
            except Exception:
                pass
            s = str(val).strip()
            if not s:
                return None
            if s.isdigit():
                return int(s)
            s_low = s.lower()
            for key in LABEL_TO_NUM:
                if key in s_low:
                    return LABEL_TO_NUM[key]
            return LABEL_TO_NUM.get(s_low, None)

        if 'predicted_label' not in df.columns or 'expected_label' not in df.columns:
            print("Error: CSV must contain 'predicted_label' and 'expected_label' columns")
            return None

        # Apply normalization
        df['pred_mapped'] = df['predicted_label'].apply(normalize_label_to_num)
        df['label_mapped'] = df['expected_label'].apply(normalize_label_to_num)

        # Drop rows missing either label or prediction
        clean = df.dropna(subset=['label_mapped', 'pred_mapped'])
        if clean.empty:
            print("No rows with both numeric label and numeric prediction available for metrics.")
            return None

        y_true = clean['label_mapped'].astype(int)
        y_pred = clean['pred_mapped'].astype(int)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        
        class_report = classification_report(
            y_true,
            y_pred,
            digits=4,
            zero_division=0
        )

        judge_model_names = None
        if 'judge_model' in df.columns:
            judge_model_names = sorted({str(m).strip() for m in df['judge_model'].dropna() if str(m).strip()})

        metrics = {
            'accuracy': accuracy,
            'precision': precision, 
            'recall': recall,
            'f1': f1,
            'classification_report': class_report,
            'judge_model_names': judge_model_names
        }

        # Write metrics to metrics.txt next to the csv file
        try:
            metrics_path = os.path.join(os.path.dirname(csv_path), 'metrics.txt')
            print(f"Writing metrics to: {metrics_path}")
            with open(metrics_path, 'w', encoding='utf-8') as mf:
                mf.write(f"Calculated: {datetime.now().isoformat()}\n")
                mf.write(f"Rows evaluated: {len(clean)}\n")
                mf.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
                mf.write(f"Precision (weighted): {metrics['precision']:.4f}\n")
                mf.write(f"Recall (weighted): {metrics['recall']:.4f}\n")
                mf.write(f"F1-score (weighted): {metrics['f1']:.4f}\n")
                mf.write("\nClassification report (per class):\n")
                mf.write(metrics['classification_report'])
                if judge_model_names:
                    mf.write("\n\nJudge models used:\n")
                    mf.write(", ".join(judge_model_names))
                # Thêm thông tin thời gian
                mf.write("\n\n" + "=" * 50 + "\n")
                mf.write("THỐNG KÊ THỜI GIAN:\n")
                mf.write("=" * 50 + "\n")
                if total_processing_time is not None:
                    mf.write(f"Tổng thời gian: {total_processing_time:.2f} giây ({total_processing_time/60:.2f} phút)\n")
                if avg_processing_time is not None:
                    mf.write(f"Thời gian trung bình mỗi sample: {avg_processing_time:.2f} giây\n")
            print(f"Successfully wrote metrics to {metrics_path}")
        except Exception as e:
            print(f"Error writing metrics file: {e}")
            raise
        
        return metrics
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_report_content():
    """Reads the current report content to extract sections."""
    if not REPORT_PATH or not os.path.exists(REPORT_PATH):
        return None, None, None, None
        
    evidence = ""
    verdict = ""
    justification = ""
    
    try:
        with open(REPORT_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Split content by sections
            sections = content.split('###')
            
            for section in sections:
                if not section.strip():
                    continue
                    
                if 'Evidence' in section:
                    evidence = section.replace('Evidence', '').strip()
                elif 'Verdict' in section:
                    verdict = section.replace('Verdict', '').strip()
                elif 'Justification' in section:
                    justification = section.replace('Justification', '').strip()
                    
        return evidence, verdict, justification
        
    except Exception as e:
        print(f"Error reading report content: {e}")
        return None, None, None, None

def log_step(step_name, data, logs_list):
    """
    Log một bước trong pipeline.
    
    Args:
        step_name: Tên bước (ví dụ: "Web Search - Input")
        data: Dict chứa dữ liệu của bước
        logs_list: List để append log vào
    """
    log_entry = {
        "step": step_name,
        "data": data,
        "timestamp": datetime.now().isoformat()
    }
    logs_list.append(log_entry)

def append_pipeline_log(title, logs):
    """
    Append pipeline log vào report.md.
    
    Args:
        title: Tiêu đề của log section (ví dụ: "BƯỚC 2: WEB SEARCH")
        logs: List các log entries từ log_step() (list of dicts) hoặc list of strings
    """
    try:
        if not REPORT_PATH:
            print("Warning: REPORT_PATH not initialized")
            return
        with open(REPORT_PATH, "a", encoding='utf-8') as f:
            f.write(f"\n### {title}\n\n")
            # Xử lý cả 2 trường hợp: list of dicts hoặc list of strings
            if logs and isinstance(logs[0], dict):
                # List of dicts (từ log_step)
                for log_entry in logs:
                    step_name = log_entry.get("step", "Unknown")
                    data = log_entry.get("data", {})
                    f.write(f"**{step_name}**\n\n")
                    for key, value in data.items():
                        if isinstance(value, (list, dict)):
                            value = str(value)
                        elif value is None:
                            value = "N/A"
                        f.write(f"- {key}: {value}\n")
                    f.write("\n")
            else:
                # List of strings (từ filter_log_lines)
                for log_line in logs:
                    f.write(f"{log_line}\n")
                f.write("\n")
    except Exception as e:
        print(f"Error appending pipeline log: {e}")

def append_evidence_info(evidence_info):
    """
    Append evidence info vào report.md.
    
    Args:
        evidence_info: Dict chứa thông tin evidence
    """
    try:
        if not REPORT_PATH:
            print("Warning: REPORT_PATH not initialized")
            return
        with open(REPORT_PATH, "a", encoding='utf-8') as f:
            f.write("\n### Evidence Info\n\n")
            f.write(f"- Total evidence: {evidence_info.get('total_evidence', 0)}\n")
            f.write(f"- Filtered evidence count: {evidence_info.get('filtered_evidence_count', 0)}\n")
            f.write(f"- Selected evidence count: {evidence_info.get('selected_evidence_count', 0)}\n")
            f.write(f"- Top_k: {evidence_info.get('top_k', 0)}\n\n")
            
            selected_evidence = evidence_info.get('selected_evidence', [])
            selected_scores = evidence_info.get('selected_scores', [])
            if selected_evidence:
                f.write("**Selected Evidence:**\n\n")
                for i, (ev, score) in enumerate(zip(selected_evidence, selected_scores)):
                    f.write(f"[E{i}] (Score: {score:.4f})\n")
                    f.write(f"{ev}\n\n")
    except Exception as e:
        print(f"Error appending evidence info: {e}")

def append_timing_info(timing_records):
    """
    Append timing information vào report.md.
    
    Args:
        timing_records: List các dict {"label": ..., "duration": ...}
    """
    try:
        if not REPORT_PATH:
            print("Warning: REPORT_PATH not initialized")
            return
        if not timing_records:
            return
        with open(REPORT_PATH, "a", encoding='utf-8') as f:
            f.write("\n### Timing Information\n\n")
            total_time = sum(record.get("duration", 0) for record in timing_records)
            f.write(f"**Total time: {total_time:.2f}s**\n\n")
            f.write("| Step | Duration (s) |\n")
            f.write("|------|-------------|\n")
            for record in timing_records:
                label = record.get("label", "Unknown")
                duration = record.get("duration", 0)
                f.write(f"| {label} | {duration:.2f} |\n")
            f.write("\n")
    except Exception as e:
        print(f"Error appending timing info: {e}")
