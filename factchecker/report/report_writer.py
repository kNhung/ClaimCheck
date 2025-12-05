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
        f.write("=" * 80 + "\n")
        f.write("📊 PIPELINE OVERVIEW\n")
        f.write("=" * 80 + "\n")
        f.write("""
Pipeline bao gồm các bước sau:

1. **Claim Filtering**: Lọc và chuẩn hóa claim ban đầu
2. **Planning**: Tạo các query tìm kiếm từ claim
3. **Web Search**: Tìm kiếm trên web với các query đã tạo
4. **Web Scraping**: Lấy nội dung từ các URL tìm được
5. **RAV (Evidence Ranking)**: Xếp hạng và chọn evidence quan trọng nhất từ mỗi URL
6. **Evidence Synthesis**: Phân tích evidence hiện có, quyết định có cần thêm evidence không
7. **Evidence Filtering & Selection**: Lọc và chọn top evidence để đưa vào judge
8. **Judge**: Đưa ra verdict cuối cùng (Supported/Refuted/Not Enough Evidence)

Tất cả các bước đều được log chi tiết bên dưới để dễ dàng theo dõi và debug.
""")
        f.write("=" * 80 + "\n\n")
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
    """Append evidence to report, avoiding duplicates"""
    try:
        if not evidence or not evidence.strip():
            return
        
        # Check if evidence already exists in report to avoid duplicates
        if REPORT_PATH and os.path.exists(REPORT_PATH):
            with open(REPORT_PATH, "r", encoding='utf-8') as f:
                existing_content = f.read()
                # Simple check: if evidence text already exists, skip
                if evidence.strip() in existing_content:
                    return
        
        with open(REPORT_PATH, "a", encoding='utf-8') as f:
            f.write(evidence + "\n\n")
    except Exception as e:
        print(f"Error appending evidence: {e}")

def append_rav_log(log_lines):
    """
    Append RAV (Retrieval-Augmented Verification) process log to report.
    
    Args:
        log_lines: List of log lines or single string
    """
    try:
        with open(REPORT_PATH, "a") as f:
            if isinstance(log_lines, str):
                f.write(log_lines + "\n")
            else:
                for line in log_lines:
                    f.write(line + "\n")
    except Exception as e:
        print(f"Error appending RAV log: {e}")

def append_evidence_filter_log(log_lines):
    """
    Append evidence filtering process log to report.
    
    Args:
        log_lines: List of log lines or single string
    """
    try:
        with open(REPORT_PATH, "a") as f:
            if isinstance(log_lines, str):
                f.write(log_lines + "\n")
            else:
                for line in log_lines:
                    f.write(line + "\n")
    except Exception as e:
        print(f"Error appending evidence filter log: {e}")

def append_pipeline_log(section_name, log_lines):
    """
    Append pipeline process log to report with standardized format.
    
    Args:
        section_name: Name of the pipeline section (e.g., "PLANNING", "WEB_SEARCH", "JUDGE")
        log_lines: List of log lines or single string
    """
    try:
        with open(REPORT_PATH, "a") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"📋 {section_name}\n")
            f.write(f"{'='*80}\n")
            if isinstance(log_lines, str):
                f.write(log_lines + "\n")
            else:
                for line in log_lines:
                    f.write(line + "\n")
            f.write(f"{'='*80}\n\n")
    except Exception as e:
        print(f"Error appending pipeline log: {e}")

def log_step(step_name, details, log_lines_list=None):
    """
    Log a single step in the pipeline.
    
    Args:
        step_name: Name of the step
        details: Details about the step (dict or string)
        log_lines_list: Optional list to append log lines to
    """
    if log_lines_list is None:
        log_lines_list = []
    
    if isinstance(details, dict):
        log_line = f"  → {step_name}:"
        for key, value in details.items():
            if isinstance(value, (list, tuple)):
                if len(value) == 0:
                    value_str = "[]"
                else:
                    # Ghi đầy đủ tất cả items, không truncate
                    value_str = str(value)
            elif isinstance(value, bool):
                value_str = "✓ Yes" if value else "✗ No"
            else:
                value_str = str(value)
            # Không truncate - ghi đầy đủ thông tin
            log_line += f"\n    • {key}: {value_str}"
    else:
        log_line = f"  → {step_name}: {details}"
        # Không truncate - ghi đầy đủ thông tin
    
    log_lines_list.append(log_line)
    print(f"[PIPELINE] {step_name}: {details}")
    return log_lines_list

def append_raw(evidence):
    try:
        with open(EVIDENCE_PATH, "a") as f:
            f.write(evidence.strip() + "\n\n")
    except Exception as e:
        print(f"Error appending evidence: {e}")

def append_action_needed(action_needed):
    try:
        with open(REPORT_PATH, "a") as f:
            f.write("### Action Needed\n\n")
            f.write(action_needed.strip() + "\n\n")
    except Exception as e:
        print(f"Error appending action_needed: {e}")

def append_evidence_info(evidence_info):
    """
    Ghi thông tin về evidence được chọn cho judge vào report.md.
    Format giống với log để đảm bảo thông tin nhất quán.
    """
    try:
        with open(REPORT_PATH, "a") as f:
            f.write("### Evidence Selected for Judge\n\n")
            f.write("=" * 80 + "\n")
            f.write("📋 DANH SÁCH BẰNG CHỨNG ĐƯỢC CHỌN CHO JUDGE:\n")
            f.write("=" * 80 + "\n")
            f.write(f"Claim: {evidence_info.get('claim', 'N/A')}\n")
            f.write(f"\nTổng số bằng chứng ban đầu: {evidence_info.get('total_evidence', 0)}\n")
            f.write(f"Số bằng chứng sau khi lọc (relevance > 0.15): {evidence_info.get('filtered_evidence_count', 0)}\n")
            f.write(f"Số bằng chứng được chọn (top_k={evidence_info.get('top_k', 0)}): {evidence_info.get('selected_evidence_count', 0)}\n")
            f.write("\n" + "-" * 80 + "\n")
            
            selected_evidence = evidence_info.get('selected_evidence', [])
            selected_scores = evidence_info.get('selected_scores', [])
            
            for i, (ev, score) in enumerate(zip(selected_evidence, selected_scores)):
                f.write(f"\n[E{i}] (Relevance score: {score:.4f})\n")
                f.write(f"{ev}\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
    except Exception as e:
        print(f"Error appending evidence info: {e}")

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

def append_timing_info(timings):
    """
    Append timing information to the report.md file.
    
    Args:
        timings: List of dicts with 'label' and 'duration' keys
    """
    try:
        if not timings:
            return
        
        with open(REPORT_PATH, "a") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("⏱️  THỐNG KÊ THỜI GIAN XỬ LÝ\n")
            f.write("=" * 80 + "\n\n")
            
            # Tính tổng thời gian
            total_duration = sum(t.get('duration', 0) for t in timings)
            
            # Nhóm timing theo loại
            timing_groups = {}
            for timing in timings:
                label = timing.get('label', 'unknown')
                duration = timing.get('duration', 0)
                
                # Phân loại timing
                if label.startswith('web_search:'):
                    group = 'Web Search'
                elif label.startswith('scrape:'):
                    group = 'Web Scraping'
                elif label.startswith('evidence_rank:'):
                    group = 'RAV (Evidence Ranking)'
                elif label.startswith('action_needed_iter_'):
                    group = 'Evidence Synthesis'
                elif label.startswith('action_needed_action_exec_'):
                    group = 'Action Execution'
                elif label == 'planning':
                    group = 'Planning'
                elif label == 'claim_filtering':
                    group = 'Claim Filtering'
                elif label == 'judge' or label.startswith('judge_try_'):
                    group = 'Judge'
                elif label == 'initial_action_execution':
                    group = 'Initial Action Execution'
                elif label == 'factcheck_run':
                    group = 'Total (FactCheck Run)'
                else:
                    group = 'Other'
                
                if group not in timing_groups:
                    timing_groups[group] = []
                timing_groups[group].append({
                    'label': label,
                    'duration': duration
                })
            
            # Ghi tổng thời gian
            f.write(f"**Tổng thời gian xử lý:** {total_duration:.2f} giây ({total_duration/60:.2f} phút)\n\n")
            
            # Ghi chi tiết theo nhóm
            for group, items in sorted(timing_groups.items()):
                group_total = sum(item['duration'] for item in items)
                group_percentage = (group_total / total_duration * 100) if total_duration > 0 else 0
                
                f.write(f"### {group}\n")
                f.write(f"**Tổng:** {group_total:.2f}s ({group_percentage:.1f}%)\n\n")
                
                # Sắp xếp items theo duration (descending)
                items_sorted = sorted(items, key=lambda x: x['duration'], reverse=True)
                
                for item in items_sorted:
                    label = item['label']
                    duration = item['duration']
                    percentage = (duration / total_duration * 100) if total_duration > 0 else 0
                    
                    # Format label cho đẹp
                    display_label = label
                    if ':' in display_label:
                        display_label = display_label.split(':', 1)[1].strip()
                    
                    f.write(f"- `{display_label}`: {duration:.2f}s ({percentage:.1f}%)\n")
                
                f.write("\n")
            
            # Top 5 bước chậm nhất
            f.write("### Top 5 Bước Chậm Nhất\n\n")
            top_5 = sorted(timings, key=lambda x: x.get('duration', 0), reverse=True)[:5]
            for i, timing in enumerate(top_5, 1):
                label = timing.get('label', 'unknown')
                duration = timing.get('duration', 0)
                percentage = (duration / total_duration * 100) if total_duration > 0 else 0
                
                # Format label
                display_label = label
                if ':' in display_label:
                    display_label = display_label.split(':', 1)[1].strip()
                
                f.write(f"{i}. `{display_label}`: {duration:.2f}s ({percentage:.1f}%)\n")
            
            f.write("\n" + "=" * 80 + "\n\n")
    except Exception as e:
        print(f"Error appending timing info: {e}")

def write_detailed_csv(claim, date, evidence, reasoning, verdict, justification, report_path, csv_path, expected_label=None, numeric_verdict=None, claim_id=None, model_name=None, processing_time=None, clean_claim=None):
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
                'reasoning',
                'verdict',
                'predicted_label',
                'expected_label',
                'compare',
                'timestamp',
                'report_path',
                'model',
                'processing_time'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()

            evidence_clean = ' '.join(evidence.strip().split()) if evidence else ""
            reasoning_clean = ' '.join(reasoning.strip().split()) if reasoning else ""
            # Use provided clean_claim if available, otherwise compute from claim
            if clean_claim is None:
                clean_claim = ' '.join(claim.strip().split()) if claim else ""
            else:
                clean_claim = ' '.join(clean_claim.strip().split()) if clean_claim else ""

            writer.writerow({
                'id': claim_id if claim_id is not None else "",
                'claim': claim,
                'clean_claim': clean_claim,
                'evidence': evidence_clean,
                'reasoning': reasoning_clean,
                'verdict': verdict,
                'predicted_label': pred_num if pred_num is not None else "",
                'expected_label': label_num if label_num is not None else "",
                'compare': 1 if (label_num is not None and pred_num is not None and label_num == pred_num) else 0,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'report_path': report_path or "",
                'model': model_name or "",
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

        model_names = None
        if 'model' in df.columns:
            model_names = sorted({str(m).strip() for m in df['model'].dropna() if str(m).strip()})

        metrics = {
            'accuracy': accuracy,
            'precision': precision, 
            'recall': recall,
            'f1': f1,
            'classification_report': class_report,
            'model_names': model_names
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
                if model_names:
                    mf.write("\n\nModels used:\n")
                    mf.write(", ".join(model_names))
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
    import re
    
    if not REPORT_PATH or not os.path.exists(REPORT_PATH):
        return None, None, None, None
        
    evidence = ""
    action_needed = ""
    verdict = ""
    justification = ""
    
    try:
        with open(REPORT_PATH, 'r', encoding='utf-8') as f:
            content = f.read()
            
            # Extract evidence từ các dòng web_search(...) Summary: ...
            evidence_pieces = []
            
            # Pattern 1: web_search('...') Summary: ... (format hiện tại)
            pattern1 = re.compile(r"web_search\([^)]+\)\s+Summary:\s*(.+?)(?=\n\n|\nweb_search\(|\n###|$)", re.DOTALL | re.IGNORECASE)
            matches1 = pattern1.findall(content)
            for match in matches1:
                summary = match.strip()
                # Filter out log text and metadata
                if summary and not summary.startswith(('📋', '🔍', '✅', '→', '•', 'BƯỚC', 'WEB SEARCH', 'WEB SCRAPING', 'RAV', 'Chunk #', 'score:', 'Content preview:', 'Snippets preview:', 'URLs:', 'Query:', 'Domain:')) and len(summary) > 10:
                    evidence_pieces.append(summary)
            
            # Pattern 2: web_search(...), Summary: ... (format có dấu phẩy)
            pattern2 = re.compile(r"web_search\([^)]+\)\s*,\s*Summary:\s*(.+?)(?=\n\n|\nweb_search\(|\n###|$)", re.DOTALL | re.IGNORECASE)
            matches2 = pattern2.findall(content)
            for match in matches2:
                summary = match.strip()
                if summary and not summary.startswith(('📋', '🔍', '✅', '→', '•', 'BƯỚC', 'WEB SEARCH', 'WEB SCRAPING', 'RAV', 'Chunk #', 'score:', 'Content preview:', 'Snippets preview:', 'URLs:', 'Query:', 'Domain:')) and len(summary) > 10:
                    evidence_pieces.append(summary)
            
            # Join evidence pieces với separator
            if evidence_pieces:
                evidence = "\n\n".join(evidence_pieces)
            
            # Extract action_needed từ section ### Action Needed
            action_needed_match = re.search(r'###\s*Action Needed\s*\n\n(.+?)(?=\n###|$)', content, re.DOTALL | re.IGNORECASE)
            if action_needed_match:
                action_needed = action_needed_match.group(1).strip()
            
            # Extract verdict từ section ### Verdict
            verdict_match = re.search(r'###\s*Verdict\s*\n\n(.+?)(?=\n###|$)', content, re.DOTALL | re.IGNORECASE)
            if verdict_match:
                verdict = verdict_match.group(1).strip()
            
            # Extract justification từ section ### Justification
            justification_match = re.search(r'###\s*Justification\s*\n\n(.+?)(?=\n###|$)', content, re.DOTALL | re.IGNORECASE)
            if justification_match:
                justification = justification_match.group(1).strip()
                    
        return evidence, action_needed, verdict, justification
        
    except Exception as e:
        print(f"Error reading report content: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None