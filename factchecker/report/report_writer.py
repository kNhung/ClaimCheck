import os
import csv
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
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
        f.write(f"# Claim: {claim}") 
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

def append_reasoning(reasoning):
    try:
        with open(REPORT_PATH, "a") as f:
            f.write("### Reasoning\n\n")
            f.write(reasoning.strip() + "\n\n")
    except Exception as e:
        print(f"Error appending reasoning: {e}")

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

def write_detailed_csv(claim, date, evidence, reasoning, verdict, justification, report_path, csv_path, expected_label=None, numeric_verdict=None):
    """Writes detailed fact-checking results to a CSV file."""
    try:
        # mapping text -> numeric
        LABEL_TO_NUM = {
            "Supported": 0,
            "Refuted": 1,
            "Not Enough Evidence": 2
        }

        # Normalize numeric_verdict: if not provided but verdict text exists, map it
        if numeric_verdict is None:
            if isinstance(verdict, (int, float)):
                numeric_verdict = int(verdict)
            elif isinstance(verdict, str):
                numeric_verdict = LABEL_TO_NUM.get(verdict.strip(), None)

        # Normalize expected_label into numeric label for metrics
        label_num = None
        if expected_label is not None:
            if isinstance(expected_label, (int, float)):
                label_num = int(expected_label)
            else:
                # try parse integer string first, else map text
                try:
                    label_num = int(str(expected_label).strip())
                except Exception:
                    label_num = LABEL_TO_NUM.get(str(expected_label).strip(), None)

        # Create header if file doesn't exist
        file_exists = os.path.isfile(csv_path)
        
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            fieldnames = [
                'claim', 
                'date',
                'evidence',
                'reasoning', 
                'verdict',
                'numeric_verdict',
                'justification',
                'report_path',
                'timestamp',
                'label'  # numeric ground-truth label
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            # Clean up text fields
            evidence = ' '.join(evidence.strip().split()) if evidence else ""
            reasoning = ' '.join(reasoning.strip().split()) if reasoning else ""
            justification = ' '.join(justification.strip().split()) if justification else ''
            
            writer.writerow({
                'claim': claim,
                'date': date,
                'evidence': evidence,
                'reasoning': reasoning,
                'verdict': verdict,
                'numeric_verdict': numeric_verdict,
                'justification': justification,
                'report_path': report_path,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'label': label_num
            })
            
        # Calculate metrics after each write if we have numeric labels
        if label_num is not None:
            metrics = calculate_metrics(csv_path)
            if metrics:
                print("\nEvaluation Metrics:")
                print(f"Accuracy: {metrics['accuracy']:.4f}")
                print(f"Precision: {metrics['precision']:.4f}")
                print(f"Recall: {metrics['recall']:.4f}")
                print(f"F1-score: {metrics['f1']:.4f}\n")
                
    except Exception as e:
        print(f"Error writing to CSV: {e}")

def calculate_metrics(csv_path):
    """Calculate metrics using numeric labels and write them to metrics.txt"""
    try:
        df = pd.read_csv(csv_path)
        # Accept text or numeric labels by mapping text -> numeric if necessary
        LABEL_TO_NUM = {
            "Supported": 0,
            "Refuted": 1,
            "Not Enough Evidence": 2
        }

        if 'numeric_verdict' not in df.columns or 'label' not in df.columns:
            print("Error: CSV must contain 'numeric_verdict' and 'label' columns")
            return None

        # Map potential text in 'label' to numeric
        if not pd.api.types.is_integer_dtype(df['label']):
            df['label_mapped'] = df['label'].map(lambda v: LABEL_TO_NUM.get(str(v).strip(), None) if pd.notna(v) else None)
        else:
            df['label_mapped'] = df['label'].astype('Int64')

        # Ensure numeric_verdict column is numeric (map text if needed)
        if not pd.api.types.is_integer_dtype(df['numeric_verdict']):
            df['pred_mapped'] = df['numeric_verdict'].map(lambda v: int(v) if pd.notna(v) and str(v).strip().isdigit() else LABEL_TO_NUM.get(str(v).strip(), None) if pd.notna(v) else None)
        else:
            df['pred_mapped'] = df['numeric_verdict'].astype('Int64')

        # Drop rows missing either label or prediction
        clean = df.dropna(subset=['label_mapped', 'pred_mapped'])
        if clean.empty:
            print("No rows with both numeric label and numeric prediction available for metrics.")
            return None

        y_true = clean['label_mapped'].astype(int)
        y_pred = clean['pred_mapped'].astype(int)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        confusion = confusion_matrix(y_true, y_pred)
        class_report = classification_report(y_true, y_pred, zero_division=0)

        metrics = {
            'accuracy': accuracy,
            'precision': precision, 
            'recall': recall,
            'f1': f1
        }

        # Write metrics to metrics.txt next to the csv file
        try:
            metrics_path = os.path.join(os.path.dirname(csv_path), 'metrics.txt')
            with open(metrics_path, 'w', encoding='utf-8') as mf:
                mf.write(f"Calculated: {datetime.now().isoformat()}\n")
                mf.write(f"Rows evaluated: {len(clean)}\n")
                mf.write(f"Accuracy: {metrics['accuracy']:.4f}\n")
                mf.write(f"Precision (weighted): {metrics['precision']:.4f}\n")
                mf.write(f"Recall (weighted): {metrics['recall']:.4f}\n")
                mf.write(f"F1-score (weighted): {metrics['f1']:.4f}\n")
                mf.write("\nConfusion Matrix:\n")
                mf.write(str(confusion) + "\n")
                mf.write("\nClassification Report:\n")
                mf.write(class_report)
        except Exception as e:
            print(f"Error writing metrics file: {e}")
        
        return metrics
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None

def get_report_content():
    """Reads the current report content to extract sections."""
    if not REPORT_PATH or not os.path.exists(REPORT_PATH):
        return None, None, None, None
        
    evidence = ""
    reasoning = ""
    verdict = ""
    justification = ""
    current_section = None
    
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
                elif 'Reasoning' in section:
                    reasoning = section.replace('Reasoning', '').strip()
                elif 'Verdict' in section:
                    verdict = section.replace('Verdict', '').strip()
                elif 'Justification' in section:
                    justification = section.replace('Justification', '').strip()
                    
        return evidence, reasoning, verdict, justification
        
    except Exception as e:
        print(f"Error reading report content: {e}")
        return None, None, None, None
