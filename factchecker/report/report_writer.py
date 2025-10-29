import os
from datetime import datetime

REPORT_PATH = None
EVIDENCE_PATH = None

def get_report_path(identifier):
    """Returns the report path based on the identifier.

    Identifier can be nested like "<ddmmyy-hhmm>/<claim_id>".
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../reports', identifier))
    filename = 'report.md'
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
