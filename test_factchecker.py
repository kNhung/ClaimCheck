"""Minimal runner for the factchecker with simple wall-clock timing."""
from env_config import *  # Load environment variables
import os
from factchecker.factchecker import factcheck


if __name__ == "__main__":
    claim = "HEHe. Ông Putin nói Nga sẽ phản ứng mạnh nếu bị Tomahawk tấn công. Câu này đúng hay sao nè!"
    cutoff_date = "24-10-2025"  # DD-MM-YYYY
    # Prefer unified FACTCHECKER_MODEL_NAME but keep backward-compatible fallback
    model_name = os.getenv("FACTCHECKER_MODEL_NAME") or os.getenv("FACTCHECK_MODEL_NAME")    
    verdict, report_path = factcheck(claim, cutoff_date, max_actions=2, model_name=model_name)
    print("Verdict:", verdict)
    print("Report saved at:", report_path)
   