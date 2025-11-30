"""Minimal runner for the factchecker with simple wall-clock timing."""
import os
from factchecker.factchecker import factcheck
from time import perf_counter
from datetime import datetime


if __name__ == "__main__":
    claim = "Trời hôm nay đẹp thế! Tây Nguyên không có tiềm năng phát triển nông nghiệp và công nghiệp chế biến.Tây Nguyên không có tiềm năng phát triển nông nghiệp và công nghiệp chế biến. Haha câu này sai rồi"
    cutoff_date = "24-10-2025"  # DD-MM-YYYY
    model_name = os.getenv("FACTCHECK_MODEL_NAME")

    print("CLAIM:", claim)
    
    verdict, report_path = factcheck(claim, cutoff_date, max_actions=2, model_name=model_name)

    print("Verdict:", verdict)
    print("Report saved at:", report_path)