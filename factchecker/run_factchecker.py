"""Minimal runner for the factchecker with simple wall-clock timing."""
import os
from .factchecker import factcheck
from time import perf_counter
from datetime import datetime


if __name__ == "__main__":
    claim = "Ông Putin nói Nga sẽ phản ứng mạnh nếu bị Tomahawk tấn công"
    cutoff_date = "24-10-2025"  # DD-MM-YYYY
    model_name = os.getenv("FACTCHECK_MODEL_NAME")

    start_ts = datetime.now()
    start = perf_counter()
    print(f"[START] {start_ts.isoformat(timespec='seconds')} | Claim: {claim}")

    verdict, report_path = factcheck(claim, cutoff_date, max_actions=2, model_name=model_name)

    end = perf_counter()
    end_ts = datetime.now()
    elapsed_s = end - start

    print("Verdict:", verdict)
    print("Report saved at:", report_path)
    print(f"[END]   {end_ts.isoformat(timespec='seconds')} | Elapsed: {elapsed_s:.2f}s (~{elapsed_s/60:.2f} min)")