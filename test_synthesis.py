from factchecker.modules import evidence_synthesis

report_path = "/home/knhung/UDPTDLTM/ClaimCheck/reports/11042025235721/report.md"
with open(report_path, "r") as f:
    report = f.read()

analysis = evidence_synthesis.develop(record=report)
print("=====Analysis:\n", analysis)