
from factchecker.factchecker import factcheck
from datetime import datetime, timedelta, timezone

VN_TIMEZONE = timezone(timedelta(hours=7))
now_vn = datetime.now(VN_TIMEZONE)
date =  now_vn.strftime("%d-%m-%Y")

claim = "Thủ tướng chính phủ Việt Nam là Phạm Minh Chính."

verdict, report_path = factcheck(claim, date)

print(f"Verdict: {verdict}")
print(f"Report saved to: {report_path}")