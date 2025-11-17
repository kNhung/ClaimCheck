from factchecker.modules import planning

claim = "Thủ tướng chính phủ Việt Nam hiện tại là ông Phạm Minh Chính"
plan = planning.plan(claim=claim)

print("=====Plan:\n", plan)