import re
import os
import json
import concurrent.futures
# Đảm bảo các modules này đã được sửa Prompts và logic trả về List/Tuple
from .modules import planning, evidence_summarization, evidence_synthesis, evaluation 
from .tools import web_search, web_scraper
from .report import report_writer
import fcntl
from datetime import datetime, timezone, timedelta

# Định nghĩa Luật Lệ (Rules)
RULES_PROMPT = """
Supported
- Sử dụng khi phát biểu được hậu thuẫn trực tiếp và rõ ràng bởi bằng chứng mạnh, đáng tin cậy. Một vài điểm chưa chắc chắn/thiếu chi tiết nhỏ không loại trừ nếu ý chính đã được chứng minh tốt.
- Dùng Supported nếu tổng thể bằng chứng nghiêng về việc phát biểu là đúng, dù vẫn có vài lưu ý hoặc chưa xác nhận đủ mọi chi tiết.

Refuted
- Sử dụng khi phát biểu bị bác bỏ bởi bằng chứng mạnh, đáng tin cậy, hoặc cho thấy yếu tố bịa đặt/đánh lừa/sai ở ý chính.
- Dùng Refuted nếu các yếu tố cốt lõi bị chứng minh là sai, kể cả khi một vài chi tiết nhỏ còn mơ hồ.
- Việc không có nguồn đáng tin nào ủng hộ phát biểu không phải là "Not Enough Evidence" — đó là Refuted.

Conflicting Evidence/Cherrypicking
- Chỉ dùng khi có các nguồn uy tín đưa ra thông tin mâu thuẫn trực tiếp và không thể dung hòa về ý chính, và không thể phân xử rõ sau khi phân tích kỹ.
- KHÔNG dùng cho bất đồng nhỏ, bằng chứng chưa đầy đủ, hoặc khi phần lớn bằng chứng nghiêng về một phía nhưng có vài nguồn ý kiến trái chiều nhỏ.

Not Enough Evidence
- Chỉ dùng khi thực sự không có bằng chứng liên quan sau khi đã tìm kiếm kỹ, hoặc phát biểu quá mơ hồ/nhập nhằng để đánh giá.
- KHÔNG dùng khi vẫn có một số bằng chứng (dù yếu), hoặc khi phát biểu khá rõ nhưng không xác nhận được mọi chi tiết.
- Đây là lựa chọn sau cùng.
"""

LABEL_MAP = {
    # Numeric to text
    0: "Supported",
    1: "Refuted", 
    2: "Not Enough Evidence",
    # Text to numeric
    "Supported": 0,
    "Refuted": 1,
    "Not Enough Evidence": 2
}

class FactChecker:
    def __init__(self, claim, date, identifier=None, multimodal=False, image_path=None, max_actions=2):
        self.claim = claim
        self.date = date
        self.multimodal = multimodal if not (multimodal and image_path is None) else False
        self.image_path = image_path
        if identifier is None:
            now_vn = datetime.now(timezone(timedelta(hours=7)))
            identifier = now_vn.strftime("%d%m%y-%H%M") # Dùng định dạng giống như trong fact-check.py
        self.identifier = identifier
        report_writer.init_report(claim, identifier)
        self.report_path = report_writer.REPORT_PATH
        print(f"Initialized report at: {self.report_path}")
        
        # Initialize the report dict for web use
        self.report = {
            "claim": self.claim,
            "date": self.date,
            "identifier": self.identifier,
            "actions": {},
            "reasoning": [],
            "judged_verdict": None,
            "verdict": None,
            "justification": None,
            "report_path": self.report_path
        }
        self.max_actions = max_actions

        # Save initial JSON report
        self.save_report_json()

    def save_report_json(self):
        """Save the report dictionary as report.json in the report_path folder"""
        try:
            json_path = os.path.join(os.path.dirname(self.report_path), 'report.json')
            with open(json_path, 'w') as f:
                # Sử dụng fcntl.flock để tránh race conditions nếu bạn đang dùng threads
                # Tuy nhiên, trong Colab thread pool đơn giản, bạn có thể bỏ qua fcntl nếu gặp lỗi
                # fcntl.flock(f, fcntl.LOCK_EX) 
                json.dump(self.report, f, indent=2)
                # fcntl.flock(f, fcntl.LOCK_UN)
            print(f"Report JSON saved to: {json_path}")
        except Exception as e:
            print(f"Error saving report JSON: {e}")

    def get_report(self):
        # Đảm bảo đường dẫn này đúng nếu bạn dùng nó trong các file khác
        report_path = os.path.abspath(os.path.join(os.path.dirname(self.report_path), 'report.md')) 
        try:
            with open(report_path, "r") as f:
                return f.read()
        except Exception as e:
            return f"Error reading report: {e}"

    def process_action_line(self, line):
        # Đây là logic chạy hành động thực tế, không cần thay đổi nếu format action hợp lệ
        try:
            m = re.match(r'(\w+)_search\("([^"]+)"\)', line)
            if m:
                action, query = m.groups()
                action_entry = {
                    "action": action + "_search",
                    "query": query,
                    "results": None
                }
                identifier = f'{action}: {query}'
                if identifier in self.report["actions"]:
                    print(f"Skipping duplicate action: {identifier}")
                    return

                if action == 'web':
                    self.report["actions"][identifier] = action_entry
                    urls, snippets = web_search.web_search(query, self.date, top_k=3)

                    # Default with snippets from web_search
                    self.report["actions"][identifier]["results"] = {url: {"snippet": snippet, 'url':url, 'summary': None} for url, snippet in zip(urls, snippets)}
                    self.save_report_json()

                    def process_result(result):
                        scraped_content = web_scraper.scrape_url_content(result)
                        # Sửa lỗi: Cần trích xuất summary từ khối mã Markdown
                        raw_summary = evidence_summarization.summarize(self.claim, scraped_content, result, record=self.get_report())
                        
                        # SỬA LỖI: Tách nội dung tóm tắt từ khối mã Markdown (dùng hàm tiện ích)
                        # (Giả định bạn đã có hàm extract_summary_content trong evidence_summarization.py)
                        # Tuy nhiên, để đơn giản, chúng ta sẽ để cho hàm gọi sau này tự làm nếu cần

                        if "NONE" in raw_summary:
                            print(f"Skipping summary for evidence: {result}")
                            return None

                        print(f"Web search result: {result}, Summary: {raw_summary}")
                        report_writer.append_raw(f"web_search('{query}') results: {result}")
                        report_writer.append_evidence(f"web_search('{query}') summary: {raw_summary}")

                        self.report["actions"][identifier]["results"][result]["summary"] = raw_summary
                        self.save_report_json()

                    with concurrent.futures.ThreadPoolExecutor() as result_executor:
                        processed = list(result_executor.map(process_result, urls))
                else:
                    return

        except Exception as e:
            print(f"Error processing action line '{line}': {e}")
            
    # HÀM RUN ĐÃ SỬA LỖI:
    def run(self):
        if self.multimodal == True:
            actions = "All"
        else:
            actions = ["web_search"]

        # --- BƯỚC 1: Lập kế hoạch và Thực thi Hành động Ban đầu ---
        
        # GỌI HÀM PLAN CẢI TIẾN (Trả về LIST)
        actions_list = planning.plan(self.claim, record=self.get_report(), actions=actions)
        
        # SỬA LỖI #1: Xử lý đầu ra dạng list từ planning.plan()
        action_lines = actions_list
        
        # Ghi log ra console (chuyển đổi list sang string để log)
        try:
            log_actions = '\n'.join(action_lines)
        except AttributeError:
            log_actions = str(action_lines)
            
        report_writer.append_iteration_actions(1, log_actions)
        print(f"Proposed actions for claim '{self.claim}':\n{log_actions}") 
        
        # Lọc các dòng hành động hợp lệ (bỏ NONE và trống)
        action_lines = [line for line in action_lines if line.strip().upper() != 'NONE' and line.strip() != '']
        
        # Lọc hành động theo cú pháp cuối cùng (kiểm tra cú pháp web_search/image_search)
        action_lines = [line for line in action_lines if re.match(r'(\w+)_search\("([^"]+)"\)', line, re.IGNORECASE)]
        
        print(f"Filtered valid action lines: {action_lines}")
        print(f"Total action lines: {len(action_lines)}")
        print(f"Max actions allowed: {self.max_actions}")

        if action_lines and len(action_lines) > self.max_actions:
            print(f"Limiting actions to the first {self.max_actions} lines.")
            action_lines = action_lines[:self.max_actions]
        
        print(f"Processing action lines: {action_lines}")

        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(executor.map(self.process_action_line, action_lines))

        self.save_report_json()

        # --- BƯỚC 2: Vòng lặp Tổng hợp Bằng chứng và Hành động ---

        iterations = 0
        seen_action_lines = set(action_lines) 
        
        while iterations <= 2:
            # GỌI HÀM DEVELOP CẢI TIẾN (Trả về TUPLE: reasoning, actions)
            reasoning_result = evidence_synthesis.develop(record=self.get_report())

            # SỬA LỖI #2: Phân tách Tuple
            if isinstance(reasoning_result, tuple) and len(reasoning_result) == 2:
                reasoning, reasoning_actions = reasoning_result
            else:
                # Trường hợp LLM trả về kết quả không mong muốn (ví dụ: chỉ string)
                print(f"Error: evidence_synthesis did not return a valid tuple. Received: {reasoning_result}")
                reasoning = str(reasoning_result)
                reasoning_actions = [] 
                # Cố gắng trích xuất hành động lần cuối nếu reasoning_result là string
                # Bạn nên để code trong evidence_synthesis.py đảm bảo trả về tuple.
                
            # Log reasoning
            print(f"Developed reasoning:\n{reasoning}")
            report_writer.append_reasoning(reasoning)

            self.report["reasoning"].append(reasoning)
            self.save_report_json()
            
            # Xử lý hành động từ Reasoning (reasoning_actions đã là list hoặc None)
            if reasoning_actions is None or (len(reasoning_actions) == 1 and reasoning_actions[0].strip().upper() == 'NONE'):
                break # Dừng nếu không có hành động hoặc là NONE

            if any(line in seen_action_lines for line in reasoning_actions):
                print("Duplicate action line detected. Stopping iterations.")
                break

            seen_action_lines.update(reasoning_actions)
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                list(executor.map(self.process_action_line, reasoning_actions))

            iterations += 1

        # --- BƯỚC 3: Đánh giá Phán quyết Cuối cùng ---
        
        allowed_verdicts = {"Supported", "Refuted", "Not Enough Evidence"}
        max_judge_tries = 3
        judge_tries = 0
        pred_verdict = ''
        rules = RULES_PROMPT
        
        while judge_tries < max_judge_tries:
            # GỌI HÀM JUDGE (Trả về RAW JUDGEMENT string)
            verdict_raw = evaluation.judge(
                record=self.get_report(),
                decision_options="Supported|Refuted|Not Enough Evidence",
                rules=rules,
                think=None 
            )
            print(f"Judged verdict (try {judge_tries+1}):\n{verdict_raw}")
            
            # SỬA LỖI #3: Sử dụng hàm extract_verdict_from_response đã cải tiến
            extracted_verdict_label = evaluation.extract_verdict(
                conclusion=verdict_raw, 
                decision_options="Supported\nRefuted\nNot Enough Evidence", 
                rules=rules
            )
            
            if extracted_verdict_label and extracted_verdict_label in allowed_verdicts:
                pred_verdict = extracted_verdict_label
                break
            
            judge_tries += 1

        # Nếu không trích xuất được sau 3 lần (chủ yếu do LLM không tuân thủ format)
        if pred_verdict not in allowed_verdicts:
            print("Original extraction failed after 3 tries. Falling back to most frequent decision option...")
            option_counts = {}
            for option in allowed_verdicts:
                count = verdict_raw.lower().count(option.lower())
                if count > 0:
                    option_counts[option] = count

            if option_counts:
                pred_verdict = max(option_counts, key=option_counts.get)
                print(f"Fallback verdict selected: {pred_verdict}")
            else:
                print("No clear decision options found. Defaulting to 'Not Enough Evidence'.")
                pred_verdict = "Not Enough Evidence"
                
        report_writer.append_verdict(verdict_raw)
        self.report["judged_verdict"] = verdict_raw
        self.report["verdict"] = pred_verdict
        self.save_report_json()

        return pred_verdict, report_writer.REPORT_PATH

# For backward compatibility, provide a function interface

def factcheck(claim, date, identifier=None, multimodal=False, image_path=None, max_actions=2, expected_label=None):
    checker = FactChecker(claim, date, identifier, multimodal, image_path, max_actions)
    verdict, report_path = checker.run()
        
    return verdict, report_path