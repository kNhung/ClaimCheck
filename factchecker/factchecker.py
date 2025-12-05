import re
import os
import json
import time
import concurrent.futures
from threading import Lock
from contextlib import contextmanager
from urllib.parse import urlparse
from .modules import planning, evidence_synthesis, evaluation, retriver_rav, llm, claim_detection
from .tools import web_search, web_scraper
from .report import report_writer
import fcntl

RULES_PROMPT = """
Supported
- Dùng khi có bằng chứng rõ ràng, trực tiếp và đáng tin cậy ỦNG HỘ yêu cầu.
- Nếu yêu cầu có nhiều khía cạnh, TẤT CẢ các khía cạnh phải được ỦNG HỘ để chọn phán quyết này.

Refuted
- Dùng khi có bằng chứng rõ ràng BÁC BỎ hoặc MÂU THUẪN trực tiếp với yêu cầu.
- Nếu yêu cầu có nhiều khía cạnh, dù chỉ 1 khía cạnh bị BÁC BỎ trong khi các khía cạnh còn lại được ủng hộ cũng đủ để chọn phán quyết này.

Not Enough Evidence
- Dùng khi KHÔNG ĐỦ bằng chứng để xác nhận hoặc bác bỏ yêu cầu.
- Cũng dùng nếu yêu cầu quá MƠ HỒ hoặc không thể kiểm chứng bằng dữ liệu hiện có.
- Nếu yêu cầu có nhiều khía cạnh, chỉ cần 1 khía cạnh không đủ bằng chứng để chọn phán quyết này.
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

class TimerTracker:
    def __init__(self):
        self.records = []

    @contextmanager
    def track(self, label: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.records.append({"label": label, "duration": duration})
            print(f"[TIMING] {label}: {duration:.2f}s")


class FactChecker:
    def __init__(self, claim, date, identifier=None, multimodal=False, image_path=None, max_actions=3, model_name=None):
        self.claim = claim
        self.date = date
        self.multimodal = multimodal if not (multimodal and image_path is None) else False
        self.image_path = image_path
        if identifier is None:
            from datetime import datetime
            identifier = datetime.now().strftime("%m%d%Y%H%M%S")
        self.identifier = identifier
        if model_name:
            llm.set_default_ollama_model(model_name)
        self.model_name = llm.get_default_ollama_model()
        report_writer.init_report(claim, identifier)
        self.report_path = report_writer.REPORT_PATH
        print(f"Initialized report at: {self.report_path}")
        # Initialize the report dict for web use
        self.report = {
            "claim": self.claim,
            "date": self.date,
            "identifier": self.identifier,
            "model": self.model_name,
            "actions": {},
            "action_needed": [],
            "verdict": None,
            "justification": None,
            "report_path": self.report_path,
            "timings": []
        }
        self.max_actions = max_actions
        self._report_lock = Lock()
        max_workers = min(8, (os.cpu_count() or 4))
        self._result_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._unsupported_domains = {"youtube.com", "youtu.be", "twitter.com", "x.com"}
        self._timers = TimerTracker()

        # Save initial JSON report
        self.save_report_json()

    def save_report_json(self):
        """Save the report dictionary as report.json in the report_path folder"""
        try:
            json_path = os.path.join(os.path.dirname(self.report_path), 'report.json')
            with self._report_lock:
                serialized = json.dumps(self.report, indent=2)
                self.report["timings"] = list(self._timers.records)
            with open(json_path, 'w') as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(serialized)
                fcntl.flock(f, fcntl.LOCK_UN)
            print(f"Report JSON saved to: {json_path}")
        except Exception as e:
            print(f"Error saving report JSON: {e}")

    def get_report(self):
        report_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../reports', self.identifier, 'report.md'))
        try:
            with open(report_path, "r") as f:
                return f.read()
        except Exception as e:
            return f"Error reading report: {e}"

    def get_trimmed_record(self, max_chars=4000):
        record = self.get_report()
        if not record or not max_chars:
            return record
        if len(record) <= max_chars:
            return record
        return record[-max_chars:]

    def process_action_line(self, line):
        try:
            m = re.match(r'(\w+)_search\("([^"]+)"\)', line, re.IGNORECASE)
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

                if action.lower() == 'web':
                    with self._report_lock:
                        self.report["actions"][identifier] = action_entry
                    with self._timers.track(f"web_search:{query}"):
                        urls, snippets = web_search.web_search(query, self.date, top_k=5)

                    # Default with snippets from web_search
                    results_payload = {url: {"snippet": snippet, 'url': url, 'summary': None} for url, snippet in zip(urls, snippets)}
                    with self._report_lock:
                        self.report["actions"][identifier]["results"] = results_payload

                    def process_result(result):
                        domain = urlparse(result).netloc.lower()
                        if any(blocked in domain for blocked in self._unsupported_domains):
                            print(f"Skipping unsupported domain: {result}")
                            return None

                        with self._timers.track(f"scrape:{domain}"):
                            scraped_content = web_scraper.scrape_url_content(result)
                        if not scraped_content:
                            print(f"Skipping empty scrape for: {result}")
                            return None

                        with self._timers.track(f"evidence_rank:{domain}"):
                            summary = retriver_rav.get_top_evidence(self.claim, scraped_content)

                        if "NONE" in summary:
                            print(f"Skipping summary for evidence: {result}")
                            return None

                        print(f"Web search result: {result}, Summmary: {summary}")
                        report_writer.append_raw(f"web_search('{query}') results: {result}")
                        report_writer.append_evidence(f"web_search('{query}') Summary: {summary}")

                        with self._report_lock:
                            if result in self.report["actions"][identifier]["results"]:
                                self.report["actions"][identifier]["results"][result]["summary"] = summary

                    list(self._result_executor.map(process_result, urls))
                    self.save_report_json()
                else:
                    return

        except Exception as e:
            print(f"Error processing action line '{line}': {e}")

    def run(self):
        try:
            with self._timers.track("factcheck_run"):
                with self._timers.track("claim_filtering"):
                    self.claim = claim_detection.claim_filter(self.claim)
                    print(f"Filtered claim: {self.claim}")

                with self._timers.track("planning"):
                    queries = planning.plan(self.claim, think=False)
                
                queries_lines = [x.strip() for x in queries.split('\n') if x.strip()]
                action_lines = ["web_search(\"" + line + "\")" for line in queries_lines if line]

                report_writer.append_iteration_actions(1, '\n'.join(action_lines))
                print(f"Proposed actions for claim '{self.claim}':\n{action_lines}")

                print(f"Total action lines: {len(action_lines)}")
                print(f"Max actions allowed: {self.max_actions}")

                if action_lines and len(action_lines) > self.max_actions:
                    print(f"Limiting actions to the first {self.max_actions} lines.")
                    action_lines = action_lines[:self.max_actions]

                print(f"Processing action lines: {action_lines}")

                with self._timers.track("initial_action_execution"):
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        list(executor.map(self.process_action_line, action_lines))

                self.save_report_json()

                iterations = 0
                seen_action_lines = set(action_lines)
                action_needed_conclusion = None
                while iterations <= 2:
                    with self._timers.track(f"action_needed_iter_{iterations+1}"):
                        # Use smaller trimmed record for faster processing
                        action_needed = evidence_synthesis.develop(record=self.get_trimmed_record(max_chars=3000), think=False)

                    print(f"Developed action_needed:\n{action_needed}")
                    report_writer.append_action_needed(action_needed)

                    with self._report_lock:
                        self.report["action_needed"].append(action_needed)
                    self.save_report_json()

                    # Check if action_needed already concluded we have enough evidence
                    action_needed_lower = action_needed.lower()
                    if 'đã đủ bằng chứng' in action_needed_lower or 'đủ bằng chứng' in action_needed_lower:
                        action_needed_conclusion = "SUFFICIENT"
                        print("[action_needed] Early exit: Sufficient evidence found")
                        break

                    action_needed_action_lines = [x.strip() for x in action_needed.split('\n')]
                    # Extract actions from format "TÌM KIẾM: <query>" (from dev-knhung)
                    extracted_actions = []
                    for line in action_needed_action_lines:
                        if line.lower() == 'none':
                            extracted_actions.append('NONE')
                        elif 'TÌM KIẾM:' in line:
                            # Find the position of "TÌM KIẾM:" and extract query after it
                            idx = line.find('TÌM KIẾM:')
                            if idx != -1:
                                query = line[idx + len('TÌM KIẾM:'):].strip()
                                extracted_actions.append("web_search(\"" + query + "\")")
                    
                    # Also check for standard action format (from HEAD)
                    if not extracted_actions:
                        action_needed_action_lines = [
                            line for line in action_needed_action_lines if re.match(r'((\w+)_search\("([^"]+)"\)|NONE)', line, re.IGNORECASE)
                        ]
                    else:
                        action_needed_action_lines = extracted_actions

                    print(f"Extracted action_needed action lines: {action_needed_action_lines}")

                    if not action_needed_action_lines or (
                        len(action_needed_action_lines) == 1 and action_needed_action_lines[0].strip().lower() == 'none'
                    ):
                        break

                    if any(line in seen_action_lines for line in action_needed_action_lines):
                        print('Duplicate action line detected. Stopping iterations.')
                        break

                    seen_action_lines.update(action_needed_action_lines)

                    with self._timers.track(f"action_needed_action_exec_{iterations+1}"):
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            list(executor.map(self.process_action_line, action_needed_action_lines))

                    iterations += 1

                allowed_verdicts = {"Supported", "Refuted", "Not Enough Evidence"}
                max_judge_tries = 3
                judge_tries = 0
                pred_verdict = ''
                rules = RULES_PROMPT
                verdict = None
                
                # If action_needed already concluded sufficient evidence, try to extract verdict from action_needed
                if action_needed_conclusion == "SUFFICIENT":
                    # Try to extract verdict directly from action_needed to skip expensive judge call
                    action_needed_text = self.report["action_needed"][-1] if self.report["action_needed"] else ""
                    if 'supported' in action_needed_text.lower() or 'hỗ trợ' in action_needed_text.lower():
                        pred_verdict = "Supported"
                        verdict = f"### Justification:\n{action_needed_text}\n\n### Verdict:\n`Supported`"
                        print("[JUDGE] Skipped judge call, using action_needed conclusion: Supported")
                
                # If we don't have a verdict yet, call judge
                if not verdict:
                    while judge_tries < max_judge_tries:
                        with self._timers.track(f"judge_try_{judge_tries+1}"):
                            verdict = evaluation.judge(
                                record=self.get_trimmed_record(max_chars=3000),
                                decision_options="Supported|Refuted|Not Enough Evidence",
                                rules=rules,
                                think=False
                            )
                        print(f"Judged verdict (try {judge_tries+1}):\n{verdict}")
                        extracted_verdict = re.search(r'`(.*?)`', verdict, re.DOTALL)
                        pred_verdict = extracted_verdict.group(1).strip() if extracted_verdict else ''

                        if not extracted_verdict:
                            extracted_verdict = re.search(r'\*\*(.*?)\*\*', verdict, re.DOTALL)
                            if extracted_verdict:
                                pred_verdict = extracted_verdict.group(1).strip()

                        vi_to_en = {
                            'có căn cứ': 'Supported',
                            'được hỗ trợ': 'Supported',
                            'được chứng minh': 'Supported',
                            'bị bác bỏ': 'Refuted',
                            'sai lệch': 'Refuted',
                            'không đủ bằng chứng': 'Not Enough Evidence',
                            'chưa đủ bằng chứng': 'Not Enough Evidence',
                            'thiếu chứng cứ': 'Not Enough Evidence'
                        }

                        en_normalize = {
                            'support': 'Supported',
                            'supported': 'Supported',
                            'refute': 'Refuted',
                            'refuted': 'Refuted',
                            'not enough': 'Not Enough Evidence',
                            'not enough evidence': 'Not Enough Evidence',
                            'insufficient evidence': 'Not Enough Evidence',
                            'insufficient': 'Not Enough Evidence'
                        }

                        if pred_verdict.lower() in vi_to_en:
                            pred_verdict = vi_to_en[pred_verdict.lower()]
                        elif pred_verdict.lower() in en_normalize:
                            pred_verdict = en_normalize[pred_verdict.lower()]

                        if pred_verdict in allowed_verdicts:
                            break
                        judge_tries += 1

                if pred_verdict not in allowed_verdicts:
                    print("No decision options found in verdict, using extract_verdict from judge.py...")
                    with self._timers.track("extract_verdict_fallback"):
                        try:
                            extracted = evaluation.extract_verdict(
                                conclusion=verdict,
                                decision_options="Supported|Refuted|Not Enough Evidence",
                                rules=rules
                            )
                            extracted_verdict = re.search(r'`(.*?)`', extracted, re.DOTALL)
                            pred_verdict = extracted_verdict.group(1).strip() if extracted_verdict else extracted.strip()
                            print(f"extract_verdict returned: {pred_verdict}")
                        except Exception as e:
                            print(f"extract_verdict failed: {e}")
                            pred_verdict = "INVALID VERDICT"
                            print("No decision options found in verdict, defaulting to 'INVALID VERDICT'.")

                report_writer.append_verdict(pred_verdict)
                report_writer.append_justification(verdict)
                with self._report_lock:
                    self.report["justification"] = verdict
                    self.report["verdict"] = pred_verdict
                self.save_report_json()

                return pred_verdict, report_writer.REPORT_PATH
        finally:
            self._result_executor.shutdown(wait=True)

# For backward compatibility, provide a function interface

def factcheck(claim, date, identifier=None, multimodal=False, image_path=None, max_actions=2, expected_label=None, model_name=None):
    checker = FactChecker(claim, date, identifier, multimodal, image_path, max_actions, model_name=model_name)
    verdict, report_path = checker.run()
    return verdict, report_path
