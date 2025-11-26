import re
import os
import json
import concurrent.futures
from .modules import planning, evidence_synthesis, evaluation, retriver_rav, llm
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

class FactChecker:
    def __init__(self, claim, date, identifier=None, multimodal=False, image_path=None, max_actions=1, model_name=None):
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
            "reasoning": [],
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
                fcntl.flock(f, fcntl.LOCK_EX)
                json.dump(self.report, f, indent=2)
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
                    self.report["actions"][identifier] = action_entry
                    urls, snippets = web_search.web_search(query, self.date, top_k=3)

                    # Default with snippets from web_search
                    self.report["actions"][identifier]["results"] = {url: {"snippet": snippet, 'url':url, 'summary': None} for url, snippet in zip(urls, snippets)}
                    self.save_report_json()

                    def process_result(result):
                        scraped_content = web_scraper.scrape_url_content(result)
                        summary = retriver_rav.get_top_evidence(self.claim, scraped_content)

                        if "NONE" in summary:
                            print(f"Skipping summary for evidence: {result}")
                            return None

                        print(f"Web search result: {result}, Summmary: {summary}")
                        report_writer.append_raw(f"web_search('{query}') results: {result}")
                        report_writer.append_evidence(f"web_search('{query}'), Summary: {summary}")

                        self.report["actions"][identifier]["results"][result]["summary"] = summary
                        self.save_report_json()

                    with concurrent.futures.ThreadPoolExecutor() as result_executor:
                        processed = list(result_executor.map(process_result, urls))
                else:
                    return

        except Exception as e:
            print(f"Error processing action line '{line}': {e}")

    def run(self):
        queries = planning.plan(self.claim)
        queries_lines = [x.strip() for x in queries.split('\n')]
        action_lines = ["web_search(\"" + line + "\")" for line in queries_lines]
        

        report_writer.append_iteration_actions(1, action_lines)
        print(f"Proposed actions for claim '{self.claim}':\n{action_lines}")

        print(f"Total action lines: {len(action_lines)}")
        print(f"Max actions allowed: {self.max_actions}")

        if action_lines and len(action_lines) > self.max_actions:
            print(f"Limiting actions to the first {self.max_actions} lines.")
            action_lines = action_lines[:self.max_actions]
        
        print(f"Processing action lines: {action_lines}")

        # block multithreading until everything above is done
        

        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(executor.map(self.process_action_line, action_lines))

        # Save the initial report after planning
        self.save_report_json()

        iterations = 0
        seen_action_lines = set(action_lines)
        while iterations <= 2:
            reasoning = evidence_synthesis.develop(record=self.get_report())

            print(f"Developed reasoning:\n{reasoning}")
            report_writer.append_reasoning(reasoning)

            self.report["reasoning"].append(reasoning)
            self.save_report_json()
            reasoning_action_lines = [x.strip() for x in reasoning.split('\n')]
            # Extract actions from format "TÌM KIẾM: <query>"
            extracted_actions = []
            for line in reasoning_action_lines:
                if line.lower() == 'none':
                    extracted_actions.append('NONE')
                elif 'TÌM KIẾM:' in line:
                    # Find the position of "TÌM KIẾM:" and extract query after it
                    idx = line.find('TÌM KIẾM:')
                    if idx != -1:
                        query = line[idx + len('TÌM KIẾM:'):].strip()
                        extracted_actions.append("web_search(\"" + query + "\")")
            reasoning_action_lines = extracted_actions

            print(f"Extracted reasoning action lines: {reasoning_action_lines}")

            if not reasoning_action_lines or (len(reasoning_action_lines) == 1 and reasoning_action_lines[0].strip().lower() == 'none'):
                break

            if any(line in seen_action_lines for line in reasoning_action_lines):
                print("Duplicate action line detected. Stopping iterations.")
                break

            seen_action_lines.update(reasoning_action_lines)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                list(executor.map(self.process_action_line, reasoning_action_lines))

            iterations += 1

        rules = RULES_PROMPT
       
        # Generate justification and verdict
        verdict = evaluation.judge(
            record=self.get_report(),
            decision_options="Supported|Refuted|Not Enough Evidence",
            rules=rules,
            think=True  # Replace with think_judge if defined
        )

        print("Justification:\n", verdict)
        report_writer.append_justification(verdict)
        self.report["justification"] = verdict

        # Extract verdict from the response
        extracted_verdict = evaluation.extract_verdict(verdict, 
            decision_options="Supported|Refuted|Not Enough Evidence",
            rules=rules
        )

        report_writer.append_verdict(extracted_verdict)
        self.report["verdict"] = extracted_verdict
        self.save_report_json()

        return extracted_verdict, report_writer.REPORT_PATH

# For backward compatibility, provide a function interface

def factcheck(claim, date, identifier=None, multimodal=False, image_path=None, max_actions=1, expected_label=None, model_name=None):
    checker = FactChecker(claim, date, identifier, multimodal, image_path, max_actions, model_name=model_name)
    verdict, report_path = checker.run()
    
    # try:
    #     # Get base directory from report path
    #     base_dir = os.path.dirname(os.path.dirname(report_path))
    #     csv_path = os.path.join(base_dir, 'detailed_results.csv')
        
    #     # Get content from report
    #     evidence, reasoning, verdict_text, justification = report_writer.get_report_content()
        
    #     # Convert numeric expected_label to text if needed
    #     if expected_label is not None and isinstance(expected_label, (int, float)):
    #         expected_label = LABEL_MAP.get(int(expected_label))
            
    #     # Convert verdict to numeric for metrics calculation
    #     numeric_verdict = LABEL_MAP.get(verdict)
        
    #     # Write to CSV with both text and numeric verdicts
    #     report_writer.write_detailed_csv(
    #         claim=claim,
    #         date=date,
    #         evidence=evidence,
    #         reasoning=reasoning,
    #         verdict=verdict,
    #         numeric_verdict=numeric_verdict,
    #         justification=justification,
    #         report_path=report_path,
    #         csv_path=csv_path,
    #         expected_label=expected_label
    #     )
    #     print(f"Detailed results written to: {csv_path}")
    # except Exception as e:
    #     print(f"Error writing detailed results to CSV: {e}")
    
    return verdict, report_path
