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
- Có bằng chứng rõ ràng, trực tiếp và đáng tin cậy ủng hộ yêu cầu.
- Dùng khi đa số bằng chứng độc lập chỉ ra rằng yêu cầu là đúng, dù một vài chi tiết nhỏ chưa được xác nhận.

Refuted
- Có bằng chứng đáng tin cậy bác bỏ hoặc mâu thuẫn trực tiếp với yêu cầu.
- Dùng khi phần chính của yêu cầu bị chứng minh sai hoặc không đúng sự thật.
- KHÔNG dùng nếu chỉ thiếu bằng chứng — chỉ dùng khi có bằng chứng phản đối rõ ràng.

Not Enough Evidence
- Dùng khi không có đủ bằng chứng để xác nhận hoặc bác bỏ yêu cầu.
- Cũng dùng nếu yêu cầu quá mơ hồ hoặc không thể kiểm chứng bằng dữ liệu hiện có.
- KHÔNG dùng nếu đã có bằng chứng rõ ràng ủng hộ hoặc phản đối.
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
    def __init__(self, claim, date, identifier=None, multimodal=False, image_path=None, max_actions=2, model_name=None):
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
                    "action": f"{action}_search",
                    "query": query,
                    "results": None
                }
                identifier = f'{action}: {query}'
                if identifier in self.report["actions"]:
                    print(f"Skipping duplicate action: {identifier}")
                    return

                if action.lower() == 'web':
                    self.report["actions"][identifier] = action_entry
                    results = web_search.web_search(query, self.date, top_k=3)
                    self.report["actions"][identifier]["results"] = {
                        r["link"]: {
                            "snippet": r.get("snippet", ""), 
                            "url": r["link"], 
                            "title": r["title"],
                            "summary": None
                        }
                        for r in results
                    }
                    self.save_report_json()
                    
                    urls = [r["link"] for r in results]

                    def process_result(result):
                        scraped_content = web_scraper.scrape_url_content(result)
                        summary = retriver_rav.get_top_evidence(self.claim, scraped_content)

                        if "NONE" in summary:
                            print(f"Skipping summary for evidence: {result}")
                            return None

                        print(f"Web search result: {result}. \nSummary: {summary}")
                        report_writer.append_raw(f"web_search('{query}') results: {result}")
                        report_writer.append_evidence(f"web_search('{query}') summary: {summary}")

                        self.report["actions"][identifier]["results"][result]["summary"] = summary
                        self.save_report_json()

                    with concurrent.futures.ThreadPoolExecutor() as result_executor:
                        processed = list(result_executor.map(process_result, urls))
                else:
                    return

        except Exception as e:
            print(f"Error processing action line '{line}': {e}")

    def run(self):
        if self.multimodal == True:
            actions = "All"
        else:
            actions = ["web_search"]#, "image_search"]
            
        actions = planning.plan(self.claim, record=self.get_report(), actions=actions)
        report_writer.append_iteration_actions(1, actions)
        print(f"Proposed actions for claim '{self.claim}':\n{actions}")

        action_lines = [x.strip() for x in actions.split('\n')]
        print(f"Extracted action lines: {action_lines}")

        # Extract all web_search actions from the entire actions string
        matches = re.findall(r'(\w+)_search\("([^"]+)"\)', actions, re.IGNORECASE)
        action_lines = [f'{action}_search("{query}")' for action, query in matches]
        print(f"Filtered valid action lines: {action_lines}")
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
            reasoning_action_lines = [line for line in reasoning_action_lines if re.match(r'((\w+)_search\("([^"]+)"\)|NONE)', line, re.IGNORECASE)]

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

        allowed_verdicts = {"Supported", "Refuted", "Not Enough Evidence"}
        max_judge_tries = 3
        judge_tries = 0
        pred_verdict = ''
        rules = RULES_PROMPT
        while judge_tries < max_judge_tries:
            verdict = evaluation.judge(
                record=self.get_report(),
                decision_options="Supported|Refuted|Not Enough Evidence",
                rules=rules,
                think=None  # Replace with think_judge if defined
            )
            print(f"Judged verdict (try {judge_tries+1}):\n{verdict}")
            extracted_verdict = re.search(r'`(.*?)`', verdict, re.DOTALL)
            pred_verdict = extracted_verdict.group(1).strip() if extracted_verdict else ''

            if not extracted_verdict:
                # Try to extract from ** **
                extracted_verdict = re.search(r'\*\*(.*?)\*\*', verdict, re.DOTALL)
                if extracted_verdict:
                    pred_verdict = extracted_verdict.group(1).strip()

            vi_to_en = {
                "có căn cứ": "Supported",
                "được hỗ trợ": "Supported",
                "được chứng minh": "Supported",
                "bị bác bỏ": "Refuted",
                "sai lệch": "Refuted",
                "không đủ bằng chứng": "Not Enough Evidence",
                "chưa đủ bằng chứng": "Not Enough Evidence",
                "thiếu chứng cứ": "Not Enough Evidence"
            }

            en_normalize = {
                "support": "Supported",
                "supported": "Supported",
                "refute": "Refuted",
                "refuted": "Refuted",
                "not enough": "Not Enough Evidence",
                "not enough evidence": "Not Enough Evidence",
                "insufficient evidence": "Not Enough Evidence",
                "insufficient": "Not Enough Evidence"
            }

            if pred_verdict.lower() in vi_to_en:
                pred_verdict = vi_to_en[pred_verdict.lower()]
            elif pred_verdict.lower() in en_normalize:
                pred_verdict = en_normalize[pred_verdict.lower()]

            if pred_verdict in allowed_verdicts:
                break
            judge_tries += 1

        # If extraction failed after max tries, find the most frequent decision option in the verdict text
        if pred_verdict not in allowed_verdicts:
            # print("Original extraction failed, falling back to most frequent decision option...")
            # option_counts = {}
            # for option in allowed_verdicts:
            #     # Count occurrences of each decision option in the verdict text
            #     count = verdict.lower().count(option.lower())
            #     if count > 0:
            #         option_counts[option] = count

            # if option_counts:
            #     # Use the most frequent option
            #     pred_verdict = max(option_counts, key=option_counts.get)
            #     print(f"Fallback verdict selected: {pred_verdict} (appeared {option_counts[pred_verdict]} times)")
            # else:
        
            # If no options found, use extract_verdict from judge.py
            print("No decision options found in verdict, using extract_verdict from judge.py...")
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

        report_writer.append_verdict(verdict)
        self.report["judged_verdict"] = verdict
        self.report["verdict"] = pred_verdict
        self.save_report_json()

        return pred_verdict, report_writer.REPORT_PATH

# For backward compatibility, provide a function interface

def factcheck(claim, date, identifier=None, multimodal=False, image_path=None, max_actions=2, expected_label=None, model_name=None):
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
