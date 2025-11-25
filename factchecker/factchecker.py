import re
import os
import json
import concurrent.futures
from .modules import planning, evidence_synthesis, evaluation, retriver_rav, llm
from .modules.token_tracker import get_global_tracker, reset_global_tracker
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
            llm.set_default_groq_model(model_name)
        self.model_name = model_name or llm.get_default_groq_model()
        report_writer.init_report(claim, identifier)
        self.report_path = report_writer.REPORT_PATH
        print(f"Initialized report at: {self.report_path}")
        # Initialize the report dict for web use
        # Limit actions dict size to avoid memory bloat
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
        self._max_actions_in_memory = 5  # Limit actions stored in memory
        self.max_actions = max_actions
        
        # Cache for report content to avoid reading file multiple times
        # Limit cache size to avoid memory issues (max 50KB)
        self._report_cache = None
        self._report_cache_dirty = True  # Flag to indicate cache needs refresh
        self._max_cache_size = 50 * 1024  # 50KB limit
        
        # Reset token tracker for this fact-check session
        reset_global_tracker()

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

    def get_report(self, max_length=None):
        """Get report content, with optional truncation to avoid rate limits.
        
        Args:
            max_length: Maximum length in characters. If None, returns full report.
                       If specified, truncates from the end, keeping the beginning.
        """
        # Use cache if available and not dirty
        if self._report_cache is not None and not self._report_cache_dirty:
            report_content = self._report_cache
        else:
            report_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../reports', self.identifier, 'report.md'))
            try:
                with open(report_path, "r") as f:
                    report_content = f.read()
                # Only cache if size is reasonable to avoid memory issues
                if len(report_content.encode('utf-8')) <= self._max_cache_size:
                    self._report_cache = report_content
                    self._report_cache_dirty = False
                else:
                    # Too large, don't cache but mark as not dirty to avoid re-reading
                    self._report_cache = None
                    self._report_cache_dirty = False
            except Exception as e:
                return f"Error reading report: {e}"
        
        # Truncate if needed (keep beginning, remove end)
        if max_length and len(report_content) > max_length:
            # Try to truncate at a reasonable point (end of a section)
            truncated = report_content[:max_length]
            # Find last newline to avoid cutting in middle of line
            last_newline = truncated.rfind('\n')
            if last_newline > max_length * 0.9:  # If we can find a newline near the limit
                truncated = truncated[:last_newline]
            truncated += f"\n\n[... Report truncated to {max_length:,} chars to avoid rate limits ...]"
            return truncated
        
        return report_content
    
    def invalidate_report_cache(self):
        """Mark report cache as dirty so it will be refreshed on next get_report() call"""
        self._report_cache_dirty = True
    
    def clear_cache(self):
        """Clear report cache to free memory"""
        self._report_cache = None
        self._report_cache_dirty = True
        import gc
        gc.collect()
    
    def cleanup_memory(self):
        """Clean up memory by clearing caches and limiting report dict size"""
        # Clear report cache
        self.clear_cache()
        
        # Limit actions dict size (keep only recent ones)
        if len(self.report["actions"]) > self._max_actions_in_memory:
            # Keep only the most recent actions
            action_items = list(self.report["actions"].items())
            self.report["actions"] = dict(action_items[-self._max_actions_in_memory:])
        
        # Limit reasoning list size
        if len(self.report["reasoning"]) > 10:
            self.report["reasoning"] = self.report["reasoning"][-10:]
        
        import gc
        gc.collect()

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
                        # Invalidate cache
                        self.invalidate_report_cache()
                        # Don't save JSON after every summary - too frequent

                    # Limit max_workers to avoid too many threads (each thread uses memory)
                    max_workers = min(3, len(urls))  # Max 3 concurrent workers
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as result_executor:
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
        

        # Limit max_workers to avoid memory issues
        max_workers = min(3, len(action_lines))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(self.process_action_line, action_lines))

        # Save report JSON after batch of actions
        self.save_report_json()

        iterations = 0
        seen_action_lines = set(action_lines)
        while iterations <= 2:
            # Get report with truncation for synthesis (keep it reasonable)
            # Estimate: ~2000 chars per evidence summary, so limit to ~8000 chars for synthesis
            record = self.get_report(max_length=8000)
            reasoning = evidence_synthesis.develop(record=record)

            print(f"Developed reasoning:\n{reasoning}")
            report_writer.append_reasoning(reasoning)

            self.report["reasoning"].append(reasoning)
            self.invalidate_report_cache()  # Cache is now stale
            # Save JSON after reasoning, but not too frequently
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

            # Limit max_workers to avoid memory issues
            max_workers = min(3, len(reasoning_action_lines))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                list(executor.map(self.process_action_line, reasoning_action_lines))
            
            # Save after processing reasoning actions
            self.save_report_json()

            iterations += 1

        allowed_verdicts = {"Supported", "Refuted", "Not Enough Evidence"}
        max_judge_tries = 3
        judge_tries = 0
        pred_verdict = ''
        rules = RULES_PROMPT
        
        # For judge, we need to truncate more aggressively to avoid rate limits
        # Different models have different limits, but 6000 chars is a safe estimate
        # We'll try progressively smaller if we hit rate limits
        max_record_lengths = [6000, 4000, 3000]  # Try progressively smaller
        
        while judge_tries < max_judge_tries:
            # Use truncated record to avoid rate limits
            max_len = max_record_lengths[min(judge_tries, len(max_record_lengths) - 1)]
            record = self.get_report(max_length=max_len)
            
            try:
                verdict = evaluation.judge(
                    record=record,
                    decision_options="Supported|Refuted|Not Enough Evidence",
                    rules=rules,
                    think=True  # Enable chain-of-thought reasoning
                )
            except RuntimeError as e:
                # If it's a rate limit error, try with smaller record
                error_str = str(e).lower()
                if '413' in str(e) or 'rate_limit' in error_str or 'too large' in error_str or 'tpm' in error_str:
                    print(f"⚠️  Rate limit hit (request too large), trying with smaller record ({max_len:,} → {max_record_lengths[min(judge_tries + 1, len(max_record_lengths) - 1)]:,} chars)")
                    judge_tries += 1
                    if judge_tries < max_judge_tries:
                        continue
                    else:
                        print("❌ Max retries reached. Consider using a model with higher rate limits or truncating record further.")
                        raise
                else:
                    # Other runtime errors - re-raise
                    raise
            except Exception as e:
                # Other exceptions - re-raise
                raise
            
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
        
        # Get and save token usage summary
        tracker = get_global_tracker()
        token_summary = tracker.get_summary()
        self.report["token_usage"] = token_summary
        
        # Print token usage summary
        print("\n" + "="*60)
        print("TOKEN USAGE SUMMARY")
        print("="*60)
        print(f"Total Requests: {token_summary['request_count']}")
        print(f"Total Tokens: {token_summary['total_tokens']:,}")
        print(f"  - Prompt Tokens: {token_summary['total_prompt_tokens']:,}")
        print(f"  - Completion Tokens: {token_summary['total_completion_tokens']:,}")
        print(f"Total Cost: ${token_summary['total_cost_usd']:.6f} USD")
        if token_summary['usage_by_model']:
            print("\nUsage by Model:")
            for model, usage in token_summary['usage_by_model'].items():
                print(f"  {model}:")
                print(f"    Tokens: {usage['total_tokens']:,} (prompt: {usage['prompt_tokens']:,}, completion: {usage['completion_tokens']:,})")
                print(f"    Cost: ${usage['cost_usd']:.6f} USD")
        print("="*60 + "\n")
        
        self.save_report_json()
        
        # Clean up memory before returning
        self.cleanup_memory()

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
