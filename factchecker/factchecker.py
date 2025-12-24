import re
import os
import json
import time
import concurrent.futures
from threading import Lock
from contextlib import contextmanager
from urllib.parse import urlparse
from .modules import planning, evaluation, retriver_rav, llm, claim_detection
from .tools import web_search, web_scraper
from .report import report_writer
import fcntl

MAX_ACTIONS = int(os.getenv("FACTCHECKER_MAX_ACTIONS", "2"))

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
    def __init__(self, claim, date, identifier=None, multimodal=False, image_path=None, max_actions=2, judge_model_name=None):
        self.claim = claim
        self.date = date
        self.multimodal = multimodal if not (multimodal and image_path is None) else False
        self.image_path = image_path
        if identifier is None:
            from datetime import datetime
            identifier = datetime.now().strftime("%Y%m%d%H%M%S")
        self.identifier = identifier
        self.judge_model_name = judge_model_name or llm.get_default_ollama_model()
        report_writer.init_report(claim, identifier)
        self.report_path = report_writer.REPORT_PATH
        print(f"Initialized report at: {self.report_path}")
        # Initialize the report dict for web use
        # Limit actions dict size to avoid memory bloat
        self.report = {
            "claim": self.claim,
            "date": self.date,
            "identifier": self.identifier,
            "judge_model": self.judge_model_name,
            "actions": {},
            "verdict": None,
            "justification": None,
            "report_path": self.report_path,
            "timings": []
        }
        self._max_actions_in_memory = 5  # Limit actions stored in memory
        self.max_actions = max_actions
        
        self._report_lock = Lock()
        max_workers = min(8, (os.cpu_count() or 4))
        self._result_executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._unsupported_domains = {"youtube.com", "youtu.be", "twitter.com", "x.com"}
        self._timers = TimerTracker()
        
        # Pre-load models before starting multi-threaded processing
        # This ensures models are loaded once and shared across threads
        retriver_rav.preload_models()
        evaluation.preload_models()
        # Pre-load accent restoration model
        from .preprocessing.preprocessing import preload_accent_model
        preload_accent_model()

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
        """
        Trim record while preserving the claim at the beginning and all evidence.
        Strategy: 
        1. Keep claim section
        2. Keep ALL evidence section (important for fact-checking)
        3. Keep recent actions/verdict from the end if space allows
        """
        record = self.get_report()
        if not record or not max_chars:
            return record
        if len(record) <= max_chars:
            return record
        
        # Find the claim section (first few lines with # Claim:)
        claim_match = re.search(r'^#\s*Claim:\s*(.+?)(?=\n##|\n###|$)', record, re.MULTILINE | re.DOTALL)
        if not claim_match:
            print("Warning: Could not find claim section in record, using end of record")
            return record[-max_chars:]
        
        # Reconstruct the full claim section including the header
        claim_content = claim_match.group(1).strip()
        claim_section = f"# Claim: {claim_content}\n"
        
        # Find Evidence section (keep ALL evidence - critical for fact-checking)
        evidence_match = re.search(r'(###\s*Evidence\s*\n\n.*?)(?=\n###|$)', record, re.DOTALL | re.IGNORECASE)
        evidence_section = evidence_match.group(1) if evidence_match else ""
        
        # Get the rest after evidence (actions, verdict, etc.)
        if evidence_match:
            rest_after_evidence = record[evidence_match.end():]
        else:
            rest_after_evidence = record[claim_match.end():]
        
        # Calculate space used
        used_chars = len(claim_section) + len(evidence_section)
        remaining_chars = max_chars - used_chars
        
        # CRITICAL: Evidence section is essential for fact-checking
        # If evidence section alone exceeds max_chars, we still keep it all
        # (better to have all evidence than truncate and lose important info)
        if len(evidence_section) > max_chars:
            # Evidence section is too long, but we MUST keep it all
            # Return claim + ALL evidence (ignore max_chars limit for evidence)
            # This ensures all evidence is available for judge
            print(f"Warning: Evidence section ({len(evidence_section)} chars) exceeds max_chars ({max_chars}). Keeping all evidence.")
            return claim_section + evidence_section
        
        if remaining_chars > 0:
            # Keep recent actions/verdict from the end if space allows
            if len(rest_after_evidence) <= remaining_chars:
                return claim_section + evidence_section + rest_after_evidence
            else:
                # Keep claim + ALL evidence + end of record (actions/verdict)
                return claim_section + evidence_section + rest_after_evidence[-remaining_chars:]
        else:
            # Claim + evidence fits exactly or slightly over
            # Truncate claim if necessary, but keep all evidence
            claim_max = max(100, max_chars - len(evidence_section))  # Keep at least 100 chars of claim
            return claim_section[:claim_max] + evidence_section

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
                    # Log web search step
                    web_search_logs = []
                    report_writer.log_step(
                        "Web Search - Input",
                        {"Query": query, "Date": self.date, "Top_k": 3},
                        web_search_logs
                    )
                    
                    with self._report_lock:
                        self.report["actions"][identifier] = action_entry
                    with self._timers.track(f"web_search:{query}"):
                        urls, snippets = web_search.web_search(query, self.date, top_k=3)
                    
                    report_writer.log_step(
                        "Web Search - Output",
                        {
                            "Number of URLs found": len(urls),
                            "URLs": urls,  # Ghi đầy đủ tất cả URLs
                            "Snippets": snippets  # Ghi đầy đủ tất cả snippets
                        },
                        web_search_logs
                    )
                    
                    # Append web search log to report
                    report_writer.append_pipeline_log(f"BƯỚC 2: WEB SEARCH - Query: {query}", web_search_logs)

                    # Default with snippets from web_search
                    results_payload = {url: {"snippet": snippet, 'url': url, 'summary': None} for url, snippet in zip(urls, snippets)}
                    with self._report_lock:
                        self.report["actions"][identifier]["results"] = results_payload

                    # Process URLs sequentially to check scores and skip URL #3 if needed
                    evidence_scores = []  # Track scores from processed URLs
                    EVIDENCE_SCORE_THRESHOLD = 0.8  # Skip URL #3 if first 2 URLs have score > this threshold
                    
                    def process_result(result, url_index):
                        domain = urlparse(result).netloc.lower()
                        
                        # Log web scraping step
                        scraping_logs = []
                        report_writer.log_step(
                            "Web Scraping - Input",
                            {"URL": result, "Domain": domain, "URL index": url_index + 1},
                            scraping_logs
                        )
                        
                        if any(blocked in domain for blocked in self._unsupported_domains):
                            report_writer.log_step(
                                "Web Scraping - Skipped",
                                {"Reason": "Unsupported domain", "Domain": domain},
                                scraping_logs
                            )
                            report_writer.append_pipeline_log(f"BƯỚC 3: WEB SCRAPING - {result}", scraping_logs)
                            print(f"Skipping unsupported domain: {result}")
                            return None, 0.0

                        with self._timers.track(f"scrape:{domain}"):
                            scraped_content = web_scraper.scrape_url_content(result)
                        if not scraped_content:
                            report_writer.log_step(
                                "Web Scraping - Failed",
                                {"Reason": "Empty content", "Content length": 0},
                                scraping_logs
                            )
                            report_writer.append_pipeline_log(f"BƯỚC 3: WEB SCRAPING - {result}", scraping_logs)
                            print(f"Skipping empty scrape for: {result}")
                            return None, 0.0
                        
                        report_writer.log_step(
                            "Web Scraping - Output",
                            {
                                "Content length": len(scraped_content),
                                "Content": scraped_content  # Ghi đầy đủ content, không truncate
                            },
                            scraping_logs
                        )
                        report_writer.append_pipeline_log(f"BƯỚC 3: WEB SCRAPING - {result}", scraping_logs)

                        # Tạo log callback để ghi lại quá trình RAV
                        rav_log_lines = []
                        def rav_log_callback(msg):
                            rav_log_lines.append(msg)
                            print(f"[RAV] {msg}")
                        
                        with self._timers.track(f"evidence_rank:{domain}"):
                            summary, relevance_score = retriver_rav.get_top_evidence(
                                self.claim, 
                                scraped_content,
                                log_callback=rav_log_callback,
                                return_score=True
                            )
                            
                            # Ghi log RAV vào report với format chuẩn
                            if rav_log_lines:
                                report_writer.append_pipeline_log(f"BƯỚC 4: RAV (Evidence Ranking) - {result}", rav_log_lines)

                        if "NONE" in summary:
                            print(f"Skipping summary for evidence: {result}")
                            return None, 0.0

                        print(f"Web search result: {result}, Summary: {summary}, Relevance score: {relevance_score:.4f}")
                        report_writer.append_raw(f"web_search('{query}') results: {result}")
                        report_writer.append_evidence(f"web_search('{query}') Summary: {summary}")

                        with self._report_lock:
                            if result in self.report["actions"][identifier]["results"]:
                                self.report["actions"][identifier]["results"][result]["summary"] = summary
                        
                        return summary, relevance_score

                    # Process URLs sequentially to enable skipping URL #3
                    url_scores = {}  # Track score for each URL by index
                    for url_index, url in enumerate(urls):
                        # Check if we should skip URL #3 (index 2)
                        if url_index == 2:
                            # Only skip if URL #1 (index 0) and URL #2 (index 1) were both processed successfully
                            # and both have good evidence (score > threshold)
                            url1_score = url_scores.get(0, 0.0)
                            url2_score = url_scores.get(1, 0.0)
                            
                            if url1_score > EVIDENCE_SCORE_THRESHOLD and url2_score > EVIDENCE_SCORE_THRESHOLD:
                                skip_logs = []
                                report_writer.log_step(
                                    "Web Scraping - Skipped (Optimization)",
                                    {
                                        "URL": url,
                                        "Reason": f"URL #1 and URL #2 already have good evidence (scores: {url1_score:.4f}, {url2_score:.4f})",
                                        "Threshold": EVIDENCE_SCORE_THRESHOLD
                                    },
                                    skip_logs
                                )
                                report_writer.append_pipeline_log(f"BƯỚC 3: WEB SCRAPING - {url} (SKIPPED - OPTIMIZATION)", skip_logs)
                                print(f"⏭️  Skipping URL #{url_index + 1} ({url}) - URLs #1 and #2 already have good evidence (scores: {url1_score:.4f}, {url2_score:.4f}, threshold: {EVIDENCE_SCORE_THRESHOLD})")
                                continue
                        
                        summary, score = process_result(url, url_index)
                        if summary is not None:
                            evidence_scores.append((summary, score))
                            url_scores[url_index] = score
                        else:
                            url_scores[url_index] = 0.0  # Mark as failed/skipped
                    
                    self.save_report_json()
                else:
                    return

        except Exception as e:
            print(f"Error processing action line '{line}': {e}")

    def run(self):
        try:
            # Initialize pipeline log
            pipeline_logs = []
            
            with self._timers.track("factcheck_run"):
                # STEP 1: Claim Filtering
                with self._timers.track("claim_filtering"):
                    original_claim = self.claim
                    self.claim = claim_detection.claim_filter(self.claim)
                    if self.claim is None or not self.claim.strip():
                        raise ValueError("Filtered claim is empty")
                    
                    claim_filtering_logs = []
                    report_writer.log_step(
                        "Claim Filtering",
                        {
                            "Original claim": original_claim,
                            "Filtered claim": self.claim,
                            "Changed": original_claim != self.claim
                        },
                        claim_filtering_logs
                    )
                    # Append claim filtering log to report
                    report_writer.append_pipeline_log("BƯỚC 0: CLAIM FILTERING", claim_filtering_logs)
                    print(f"Filtered claim: {self.claim}")

                # STEP 2: Planning
                planning_logs = []
                with self._timers.track("planning"):
                    report_writer.log_step("Planning - Input", {"Claim": self.claim}, planning_logs)
                    queries = planning.plan(self.claim)
                    queries_lines = [x.strip() for x in queries.split('\n') if x.strip()]
                    action_lines = ["web_search(\"" + line + "\")" for line in queries_lines if line]
                    
                    report_writer.log_step(
                        "Planning - Output",
                        {
                            "Generated queries": queries_lines,
                            "Number of queries": len(queries_lines),
                            "Action lines": action_lines
                        },
                        planning_logs
                    )
                    
                    # Append planning log to report
                    report_writer.append_pipeline_log("BƯỚC 1: PLANNING (Tạo Query Tìm Kiếm)", planning_logs)
                
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

                # STEP 3: Verdict Judging
                allowed_verdicts = {"Supported", "Refuted", "Not Enough Evidence"}
                pred_verdict = ''
                verdict = None
                evidence_info = None

                with self._timers.track(f"judging"):
                    result = evaluation.judge(record=self.get_trimmed_record(max_chars=3000))
                    # Judge now returns tuple (verdict_string, evidence_info)
                    if isinstance(result, tuple) and len(result) == 2:
                        verdict, evidence_info = result
                    else:
                        # Backward compatibility: if judge returns string only
                        verdict = result
                        evidence_info = None
                print(f"Judged verdict:\n{verdict}")
                extracted_verdict = re.search(r'`(.*?)`', verdict, re.DOTALL)
                pred_verdict = extracted_verdict.group(1).strip() if extracted_verdict else ''

                if pred_verdict not in allowed_verdicts:
                    print("No decision options found in verdict, using extract_verdict from judge.py...")
                    with self._timers.track("extract_verdict_fallback"):
                        try:
                            extracted = evaluation.extract_verdict(conclusion=verdict)
                            extracted_verdict = re.search(r'`(.*?)`', extracted, re.DOTALL)
                            pred_verdict = extracted_verdict.group(1).strip() if extracted_verdict else extracted.strip()
                            print(f"extract_verdict returned: {pred_verdict}")
                        except Exception as e:
                            print(f"extract_verdict failed: {e}")
                            pred_verdict = "INVALID VERDICT"
                            print("No decision options found in verdict, defaulting to 'INVALID VERDICT'.")

                # Ghi thông tin evidence vào report.md trước khi ghi verdict
                if evidence_info:
                    # Ghi log quá trình filter evidence nếu có
                    if 'filter_log' in evidence_info:
                        report_writer.append_pipeline_log("BƯỚC 5: EVIDENCE FILTERING & SELECTION", evidence_info['filter_log'])
                    report_writer.append_evidence_info(evidence_info)
                    
                    # Log judge step
                    judge_logs = []
                    report_writer.log_step(
                        "Judge - Input",
                        {
                            "Claim": evidence_info.get('claim', 'N/A'),
                            "Selected evidence count": evidence_info.get('selected_evidence_count', 0),
                            "Top_k": evidence_info.get('top_k', 0)
                        },
                        judge_logs
                    )
                    report_writer.log_step(
                        "Judge - Output",
                        {
                            "Verdict": pred_verdict,
                            "Justification preview": verdict[:200] + "..." if len(verdict) > 200 else verdict
                        },
                        judge_logs
                    )
                    report_writer.append_pipeline_log("BƯỚC 6: JUDGE (Final Verdict)", judge_logs)
                
                report_writer.append_verdict(pred_verdict)
                report_writer.append_justification(verdict)
                
                # Ghi thông tin timing vào report
                report_writer.append_timing_info(self._timers.records)
                
                with self._report_lock:
                    self.report["justification"] = verdict
                    self.report["verdict"] = pred_verdict
                    if evidence_info:
                        self.report["evidence_info"] = evidence_info
                self.save_report_json()

                return pred_verdict, report_writer.REPORT_PATH
        finally:
            self._result_executor.shutdown(wait=True)

# For backward compatibility, provide a function interface

def factcheck(claim, date, identifier=None, multimodal=False, image_path=None, max_actions=2, judge_model_name=None):
    try:
        checker = FactChecker(claim, date, identifier, multimodal, image_path, max_actions, judge_model_name=judge_model_name)
        verdict, report_path = checker.run()
        return verdict, report_path
    except ValueError as e:
        # Giữ nguyên ValueError với message thân thiện
        # Đây thường là lỗi validation (claim không hợp lệ)
        error_message = str(e)
        # Nếu là lỗi "Filtered claim is empty", thay bằng message thân thiện hơn
        if "Filtered claim is empty" in error_message or "empty" in error_message.lower():
            error_message = "Vui lòng nhập một câu claim hợp lệ để kiểm chứng. Câu bạn nhập không phải là một claim có thể kiểm chứng được."
        print(f"Error in factcheck: {e}")
        print(f"User-friendly message: {error_message}")
        raise ValueError(error_message) from e
    except Exception as e:
        # Các lỗi khác
        error_message = f"Đã xảy ra lỗi khi kiểm chứng: {str(e)}"
        print(f"Error in factcheck: {e}")
        print(error_message)
        raise ValueError(error_message) from e
