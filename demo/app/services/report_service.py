import os
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

from core.config import settings
from models.report import (
    ReportResponse,
    ReportSummaryResponse,
    ActionInfo,
    ActionResult,
    TimingRecord,
)


class ReportService:
    """Service for managing fact-check reports."""
    
    def __init__(self):
        self.reports_dir = Path(settings.REPORTS_DIR)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_report_dir(self, report_id: str) -> Path:
        """Get the directory path for a report ID."""
        return self.reports_dir / report_id
    
    def _get_report_json_path(self, report_id: str) -> Path:
        """Get the path to report.json for a report ID."""
        return self._get_report_dir(report_id) / "report.json"
    
    def _get_report_md_path(self, report_id: str) -> Path:
        """Get the path to report.md for a report ID."""
        return self._get_report_dir(report_id) / "report.md"
    
    def get_report_json(self, report_id: str) -> Optional[Dict[str, Any]]:
        """
        Load report.json for a given report ID.
        
        Args:
            report_id: The report identifier
            
        Returns:
            Dictionary with report data, or None if not found
        """
        json_path = self._get_report_json_path(report_id)
        
        if not json_path.exists():
            return None
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Failed to read report JSON: {str(e)}") from e
    
    def get_report_markdown(self, report_id: str) -> Optional[str]:
        """
        Load report.md for a given report ID.
        
        Args:
            report_id: The report identifier
            
        Returns:
            Markdown content as string, or None if not found
        """
        md_path = self._get_report_md_path(report_id)
        
        if not md_path.exists():
            return None
        
        try:
            with open(md_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise Exception(f"Failed to read report Markdown: {str(e)}") from e
    
    def _convert_action_results(self, results: Dict[str, Any]) -> Dict[str, ActionResult]:
        """Convert action results from JSON format to ActionResult models."""
        converted = {}
        for url, result_data in results.items():
            if isinstance(result_data, dict):
                converted[url] = ActionResult(
                    snippet=result_data.get("snippet"),
                    url=result_data.get("url", url),
                    summary=result_data.get("summary")
                )
            else:
                # Fallback if result_data is not a dict
                converted[url] = ActionResult(url=url)
        return converted
    
    def _convert_actions(self, actions: Dict[str, Any]) -> Dict[str, ActionInfo]:
        """Convert actions from JSON format to ActionInfo models."""
        converted = {}
        for key, action_data in actions.items():
            if isinstance(action_data, dict):
                results = action_data.get("results", {})
                converted[key] = ActionInfo(
                    action=action_data.get("action", "unknown"),
                    query=action_data.get("query", ""),
                    results=self._convert_action_results(results) if results else None
                )
        return converted
    
    def _convert_timings(self, timings: List[Dict[str, Any]]) -> List[TimingRecord]:
        """Convert timing records from JSON format to TimingRecord models."""
        return [
            TimingRecord(
                label=timing.get("label", ""),
                duration=timing.get("duration", 0.0)
            )
            for timing in timings
            if isinstance(timing, dict)
        ]
    
    async def get_report(self, report_id: str) -> Optional[ReportResponse]:
        """
        Get full report data for a given report ID.
        
        Args:
            report_id: The report identifier
            
        Returns:
            ReportResponse object, or None if report not found
        """
        report_data = self.get_report_json(report_id)
        
        if not report_data:
            return None
        
        # Convert actions
        actions = self._convert_actions(report_data.get("actions", {}))
        
        # Convert timings
        timings = self._convert_timings(report_data.get("timings", []))
        
        # Get report path
        report_dir = self._get_report_dir(report_id)
        report_path = str(report_dir)
        
        return ReportResponse(
            claim=report_data.get("claim", ""),
            date=report_data.get("date", ""),
            identifier=report_data.get("identifier", report_id),
            model=report_data.get("model"),
            verdict=report_data.get("verdict"),
            justification=report_data.get("justification"),
            actions=actions,
            action_needed=report_data.get("action_needed", []),
            report_path=report_path,
            timings=timings
        )
    
    async def get_report_list(
        self,
        page: int = 1,
        page_size: int = 10
    ) -> List[ReportSummaryResponse]:
        """
        List all available reports (paginated).
        
        Args:
            page: Page number (1-indexed)
            page_size: Number of items per page
            
        Returns:
            List of ReportSummaryResponse objects
        """
        # Scan reports directory
        reports = []
        
        if not self.reports_dir.exists():
            return []
        
        # Iterate through report directories
        for report_dir in self.reports_dir.iterdir():
            if not report_dir.is_dir():
                continue
            
            report_id = report_dir.name
            report_data = self.get_report_json(report_id)
            
            if not report_data:
                continue
            
            # Get creation time from directory
            try:
                created_at = datetime.fromtimestamp(report_dir.stat().st_mtime)
            except Exception:
                created_at = None
            
            reports.append(
                ReportSummaryResponse(
                    report_id=report_id,
                    claim=report_data.get("claim", ""),
                    verdict=report_data.get("verdict"),
                    date=report_data.get("date", ""),
                    created_at=created_at,
                    report_path=str(report_dir)
                )
            )
        
        # Sort by creation time (newest first)
        reports.sort(key=lambda x: x.created_at or datetime.min, reverse=True)
        
        # Paginate
        start = (page - 1) * page_size
        end = start + page_size
        
        return reports[start:end]
    
    def report_exists(self, report_id: str) -> bool:
        """Check if a report exists."""
        json_path = self._get_report_json_path(report_id)
        return json_path.exists()

