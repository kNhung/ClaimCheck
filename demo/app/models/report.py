from typing import Optional, Dict, List, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ActionResult(BaseModel):
    """Model for a single search action result."""
    snippet: Optional[str] = Field(None, description="Snippet from search result")
    url: str = Field(..., description="URL of the source")
    summary: Optional[str] = Field(None, description="Evidence summary extracted from the source")


class ActionInfo(BaseModel):
    """Model for action information."""
    action: str = Field(..., description="Type of action (e.g., 'web_search')")
    query: str = Field(..., description="Search query used")
    results: Optional[Dict[str, ActionResult]] = Field(
        None, 
        description="Dictionary mapping URLs to action results"
    )


class TimingRecord(BaseModel):
    """Model for timing information."""
    label: str = Field(..., description="Timing label")
    duration: float = Field(..., description="Duration in seconds")


class ReportResponse(BaseModel):
    """Complete report response model."""
    claim: str = Field(..., description="The claim that was checked")
    date: str = Field(..., description="Cut-off date")
    identifier: str = Field(..., description="Report identifier")
    model: Optional[str] = Field(None, description="Model name used")
    verdict: Optional[str] = Field(
        None,
        description="Fact-check verdict",
        pattern="^(Supported|Refuted|Not Enough Evidence)$"
    )
    justification: Optional[str] = Field(None, description="Detailed justification for the verdict")
    actions: Dict[str, ActionInfo] = Field(
        default_factory=dict,
        description="Dictionary of actions performed during fact-checking"
    )
    action_needed: List[str] = Field(
        default_factory=list,
        description="List of additional actions needed (from iterations)"
    )
    report_path: str = Field(..., description="Path to report directory")
    timings: List[TimingRecord] = Field(
        default_factory=list,
        description="Timing records for performance analysis"
    )


class ReportSummaryResponse(BaseModel):
    """Summary response for report listing."""
    report_id: str = Field(..., description="Report identifier")
    claim: str = Field(..., description="The claim")
    verdict: Optional[str] = Field(None, description="Fact-check verdict")
    date: str = Field(..., description="Cut-off date")
    created_at: Optional[datetime] = Field(None, description="Report creation timestamp")
    report_path: str = Field(..., description="Path to report directory")


class ReportListResponse(BaseModel):
    """Response for listing reports."""
    reports: List[ReportSummaryResponse] = Field(..., description="List of reports")
    total: int = Field(..., description="Total number of reports")
    page: int = Field(..., description="Current page number")
    page_size: int = Field(..., description="Page size")

