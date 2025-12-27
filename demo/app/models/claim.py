from typing import Optional
from pydantic import BaseModel, Field


class FactCheckRequest(BaseModel):
    """Request model for fact-checking a claim."""
    claim: str = Field(
        ..., 
        description="The claim to be fact-checked",
        min_length=1,
        max_length=5000,
        example="Ông Putin nói Nga sẽ phản ứng mạnh nếu bị Tomahawk tấn công"
    )
    date: str = Field(
        ..., 
        description="Cut-off date in DD-MM-YYYY format",
        pattern=r"^\d{2}-\d{2}-\d{4}$",
        example="01-01-2024"
    )
    max_actions: Optional[int] = Field(
        default=2,
        ge=1,
        le=10,
        description="Maximum number of search actions to perform"
    )
    judge_model_name: Optional[str] = Field(
        default=None,
        description="Model to use for judging (e.g., 'qwen2.5:0.5b'). If not provided, uses default from config."
    )


class FactCheckResponse(BaseModel):
    """Response model for fact-checking result."""
    report_id: str = Field(..., description="Unique identifier for the report")
    verdict: str = Field(
        ..., 
        description="Fact-check verdict",
        example="Supported",
        pattern="^(Supported|Refuted|Not Enough Evidence)$"
    )
    report_path: str = Field(..., description="Path to the generated report directory")
    claim: str = Field(..., description="The original claim that was checked")
    date: str = Field(..., description="Cut-off date used")
    model: Optional[str] = Field(None, description="Model name used for fact-checking")


class FactCheckStatusResponse(BaseModel):
    """Response model for checking fact-check task status."""
    report_id: str = Field(..., description="Report identifier")
    status: str = Field(
        ..., 
        description="Task status",
        pattern="^(pending|processing|completed|failed)$"
    )
    progress: Optional[float] = Field(
        None, 
        ge=0.0, 
        le=100.0, 
        description="Progress percentage (0-100)"
    )
    verdict: Optional[str] = Field(
        None,
        description="Fact-check verdict (only available when status is 'completed')"
    )
    error: Optional[str] = Field(None, description="Error message (only available when status is 'failed')")

