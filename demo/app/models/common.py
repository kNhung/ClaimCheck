from pydantic import BaseModel, Field
from typing import Optional


class ErrorResponse(BaseModel):
    """Standard error response format."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")


class MessageResponse(BaseModel):
    """Simple message response."""
    message: str = Field(..., description="Response message")


class PaginationParams(BaseModel):
    """Pagination parameters for list endpoints."""
    page: int = Field(default=1, ge=1, description="Page number (starts from 1)")
    page_size: int = Field(default=10, ge=1, le=100, description="Number of items per page")


class PaginatedResponse(BaseModel):
    """Paginated response wrapper."""
    page: int
    page_size: int
    total: int
    total_pages: int

