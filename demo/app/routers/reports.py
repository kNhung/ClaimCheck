"""Report management endpoints."""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import PlainTextResponse
from typing import Optional
from models.report import ReportResponse, ReportListResponse, ReportSummaryResponse
from models.common import PaginationParams, MessageResponse
from services.report_service import ReportService

router = APIRouter()
service = ReportService()


@router.get("/{report_id}", response_model=ReportResponse)
async def get_report(report_id: str):
    """
    Get full report by ID.
    
    Args:
        report_id: The report identifier
        
    Returns:
        ReportResponse with complete report data
        
    Raises:
        HTTPException: If report not found
    """
    report = await service.get_report(report_id)
    
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report '{report_id}' not found"
        )
    
    return report


@router.get("/{report_id}/json")
async def get_report_json(report_id: str):
    """
    Get report as JSON.
    
    Returns the raw report.json file content.
    
    Args:
        report_id: The report identifier
        
    Returns:
        JSON object with report data
        
    Raises:
        HTTPException: If report not found
    """
    report_data = service.get_report_json(report_id)
    
    if not report_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report '{report_id}' not found"
        )
    
    return report_data


@router.get("/{report_id}/markdown", response_class=PlainTextResponse)
async def get_report_markdown(report_id: str):
    """
    Get report as Markdown.
    
    Returns the raw report.md file content.
    
    Args:
        report_id: The report identifier
        
    Returns:
        Plain text Markdown content
        
    Raises:
        HTTPException: If report not found
    """
    markdown = service.get_report_markdown(report_id)
    
    if not markdown:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report '{report_id}' not found"
        )
    
    return markdown


@router.get("", response_model=ReportListResponse)
async def list_reports(
    page: int = 1,
    page_size: int = 10
):
    """
    List all available reports with pagination.
    
    Args:
        page: Page number (default: 1)
        page_size: Number of items per page (default: 10, max: 100)
        
    Returns:
        ReportListResponse with paginated list of reports
    """
    # Validate pagination params
    if page < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Page must be >= 1"
        )
    
    if page_size < 1 or page_size > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Page size must be between 1 and 100"
        )
    
    reports = await service.get_report_list(page=page, page_size=page_size)
    
    # Calculate total (simplified - in production, use database count)
    # For now, we return all reports and slice them
    all_reports = await service.get_report_list(page=1, page_size=1000)
    total = len(all_reports)
    
    return ReportListResponse(
        reports=reports,
        total=total,
        page=page,
        page_size=page_size
    )


@router.get("/{report_id}/exists")
async def check_report_exists(report_id: str) -> MessageResponse:
    """
    Check if a report exists.
    
    Args:
        report_id: The report identifier
        
    Returns:
        MessageResponse indicating if report exists
    """
    exists = service.report_exists(report_id)
    
    if exists:
        return MessageResponse(message=f"Report '{report_id}' exists")
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report '{report_id}' not found"
        )


