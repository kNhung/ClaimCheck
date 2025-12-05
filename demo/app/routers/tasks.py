"""Task management endpoints (for async fact-checking)."""

from fastapi import APIRouter, HTTPException, status
from models.claim import FactCheckStatusResponse
from models.common import MessageResponse
from services.task_service import TaskService

router = APIRouter()
service = TaskService()


@router.get("/{task_id}", response_model=FactCheckStatusResponse)
async def get_task_status(task_id: str):
    """
    Get task status by ID.
    
    Args:
        task_id: The task identifier (same as report_id)
        
    Returns:
        FactCheckStatusResponse with task status and progress
        
    Raises:
        HTTPException: If task not found
    """
    task_status = service.get_task_status(task_id)
    
    if not task_status:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found"
        )
    
    return task_status


@router.delete("/{task_id}", response_model=MessageResponse)
async def delete_task(task_id: str):
    """
    Delete a task.
    
    Args:
        task_id: The task identifier
        
    Returns:
        MessageResponse confirming deletion
        
    Raises:
        HTTPException: If task not found
    """
    deleted = service.delete_task(task_id)
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task '{task_id}' not found"
        )
    
    return MessageResponse(message=f"Task '{task_id}' deleted successfully")


