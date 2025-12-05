from typing import Dict, Optional
from datetime import datetime
from enum import Enum

from models.claim import FactCheckStatusResponse


class TaskStatus(str, Enum):
    """Task status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskService:
    """Service for managing background tasks."""
    
    def __init__(self):
        # In-memory task store (for simple implementation)
        # In production, use Redis or database
        self._tasks: Dict[str, Dict] = {}
    
    def create_task(self, report_id: str) -> str:
        """
        Create a new task for fact-checking.
        
        Args:
            report_id: The report identifier
            
        Returns:
            Task ID (same as report_id for simplicity)
        """
        task_id = report_id
        self._tasks[task_id] = {
            "report_id": report_id,
            "status": TaskStatus.PENDING.value,
            "progress": 0.0,
            "created_at": datetime.now(),
            "verdict": None,
            "error": None
        }
        return task_id
    
    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        progress: Optional[float] = None,
        verdict: Optional[str] = None,
        error: Optional[str] = None
    ) -> None:
        """
        Update task status.
        
        Args:
            task_id: The task identifier
            status: New status
            progress: Progress percentage (0-100)
            verdict: Verdict (if completed)
            error: Error message (if failed)
        """
        if task_id not in self._tasks:
            raise ValueError(f"Task {task_id} not found")
        
        self._tasks[task_id]["status"] = status.value
        
        if progress is not None:
            self._tasks[task_id]["progress"] = progress
        
        if verdict is not None:
            self._tasks[task_id]["verdict"] = verdict
        
        if error is not None:
            self._tasks[task_id]["error"] = error
    
    def get_task_status(self, task_id: str) -> Optional[FactCheckStatusResponse]:
        """
        Get task status.
        
        Args:
            task_id: The task identifier
            
        Returns:
            FactCheckStatusResponse or None if task not found
        """
        if task_id not in self._tasks:
            return None
        
        task = self._tasks[task_id]
        
        return FactCheckStatusResponse(
            report_id=task["report_id"],
            status=task["status"],
            progress=task.get("progress"),
            verdict=task.get("verdict"),
            error=task.get("error")
        )
    
    def delete_task(self, task_id: str) -> bool:
        """
        Delete a task.
        
        Args:
            task_id: The task identifier
            
        Returns:
            True if deleted, False if not found
        """
        if task_id in self._tasks:
            del self._tasks[task_id]
            return True
        return False
    
    def task_exists(self, task_id: str) -> bool:
        """Check if a task exists."""
        return task_id in self._tasks

