"""Health check endpoints."""

from fastapi import APIRouter
from models.common import MessageResponse

router = APIRouter()


@router.get("", response_model=MessageResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        MessageResponse with status message
    """
    return MessageResponse(message="healthy")


@router.get("/ready", response_model=MessageResponse)
async def readiness_check():
    """
    Readiness probe endpoint.
    
    Checks if the service is ready to accept requests.
    
    Returns:
        MessageResponse with status message
    """
    return MessageResponse(message="ready")


