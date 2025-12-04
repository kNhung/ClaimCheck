"""Fact-checking endpoints."""

from fastapi import APIRouter, HTTPException, status
from models.claim import FactCheckRequest, FactCheckResponse
from services.factcheck_service import FactCheckService

router = APIRouter()
service = FactCheckService()


@router.post("/verify", response_model=FactCheckResponse, status_code=status.HTTP_200_OK)
async def verify_claim(request: FactCheckRequest):
    """
    Verify a claim using the fact-checking pipeline.
    
    This endpoint takes a claim and verifies it using the fact-checking system.
    The process includes:
    - Planning search queries
    - Gathering evidence from web sources
    - Synthesizing evidence
    - Evaluating and judging the claim
    - Generating a verdict
    
    Args:
        request: FactCheckRequest containing:
            - claim: The claim to verify
            - date: Cut-off date (DD-MM-YYYY)
            - max_actions: Maximum search actions (optional)
            - model_name: Ollama model name (optional)
    
    Returns:
        FactCheckResponse with:
            - report_id: Unique identifier
            - verdict: "Supported", "Refuted", or "Not Enough Evidence"
            - report_path: Path to generated report
            - claim, date, model: Original parameters
    
    Raises:
        HTTPException: If fact-checking fails
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"Received fact-check request: claim={request.claim[:50]}..., date={request.date}")
    
    try:
        logger.info("Calling FactCheckService.verify_claim...")
        result = await service.verify_claim(
            claim=request.claim,
            date=request.date,
            max_actions=request.max_actions,
            model_name=request.model_name
        )
        logger.info(f"Fact-check completed: verdict={result.verdict}, report_id={result.report_id}")
        return result
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except ImportError as e:
        logger.error(f"Import error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to import factchecker module: {str(e)}. Please ensure all dependencies are installed."
        )
    except Exception as e:
        logger.error(f"Fact-checking error: {type(e).__name__}: {str(e)}", exc_info=True)
        # Get full traceback for debugging
        import traceback
        error_traceback = traceback.format_exc()
        logger.error(f"Full traceback:\n{error_traceback}")
        
        # Return more detailed error message
        error_detail = str(e)
        if settings.DEBUG:
            error_detail += f"\n\nTraceback:\n{error_traceback}"
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fact-checking failed: {error_detail}"
        )


