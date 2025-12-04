import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

# Add project root to path to import factchecker
# From backend/app/services/ to project root: ../../../
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from factchecker.factchecker import factcheck
except ImportError:
    # Try alternative path
    alt_path = Path(__file__).parent.parent.parent.parent.parent
    if alt_path.exists():
        sys.path.insert(0, str(alt_path))
    from factchecker.factchecker import factcheck

from core.config import settings
from models.claim import FactCheckResponse


class FactCheckService:
    """Service for handling fact-checking operations."""
    
    def __init__(self):
        self.reports_dir = settings.REPORTS_DIR
    
    async def verify_claim(
        self,
        claim: str,
        date: str,
        max_actions: Optional[int] = None,
        model_name: Optional[str] = None,
        identifier: Optional[str] = None
    ) -> FactCheckResponse:
        """
        Verify a claim using the factchecker pipeline.
        
        Args:
            claim: The claim to verify
            date: Cut-off date in DD-MM-YYYY format
            max_actions: Maximum number of search actions (defaults to config)
            model_name: Ollama model name (defaults to config)
            identifier: Optional identifier for the report
            
        Returns:
            FactCheckResponse with verdict and report information
            
        Raises:
            Exception: If fact-checking fails
        """
        # Use defaults from config if not provided
        max_actions = max_actions or settings.FACTCHECKER_MAX_ACTIONS
        model_name = model_name or settings.FACTCHECKER_MODEL_NAME
        
        # Generate identifier if not provided
        if not identifier:
            identifier = datetime.now().strftime("%m%d%Y%H%M%S")
        
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            logger.info(f"Starting fact-check: claim='{claim[:50]}...', date={date}, identifier={identifier}")
            logger.info(f"Using max_actions={max_actions}, model_name={model_name}")
            
            # Check if factcheck function is available
            try:
                from factchecker.factchecker import factcheck as factcheck_func
                logger.info("factcheck function imported successfully")
            except ImportError as import_err:
                logger.error(f"Failed to import factcheck function: {import_err}")
                raise ImportError(
                    f"Cannot import factchecker.factchecker. "
                    f"Make sure factchecker package is accessible. "
                    f"Error: {str(import_err)}"
                )
            
            # Call the factcheck function
            logger.info("Calling factchecker.factcheck function...")
            try:
                verdict, report_path = factcheck_func(
                    claim=claim,
                    date=date,
                    identifier=identifier,
                    max_actions=max_actions,
                    model_name=model_name
                )
                logger.info(f"Fact-check completed: verdict={verdict}, report_path={report_path}")
            except Exception as factcheck_err:
                logger.error(f"factcheck function raised error: {type(factcheck_err).__name__}: {str(factcheck_err)}")
                import traceback
                logger.error(f"factcheck traceback:\n{traceback.format_exc()}")
                raise
            
            # Extract report_id from report_path
            # report_path format: reports/{identifier}/report.md
            report_id = identifier
            
            return FactCheckResponse(
                report_id=report_id,
                verdict=verdict,
                report_path=report_path,
                claim=claim,
                date=date,
                model=model_name
            )
        except ImportError as e:
            # Re-raise ImportError as-is
            logger.error(f"Import error in verify_claim: {str(e)}")
            raise
        except Exception as e:
            # Re-raise with context
            logger.error(f"Unexpected error in verify_claim: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise Exception(f"Fact-checking failed: {type(e).__name__}: {str(e)}") from e

