"""FastAPI application main entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from core.config import settings
from routers import (
    health_router,
    factcheck_router,
    reports_router,
    tasks_router,
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="API for fact-checking claims using advanced AI techniques",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "An unexpected error occurred"
        }
    )


# Include routers
app.include_router(
    health_router,
    prefix="/health",
    tags=["health"]
)

app.include_router(
    factcheck_router,
    prefix="/factcheck",
    tags=["factcheck"]
)

app.include_router(
    reports_router,
    prefix="/reports",
    tags=["reports"]
)

app.include_router(
    tasks_router,
    prefix="/tasks",
    tags=["tasks"]
)


@app.on_event("startup")
async def startup_event():
    """Called when the application starts."""
    logger.info(f"Starting {settings.APP_NAME} API")
    logger.info(f"Debug mode: {settings.DEBUG}")
    logger.info(f"Reports directory: {settings.REPORTS_DIR}")
    logger.info("API documentation available at /docs")


@app.on_event("shutdown")
async def shutdown_event():
    """Called when the application shuts down."""
    logger.info(f"Shutting down {settings.APP_NAME} API")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info"
    )

