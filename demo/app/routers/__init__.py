"""Routers package."""

from .health import router as health_router
from .factcheck import router as factcheck_router
from .reports import router as reports_router
from .tasks import router as tasks_router

__all__ = [
    "health_router",
    "factcheck_router",
    "reports_router",
    "tasks_router",
]


