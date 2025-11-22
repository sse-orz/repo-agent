from fastapi import APIRouter
from app.api import agents

# Create main API router
api_router = APIRouter()

# Include sub-routers with prefixes and tags
api_router.include_router(agents.router, prefix="/agents", tags=["agents"])
