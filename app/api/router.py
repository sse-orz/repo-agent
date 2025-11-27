from fastapi import APIRouter
from app.api import agents, rag

# Create main API router
api_router = APIRouter()

# Include sub-routers with prefixes and tags
api_router.include_router(agents.router, prefix="/agents", tags=["agents"])
api_router.include_router(rag.router, prefix="/rag", tags=["rag"])
