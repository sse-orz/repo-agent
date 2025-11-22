from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.api.router import api_router
from app.models.common import BaseResponse
from app.config import settings
import os

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Repository documentation generation API",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for generated wikis
wikis_path = os.path.join(os.path.dirname(__file__), ".wikis")
os.makedirs(wikis_path, exist_ok=True)
app.mount("/wikis", StaticFiles(directory=wikis_path), name="wikis")

# Include API routes
app.include_router(api_router, prefix=settings.API_PREFIX)


@app.get("/")
async def root() -> BaseResponse:
    return BaseResponse(
        code=200,
        message="Welcome to the Repo Agent API!",
    )


@app.get("/health")
async def health_check() -> BaseResponse:
    return BaseResponse(
        code=200,
        message="Repo Agent API is healthy.",
    )
