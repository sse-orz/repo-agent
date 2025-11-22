from fastapi import APIRouter, Depends, Body

from app.models.common import BaseResponse
from app.models.agents import GenerateRequest
from app.services.agent_service import AgentService

router = APIRouter()


def get_agent_service() -> AgentService:
    return AgentService()


@router.post("/generate")
async def generate_agent_documentation(
    request: GenerateRequest = Body(...),
    agent_service: AgentService = Depends(get_agent_service),
) -> BaseResponse:
    # generate documentation for a repository
    data = agent_service.generate_documentation(request)

    return BaseResponse(
        message="Agent documentation generated successfully.",
        code=200,
        data=data,
    )


@router.get("/list")
async def list_documentation(
    agent_service: AgentService = Depends(get_agent_service),
) -> BaseResponse:
    # list wikis that have been generated
    data = agent_service.list_wikis()

    return BaseResponse(
        message="List of generated wikis retrieved successfully.",
        code=200,
        data=data,
    )
