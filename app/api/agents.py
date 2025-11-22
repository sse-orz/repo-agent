from fastapi import APIRouter, Depends, Body

from app.models.common import BaseResponse
from app.models.agents import GenerateRequest
from app.services.agent_service import AgentService

router = APIRouter()


def get_agent_service() -> AgentService:
    return AgentService()


# curl -X POST http://localhost:8000/api/v1/agents/generate -H "Content-Type: application/json" -d '{"owner": "octocat", "repo": "Hello-World"}'
@router.post("/generate")
async def generate_agent_documentation(
    request: GenerateRequest = Body(...),
    agent_service: AgentService = Depends(get_agent_service),
) -> BaseResponse:
    # generate documentation for a repository

    # existing documentation check
    # and request parameter need_update is False
    existing_wiki = agent_service.get_wiki_info(request.owner, request.repo)
    if existing_wiki and not request.need_update:
        return BaseResponse(
            message="Existing documentation found.",
            code=200,
            data=existing_wiki,
        )

    # normally generate or update documentation
    data = agent_service.generate_documentation(request)

    return BaseResponse(
        message="Agent documentation generated successfully.",
        code=200,
        data=data,
    )


# curl http://localhost:8000/api/v1/agents/list
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
