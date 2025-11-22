from fastapi import APIRouter, Depends
from typing import List
from datetime import datetime

from app.models.common import BaseResponse
from app.services.agent_service import AgentService
from agents.sub_graph.parent import ParentGraphBuilder
from config import CONFIG

router = APIRouter()


def get_agent_service() -> AgentService:
    return AgentService()


@router.get("/generate")
async def generate_agent_documentation(
    agent_service: AgentService = Depends(get_agent_service),
) -> BaseResponse:
    CONFIG.display()
    parent_graph_builder = ParentGraphBuilder(branch_mode="all")
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parent_graph_builder.run(
        inputs={
            "owner": "octocat",
            "repo": "Hello-World",
            "platform": "github",
            "mode": "fast",  # "fast" or "smart"
            "max_workers": 50,  # 20 worker -> 3 - 4 minutes
            "date": date,
            "log": False,
        },
        config={
            "configurable": {
                "thread_id": f"wiki-generation-{date}",
            }
        },
        count_time=True,
    )
    return BaseResponse(message="Agent documentation generated successfully.")
