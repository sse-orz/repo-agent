from fastapi import APIRouter, Depends, Body
from fastapi.responses import StreamingResponse
from queue import Queue
from threading import Thread
import asyncio

from app.models.common import BaseResponse
from app.models.agents import GenerateRequestWrapper, WikiFilesResponse, AgentMode
from app.services.agent_service import AgentService

router = APIRouter()


def get_agent_service() -> AgentService:
    return AgentService()


# curl -X POST http://localhost:8000/api/v1/agents/generate
# -H "Content-Type: application/json" -d '{"mode": "sub", "request": {"owner": "octocat", "repo": "Hello-World"}}'
@router.post("/generate")
async def generate_agent_documentation(
    wrapper: GenerateRequestWrapper = Body(...),
    agent_service: AgentService = Depends(get_agent_service),
) -> BaseResponse:
    # Generate or update documentation normally
    if existing_wiki := agent_service.check_existing_wiki(
        wrapper.request, wrapper.mode
    ):
        return BaseResponse(
            message="Existing documentation found.", code=200, data=existing_wiki
        )

    if not agent_service.preprocess_repo(wrapper.request):
        return BaseResponse(
            message="Failed to preprocess repository.", code=500, data=None
        )

    data = agent_service.generate_documentation(wrapper.mode, wrapper.request)

    return BaseResponse(
        message="Agent documentation generated successfully.", code=200, data=data
    )


# curl -N http://localhost:8000/api/v1/agents/generate-stream
# -X POST -H "Content-Type: application/json" -d '{"mode": "sub", "request": {"owner": "octocat", "repo": "Hello-World"}}'
@router.post("/generate-stream")
async def generate_agent_documentation_stream(
    wrapper: GenerateRequestWrapper = Body(...),
    agent_service: AgentService = Depends(get_agent_service),
):
    # Generate or update documentation in streaming mode
    if existing_wiki := agent_service.check_existing_wiki(
        wrapper.request, wrapper.mode
    ):
        return BaseResponse(
            message="Existing documentation found.", code=200, data=existing_wiki
        )

    if not agent_service.preprocess_repo(wrapper.request):
        return BaseResponse(
            message="Failed to preprocess repository.", code=500, data=None
        )

    progress_queue = Queue()
    result_container = {"data": None, "error": None}

    def run_generation():
        try:
            result_container["data"] = agent_service.generate_documentation(
                wrapper.mode, wrapper.request, progress_queue.put
            )
        except Exception as e:
            result_container["error"] = str(e)
        finally:
            progress_queue.put({"stage": "done", "message": "Generation finished"})

    Thread(target=run_generation, daemon=True).start()

    async def event_generator():
        while True:
            await asyncio.sleep(0.1)

            if not progress_queue.empty():
                progress_data = progress_queue.get()
                wiki_info = agent_service.get_wiki_info(
                    wrapper.request.owner, wrapper.request.repo, wrapper.mode
                )
                progress_data["wiki_info"] = wiki_info
                yield f"data: {BaseResponse(message=progress_data.get('message', ''), code=200, data=progress_data).model_dump_json()}\n\n"

                if progress_data.get("stage") == "done":
                    if error := result_container["error"]:
                        yield f"data: {BaseResponse(message='Documentation generation failed', code=500, data={'error': error}).model_dump_json()}\n\n"
                    else:
                        result_dict = (
                            result_container["data"].model_dump()
                            if result_container["data"]
                            else None
                        )
                        # Add completion flag for frontend to stop polling
                        if result_dict:
                            result_dict["generation_complete"] = True
                        yield f"data: {BaseResponse(message='Documentation generation completed successfully', code=200, data=result_dict).model_dump_json()}\n\n"
                    break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# curl http://localhost:8000/api/v1/agents/list
@router.get("/list")
async def list_documentation(
    agent_service: AgentService = Depends(get_agent_service),
) -> BaseResponse:
    # List all generated wikis
    return BaseResponse(
        message="List of generated wikis retrieved successfully.",
        code=200,
        data=agent_service.list_wikis(),
    )


# curl "http://localhost:8000/api/v1/agents/wikis/octocat/Hello-World?mode=moe"
@router.get("/wikis/{owner}/{repo}")
async def get_wiki_files(
    owner: str,
    repo: str,
    mode: AgentMode = AgentMode.SUB,
    agent_service: AgentService = Depends(get_agent_service),
) -> BaseResponse:
    """Get currently generated wiki files for a repository.
    
    This endpoint is used for progressive loading to query files that have
    been generated so far, without waiting for the entire generation to complete.
    """
    wiki_info = agent_service.get_wiki_info(owner, repo, mode)
    
    if not wiki_info:
        return BaseResponse(
            code=200,
            message="No wiki files found yet",
            data=WikiFilesResponse(files=[], total_files=0)
        )
    
    return BaseResponse(
        code=200,
        message="Success",
        data=WikiFilesResponse(
            files=wiki_info.files,
            total_files=wiki_info.total_files
        )
    )
