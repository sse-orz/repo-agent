from fastapi import APIRouter, Body, Depends
from fastapi.responses import StreamingResponse

from app.models.common import BaseResponse
from app.models.rag import RAGQueryRequest
from app.services.rag_service import RAGService


router = APIRouter()


def get_rag_service() -> RAGService:
    # FastAPI dependency to provide RAGService
    return RAGService()


@router.post("/ask")
async def ask_repository_question(
    request: RAGQueryRequest = Body(...),
    rag_service: RAGService = Depends(get_rag_service),
) -> BaseResponse:
    # Ask a question about a specific repository using RAG
    answer = rag_service.ask(request)
    return BaseResponse(
        message="RAG query executed successfully.",
        code=200,
        data={"answer": answer},
    )


@router.post("/ask-stream")
async def ask_repository_question_stream(
    request: RAGQueryRequest = Body(...),
    rag_service: RAGService = Depends(get_rag_service),
):
    # Ask a question about a specific repository using RAG with SSE streaming

    def event_generator():
        for item in rag_service.stream(request):
            yield f"data: {BaseResponse(message='RAG update', code=200, data=item).model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
