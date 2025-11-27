from pydantic import BaseModel, Field


class RAGQueryRequest(BaseModel):
    # Request body for repository RAG question answering

    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")
    platform: str = Field(
        "github", description="Code hosting platform (github, gitlab, etc.)"
    )
    question: str = Field(..., description="User question about the repository")
