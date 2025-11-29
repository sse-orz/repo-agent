from pydantic import BaseModel, Field
from typing import List
from enum import Enum


class BranchMode(str, Enum):
    ALL = "all"
    CODE = "code"
    REPO = "repo"
    CHECK = "check"


class AgentMode(str, Enum):
    SUB = "sub"
    MOE = "moe"


class GenerateRequest(BaseModel):
    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")
    platform: str = Field("github", description="Platform (github, gitlab, etc.)")
    need_update: bool = Field(
        False, description="Whether to update existing documentation"
    )
    branch_mode: BranchMode = Field(
        BranchMode.ALL,
        description="Branch mode: all (repo+code), code only, repo only, or check",
    )
    mode: str = Field("fast", description="Generation mode (fast or smart)")
    max_workers: int = Field(50, description="Maximum number of workers")
    log: bool = Field(False, description="Enable logging")


class GenerateRequestWrapper(BaseModel):
    mode: AgentMode = Field(AgentMode.SUB, description="Agent mode: sub or moe")
    request: GenerateRequest = Field(..., description="Generation request parameters")


class FileInfo(BaseModel):
    name: str = Field(..., description="File name")
    path: str = Field(..., description="Relative file path")
    url: str = Field(..., description="URL to access the file")
    size: int = Field(..., description="File size in bytes")


class GenerateResponseData(BaseModel):
    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")
    wiki_path: str = Field(..., description="Local wiki directory path")
    wiki_url: str = Field(..., description="Base URL to access wiki files")
    files: List[FileInfo] = Field(
        default_factory=list, description="List of generated files"
    )
    total_files: int = Field(0, description="Total number of files generated")


class WikiItem(BaseModel):
    owner: str = Field(..., description="Repository owner")
    repo: str = Field(..., description="Repository name")
    wiki_path: str = Field(..., description="Local wiki directory path")
    wiki_url: str = Field(..., description="URL to access wiki files")
    total_files: int = Field(0, description="Total number of files")
    generated_at: str = Field(..., description="Last modification time")


class ListWikisResponseData(BaseModel):
    wikis: List[WikiItem] = Field(default_factory=list, description="List of wikis")
    total_wikis: int = Field(0, description="Total number of wikis")
