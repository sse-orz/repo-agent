import asyncio
import json
import time
from enum import Enum
from typing import List, Optional, Any, Dict
from fastapi import FastAPI, Body, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="Mock Agent API")

# 允许跨域，方便前端调试
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. 复刻数据模型 (保持与后端一致)
# ==========================================


class AgentMode(str, Enum):
    SUB = "sub"
    MOE = "moe"


class RequestData(BaseModel):
    owner: str
    repo: str


class GenerateRequestWrapper(BaseModel):
    mode: str = "sub"
    request: RequestData


class BaseResponse(BaseModel):
    message: str
    code: int
    data: Optional[Any] = None


class WikiFilesResponse(BaseModel):
    files: List[str]
    total_files: int


# ==========================================
# 2. 模拟内存数据库 (Mock Data)
# ==========================================

# 预存一些数据，用于测试 /list 和 /wikis 接口
MOCK_DB = [
    {
        "owner": "octocat",
        "repo": "legacy-project",
        "mode": "sub",
        "updated_at": "2023-10-01T12:00:00",
        "files": ["README.md", "main.py"],
        "total_files": 2,
    }
]


def find_mock_wiki(owner: str, repo: str, mode: str):
    return next(
        (
            item
            for item in MOCK_DB
            if item["owner"] == owner and item["repo"] == repo and item["mode"] == mode
        ),
        None,
    )


# ==========================================
# 3. 接口实现
# ==========================================


# ---------------------------------------------------
# 接口 1: 同步生成 /generate
# ---------------------------------------------------
@app.post("/api/v1/agents/generate")
async def generate_agent_documentation(
    wrapper: GenerateRequestWrapper = Body(...),
):
    owner = wrapper.request.owner
    repo = wrapper.request.repo

    print(f"[Sync] Received generate request: {owner}/{repo}")

    # 模拟耗时
    await asyncio.sleep(2.0)

    # 1. 模拟: 已存在 (Repo名为 "exist-sync")
    if repo == "exist-sync":
        existing_data = {"id": 999, "content": "Old Content", "status": "done"}
        return BaseResponse(
            message="Existing documentation found.", code=200, data=existing_data
        )

    # 2. 模拟: 预处理失败 (Repo名为 "fail-sync")
    if repo == "fail-sync":
        return BaseResponse(
            message="Failed to preprocess repository.", code=500, data=None
        )

    # 3. 模拟: 成功生成
    new_data = {
        "owner": owner,
        "repo": repo,
        "files": ["README.md", "api.py", "utils.py"],
        "summary": "Generated via sync mode",
    }

    # 将新生成的假数据加入 mock db，这样调用 /list 就能看到了
    if not find_mock_wiki(owner, repo, wrapper.mode):
        MOCK_DB.append(
            {
                "owner": owner,
                "repo": repo,
                "mode": wrapper.mode,
                "files": new_data["files"],
                "total_files": len(new_data["files"]),
            }
        )

    return BaseResponse(
        message="Agent documentation generated successfully.", code=200, data=new_data
    )


# ---------------------------------------------------
# 接口 2: 流式生成 /generate-stream
# ---------------------------------------------------
@app.post("/api/v1/agents/generate-stream")
async def generate_agent_documentation_stream(
    wrapper: GenerateRequestWrapper = Body(...),
):
    owner = wrapper.request.owner
    repo = wrapper.request.repo
    mode = wrapper.mode

    print(f"[Stream] Received stream request: {owner}/{repo}")

    # 1. 模拟: 已存在 (Repo名为 "exist-stream")
    # 注意：原代码逻辑是如果在 Stream 接口发现已存在，直接返回 JSON Response，而不是 StreamResponse
    if repo == "exist-stream":
        existing_wiki = {"title": "Existing Wiki", "status": "done"}
        return BaseResponse(
            message="Existing documentation found.", code=200, data=existing_wiki
        )

    # 2. 模拟: 预处理失败 (Repo名为 "fail-stream")
    if repo == "fail-stream":
        return BaseResponse(
            message="Failed to preprocess repository.", code=500, data=None
        )

    # 3. 模拟流式生成器
    async def event_generator():
        # 模拟生成过程中的阶段
        stages = [
            {"stage": "init", "msg": "Initializing agent environment..."},
            {"stage": "cloning", "msg": f"Cloning repository {owner}/{repo}..."},
            {"stage": "analyzing", "msg": "Analyzing file dependency graph..."},
            {"stage": "generating", "msg": "LLM is generating documentation..."},
            {"stage": "saving", "msg": "Saving results to database..."},
        ]

        # 模拟 wiki_info 对象 (原代码中每次循环都会获取这个对象)
        mock_wiki_info = {
            "owner": owner,
            "repo": repo,
            "status": "processing",
            "files": [],
            "total_files": 0,
        }

        should_error = repo == "error-stream"  # Repo名为 "error-stream" 时模拟中断

        for i, step in enumerate(stages):
            await asyncio.sleep(1.0)  # 模拟每个步骤耗时

            # 动态更新 wiki_info 以模拟进度的变化（供前端展示）
            mock_wiki_info["total_files"] = i * 2
            mock_wiki_info["files"] = [f"file_{x}.py" for x in range(i * 2)]

            progress_data = {
                "stage": step["stage"],
                "message": step["msg"],
                "wiki_info": mock_wiki_info,  # 关键：原代码会返回这个字段
            }

            # 构造符合 SSE 规范的数据: data: {...}\n\n
            resp_json = BaseResponse(
                message=progress_data["message"], code=200, data=progress_data
            ).model_dump_json()

            yield f"data: {resp_json}\n\n"

        # 模拟结束或报错
        if should_error:
            # 模拟最后阶段报错
            error_data = {"error": "Simulated connection timeout during generation"}
            err_resp = BaseResponse(
                message="Documentation generation failed", code=500, data=error_data
            )
            yield f"data: {err_resp.model_dump_json()}\n\n"
        else:
            # 模拟完成
            progress_data_done = {
                "stage": "done",
                "message": "Generation finished",
                "wiki_info": mock_wiki_info,
            }

            # 发送最后一条进度 done
            yield f"data: {BaseResponse(message='Generation finished', code=200, data=progress_data_done).model_dump_json()}\n\n"

            # 发送最终结果 Result (generation_complete=True)
            final_data = {
                "wiki_content": "# Generated Doc\nMock content...",
                "generation_complete": True,  # 这是一个让前端停止轮询的重要标志
                "files": ["file_1.py", "file_2.py", "README.md"],
                "total_files": 3,
            }

            # 更新 Mock DB
            if not find_mock_wiki(owner, repo, mode):
                MOCK_DB.append(
                    {
                        "owner": owner,
                        "repo": repo,
                        "mode": mode,
                        "files": final_data["files"],
                        "total_files": final_data["total_files"],
                    }
                )

            success_resp = BaseResponse(
                message="Documentation generation completed successfully",
                code=200,
                data=final_data,
            )
            yield f"data: {success_resp.model_dump_json()}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------
# 接口 3: 获取列表 /list
# ---------------------------------------------------
@app.get("/api/v1/agents/list")
async def list_documentation():
    print("[List] Returning wiki list")
    return BaseResponse(
        message="List of generated wikis retrieved successfully.",
        code=200,
        data=MOCK_DB,
    )


# ---------------------------------------------------
# 接口 4: 获取详情/文件 /wikis/{owner}/{repo}
# ---------------------------------------------------
@app.get("/api/v1/agents/wikis/{owner}/{repo}")
async def get_wiki_files(owner: str, repo: str, mode: AgentMode = Query(AgentMode.SUB)):
    print(f"[Wiki Info] Fetching info for {owner}/{repo} mode={mode}")

    # 查找 mock 数据
    wiki = find_mock_wiki(owner, repo, mode)

    if not wiki:
        # 没找到，返回空列表 (模拟处理中或未开始)
        return BaseResponse(
            code=200,
            message="No wiki files found yet",
            data=WikiFilesResponse(files=[], total_files=0),
        )

    # 找到了
    return BaseResponse(
        code=200,
        message="Success",
        data=WikiFilesResponse(
            files=wiki.get("files", []), total_files=wiki.get("total_files", 0)
        ),
    )


if __name__ == "__main__":
    import uvicorn

    # 运行在 localhost:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
