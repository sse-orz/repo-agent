from typing import Dict, Literal, Optional
from typing_extensions import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from .planner import Planner
from .joiner import Joiner
from .task_fetching_unit import TaskFetchingUnit
from .output_parser import Task
from .log import setup_logging, get_logger

logger = get_logger("llm_compiler.main")


class LLMCompilerState(TypedDict, total=False):
    """LLMCompiler state definition."""

    # === I/O ===
    input: str  # Analysis task description
    output: Optional[str]  # Final answer

    # === Repository info ===
    repo_path: str  # Local repository path
    wiki_path: str  # Wiki output path
    owner: Optional[str]  # Repository owner
    repo_name: Optional[str]  # Repository name

    # === Control flow ===
    is_replan: bool  # Whether a re-planning round is needed
    replan_count: int  # Current re-planning count
    is_final_iteration: bool  # Whether this is the final iteration

    # === Data ===
    tasks: Dict[
        int, Dict
    ]  # Task dict {idx: Task dict} - serialized Task dicts, not Task objects
    context: str  # Re-planning context

    # === Joiner output ===
    joiner_thought: Optional[str]  # Joiner's reasoning


class LLMCompilerAgent:
    def __init__(
        self,
        llm: BaseChatModel,
        tools: list[BaseTool],
        repo_path: str,
        wiki_path: str,
        owner: Optional[str] = None,
        repo_name: Optional[str] = None,
        max_replans: int = 10,
        log_dir: str = ".logs",
    ):
        """Initialize the LLMCompiler Agent.

        Args:
            llm: The chat model.
            tools: List of tools the agent can use.
            repo_path: Local repository path.
            wiki_path: Wiki output path.
            owner: Repository owner.
            repo_name: Repository name.
            max_replans: Max number of re-plans.
            log_dir: Log directory path.
        """
        # Initialize logging subsystem
        setup_logging(log_dir)
        logger.info(
            f"[Main] Initialize LLMCompiler Agent - repo_path={repo_path}, wiki_path={wiki_path}"
        )

        self.llm = llm
        self.tools = tools
        self.repo_path = repo_path
        self.wiki_path = wiki_path
        self.owner = owner
        self.repo_name = repo_name
        self.max_replans = max_replans

        logger.info(
            f"[Main] Config - max_replans={max_replans}, num_tools={len(tools)}"
        )

        # Create core components: planner, task_fetching_unit, joiner
        self.planner = Planner(llm, tools, repo_path, wiki_path, owner, repo_name)
        self.joiner = Joiner(llm)
        self.task_fetching_unit_class = TaskFetchingUnit

        # Create checkpoint memory
        self.memory = InMemorySaver()

        # Build the LangGraph application
        self.app = self._build_graph()
        logger.info("[Main] LLMCompiler Agent initialized")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(LLMCompilerState)

        # Add nodes
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("joiner", self._joiner_node)

        # Set edges for the main flow
        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "executor")
        workflow.add_edge("executor", "joiner")

        # Conditional edges: continue to planner if re-planning is needed
        workflow.add_conditional_edges(
            "joiner",
            self._should_continue,
            {
                "planner": "planner",
                "end": END,
            },
        )

        return workflow.compile(checkpointer=self.memory)

    async def _planner_node(self, state: LLMCompilerState) -> Dict:
        """Planner node."""
        replan_count = state.get("replan_count", 0)
        is_replan = state.get("is_replan", False)

        logger.info(
            f"[Main] === Planner Node === iterations: {replan_count}, is_replan: {is_replan}"
        )

        # Run the planner
        tasks = await self.planner.plan(
            input=state["input"], context=state.get("context", ""), is_replan=is_replan
        )

        logger.info(f"[Main] Planner node finished, generated {len(tasks)} tasks")

        tasks_dict = {idx: task.to_dict() for idx, task in tasks.items()}
        return {"tasks": tasks_dict}

    async def _executor_node(self, state: LLMCompilerState) -> Dict:
        """Executor node."""
        logger.info(f"[Main] === Executor Node === start executing tasks")
        tasks_dict = state["tasks"]
        # Rebuild Task objects from dictionary
        tasks = {
            idx: Task.from_dict(task_data, tools=self.tools)
            for idx, task_data in tasks_dict.items()
        }

        # Create scheduling unit instance
        tfu = self.task_fetching_unit_class()
        tfu.set_tasks(tasks)

        # Schedule execution
        executed_tasks = await tfu.schedule()
        executed_tasks_dict = {
            idx: task.to_dict() for idx, task in executed_tasks.items()
        }

        logger.info(f"[Main] Executor node finished, all tasks executed")

        return {"tasks": executed_tasks_dict}

    async def _joiner_node(self, state: LLMCompilerState) -> Dict:
        """Joiner node."""
        replan_count = state.get("replan_count", 0)
        # Determine whether this is the final iteration
        is_final = replan_count >= self.max_replans

        logger.info(
            f"[Main] === Joiner Node === iterations: {replan_count}, is_final: {is_final}"
        )

        tasks_dict = state["tasks"]
        tasks = {
            idx: Task.from_dict(task_data, tools=self.tools)
            for idx, task_data in tasks_dict.items()
        }

        # Invoke joiner for decision making
        thought, answer, is_replan = await self.joiner.join(
            input=state["input"], tasks=tasks, is_final=is_final
        )

        # Build state updates
        updates = {
            "joiner_thought": thought,
            "is_replan": is_replan,
        }

        if is_replan:
            # Generate context for re-planning
            context = self.joiner._generate_context_for_replanner(tasks, thought)
            updates["context"] = context
            new_replan_count = state.get("replan_count", 0) + 1
            updates["replan_count"] = new_replan_count
            logger.info(
                f"[Main] Joiner decision: Replan, will run re-plan round {new_replan_count}"
            )
        else:
            # Final answer
            updates["output"] = answer
            answer_preview = (
                answer[:200] + "..." if answer and len(answer) > 200 else answer or ""
            )
            logger.info(f"[Main] Joiner decision: Finish, task completed")
            logger.info(f"[Main] Final answer: {answer_preview}")

        return updates

    def _should_continue(self, state: LLMCompilerState) -> Literal["planner", "end"]:
        """Decide whether to continue with re-planning or finish."""
        if (
            state.get("is_replan", False)
            and state.get("replan_count", 0) < self.max_replans
        ):
            return "planner"
        return "end"

    async def ainvoke(self, query: str, config: Optional[Dict] = None) -> str:
        """Asynchronously invoke the agent and return the final answer.

        Args:
            query: The user query.
            config: Optional LangGraph config.

        Returns:
            The final answer string.
        """
        import time

        start_time = time.time()
        query_preview = query[:200] + "..." if len(query) > 200 else query
        logger.info(f"[Main] ========== Start running LLMCompiler ==========")
        logger.info(f"[Main] User query: {query_preview}")

        if config is None:
            config = {
                "configurable": {"thread_id": "default"},
                "recursion_limit": self.max_replans * 3 + 10,
            }

        # Initial state
        initial_state: LLMCompilerState = {
            "input": query,
            "output": None,
            "repo_path": self.repo_path,
            "wiki_path": self.wiki_path,
            "owner": self.owner,
            "repo_name": self.repo_name,
            "is_replan": False,
            "replan_count": 0,
            "tasks": {},
            "context": "",
            "joiner_thought": None,
            "is_final_iteration": False,
        }

        logger.info(f"[Main] Initial state ready, max_replans={self.max_replans}")

        # Run the graph
        final_state = await self.app.ainvoke(initial_state, config)

        total_time = time.time() - start_time
        final_replan_count = final_state.get("replan_count", 0)
        output = final_state.get("output")
        output_preview = (
            output[:200] + "..." if output and len(output) > 200 else output or ""
        )
        logger.info(f"[Main] ========== LLMCompiler finished ==========")
        logger.info(
            f"[Main] Total time: {total_time:.2f}s, re-plans: {final_replan_count}"
        )
        logger.info(f"[Main] Final output: {output_preview}")

        return final_state["output"]


# ============= LLMCompilerAgent Test =============


# ============= Added Tests per Plan =============
from config import CONFIG
from agents.tools import (
    read_file_tool,
    write_file_tool,
    get_repo_structure_tool,
    code_file_analysis_tool,
    get_repo_basic_info_tool,
    get_repo_commit_info_tool,
)
from langchain.tools import tool
import asyncio
import os
import json
import re
import glob

# Agents to be wrapped as tools
from agents.code_analysis_agent import CodeAnalysisAgent
from agents.doc_generation_agent import DocGenerationAgent
from agents.repo_info_agent import RepoInfoAgent
from agents.summary_agent import SummaryAgent


# -------- Helpers: wrap sync tools as async tools --------
@tool(
    description="read_file_tool_async(file_path: str) -> dict: Read the content of a file in the repository."
)
async def read_file_tool_async(file_path: str):
    return read_file_tool.func(file_path)


@tool(
    description="write_file_tool_async(file_path: str, content: str) -> dict: Write content to a file in the repository or wiki."
)
async def write_file_tool_async(file_path: str, content: str):
    return write_file_tool.func(file_path=file_path, content=content)


@tool(
    description="get_repo_structure_tool_async(repo_path: str) -> dict: Retrieve the directory and file structure of the repository."
)
async def get_repo_structure_tool_async(repo_path: str):
    return get_repo_structure_tool.func(repo_path)


@tool(
    description="code_file_analysis_tool_async(file_path: str) -> dict: Analyze a code file using Tree-sitter."
)
async def code_file_analysis_tool_async(file_path: str):
    return code_file_analysis_tool.func(file_path)


@tool(
    description="get_repo_basic_info_tool_async(owner: str, repo: str, platform: str = 'github') -> dict: Get basic repository information (name, description, topics, etc.)."
)
async def get_repo_basic_info_tool_async(
    owner: str, repo: str, platform: str = "github"
):
    return get_repo_basic_info_tool.func(owner=owner, repo=repo, platform=platform)


@tool(
    description="get_repo_commit_info_tool_async(owner: str, repo: str, platform: str = 'github', max_num: int = 10) -> dict: Get the latest commits and commit metadata."
)
async def get_repo_commit_info_tool_async(
    owner: str, repo: str, platform: str = "github", max_num: int = 10
):
    return get_repo_commit_info_tool.func(
        owner=owner, repo=repo, platform=platform, max_num=max_num
    )


# -------- Agents wrapped as tools --------
@tool(
    description="repo_info_agent_tool(repo_path: str, wiki_path: str, owner: str, repo_name: str) -> dict: Aggregate repository information using structure, basic info, and commits."
)
async def repo_info_agent_tool(
    repo_path: str, wiki_path: str, owner: str, repo_name: str
) -> dict:
    from langchain_core.messages import SystemMessage, HumanMessage

    # First call RepoInfoAgent to collect basic repository information
    ria = RepoInfoAgent(
        repo_path=repo_path,
        wiki_path=wiki_path,
    )
    result = ria.run(owner=owner, repo_name=repo_name)

    # Find and analyze Markdown files within the repository
    md_files = []
    file_exts = {".md"}
    ignore_dirs = [".git", "node_modules", "__pycache__", ".vscode", ".idea"]
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for f in files:
            if os.path.splitext(f)[1] in file_exts:
                md_files.append(os.path.join(root, f))

    # Analyze discovered Markdown files with the LLM to extract documentation insights
    markdown_analysis = {}
    if md_files:
        llm = CONFIG.get_llm()

        for md_file in md_files:
            try:
                # Read file content
                file_content = read_file_tool.func(md_file)
                content = file_content.get("content", "")

                if not content or len(content) < 50:  # Skip files that are too short
                    continue

                # Limit content length to avoid exceeding LLM context window
                max_content_length = 10000
                if len(content) > max_content_length:
                    content = (
                        content[:max_content_length] + "\n\n[Content truncated...]"
                    )

                # Use the LLM to analyze the Markdown content
                analysis_prompt = SystemMessage(
                    content="""You are an expert in analyzing code repositories. Analyze the following Markdown document and extract the following key information:
1. Project Overview (main functionality, goals, intended use)
2. Key Features (major features or highlights)
3. Quick Start Guide (installation, configuration, basic usage)
4. Architecture Overview (if available)
5. Main Components/Modules (if available)

Please return the analysis in JSON format as follows:
{
    "overview": "Project overview",
    "key_features": ["Feature 1", "Feature 2", ...],
    "quick_start": "Summary of quick start guide",
    "architecture": "Architecture overview (if available)",
    "main_components": ["Component 1", "Component 2", ...]
}

If some information is not present in the document, set those fields to null or empty arrays."""
                )

                human_prompt = HumanMessage(
                    content=f"Please analyze the following Markdown content.\n\nFile path: {md_file}\n\nContent:\n{content}"
                )

                # Invoke the LLM for analysis
                response = await llm.ainvoke([analysis_prompt, human_prompt])
                analysis_text = (
                    response.content if hasattr(response, "content") else str(response)
                )

                # Attempt to parse JSON from the response
                try:
                    # Try extracting the JSON segment
                    json_match = re.search(r"\{[\s\S]*\}", analysis_text)
                    if json_match:
                        analysis_json = json.loads(json_match.group())
                    else:
                        # If no JSON found, build a simple fallback structure
                        analysis_json = {
                            "overview": (
                                analysis_text[:500]
                                if len(analysis_text) > 500
                                else analysis_text
                            ),
                            "raw_analysis": analysis_text,
                        }
                except json.JSONDecodeError:
                    # JSON parsing failed; use raw text as fallback
                    analysis_json = {
                        "overview": (
                            analysis_text[:500]
                            if len(analysis_text) > 500
                            else analysis_text
                        ),
                        "raw_analysis": analysis_text,
                    }

                # Use the filename as the key (strip the path)
                file_key = os.path.basename(md_file)
                markdown_analysis[file_key] = analysis_json

            except Exception as e:
                # Log error but continue the overall flow when analysis fails
                print(f"  ⚠ Error analyzing Markdown file {md_file}: {str(e)}")
                continue

    # Add Markdown analysis results to the return payload
    if markdown_analysis:
        result["markdown_analysis"] = markdown_analysis
        result["markdown_files_analyzed"] = list(markdown_analysis.keys())

    return result


@tool(
    description="code_analysis_agent_tool(repo_path: str, max_files: int = 80) -> dict: Perform repository code analysis using Tree-sitter across selected files."
)
async def code_analysis_agent_tool(repo_path: str, max_files: int = 80) -> dict:
    ca = CodeAnalysisAgent(repo_path=repo_path)
    # Collect a subset of code files to control scale
    code_exts = {".py", ".ts", ".js", ".tsx", ".jsx", ".go"}
    file_list = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in [".git", "node_modules", "__pycache__"]]
        for f in files:
            if len(file_list) >= max_files:
                break
            if os.path.splitext(f)[1] in code_exts:
                file_list.append(os.path.join(root, f))
        if len(file_list) >= max_files:
            break
    return ca.run(
        file_list=file_list,
        batch_size=10,
        parallel_batches=True,
        max_workers=8,
    )


@tool(
    description="doc_generation_agent_tool(wiki_path: str, repo_info_json: str, code_analysis_json: str) -> dict: Generate docs from provided repo info and code analysis."
)
async def doc_generation_agent_tool(
    wiki_path: str, repo_info_json: str, code_analysis_json: str
) -> dict:
    llm = CONFIG.get_llm()
    # 解析 repo_info
    try:
        repo_info = (
            json.loads(repo_info_json)
            if isinstance(repo_info_json, str)
            else repo_info_json
        )
    except Exception:
        repo_info = {
            "repo_name": "",
            "description": "",
            "main_language": "",
            "structure": [],
            "commits": [],
        }
    # 解析 code_analysis
    try:
        code_analysis = (
            json.loads(code_analysis_json)
            if isinstance(code_analysis_json, str)
            else code_analysis_json
        )
    except Exception:
        code_analysis = {"files": {}, "summary": {}, "analyzed_files": 0}
    dga = DocGenerationAgent(llm, [write_file_tool, read_file_tool])
    return dga.run(
        repo_info=repo_info, code_analysis=code_analysis, wiki_path=wiki_path
    )


@tool(
    description='summary_agent_tool(wiki_path: str, docs_json: str, repo_info_json: str = "", code_analysis_json: str = "") -> dict: Summarize generated docs, optionally incorporating repo info and code analysis.'
)
async def summary_agent_tool(
    wiki_path: str,
    docs_json: str,
    repo_info_json: str = "",
    code_analysis_json: str = "",
) -> dict:
    llm = CONFIG.get_llm()
    try:
        docs = (
            json.loads(docs_json) if isinstance(docs_json, str) else (docs_json or [])
        )
    except Exception:
        docs = []
    try:
        repo_info = (
            json.loads(repo_info_json)
            if isinstance(repo_info_json, str) and repo_info_json
            else None
        )
    except Exception:
        repo_info = None
    try:
        code_analysis = (
            json.loads(code_analysis_json)
            if isinstance(code_analysis_json, str) and code_analysis_json
            else None
        )
    except Exception:
        code_analysis = None
    sa = SummaryAgent(llm, [write_file_tool, read_file_tool])
    return sa.run(
        docs=docs, wiki_path=wiki_path, repo_info=repo_info, code_analysis=code_analysis
    )


async def test_tools_only():
    llm = CONFIG.get_llm()
    repo_path = "./.repos/langchain-ai_langchain"
    wiki_path = "./.wikis/langchain-ai_langchain/tools_only"
    tools = [
        read_file_tool_async,
        get_repo_structure_tool_async,
        code_file_analysis_tool_async,
        write_file_tool_async,
        get_repo_basic_info_tool_async,
        get_repo_commit_info_tool_async,
    ]
    agent = LLMCompilerAgent(
        llm=llm,
        tools=tools,
        repo_path=repo_path,
        wiki_path=wiki_path,
        owner="langchain-ai",
        repo_name="langchain",
    )
    query = (
        "Please analyze the langchain-ai/langchain repository. (you may use the tools provided) "
        "You should generate README, ARCHITECTURE, API, and DEVELOPMENT documents along with an index in the wiki directory, "
        "and finally return a summary."
    )
    output = await agent.ainvoke(query)
    assert output is not None and len(str(output).strip()) > 0
    print("[tools_only] output:\n", str(output)[:500])


async def test_agents_as_tools():
    llm = CONFIG.get_llm()
    # repo_path = "./.repos/langchain-ai_langchain"
    # wiki_path = "./.wikis/langchain-ai_langchain/agents_as_tools"
    repo_path = "./.repos/cloudwego_eino"
    wiki_path = "./.wikis/cloudwego_eino"
    tools = [
        repo_info_agent_tool,
        code_analysis_agent_tool,
        doc_generation_agent_tool,
        summary_agent_tool,
    ]
    agent = LLMCompilerAgent(
        llm=llm,
        tools=tools,
        repo_path=repo_path,
        wiki_path=wiki_path,
        # owner="langchain-ai",
        # repo_name="langchain",
        owner="cloudwego",
        repo_name="eino",
    )
    # query = (
    #     "Please analyze the langchain-ai/langchain repository. (you may use the encapsulated Agent tools) "
    #     "then generate README, ARCHITECTURE, API, and DEVELOPMENT documents along with an index in the wiki directory, "
    #     "and finally return a summary."
    # )
    query = (
        "Please analyze the cloudwego/eino repository. (you may use the encapsulated Agent tools) "
        "then generate README, ARCHITECTURE, API, and DEVELOPMENT documents along with an index in the wiki directory, "
        "and finally return a summary."
    )
    output = await agent.ainvoke(query)
    assert output is not None and len(str(output).strip()) > 0
    print("[agents_as_tools] output:\n", str(output)[:500])


if __name__ == "__main__":
    # asyncio.run(test_tools_only())
    asyncio.run(test_agents_as_tools())
