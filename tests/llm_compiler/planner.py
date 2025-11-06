from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import BaseTool
from typing import Dict, List, Optional, Sequence

from .constants import END_OF_PLAN
from .output_parser import LLMCompilerPlanParser, Task
from .log import get_logger

logger = get_logger("llm_compiler.planner")

# Join action description
JOIN_DESCRIPTION = (
    "join() -> str:\n"
    " - Collects and combines results from prior actions.\n"
    " - A LLM agent is called upon invoking join to either finalize the user query or wait until the plans are executed.\n"
    " - join should always be the last action in the plan, and will be called in two scenarios:\n"
    "   (a) if the answer can be determined by gathering the outputs from tasks to generate the final response.\n"
    "   (b) if the answer cannot be determined in the planning phase before you execute the plans."
)

# The end of the planner prompt. It is used to ensure the format of the LLM response is correct.
PlannerPromptEnd = (
    "Remember, ONLY respond with the task list in the correct format! E.g.:\n"
    "idx. tool(arg_name=args)\n"
    "None\n"
)


def _generate_llm_compiler_prompt(
    tools: Sequence[BaseTool],
    repo_path: str,
    wiki_path: str,
    owner: str,
    repo_name: str,
    is_replan: bool = False,
) -> str:
    """Generate the system prompt for LLMCompiler.

    Args:
        tools: List of available tools.
        repo_path: Local repository path.
        wiki_path: Output wiki path.
        owner: Repository owner.
        repo_name: Repository name.
        is_replan: Whether the prompt is for a re-planning round.

    Returns:
        The system prompt string.
    """
    prefix = (
        "Given a repository analysis task, create a plan to analyze the code repository and generate documentation with the utmost parallelizability. "
        f"Each plan should comprise an action from the following {len(tools) + 1} types:\n"
    )

    # Tool descriptions
    for i, tool in enumerate(tools):
        prefix += f"{i+1}. {tool.description}\n"

    # Repository context information appended to the prompt
    prefix += f"Repository Analysis Task Information:\n"
    prefix += f" - Repository Local Path: {repo_path}\n"
    prefix += f" - Output Wiki Path: {wiki_path}\n"
    prefix += f" - Repository Owner: {owner}\n"
    prefix += f" - Repository Name: {repo_name}\n"

    # Description for the final join action
    prefix += f"{len(tools)+1}. {JOIN_DESCRIPTION}\n\n"

    # Guidelines
    prefix += (
        "Guidelines for Repository Analysis:\n"
        " - Each action described above contains input/output types and description.\n"
        "    - You must strictly adhere to the input and output types for each action.\n"
        "    - The action descriptions contain the guidelines. You MUST strictly follow those guidelines when you use the actions.\n"
        " - Each action in the plan should strictly be one of the above types. Follow the Python conventions for each action.\n"
        " - Each action MUST have a unique ID, which is strictly increasing.\n"
        " - **CRITICAL: Action Format Requirement** - Each action MUST be written in the EXACT format: `ID. tool_name(arguments)`\n"
        "    - Use a period (.) followed by a space after the ID number, NOT a colon (:) or equals sign (=)\n"
        "    - Correct format: `1. tool_name(arguments)`\n"
        "    - Incorrect formats (DO NOT USE): `1: tool_name(...)`, `1 = tool_name(...)`, `1.tool_name(...)`\n"
        "    - The format is strictly enforced by the parser. Using wrong format will cause parsing failure.\n"
        " - Inputs for actions can either be constants or outputs from preceding actions. "
        "In the latter case, use the format $id to denote the ID of the previous action whose output will be the input.\n"
        f" - Always call join as the last action in the plan. Say '{END_OF_PLAN}' after you call join\n"
        " - Ensure the plan maximizes parallelizability.\n"
        " - Only use the provided action types. If a query cannot be addressed using these, invoke the join action for the next steps.\n"
        " - Never explain the plan with comments (e.g. #).\n"
        " - Never introduce new actions other than the ones provided.\n\n"
        "Repository Analysis Workflow (Priority Order):\n"
        " - **Phase 1 - Information Collection (Parallel Execution):**\n"
        "    - Use get_repo_basic_info_tool to gather repository metadata (name, description, language, etc.)\n"
        "    - Use get_repo_structure_tool to explore directory hierarchy and file structure\n"
        "    - Use get_repo_commit_info_tool to analyze commit history and development activity\n"
        "    - These information collection tasks CAN and SHOULD be executed in parallel\n"
        " - **Phase 2 - Code Analysis (Parallel Execution):**\n"
        "    - Use code_file_analysis_tool to analyze key code files (functions, classes, dependencies)\n"
        "    - Use read_file_tool if detailed file content is needed\n"
        "    - Multiple file analyses CAN and SHOULD be executed in parallel\n"
        " - **Phase 3 - Documentation Generation (After All Analysis):**\n"
        "    - Use write_file_tool ONLY after completing all information collection and code analysis\n"
        "    - write_file_tool should depend on outputs from Phase 1 and Phase 2\n"
        "    - Generate structured wiki documents based on collected information\n\n"
        "Critical Constraints:\n"
        " - **DO NOT** use write_file_tool until all repository information has been collected and analyzed\n"
        " - Maximize parallelization within each phase (e.g., analyze multiple files simultaneously)\n"
        " - Respect phase dependencies: Phase 2 depends on Phase 1, Phase 3 depends on Phase 1 & 2\n\n"
    )

    if is_replan:
        prefix += (
            ' - You are given "Previous Plan" which is the plan that the previous agent created along with the execution results '
            "(given as Observation) of each plan and a general thought (given as Thought) about the executed results."
            'You MUST use these information to create the next plan under "Current Plan".\n'
            ' - When starting the Current Plan, you should start with "Thought" that outlines the strategy for the next plan.\n'
            " - In the Current Plan, you should NEVER repeat the actions that are already executed in the Previous Plan.\n"
            " - You must continue the task index from the end of the previous one. Do not repeat task indices.\n"
        )

    return prefix


class Planner:
    """Planner for LLMCompiler.

    Responsible for constructing the appropriate prompts and parsing the LLM's
    response into a structured set of executable tasks.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        tools: List[BaseTool],
        repo_path: str,
        wiki_path: str,
        owner: str,
        repo_name: str,
        stop: Optional[List[str]] = None,
    ):
        """Initialize the planner.

        Args:
            llm: The chat model.
            tools: List of tools available to the planner and executor.
            repo_path: Local repository path.
            wiki_path: Output wiki path.
            owner: Repository owner.
            repo_name: Repository name.
            stop: Stop words to terminate LLM generation (defaults to END_OF_PLAN).
        """
        self.llm = llm
        self.tools = tools
        self.stop = stop or [END_OF_PLAN]

        # Prepare both the initial and re-plan system prompts
        self.system_prompt = _generate_llm_compiler_prompt(
            tools=tools,
            repo_path=repo_path,
            wiki_path=wiki_path,
            owner=owner,
            repo_name=repo_name,
            is_replan=False,
        )
        self.system_prompt_replan = _generate_llm_compiler_prompt(
            tools=tools,
            repo_path=repo_path,
            wiki_path=wiki_path,
            owner=owner,
            repo_name=repo_name,
            is_replan=True,
        )

        # Create the output parser for translating LLM output into tasks
        self.parser = LLMCompilerPlanParser(tools=tools)

    async def plan(
        self, input: str, context: str = "", is_replan: bool = False
    ) -> Dict[int, Task]:
        """Generate a plan of executable tasks from the LLM.

        Args:
            input: The user query.
            context: Re-planning context (required when is_replan is True).
            is_replan: Whether this is a re-planning round.

        Returns:
            A dictionary mapping task -> {idx: Task}
        """
        # Select system prompt according to planning mode
        system_prompt = self.system_prompt_replan if is_replan else self.system_prompt

        # Build the human message
        if is_replan:
            assert context, "Context is required when is_replan=True"
            human_prompt = f"Question: {input}\n{context}"
        else:
            human_prompt = f"Question: {input}"

        # Invoke the LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
            SystemMessage(content=PlannerPromptEnd),
        ]

        logger.info(
            f"[Planner] Start planning - is_replan={is_replan}, input={input[:100]}..."
        )

        response = await self.llm.ainvoke(messages, stop=self.stop)

        # Log raw LLM response for traceability
        logger.info(
            f"[Planner] LLM response received (length: {len(response.content)} chars)"
        )
        logger.debug(f"[Planner] Raw LLM response:\n{response.content}")

        # Parse output into structured tasks
        tasks = self.parser.parse(response.content + "\n")

        # Log parsed tasks
        task_count = len(tasks)
        task_ids = sorted(tasks.keys()) if tasks else []
        logger.info(f"[Planner] Tasks parsed - count: {task_count}, ids: {task_ids}")

        # Detailed per-task logging to aid debugging
        for idx, task in tasks.items():
            logger.debug(
                f"[Planner] Task {idx}: name={task.name}, "
                f"args={task.args}, dependencies={task.dependencies}, "
                f"is_join={task.is_join}, thought={task.thought}"
            )

        return tasks
