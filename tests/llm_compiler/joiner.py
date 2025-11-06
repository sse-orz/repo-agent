from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Dict, Optional, Tuple

from .constants import JOINER_FINISH, JOINER_REPLAN
from .output_parser import Task
from .log import get_logger

logger = get_logger("llm_compiler.joiner")


JOINER_PROMPT_TEMPLATE = """Given a repository analysis task and observations from executed tasks, decide the next action.

Your goal is to analyze a code repository and generate comprehensive documentation. Evaluate whether:
1. All necessary repository information has been collected (basic info, structure, commits)
2. Key code files have been analyzed (functions, classes, dependencies)
3. The information is sufficient to generate complete and accurate documentation

Output format:
Thought: <your reasoning about the analysis completeness>
Action: Finish(<summary of analysis and documentation plan>) OR Replan(<feedback for next plan>)

Examples:

Example 1 (Finish - Analysis Complete):
Thought: We have collected repository metadata, explored the complete directory structure, analyzed commit history, and examined all key code files including main.py, utils.py, and api.py. The observations contain comprehensive information about the repository's purpose, architecture, and key components. We have sufficient data to generate complete documentation.
Action: Finish(Repository analysis complete. The codebase is a REST API framework with 3 main modules: authentication, data processing, and API routing. Ready to generate structured wiki including overview, architecture, API documentation, and usage guide.)

Example 2 (Replan - Missing Code Analysis):
Thought: We successfully collected repository metadata and structure, but have only analyzed 2 files (main.py, config.py). The structure shows there are critical modules in src/core/ and src/services/ that haven't been analyzed yet. Without analyzing these core components, the documentation would be incomplete.
Action: Replan(Need to analyze additional critical files: src/core/engine.py, src/services/database.py, and src/services/cache.py to understand the core functionality before generating documentation)

Example 3 (Replan - Insufficient Repository Information):
Thought: We analyzed several code files, but we haven't collected basic repository information or explored the full directory structure. Without understanding the overall repository context, we cannot properly organize the documentation or understand the relationships between analyzed files.
Action: Replan(Need to first collect repository basic information and structure before continuing with code analysis. Use get_repo_basic_info_tool and get_repo_structure_tool to establish context.)
"""


class Joiner:
    """Joiner for LLMCompiler.

    Decides whether to finish or request a re-plan based on task observations
    and constructs context for the planner when needed.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        prompt_template: Optional[str] = None,
        prompt_template_final: Optional[str] = None,
    ):
        """Initialize the Joiner.

        Args:
            llm: The chat model.
            prompt_template: Default prompt template for Joiner.
            prompt_template_final: Prompt template used for final iteration (no re-plan).
        """
        self.llm = llm
        self.prompt_template = prompt_template or JOINER_PROMPT_TEMPLATE
        self.prompt_template_final = prompt_template_final or self.prompt_template

    def _parse_output(self, raw_answer: str) -> Tuple[str, str, bool]:
        """Parse the Joiner's raw output.

        Args:
            raw_answer: Raw LLM response text.

        Returns:
            thought: The reasoning text.
            answer: The content inside the action parentheses.
            is_replan: Whether the decision is to re-plan.
        """
        thought, answer, is_replan = "", "", False

        for line in raw_answer.split("\n"):
            if line.startswith("Action:"):
                # Extract content inside parentheses
                start = line.find("(")
                end = line.rfind(")")
                if start != -1 and end != -1:
                    answer = line[start + 1 : end]
                is_replan = JOINER_REPLAN in line
            elif line.startswith("Thought:"):
                thought = line.split("Thought:", 1)[1].strip()

        return thought, answer, is_replan

    def _build_scratchpad(self, tasks: Dict[int, Task]) -> str:
        """Build agent scratchpad (task-observation summary)."""
        scratchpad = ""

        for idx in sorted(tasks.keys()):
            task = tasks[idx]
            if task.is_join:
                continue

            scratchpad += task.get_thought_action_observation(
                include_action=True, include_thought=True, include_action_idx=False
            )
            scratchpad += "\n"

        return scratchpad.strip()

    def _generate_context_for_replanner(
        self, tasks: Dict[int, Task], joiner_thought: str
    ) -> str:
        """Generate re-planning context for the planner.

        Format:
        Previous Plan:

        1. action1(args)
        Observation: xxx
        2. action2(args)
        Observation: yyy

        Thought: <joiner_thought>

        Current Plan:
        """
        # Build the execution summary of the previous round
        previous_plan = ""
        for idx in sorted(tasks.keys()):
            task = tasks[idx]
            if task.is_join:
                continue

            previous_plan += task.get_thought_action_observation(
                include_action=True,
                include_thought=False,  # do not include per-task thought
                include_action_idx=True,  # include index prefix
            )
            previous_plan += "\n"

        previous_plan = previous_plan.strip()

        # Combine into the final context for the re-planner
        context = (
            f"Previous Plan:\n\n{previous_plan}\n\n"
            f"Thought: {joiner_thought}\n\n"
            f"Current Plan:\n\n"
        )

        return context

    async def join(
        self, input: str, tasks: Dict[int, Task], is_final: bool = False
    ) -> Tuple[str, str, bool]:
        """Run the Join step to decide Finish or Replan.

        Args:
            input: User query.
            tasks: Executed task dict including observations.
            is_final: If True, do not allow re-planning.

        Returns:
            A tuple of (thought, answer, is_replan).
        """
        logger.info(f"[Joiner] Start Join - is_final={is_final}, tasks={len(tasks)}")

        # Pick template depending on finality
        prompt_template = (
            self.prompt_template_final if is_final else self.prompt_template
        )

        # Build scratchpad from tasks
        scratchpad = self._build_scratchpad(tasks)
        logger.debug(f"[Joiner] Scratchpad length: {len(scratchpad)} chars")

        # Build user message (Question + observations)
        human_prompt = f"Question: {input}\n\n{scratchpad}"

        messages = [
            SystemMessage(content=prompt_template),
            HumanMessage(content=human_prompt),
        ]

        logger.debug(
            f"[Joiner] Calling LLM, human_prompt length: {len(human_prompt)} chars"
        )
        response = await self.llm.ainvoke(messages)

        # Log LLM response
        logger.info(
            f"[Joiner] LLM response received (length: {len(response.content)} chars)"
        )
        logger.debug(f"[Joiner] Raw LLM response:\n{response.content}")

        # Parse decision output
        thought, answer, is_replan = self._parse_output(response.content)

        # Force Finish in the final iteration even if model suggests Replan
        if is_final:
            original_is_replan = is_replan
            is_replan = False
            if original_is_replan:
                logger.info(
                    f"[Joiner] Final iteration, force Finish (original: Replan)"
                )

        # Log decision results
        decision = "Replan" if is_replan else "Finish"
        thought_preview = (
            thought[:100] + "..." if thought and len(thought) > 100 else thought or ""
        )
        answer_preview = (
            answer[:100] + "..." if answer and len(answer) > 100 else answer or ""
        )
        logger.info(
            f"[Joiner] Decision: {decision} - "
            f"thought={thought_preview}, answer={answer_preview}"
        )
        logger.debug(f"[Joiner] Full thought: {thought}")
        logger.debug(f"[Joiner] Full answer: {answer}")

        return thought, answer, is_replan
