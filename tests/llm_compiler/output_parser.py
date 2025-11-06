import ast
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.transform import BaseTransformOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool

from .constants import THOUGHT_PATTERN, ACTION_PATTERN, ID_PATTERN
from .log import get_logger

logger = get_logger("llm_compiler.parser")


@dataclass
class Task:
    """Task definition."""

    idx: int
    name: str
    tool: Union[
        Callable, BaseTool
    ]  # Tool function or BaseTool instance (lambda for join)
    args: tuple
    dependencies: List[int]
    stringify_rule: Optional[Callable] = None
    thought: Optional[str] = None
    observation: Optional[str] = None
    is_join: bool = False

    async def __call__(self) -> Any:
        """Execute the task via its bound tool.

        Supports both LangChain BaseTool and plain async callables.
        """

        if isinstance(self.tool, BaseTool):
            # Convert args to a dict (LangChain tools expect dict input)
            if self.args:
                # If first arg is a dict (keyword style), use it as-is
                if len(self.args) == 1 and isinstance(self.args[0], dict):
                    tool_input = self.args[0]
                # Single string might be raw input or key=value string
                elif len(self.args) == 1 and isinstance(self.args[0], str):
                    # If it looks like keyword string, parse it
                    if "=" in self.args[0]:
                        tool_input = _parse_keyword_args(self.args[0])
                    else:
                        # Otherwise, pass as single input field
                        tool_input = {"input": self.args[0]}
                else:
                    # Fallback: pack into list under 'args'
                    tool_input = (
                        self.args[0]
                        if len(self.args) == 1
                        else {"args": list(self.args)}
                    )
            else:
                tool_input = {}
            return await self.tool.ainvoke(tool_input)
        # Plain callable branch
        elif callable(self.tool):
            # If args contain a single dict, treat it as kwargs
            if self.args and len(self.args) == 1 and isinstance(self.args[0], dict):
                return await self.tool(**self.args[0])
            else:
                return await self.tool(*self.args)
        else:
            raise TypeError(f"Tool {self.name} is not callable: {type(self.tool)}")

    def get_thought_action_observation(
        self,
        include_action: bool = True,
        include_thought: bool = True,
        include_action_idx: bool = False,
    ) -> str:
        """Generate a Thought-Action-Observation string snippet."""
        result = ""

        if self.thought and include_thought:
            result += f"Thought: {self.thought}\n"

        if include_action:
            idx_prefix = f"{self.idx}. " if include_action_idx else ""
            if self.stringify_rule:
                result += f"{idx_prefix}{self.stringify_rule(self.args)}\n"
            else:
                args_str = str(self.args[0]) if len(self.args) == 1 else str(self.args)
                result += f"{idx_prefix}{self.name}({args_str})\n"

        if self.observation is not None:
            result += f"Observation: {self.observation}\n"

        return result

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the Task to a dict.

        Returns:
            A JSON-serializable dict (excluding tool and stringify_rule).
        """

        def _make_serializable(obj):
            """Recursively convert nested objects into serializable values."""
            if isinstance(obj, dict):
                return {k: _make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_make_serializable(item) for item in obj]
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            else:
                # Fallback: stringify unknown types
                return str(obj)

        # Convert args into a JSON-serializable form
        args_list = _make_serializable(self.args)

        return {
            "idx": self.idx,
            "name": self.name,
            "args": args_list,
            "dependencies": self.dependencies,
            "thought": self.thought,
            "observation": self.observation,
            "is_join": self.is_join,
        }

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], tools: Optional[Sequence[BaseTool]] = None
    ) -> "Task":
        """Reconstruct a Task from a serialized dict.

        Args:
            data: Serialized task dictionary.
            tools: Optional tools list to rebind tool and stringify_rule.

        Returns:
            Task instance.
        """
        # Restore args tuple
        args = tuple(data.get("args", []))

        # Re-bind tool and stringify_rule if tools are provided
        tool = None
        stringify_rule = None

        if data.get("is_join", False):
            tool = lambda: None
        elif tools:
            tool_name = data.get("name", "")
            try:
                found_tool = tools[[t.name for t in tools].index(tool_name)]
                # Obtain stringify_rule from the original tool object
                stringify_rule = getattr(found_tool, "stringify_rule", None)

                if isinstance(found_tool, BaseTool):
                    # Directly use BaseTool
                    tool = found_tool
                elif hasattr(found_tool, "func"):
                    tool = found_tool.func
                elif hasattr(found_tool, "coroutine"):
                    tool = found_tool.coroutine
                else:
                    tool = found_tool
            except (ValueError, IndexError):
                # Tool not found; keep None (will raise at call time)
                tool = None
                stringify_rule = None

        return cls(
            idx=data["idx"],
            name=data["name"],
            tool=tool,
            args=args,
            dependencies=data.get("dependencies", []),
            stringify_rule=stringify_rule,
            thought=data.get("thought"),
            observation=data.get("observation"),
            is_join=data.get("is_join", False),
        )


def _parse_keyword_args(args_str: str) -> dict:
    """Parse a keyword-argument string into a dict.

    Args:
        args_str: String like 'file_path="README.md", max_num=10'.

    Returns:
        Parsed dict.
    """
    if not args_str or args_str.strip() == "":
        return {}

    result = {}
    # Regex to match key=value pairs, handling quoted values
    pattern = r'(\w+)\s*=\s*("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[^,]+)'
    matches = re.findall(pattern, args_str)

    for key, value in matches:
        key = key.strip()
        value = value.strip()

        # Attempt to parse values, unquoting strings, literal-eval others
        try:
            # Strip quotes for quoted strings
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                result[key] = value[1:-1]
            else:
                # Try parsing as Python literal
                result[key] = ast.literal_eval(value)
        except Exception:
            # Fallback to raw string (without quotes)
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                result[key] = value[1:-1]
            else:
                result[key] = value

    return result


def _parse_llm_compiler_action_args(args: str) -> tuple:
    """Parse action argument string emitted by the LLM into a tuple.

    Args:
        args: String like '"search term"' or 'file_path="README.md", max_num=10'.

    Returns:
        Parsed argument tuple; for keyword-style args, returns a single dict in a tuple.
    """
    if args == "":
        return ()

    # Keyword-style args contain '='
    if "=" in args:
        # Parse into a dict
        parsed_dict = _parse_keyword_args(args)
        return (parsed_dict,)

    try:
        # Try literal-eval for positional args
        parsed = ast.literal_eval(args)
        if not isinstance(parsed, (list, tuple)):
            parsed = (parsed,)
        return tuple(parsed) if isinstance(parsed, list) else parsed
    except Exception:
        # Fallback to raw string
        return (args,)


def default_dependency_rule(idx: int, args: str) -> bool:
    """Default dependency rule: whether args reference task idx via $idx.

    Args:
        idx: Task index to check.
        args: Argument string.

    Returns:
        True if dependency exists, else False.
    """
    matches = re.findall(ID_PATTERN, args)
    numbers = [int(match) for match in matches]
    return idx in numbers


def _get_dependencies_from_graph(idx: int, tool_name: str, args: str) -> List[int]:
    """Extract dependencies from the argument string.

    Args:
        idx: Current task index.
        tool_name: Tool name.
        args: Raw argument string.

    Returns:
        List of dependency indices.
    """
    if tool_name == "join":
        # join depends on all previous tasks
        return list(range(1, idx))

    # Find referenced tasks via $<id>
    return [i for i in range(1, idx) if default_dependency_rule(i, args)]


def instantiate_task(
    tools: Sequence[BaseTool],
    idx: int,
    tool_name: str,
    args: str,
    thought: Optional[str] = None,
) -> Task:
    """Instantiate a Task from the parsed components.

    Args:
        tools: Available tools list.
        idx: Task index.
        tool_name: Tool name.
        args: Argument string.
        thought: Optional associated thought.

    Returns:
        Task object.
    """
    logger.debug(
        f"[Parser] Instantiate task {idx}: tool_name={tool_name}, args={args}, thought={thought}"
    )

    # Parse arguments and compute dependencies
    tool_args = _parse_llm_compiler_action_args(args)
    dependencies = _get_dependencies_from_graph(idx, tool_name, args)

    logger.debug(
        f"[Parser] Task {idx} parsed: tool_args={tool_args}, dependencies={dependencies}"
    )

    # Handle join
    if tool_name == "join":
        tool_func = lambda: None  # join does not execute
        stringify_rule = None
        is_join = True
        logger.debug(f"[Parser] Task {idx} is a join operation")
    else:
        # Find tool by name
        try:
            tool = tools[[t.name for t in tools].index(tool_name)]
            logger.debug(f"[Parser] Task {idx} found tool: {tool_name}")
        except ValueError as e:
            logger.error(f"[Parser] Task {idx} tool not found: {tool_name}")
            raise OutputParserException(f"Tool {tool_name} not found.") from e

        # For LangChain BaseTool, use the tool object directly.
        # Task.__call__ will handle the invocation.
        if isinstance(tool, BaseTool):
            tool_func = tool
        # For plain callable, try common attributes
        elif hasattr(tool, "func"):
            tool_func = tool.func
        elif hasattr(tool, "coroutine"):
            tool_func = tool.coroutine
        elif callable(tool):
            tool_func = tool
        else:
            logger.error(f"[Parser] Task {idx} tool not callable: {tool_name}")
            raise OutputParserException(f"Tool {tool_name} is not callable.")

        # Get stringify_rule if provided by the tool
        stringify_rule = getattr(tool, "stringify_rule", None)
        is_join = False

    task = Task(
        idx=idx,
        name=tool_name,
        tool=tool_func,
        args=tool_args,
        dependencies=dependencies,
        stringify_rule=stringify_rule,
        thought=thought,
        observation=None,
        is_join=is_join,
    )

    logger.info(f"[Parser] Task {idx} instantiated: {tool_name}({args})")

    return task


class LLMCompilerPlanParser(BaseTransformOutputParser[Task]):
    """LLMCompiler plan output parser with streaming support."""

    tools: List[BaseTool]

    def __init__(self, tools: List[BaseTool], **kwargs):
        super().__init__(tools=tools, **kwargs)

    def _transform(self, input: Iterator[Union[str, BaseMessage]]) -> Iterator[Task]:
        """Stream-transform input into tasks.

        Args:
            input: Iterator of strings or messages.

        Yields:
            Parsed Task objects.
        """
        texts = []
        thought = None

        for chunk in input:
            # Normalize to string
            text = chunk if isinstance(chunk, str) else str(chunk.content)
            for task, thought in self.ingest_token(text, texts, thought):
                yield task

        # Flush remaining buffer
        if texts:
            task, _ = self._parse_task("".join(texts), thought)
            if task:
                yield task

    def parse(self, text: str) -> Dict[int, Task]:
        """Parse full text into a task dictionary.

        Args:
            text: The full LLM output text.

        Returns:
            Dict {idx: Task}.
        """
        logger.debug(f"[Parser] Begin parsing text, length: {len(text)} chars")
        try:
            tasks = list(self._transform([text]))
            task_dict = {task.idx: task for task in tasks}
            logger.info(f"[Parser] Parsing complete, parsed {len(task_dict)} tasks")
            return task_dict
        except Exception as e:
            logger.error(f"[Parser] Parse failed: {str(e)}")
            raise

    def stream(
        self,
        input: Union[str, BaseMessage],
        config: Optional[RunnableConfig] = None,
        **kwargs: Any,
    ) -> Iterator[Task]:
        """Stream parse input into Task objects.

        Args:
            input: String or message.
            config: Optional run configuration.
            **kwargs: Additional params.

        Yields:
            Parsed Task objects.
        """
        yield from self.transform([input], config, **kwargs)

    def ingest_token(
        self, token: str, buffer: List[str], thought: Optional[str]
    ) -> Iterator[Tuple[Optional[Task], Optional[str]]]:
        """Ingest a token and try to parse tasks line-by-line.

        Args:
            token: Current token chunk.
            buffer: Text buffer accumulator.
            thought: Current thought string, if any.

        Yields:
            Tuples of (Task or None, updated thought).
        """
        buffer.append(token)
        if "\n" in token:
            buffer_ = "".join(buffer).split("\n")
            suffix = buffer_[-1]
            for line in buffer_[:-1]:
                task, thought = self._parse_task(line, thought)
                if task:
                    yield task, thought
            buffer.clear()
            buffer.append(suffix)

    def _parse_task(
        self, line: str, thought: Optional[str] = None
    ) -> Tuple[Optional[Task], Optional[str]]:
        """Parse a single line into a Task if it matches the action pattern.

        Args:
            line: Single line of text.
            thought: Current thought string carried over from previous lines.

        Returns:
            A tuple of (Task or None, updated thought).
        """
        task = None
        if match := re.match(THOUGHT_PATTERN, line):
            # Thought line
            thought = match.group(1)
        elif match := re.match(ACTION_PATTERN, line):
            # Action line
            idx, tool_name, args, _ = match.groups()
            idx = int(idx)
            task = instantiate_task(
                tools=self.tools,
                idx=idx,
                tool_name=tool_name,
                args=args,
                thought=thought,
            )
            thought = None
        # Ignore other lines
        return task, thought
