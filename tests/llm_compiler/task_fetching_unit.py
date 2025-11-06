import asyncio
import json
import time
from typing import Any, Dict, List

from .constants import ID_PATTERN, SCHEDULING_INTERVAL
from .output_parser import Task
from .log import get_logger

logger = get_logger("llm_compiler.executor")


def _serialize_observation(observation: Any) -> str:
    """Serialize an observation into a string.

    - For JSON-serializable types like dict/list, use json.dumps.
    - For strings, return as-is.
    - For other types, convert via str().

    Args:
        observation: Task execution result.

    Returns:
        Serialized string representation.
    """
    if isinstance(observation, (dict, list)):
        return json.dumps(observation, ensure_ascii=False)
    elif isinstance(observation, str):
        return observation
    else:
        return str(observation)


def _replace_arg_mask_with_real_value(
    arg: Any, dependencies: List[int], tasks: Dict[int, Task]
) -> Any:
    """Recursively replace dependency masks ($1, ${1}) in arguments.

    Supports nested structures:
    - Strings: replace masks inline.
    - Dicts: recursively process values.
    - Lists/Tuples: recursively process elements.

    Args:
        arg: The argument value to process.
        dependencies: List of dependent task indices.
        tasks: Tasks dict containing observations.

    Returns:
        The argument value with masks replaced by observations.
    """
    # Handle string argument: inline mask replacement
    if isinstance(arg, str):
        original_arg = arg
        for dep_idx in sorted(dependencies, reverse=True):
            if tasks[dep_idx].observation is not None:
                obs_value = _serialize_observation(tasks[dep_idx].observation)
                for mask in [f"${{{dep_idx}}}", f"${dep_idx}"]:
                    if mask in arg:
                        arg = arg.replace(mask, obs_value)
                        logger.debug(
                            f"[Executor] Replace mask: {mask} -> {obs_value[:100]}..."
                            if len(obs_value) > 100
                            else f"[Executor] Replace mask: {mask} -> {obs_value}"
                        )
        if arg != original_arg:
            logger.debug(
                f"[Executor] String arg replaced: '{original_arg[:100]}...' -> '{arg[:100]}...'"
                if len(arg) > 100 or len(original_arg) > 100
                else f"[Executor] String arg replaced: '{original_arg}' -> '{arg}'"
            )
        return arg

    # Handle dict argument: recurse into values
    elif isinstance(arg, dict):
        result = {}
        for key, value in arg.items():
            result[key] = _replace_arg_mask_with_real_value(value, dependencies, tasks)
        if result != arg:
            logger.debug(
                f"[Executor] Dict arg replaced: {list(arg.keys())} -> updated values"
            )
        return result

    # Handle list argument: recurse into elements
    elif isinstance(arg, list):
        result = [
            _replace_arg_mask_with_real_value(item, dependencies, tasks) for item in arg
        ]
        if result != arg:
            logger.debug(
                f"[Executor] List arg replaced: length={len(arg)} -> updated elements"
            )
        return result

    # Handle tuple argument: recurse into elements
    elif isinstance(arg, tuple):
        result = tuple(
            _replace_arg_mask_with_real_value(item, dependencies, tasks) for item in arg
        )
        if result != arg:
            logger.debug(
                f"[Executor] Tuple arg replaced: length={len(arg)} -> updated elements"
            )
        return result

    # Other types: return as-is
    else:
        return arg


class TaskFetchingUnit:
    """Task scheduling unit respecting dependencies and enabling concurrency."""

    def __init__(self):
        """Initialize the scheduling unit."""
        self.tasks: Dict[int, Task] = {}
        self.tasks_done: Dict[int, asyncio.Event] = {}
        self.remaining_tasks: set = set()

    def set_tasks(self, tasks: Dict[int, Task]):
        """Register tasks to be scheduled.

        Args:
            tasks: Task dict {idx: Task}.
        """
        self.tasks.update(tasks)
        self.tasks_done.update({idx: asyncio.Event() for idx in tasks})
        self.remaining_tasks.update(set(tasks.keys()))
        logger.info(
            f"[Executor] Tasks registered: {len(tasks)} tasks, ids: {sorted(tasks.keys())}"
        )

    def _all_tasks_done(self) -> bool:
        """Check whether all tasks are completed."""
        return all(self.tasks_done[idx].is_set() for idx in self.tasks_done)

    def _get_executable_tasks(self) -> List[int]:
        """Get all executable tasks whose dependencies are satisfied."""
        executable = []
        for idx in self.remaining_tasks:
            task = self.tasks[idx]
            if all(self.tasks_done[dep].is_set() for dep in task.dependencies):
                executable.append(idx)
        if executable:
            logger.debug(
                f"[Executor] Found {len(executable)} executable tasks: {executable}"
            )
        return executable

    def _preprocess_args(self, task: Task):
        """Preprocess task arguments by replacing dependency masks.

        Args:
            task: Task to preprocess.
        """
        if not task.dependencies:
            logger.debug(
                f"[Executor] Task {task.idx} has no dependencies, skip arg preprocessing"
            )
            return

        original_args = task.args
        processed_args = []

        logger.debug(
            f"[Executor] Task {task.idx} arg preprocessing start: "
            f"original={str(original_args)[:200]}..."
            if len(str(original_args)) > 200
            else f"[Executor] Task {task.idx} arg preprocessing start: original={original_args}"
        )

        for i, arg in enumerate(task.args):
            processed_arg = _replace_arg_mask_with_real_value(
                arg, task.dependencies, self.tasks
            )
            processed_args.append(processed_arg)

            # Log per-argument replacement result
            if processed_arg != arg:
                logger.debug(
                    f"[Executor] Task {task.idx} arg {i} replaced: "
                    f"{type(arg).__name__} -> {type(processed_arg).__name__}"
                )

        task.args = tuple(processed_args)

        logger.debug(
            f"[Executor] Task {task.idx} arg preprocessing done: "
            f"processed={str(task.args)[:200]}..."
            if len(str(task.args)) > 200
            else f"[Executor] Task {task.idx} arg preprocessing done: processed={task.args}"
        )

    async def _run_task(self, task: Task):
        """Execute a single task and record its observation.

        Args:
            task: Task to execute.
        """
        start_time = time.time()
        logger.info(f"[Executor] Start task {task.idx}: {task.name}({task.args})")

        # Preprocess arguments prior to execution
        self._preprocess_args(task)
        if task.dependencies:
            logger.debug(
                f"[Executor] Task {task.idx} dependencies: {task.dependencies}"
            )

        # Execute task (join is a no-op)
        if not task.is_join:
            try:
                logger.debug(f"[Executor] Task {task.idx} invoking tool: {task.name}")
                observation = await task()  # Calls Task.__call__()
                task.observation = _serialize_observation(observation)
                elapsed = time.time() - start_time
                obs_preview = (
                    task.observation[:200] + "..."
                    if len(task.observation) > 200
                    else task.observation
                )
                logger.info(
                    f"[Executor] Task {task.idx} done (elapsed: {elapsed:.2f}s): "
                    f"observation={obs_preview}"
                )
                logger.debug(
                    f"[Executor] Task {task.idx} full observation: {task.observation}"
                )
            except Exception as e:
                elapsed = time.time() - start_time
                task.observation = f"Error: {str(e)}"
                logger.error(
                    f"[Executor] Task {task.idx} failed (elapsed: {elapsed:.2f}s): {str(e)}",
                    exc_info=True,
                )
        else:
            logger.info(
                f"[Executor] Task {task.idx} is a join operation, skip execution"
            )

        # Mark as done
        self.tasks_done[task.idx].set()
        logger.debug(f"[Executor] Task {task.idx} marked as done")

    async def schedule(self) -> Dict[int, Task]:
        """Schedule and execute all tasks concurrently, respecting dependencies.

        Returns:
            Dict of executed tasks.
        """
        logger.info(f"[Executor] Start scheduling tasks, total: {len(self.tasks)}")
        schedule_start = time.time()
        iteration = 0

        while not self._all_tasks_done():
            iteration += 1
            # Get executable tasks
            executable = self._get_executable_tasks()

            # Launch executions
            if executable:
                logger.info(
                    f"[Executor] Scheduling iter {iteration}: launching {len(executable)} task(s)"
                )
                for idx in executable:
                    asyncio.create_task(self._run_task(self.tasks[idx]))
                    self.remaining_tasks.remove(idx)
            else:
                logger.debug(
                    f"[Executor] Scheduling iter {iteration}: no tasks ready, waiting"
                )

            # Brief sleep between iterations
            await asyncio.sleep(SCHEDULING_INTERVAL)

        total_time = time.time() - schedule_start
        logger.info(
            f"[Executor] All tasks finished, total: {total_time:.2f}s, iterations: {iteration}"
        )

        return self.tasks
