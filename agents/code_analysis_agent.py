from config import CONFIG
from agents.tools import (
    read_file_tool,
    code_file_analysis_tool,
)
from .base_agent import AgentState

from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    ToolMessage,
    HumanMessage,
)
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Sequence, Literal, Dict, Any
from contextlib import redirect_stdout
from datetime import datetime
import json
import os
import io

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import copy


class CodeAnalysisAgent:
    def __init__(self, llm, tools):
        self.llm = llm.bind_tools(tools)
        self.tools = tools
        self.tool_executor = ToolNode(tools)
        self.memory = InMemorySaver()
        self.app = self._build_app()

    def _build_app(self):
        """Build the agent workflow graph."""
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self.tool_executor)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END,
            },
        )
        workflow.add_edge("tools", "agent")
        return workflow.compile(checkpointer=self.memory)

    def _should_continue(self, state: AgentState) -> str:
        """Determine whether to continue or end."""
        messages = state["messages"]
        last_message = messages[-1]
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return "end"
        else:
            return "continue"

    def _agent_node(self, state: AgentState) -> AgentState:
        """Call LLM with current state."""
        system_prompt = SystemMessage(
            content="""You are a code analysis expert. Your task is to analyze code files efficiently.

                        IMPORTANT RULES:
                        1. Call code_file_analysis_tool ONLY ONCE per file
                        2. After analyzing all files, return results immediately in JSON format
                        3. Do NOT repeat tool calls if you already have the results

                        For each file, extract:
                        - Main functions/classes with signatures
                        - Dependencies and imports
                        - Complexity score (1-10)
                        - Lines of code

                        Return results in this exact JSON structure:
                        {
                            "files": {
                                "file_path": {
                                    "language": "string",
                                    "functions": [{"name": "...", "signature": "...", "line_start": 0, "line_end": 0}],
                                    "classes": [{"name": "...", "methods": [], "line_start": 0, "line_end": 0}],
                                    "imports": [],
                                    "complexity_score": 5,
                                    "lines_of_code": 100,
                                    "summary": "Brief description"
                                }
                            }
                        }

                        Do NOT include any text outside the JSON structure.
                    """
        )
        response = self.llm.invoke([system_prompt] + state["messages"])
        return {"messages": [response]}

    def _create_batch_agent(self):
        """Create a new agent instance for parallel batch processing.

        This is necessary because each batch needs its own memory/state.
        """
        batch_memory = InMemorySaver()
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self.tool_executor)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END,
            },
        )
        workflow.add_edge("tools", "agent")
        return workflow.compile(checkpointer=batch_memory)

    def _analyze_single_file_with_agent(
        self, file_path: str, batch_num: int, file_num: int
    ) -> Tuple[str, dict]:
        """Analyze a single file using a fresh LLM agent instance.

        Args:
            file_path (str): Path to the file to analyze
            batch_num (int): Batch number for logging
            file_num (int): File number within batch for logging

        Returns:
            Tuple[str, dict]: (file_path, analysis_result)
        """
        # Create a fresh agent for each file to avoid context accumulation
        file_agent = self._create_batch_agent()

        # Prepare the prompt for this single file
        prompt = f"""
                    Analyze the following code file:

                    File to analyze: {file_path}

                    Tasks:
                    1. Use code_file_analysis_tool with the file path AS IS
                    2. Extract the analysis information as specified in the system prompt
                    3. Calculate a complexity score based on the analysis results

                    Return the result in this exact JSON format:
                    {{
                        "language": "...",
                        "functions": [...],
                        "classes": [...],
                        "imports": [...],
                        "complexity_score": 5,
                        "lines_of_code": 100,
                        "summary": "Brief description"
                    }}

                    Important: 
                    - Call code_file_analysis_tool for this file
                    - Return results in the exact JSON format above
                    - Do NOT add extra text outside the JSON structure
                """

        initial_state = AgentState(
            messages=[HumanMessage(content=prompt)],
            repo_path="",
            wiki_path="",
        )

        # Run the file agent workflow
        final_state = None
        try:
            for state in file_agent.stream(
                initial_state,
                stream_mode="values",
                config={
                    "configurable": {
                        "thread_id": f"file-analysis-{batch_num}-{file_num}-{datetime.now().timestamp()}"
                    },
                    "recursion_limit": 50,  # Reduced for single file
                },
            ):
                final_state = state

            # Extract results from final state
            if final_state:
                last_message = final_state["messages"][-1]
                if hasattr(last_message, "content"):
                    try:
                        content = last_message.content
                        import re

                        # Try to extract JSON
                        json_code_block = re.search(
                            r"```json\s*([\s\S]*?)\s*```", content
                        )
                        if json_code_block:
                            json_str = json_code_block.group(1)
                        else:
                            json_match = re.search(r"\{[\s\S]*\}", content)
                            if json_match:
                                json_str = json_match.group()
                            else:
                                json_str = None

                        if json_str:
                            # Clean and parse
                            json_str = re.sub(
                                r",(\s*[}\]])", r"\1", json_str
                            )  # Remove trailing commas
                            result = json.loads(json_str)
                            print(
                                f"    [Batch {batch_num}] ✓ {os.path.basename(file_path)}"
                            )
                            return file_path, result
                    except json.JSONDecodeError as e:
                        print(
                            f"    [Batch {batch_num}] ✗ {os.path.basename(file_path)}: JSON parse error"
                        )
                        return file_path, {
                            "error": f"JSON parse error: {e}",
                            "raw_output": last_message.content[:500],
                        }

            # If no valid result
            print(
                f"    [Batch {batch_num}] ✗ {os.path.basename(file_path)}: No valid result"
            )
            return file_path, {
                "error": "No valid result from agent",
            }

        except Exception as e:
            print(f"    [Batch {batch_num}] ✗ {os.path.basename(file_path)}: {e}")
            return file_path, {
                "error": str(e),
            }

    def _analyze_single_file_with_agent(
        self, file_path: str, batch_num: int, file_num: int, repo_path: str
    ) -> Tuple[str, dict]:
        """Analyze a single file with a fresh agent to avoid context accumulation.

        Args:
            file_path (str): File path to analyze
            batch_num (int): Batch number for logging
            file_num (int): File number within the batch
            repo_path (str): Repository path

        Returns:
            Tuple[str, dict]: (file_path, file_analysis_result)
        """
        # Create a fresh agent for each file
        file_agent = self._create_batch_agent()

        # Create a single-file prompt
        prompt = f"""
                    Analyze this code file:

                    File path: {file_path}

                    Use code_file_analysis_tool to analyze the file, then provide the analysis in this exact JSON format:
                    {{
                        "language": "...",
                        "functions": [...],
                        "classes": [...],
                        "complexity_score": 5,
                        "lines_of_code": 100,
                        "summary": "..."
                    }}

                    Important:
                    - Call code_file_analysis_tool with the file path AS IS
                    - Return ONLY the JSON structure above
                    - Do NOT add extra text outside the JSON
                """

        initial_state = AgentState(
            messages=[HumanMessage(content=prompt)],
            repo_path=repo_path,
            wiki_path="",
        )

        # Run the agent workflow with fresh context
        final_state = None
        try:
            for state in file_agent.stream(
                initial_state,
                stream_mode="values",
                config={
                    "configurable": {
                        "thread_id": f"batch-{batch_num}-file-{file_num}-{datetime.now().timestamp()}"
                    },
                    "recursion_limit": 50,
                },
            ):
                final_state = state

            # Extract result
            if final_state:
                last_message = final_state["messages"][-1]
                if hasattr(last_message, "content"):
                    try:
                        content = last_message.content
                        import re

                        json_match = re.search(r"\{[\s\S]*\}", content)
                        if json_match:
                            parsed_result = json.loads(json_match.group())
                            return file_path, parsed_result
                    except json.JSONDecodeError as e:
                        return file_path, {
                            "error": "JSON parse error",
                            "raw_output": last_message.content[:300],
                        }

            return file_path, {"error": "No result from agent"}

        except Exception as e:
            return file_path, {"error": str(e)}

    def _analyze_batch_with_agent(
        self, batch: List[str], batch_num: int, repo_path: str
    ) -> Tuple[int, dict]:
        """Analyze a batch of files using dedicated LLM agent instances (one per file).

        Args:
            batch (List[str]): List of file paths to analyze
            batch_num (int): Batch number for logging
            repo_path (str): Repository path

        Returns:
            Tuple[int, dict]: (batch_num, batch_results)
        """
        print(f"  [Batch {batch_num}] Starting analysis of {len(batch)} files...")

        batch_results = {}

        # Analyze each file with a fresh agent to avoid context accumulation
        for i, file_path in enumerate(batch, 1):
            print(
                f"    [Batch {batch_num}] Analyzing file {i}/{len(batch)}: {os.path.basename(file_path)}"
            )
            file_path_result, file_result = self._analyze_single_file_with_agent(
                file_path, batch_num, i, repo_path
            )
            batch_results[file_path_result] = file_result

        print(
            f"  [Batch {batch_num}] ✓ Completed ({len(batch_results)} files analyzed)"
        )
        return batch_num, batch_results

    def run(
        self,
        repo_path: str,
        file_list: list,
        batch_size: int = 1,
        parallel_batches: bool = True,
        max_workers: int = 100,
    ) -> dict:
        """Analyze core code files with parallel batch processing.

        Args:
            repo_path (str): Local path to the repository
            file_list (list): List of file paths to analyze
            batch_size (int): Number of files to analyze in one batch (default 1)
            parallel_batches (bool): Whether to process batches in parallel (default True)
            max_workers (int): Maximum number of parallel batch workers (default 100)

        Returns:
            dict: Analysis results for all files
        """
        if not file_list:
            return {
                "total_files": 0,
                "analyzed_files": 0,
                "files": {},
                "summary": {
                    "total_functions": 0,
                    "total_classes": 0,
                    "average_complexity": 0,
                    "total_lines": 0,
                },
            }

        # Filter and validate files
        code_extensions = {
            ".py",
            ".js",
            ".java",
            ".cpp",
            ".c",
            ".go",
            ".rs",
            ".ts",
            ".jsx",
            ".tsx",
            ".h",
            ".hpp",
        }
        valid_files = []
        for f in file_list:
            if os.path.isfile(f) and os.path.splitext(f)[1] in code_extensions:
                valid_files.append(f)
                print(f"✓ Valid code file found: {f}")

        if not valid_files:
            return {
                "total_files": len(file_list),
                "analyzed_files": 0,
                "files": {},
                "summary": {
                    "total_functions": 0,
                    "total_classes": 0,
                    "average_complexity": 0,
                    "total_lines": 0,
                },
                "warning": "No valid code files found to analyze",
            }

        # Limit files
        max_files = 1000
        if len(valid_files) > max_files:
            valid_files = valid_files[:max_files]
            print(f"Warning: Analyzing only first {max_files} files to avoid overflow")

        # Sort files by size (largest first) for better load balancing
        print("\nSorting files by size...")
        file_sizes = []
        for f in valid_files:
            try:
                size = os.path.getsize(f)
                file_sizes.append((f, size))
            except OSError:
                file_sizes.append((f, 0))

        # Sort by size in descending order
        file_sizes.sort(key=lambda x: x[1], reverse=True)
        sorted_files = [f for f, _ in file_sizes]

        total_size = sum(size for _, size in file_sizes)
        print(f"Total size: {total_size / 1024:.2f} KB")
        print(f"Average file size: {total_size / len(file_sizes) / 1024:.2f} KB")
        print(f"Largest file: {file_sizes[0][1] / 1024:.2f} KB")
        print(f"Smallest file: {file_sizes[-1][1] / 1024:.2f} KB")

        # Split into batches using greedy algorithm for balanced batch sizes
        num_batches = (len(sorted_files) + batch_size - 1) // batch_size
        batches = [[] for _ in range(num_batches)]
        batch_sizes = [0] * num_batches

        # Greedy assignment: assign each file to the batch with smallest current size
        for file_path, file_size in file_sizes:
            # Find batch with minimum total size
            min_batch_idx = batch_sizes.index(min(batch_sizes))
            batches[min_batch_idx].append(file_path)
            batch_sizes[min_batch_idx] += file_size

        # Remove empty batches
        batches = [b for b in batches if b]

        print(f"\nCreated {len(batches)} balanced batches:")
        for i, (batch, size) in enumerate(zip(batches, batch_sizes[: len(batches)]), 1):
            print(f"  Batch {i}: {len(batch)} files, {size / 1024:.2f} KB")

        all_results = {}

        if parallel_batches and len(batches) > 1:
            # ========== 批次间并行处理 ==========
            print(
                f"\n Processing {len(batches)} batches in PARALLEL (max {max_workers} concurrent batches)..."
            )

            # start_time = datetime.now()

            with ThreadPoolExecutor(
                max_workers=min(len(batches), max_workers)
            ) as executor:
                # 为每个批次提交一个独立的 agent 任务
                future_to_batch = {
                    executor.submit(
                        self._analyze_batch_with_agent, batch, i + 1, repo_path
                    ): i
                    for i, batch in enumerate(batches)
                }

                # 收集结果（按完成顺序）
                for future in as_completed(future_to_batch):
                    batch_idx = future_to_batch[future]
                    try:
                        batch_num, batch_results = future.result()
                        print(f"DEBUG: {batch_results}")
                        all_results.update(batch_results)
                        print(f"✓ Batch {batch_num} results merged into final results")
                    except Exception as e:
                        print(f"✗ Batch {batch_idx + 1} failed with exception: {e}")
                        # Add error results for failed batch
                        for file_path in batches[batch_idx]:
                            all_results[file_path] = {
                                "error": f"Batch execution failed: {e}"
                            }
        else:
            # ========== 批次间串行处理 ==========
            print(f"\n Processing {len(batches)} batches SEQUENTIALLY...")
            for i, batch in enumerate(batches):
                batch_num, batch_results = self._analyze_batch_with_agent(
                    batch, i + 1, repo_path
                )
                all_results.update(batch_results)

        # Calculate summary statistics
        total_functions = 0
        total_classes = 0
        total_complexity = 0
        total_lines = 0
        analyzed_count = 0

        for file_path, analysis in all_results.items():
            if "error" not in analysis:
                analyzed_count += 1
                total_functions += len(analysis.get("functions", []))
                total_classes += len(analysis.get("classes", []))
                total_complexity += analysis.get("complexity_score", 0)
                total_lines += analysis.get("lines_of_code", 0)

        avg_complexity = total_complexity / analyzed_count if analyzed_count > 0 else 0

        # end_time = datetime.now()

        # print(f"\n Total analysis time: {end_time - start_time}\n")

        return {
            "total_files": len(file_list),
            "analyzed_files": analyzed_count,
            "files": all_results,
            "summary": {
                "total_functions": total_functions,
                "total_classes": total_classes,
                "average_complexity": round(avg_complexity, 2),
                "total_lines": total_lines,
                "languages": list(
                    set(
                        analysis.get("language", "unknown")
                        for analysis in all_results.values()
                        if "error" not in analysis
                    )
                ),
            },
            "parallel_mode": (
                "batches_parallel"
                if parallel_batches and len(batches) > 1
                else "sequential"
            ),
        }


# ========== CodeAnalysisAgentTest ==========
def CodeAnalysisAgentTest():
    llm = CONFIG.get_llm()
    tools = [code_file_analysis_tool, read_file_tool]
    code_agent = CodeAnalysisAgent(llm, tools)

    repo_path = (
        "/home/dunjia/workspace/mid-grade-project/core/repo-agent/.repos/facebook_zstd"
    )

    # 收集文件
    file_list = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in [".git", "node_modules", "__pycache__"]]
        for file in files:
            if file.endswith((".c", ".h")) and len(file_list) < 1000:
                file_list.append(os.path.join(root, file))

    print(f"Found {len(file_list)} files for analysis")

    code_analysis = code_agent.run(
        repo_path=repo_path,
        file_list=file_list,
        batch_size=1,
        parallel_batches=True,
        max_workers=100,  # 最大并行批次数
    )

    print("\n" + "=" * 60)
    print(f"Analysis Mode: {code_analysis.get('parallel_mode', 'unknown')}")
    print(f"Total Files: {code_analysis['total_files']}")
    print(f"Analyzed Files: {code_analysis['analyzed_files']}")
    print(f"Total Functions: {code_analysis['summary']['total_functions']}")
    print(f"Total Classes: {code_analysis['summary']['total_classes']}")
    print(f"Average Complexity: {code_analysis['summary']['average_complexity']}")
    print("=" * 60)


if __name__ == "__main__":
    CodeAnalysisAgentTest()
