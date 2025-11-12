from config import CONFIG
from agents.tools import (
    read_file_tool,
    code_file_analysis_tool,
)
from .base_agent import BaseAgent, AgentState

from langchain_core.messages import SystemMessage, HumanMessage
from datetime import datetime
import json
import os
import re
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple


class CodeAnalysisAgent(BaseAgent):
    """
    Agent for analyzing code files.
    """

    def __init__(self, repo_path: str, wiki_path: str = ""):
        """Initialize the CodeAnalysisAgent.

        Args:
            repo_path (str): Local path to the repository
            wiki_path (str): Path to wiki (optional)
        """
        system_prompt = SystemMessage(
            content="""You are a code analysis expert. Your task is to analyze multiple code files efficiently in batch.

                        IMPORTANT RULES:
                        1. For each file, call code_file_analysis_tool EXACTLY ONCE
                        2. Analyze all files in the batch sequentially
                        3. Return results for ALL files in a single JSON response

                        For each file, extract:
                        - Main functions/classes with signatures
                        - Dependencies and imports  
                        - Complexity score (1-10)
                        - Lines of code

                        Return results in this exact JSON structure:
                        {
                            "results": {
                                "/path/to/file1.c": {
                                    "language": "C",
                                    "functions": [...],
                                    "classes": [...],
                                    "imports": [],
                                    "complexity_score": 5,
                                    "lines_of_code": 100,
                                    "summary": "Brief description"
                                },
                                "/path/to/file2.h": {
                                    ...
                                }
                            }
                        }

                        CRITICAL:
                        - Return ONLY the JSON object, no additional text
                        - Include ALL files from the batch in the results
                        - Use absolute file paths as keys
                        - If a file analysis fails, include an error field for that file
                    """
        )

        tools = [
            code_file_analysis_tool,
            read_file_tool,
        ]

        super().__init__(
            tools=tools,
            system_prompt=system_prompt,
            repo_path=repo_path,
            wiki_path=wiki_path,
        )

    def _analyze_batch(self, batch: List[str], batch_num: int) -> Tuple[int, dict]:
        """Analyze a batch of files in a SINGLE LLM call.

        Args:
            batch (List[str]): List of file paths
            batch_num (int): Batch number for logging

        Returns:
            Tuple[int, dict]: (batch_num, batch_results)
        """
        print(f"  [Batch {batch_num}] Analyzing {len(batch)} files in ONE LLM call...")
        start_time = time.time()

        # Build prompt for batch analysis
        file_list_str = "\n".join([f"  - {f}" for f in batch])

        prompt = f"""
                    Analyze the following {len(batch)} code files:

                    {file_list_str}

                    For each file:
                    1. Use code_file_analysis_tool with the EXACT file path
                    2. Extract all required information
                    3. Calculate complexity score based on:
                    - Number of functions/classes
                    - Code length
                    - Nesting depth
                    - Dependencies

                    Return results for ALL files in the JSON format specified in the system prompt.

                    IMPORTANT: 
                    - Analyze each file ONCE
                    - Include ALL {len(batch)} files in your response
                    - Use the absolute file paths as JSON keys
                """

        initial_state = AgentState(
            messages=[HumanMessage(content=prompt)],
            repo_path=self.repo_path,
            wiki_path=self.wiki_path,
        )

        # Run batch analysis
        final_state = None
        try:
            for state in self.app.stream(
                initial_state,
                stream_mode="values",
                config={
                    "configurable": {
                        "thread_id": f"batch-{batch_num}-{datetime.now().timestamp()}"
                    },
                    "recursion_limit": 100,  # Increased for multiple files
                },
            ):
                final_state = state

                # Log tool calls
                last_msg = state["messages"][-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    for tool_call in last_msg.tool_calls:
                        if tool_call["name"] == "code_file_analysis_tool":
                            file_path = tool_call.get("args", {}).get(
                                "file_path", "unknown"
                            )
                            file_name = os.path.basename(file_path)
                            print(f"    [Batch {batch_num}] Analyzing: {file_name}")

            # Extract results
            if final_state:
                batch_results = self._extract_batch_results(
                    final_state, batch, batch_num
                )

                # Print summary
                success_count = sum(
                    1 for r in batch_results.values() if "error" not in r
                )
                error_count = len(batch_results) - success_count
                print(
                    f"  [Batch {batch_num}] ✓ Completed in {time.time() - start_time:.2f}s: {success_count} success, {error_count} errors"
                )

                return batch_num, batch_results
            else:
                print(f"  [Batch {batch_num}] ✗ No result from LLM")
                return batch_num, {f: {"error": "No result from LLM"} for f in batch}

        except Exception as e:
            print(f"  [Batch {batch_num}] ✗ Batch failed: {e}")
            return batch_num, {f: {"error": f"Batch failed: {e}"} for f in batch}

    def _extract_batch_results(
        self, final_state: AgentState, expected_files: List[str], batch_num: int
    ) -> dict:
        """Extract results for all files in the batch.

        Args:
            final_state (AgentState): The final agent state
            expected_files (List[str]): List of file paths we expected to analyze
            batch_num (int): Batch number for logging

        Returns:
            dict: Results for each file
        """
        if not final_state:
            return {f: {"error": "No final state"} for f in expected_files}

        last_message = final_state["messages"][-1]

        if not hasattr(last_message, "content"):
            return {f: {"error": "No content in message"} for f in expected_files}

        try:
            content = last_message.content

            # Try to extract JSON
            json_code_block = re.search(r"```json\s*([\s\S]*?)\s*```", content)
            if json_code_block:
                json_str = json_code_block.group(1)
            else:
                json_match = re.search(r"\{[\s\S]*\}", content)
                if json_match:
                    json_str = json_match.group()
                else:
                    print(f"  [Batch {batch_num}] ⚠ No JSON found in response")
                    return {f: {"error": "No JSON in response"} for f in expected_files}

            # Clean and parse JSON
            json_str = re.sub(r",(\s*[}\]])", r"\1", json_str)
            result = json.loads(json_str)

            # Extract results for each file
            batch_results = {}

            # Check if results are in a "results" key
            if "results" in result:
                file_results = result["results"]
            else:
                file_results = result

            # Match results to expected files
            for file_path in expected_files:
                # Try exact match first
                if file_path in file_results:
                    batch_results[file_path] = file_results[file_path]

                    # Validate required fields
                    required_fields = [
                        "language",
                        "functions",
                        "classes",
                        "complexity_score",
                        "lines_of_code",
                    ]
                    missing_fields = [
                        f for f in required_fields if f not in batch_results[file_path]
                    ]

                    if missing_fields:
                        batch_results[file_path][
                            "warning"
                        ] = f"Missing fields: {missing_fields}"
                else:
                    # Try to match by basename
                    basename = os.path.basename(file_path)
                    matched = False

                    for result_key, result_value in file_results.items():
                        if os.path.basename(result_key) == basename:
                            batch_results[file_path] = result_value
                            matched = True
                            break

                    if not matched:
                        batch_results[file_path] = {
                            "error": f"No result found for this file in LLM response"
                        }

            return batch_results

        except json.JSONDecodeError as e:
            print(f"  [Batch {batch_num}] ✗ JSON parse error: {e}")
            return {f: {"error": f"JSON parse error: {e}"} for f in expected_files}
        except Exception as e:
            print(f"  [Batch {batch_num}] ✗ Unexpected error: {e}")
            return {f: {"error": f"Unexpected error: {e}"} for f in expected_files}

    def run(
        self,
        file_list: list,
        batch_size: int = 5,
        parallel_batches: bool = True,
        max_workers: int = 10,
        max_files: int = 500,
    ) -> dict:
        """Analyze code files with optional parallel batch processing.

        Args:
            file_list (list): List of file paths to analyze
            batch_size (int): Number of files per batch (default: 5)
            parallel_batches (bool): Whether to process batches in parallel (default: True)
            max_workers (int): Maximum number of parallel workers (default: 10)
            max_files (int): Maximum number of files to analyze (default: 500)

        Returns:
            dict: Analysis results including summary statistics
        """

        start_time = time.time()

        if not file_list:
            return self._empty_result()

        # Filter valid code files
        valid_files = self._filter_valid_files(file_list)

        if not valid_files:
            return self._empty_result(warning="No valid code files found")

        # Limit files to avoid overwhelming the system
        if len(valid_files) > max_files:
            print(f" Limiting to first {max_files} files")
            valid_files = valid_files[:max_files]

        # Sort files by size and create balanced batches
        batches = self._create_balanced_batches(valid_files, batch_size)

        # Process batches
        all_results = {}

        if parallel_batches and len(batches) > 1:
            print(
                f"\n   Processing {len(batches)} batches in PARALLEL (max {max_workers} workers)..."
            )
            print(f"   Each batch processes {batch_size} files in ONE LLM call")
            all_results = self._process_batches_parallel(batches, max_workers)
        else:
            print(f"\n   Processing {len(batches)} batches SEQUENTIALLY...")
            print(f"   Each batch processes {batch_size} files in ONE LLM call")
            all_results = self._process_batches_sequential(batches)

        print(
            f"\n✓ Completed analysis of {len(valid_files)} files in {time.time() - start_time:.2f}s"
        )

        # Calculate summary
        return self._build_result(
            file_list, all_results, parallel_batches, len(batches)
        )

    def _filter_valid_files(self, file_list: list) -> list:
        """Filter and validate code files.

        Args:
            file_list (list): List of file paths

        Returns:
            list: List of valid code file paths
        """
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
            ".cs",
            ".rb",
            ".php",
        }

        valid_files = []
        for f in file_list:
            if os.path.isfile(f) and os.path.splitext(f)[1] in code_extensions:
                valid_files.append(f)

        print(f"✓ Found {len(valid_files)} valid code files")
        return valid_files

    def _create_balanced_batches(self, files: list, batch_size: int) -> list:
        """Create balanced batches using round-robin distribution.

        Args:
            files (list): List of file paths
            batch_size (int): Target number of files per batch

        Returns:
            list: List of batches (each batch is a list of file paths)
        """
        print("\nCreating balanced batches...")

        # Get file sizes
        file_sizes = []
        for f in files:
            try:
                size = os.path.getsize(f)
                file_sizes.append((f, size))
            except OSError:
                file_sizes.append((f, 0))

        # Sort by size (largest first)
        file_sizes.sort(key=lambda x: x[1], reverse=True)

        # Print statistics
        total_size = sum(size for _, size in file_sizes)
        print(f"Total files: {len(files)}")
        print(f"Total size: {total_size / 1024:.2f} KB")
        print(f"Average file size: {total_size / len(file_sizes) / 1024:.2f} KB")
        print(f"Largest file: {file_sizes[0][1] / 1024:.2f} KB")
        print(f"Smallest file: {file_sizes[-1][1] / 1024:.2f} KB")

        # Calculate number of batches
        num_batches = (len(files) + batch_size - 1) // batch_size

        batches = [[] for _ in range(num_batches)]
        batch_sizes = [0] * num_batches

        # Distribute files round-robin (largest files distributed first)
        for idx, (file_path, file_size) in enumerate(file_sizes):
            batch_idx = idx % num_batches
            batches[batch_idx].append(file_path)
            batch_sizes[batch_idx] += file_size

        # Remove empty batches
        batches = [b for b in batches if b]
        batch_sizes = batch_sizes[: len(batches)]

        print(f"\nCreated {len(batches)} batches (round-robin distribution):")
        for i, (batch, size) in enumerate(zip(batches, batch_sizes), 1):
            print(f"  Batch {i}: {len(batch)} files, {size / 1024:.2f} KB")

        return batches

    def _process_batches_parallel(self, batches: list, max_workers: int) -> dict:
        """Process batches in parallel using ThreadPoolExecutor.

        Args:
            batches (list): List of batches to process
            max_workers (int): Maximum number of parallel workers

        Returns:
            dict: Combined results from all batches
        """
        all_results = {}

        with ThreadPoolExecutor(max_workers=min(len(batches), max_workers)) as executor:
            # Submit all batch jobs
            future_to_batch = {
                executor.submit(self._analyze_batch, batch, i + 1): i
                for i, batch in enumerate(batches)
            }

            # Collect results as they complete
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_num, batch_results = future.result()
                    all_results.update(batch_results)
                    print(f"✓ Batch {batch_num} results merged into final results")
                except Exception as e:
                    print(f"✗ Batch {batch_idx + 1} failed: {e}")
                    for file_path in batches[batch_idx]:
                        all_results[file_path] = {"error": f"Batch failed: {e}"}

        return all_results

    def _process_batches_sequential(self, batches: list) -> dict:
        """Process batches sequentially.

        Args:
            batches (list): List of batches to process

        Returns:
            dict: Combined results from all batches
        """
        all_results = {}

        for i, batch in enumerate(batches, 1):
            _, batch_results = self._analyze_batch(batch, i)
            all_results.update(batch_results)

        return all_results

    def _build_result(
        self,
        original_files: list,
        analyzed_results: dict,
        parallel: bool,
        num_batches: int,
    ) -> dict:
        """Build final result with summary statistics.

        Args:
            original_files (list): Original list of files
            analyzed_results (dict): Analysis results
            parallel (bool): Whether parallel processing was used
            num_batches (int): Number of batches processed

        Returns:
            dict: Complete result with statistics
        """
        # Calculate statistics
        total_functions = 0
        total_classes = 0
        total_complexity = 0
        total_lines = 0
        analyzed_count = 0
        languages = set()

        for file_path, analysis in analyzed_results.items():
            if "error" not in analysis:
                analyzed_count += 1
                total_functions += len(analysis.get("functions", []))
                total_classes += len(analysis.get("classes", []))
                total_complexity += analysis.get("complexity_score", 0)
                total_lines += analysis.get("lines_of_code", 0)
                languages.add(analysis.get("language", "unknown"))

        avg_complexity = total_complexity / analyzed_count if analyzed_count > 0 else 0

        return {
            "total_files": len(original_files),
            "analyzed_files": analyzed_count,
            "files": analyzed_results,
            "summary": {
                "total_functions": total_functions,
                "total_classes": total_classes,
                "average_complexity": round(avg_complexity, 2),
                "total_lines": total_lines,
                "languages": list(languages),
            },
            "processing": {
                "mode": "parallel" if parallel and num_batches > 1 else "sequential",
                "num_batches": num_batches,
                "files_per_batch": (
                    len(analyzed_results) // num_batches if num_batches > 0 else 0
                ),
                "llm_calls": num_batches,
            },
        }

    def _empty_result(self, warning: str = None) -> dict:
        """Generate empty result structure.

        Args:
            warning (str, optional): Warning message

        Returns:
            dict: Empty result structure
        """
        result = {
            "total_files": 0,
            "analyzed_files": 0,
            "files": {},
            "summary": {
                "total_functions": 0,
                "total_classes": 0,
                "average_complexity": 0,
                "total_lines": 0,
                "languages": [],
            },
            "processing": {
                "mode": "none",
                "num_batches": 0,
                "files_per_batch": 0,
                "llm_calls": 0,
            },
        }

        if warning:
            result["warning"] = warning

        return result


# ========== CodeAnalysisAgentTest ==========
def CodeAnalysisAgentTest():
    """Test the CodeAnalysisAgent."""
    print("=" * 80)
    print("Testing CodeAnalysisAgent with Batch LLM Calls")
    print("=" * 80)

    repo_path = "./.repos/facebook_zstd"

    # Collect test files
    file_list = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [
            d
            for d in dirs
            if d not in [".git", "node_modules", "__pycache__", "build", "dist"]
        ]

        for file in files:
            if file.endswith((".c", ".h")):
                file_path = os.path.join(root, file)
                file_list.append(file_path)

                if len(file_list) >= 20:
                    break

        if len(file_list) >= 20:
            break

    print(f"\nCollected {len(file_list)} files for testing")

    # Create agent
    agent = CodeAnalysisAgent(repo_path=repo_path)

    # Run analysis
    print("\n" + "=" * 80)
    print("Running Analysis")
    print("=" * 80)

    results = agent.run(
        file_list=file_list, batch_size=2, parallel_batches=True, max_workers=10
    )

    # Print results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Processing Mode: {results['processing']['mode']}")
    print(f"Total Files: {results['total_files']}")
    print(f"Analyzed Files: {results['analyzed_files']}")
    print(f"Files per Batch: {results['processing']['files_per_batch']}")
    print(f"Total LLM Calls: {results['processing']['llm_calls']}")
    print(f"Total Functions: {results['summary']['total_functions']}")
    print(f"Total Classes: {results['summary']['total_classes']}")
    print(f"Average Complexity: {results['summary']['average_complexity']}")
    print(f"Total Lines: {results['summary']['total_lines']}")
    print(f"Languages: {', '.join(results['summary']['languages'])}")


if __name__ == "__main__":
    CodeAnalysisAgentTest()
