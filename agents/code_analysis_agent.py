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

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple


class CodeAnalysisAgent(BaseAgent):
    """Agent for analyzing code files.
    
    Inherits from BaseAgent to leverage common workflow patterns.
    Supports batch processing and parallel execution for efficiency.
    """
    
    def __init__(self, repo_path: str, wiki_path: str = ""):
        """Initialize the CodeAnalysisAgent.
        
        Args:
            repo_path (str): Local path to the repository
            wiki_path (str): Path to wiki (optional)
        """
        system_prompt = SystemMessage(
            content="""You are a code analysis expert. Your task is to analyze code files efficiently.

                        IMPORTANT RULES:
                        1. Call code_file_analysis_tool ONLY ONCE per file
                        2. After analyzing the file, return results immediately in JSON format
                        3. Do NOT repeat tool calls if you already have the results

                        For the file, extract:
                        - Main functions/classes with signatures
                        - Dependencies and imports
                        - Complexity score (1-10)
                        - Lines of code

                        Return results in this exact JSON structure:
                        {
                            "language": "string",
                            "functions": [{"name": "...", "signature": "...", "line_start": 0, "line_end": 0}],
                            "classes": [{"name": "...", "methods": [], "line_start": 0, "line_end": 0}],
                            "imports": [],
                            "complexity_score": 5,
                            "lines_of_code": 100,
                            "summary": "Brief description"
                        }

                        CRITICAL:
                        - Return ONLY the JSON object, no additional text
                        - Do NOT wrap in markdown code blocks
                        - Ensure all fields are present
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
            wiki_path=wiki_path
        )

    def _analyze_single_file(
        self, 
        file_path: str, 
        batch_num: int, 
        file_num: int
    ) -> Tuple[str, dict]:
        """Analyze a single file using a fresh agent instance.
        
        Each file gets its own agent to avoid context accumulation and token overflow.
        
        Args:
            file_path (str): Path to the file to analyze
            batch_num (int): Batch number for logging
            file_num (int): File number within batch
            
        Returns:
            Tuple[str, dict]: (file_path, analysis_result)
        """
        # Create fresh agent for this file (avoids context accumulation)
        file_agent = CodeAnalysisAgent(repo_path=self.repo_path, wiki_path=self.wiki_path)
        
        # Prepare prompt for single file
        prompt = f"""
                    Analyze this code file:

                    File path: {file_path}

                    Tasks:
                    1. Use code_file_analysis_tool with the file path AS IS
                    2. Extract all required information
                    3. Calculate complexity score based on:
                    - Number of functions/classes
                    - Code length
                    - Nesting depth
                    - Dependencies

                    Return the analysis in the exact JSON format specified in the system prompt.

                    IMPORTANT: Return ONLY the JSON, no extra text.
                """

        initial_state = AgentState(
            messages=[HumanMessage(content=prompt)],
            repo_path=self.repo_path,
            wiki_path=self.wiki_path,
        )

        # Run analysis
        final_state = None
        try:
            for state in file_agent.app.stream(
                initial_state,
                stream_mode="values",
                config={
                    "configurable": {
                        "thread_id": f"file-{batch_num}-{file_num}-{datetime.now().timestamp()}"
                    },
                    "recursion_limit": 50,
                },
            ):
                final_state = state

            # Extract result
            if final_state:
                result = self._extract_file_result(final_state)
                if "error" not in result:
                    print(f"    [Batch {batch_num}] ✓ {os.path.basename(file_path)}")
                else:
                    print(f"    [Batch {batch_num}] ⚠ {os.path.basename(file_path)}: {result['error']}")
                return file_path, result
            else:
                print(f"    [Batch {batch_num}] ✗ {os.path.basename(file_path)}: No result")
                return file_path, {"error": "No result from agent"}

        except Exception as e:
            print(f"    [Batch {batch_num}] ✗ {os.path.basename(file_path)}: {e}")
            return file_path, {"error": str(e)}

    def _extract_file_result(self, final_state: AgentState) -> dict:
        """Extract analysis result from final state.
        
        Args:
            final_state (AgentState): The final agent state
            
        Returns:
            dict: Parsed analysis result or error
        """
        if not final_state:
            return {"error": "No final state"}

        last_message = final_state["messages"][-1]
        
        if not hasattr(last_message, "content"):
            return {"error": "No content in message"}

        try:
            content = last_message.content
            
            # Try to extract JSON
            # Method 1: Look for JSON code block
            json_code_block = re.search(r'```json\s*([\s\S]*?)\s*```', content)
            if json_code_block:
                json_str = json_code_block.group(1)
            else:
                # Method 2: Look for plain JSON object
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    json_str = json_match.group()
                else:
                    return {
                        "error": "No JSON found in response",
                        "raw_output": content[:500]
                    }
            
            # Clean JSON string
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)  # Remove trailing commas
            
            # Parse JSON
            result = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["language", "functions", "classes", "complexity_score", "lines_of_code"]
            missing_fields = [f for f in required_fields if f not in result]
            
            if missing_fields:
                return {
                    "error": f"Missing fields: {missing_fields}",
                    "partial_data": result
                }
            
            return result

        except json.JSONDecodeError as e:
            return {
                "error": f"JSON parse error: {e}",
                "raw_output": content[:500]
            }
        except Exception as e:
            return {
                "error": f"Unexpected error: {e}",
                "raw_output": str(content)[:500] if content else ""
            }

    def _analyze_batch(
        self, 
        batch: List[str], 
        batch_num: int
    ) -> Tuple[int, dict]:
        """Analyze a batch of files sequentially.
        
        Args:
            batch (List[str]): List of file paths
            batch_num (int): Batch number for logging
            
        Returns:
            Tuple[int, dict]: (batch_num, batch_results)
        """
        print(f"  [Batch {batch_num}] Starting analysis of {len(batch)} files...")
        
        batch_results = {}
        
        for i, file_path in enumerate(batch, 1):
            file_path_key, file_result = self._analyze_single_file(
                file_path, batch_num, i
            )
            batch_results[file_path_key] = file_result
        
        print(f"  [Batch {batch_num}] ✓ Completed ({len(batch_results)} files)")
        return batch_num, batch_results

    def run(
        self,
        file_list: list,
        batch_size: int = 2,
        parallel_batches: bool = True,
        max_workers: int = 10,
        max_files: int = 500
    ) -> dict:
        """Analyze code files with optional parallel batch processing.

        Args:
            file_list (list): List of file paths to analyze
            batch_size (int): Number of files per batch (default: 10)
            parallel_batches (bool): Whether to process batches in parallel (default: True)
            max_workers (int): Maximum number of parallel workers (default: 3)

        Returns:
            dict: Analysis results including summary statistics
        """
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
            print(f"\n Processing {len(batches)} batches in PARALLEL (max {max_workers} workers)...")
            all_results = self._process_batches_parallel(batches, max_workers)
        else:
            print(f"\n Processing {len(batches)} batches SEQUENTIALLY...")
            all_results = self._process_batches_sequential(batches)

        # Calculate summary
        return self._build_result(file_list, all_results, parallel_batches, len(batches))

    def _filter_valid_files(self, file_list: list) -> list:
        """Filter and validate code files.
        
        Args:
            file_list (list): List of file paths
            
        Returns:
            list: List of valid code file paths
        """
        code_extensions = {
            '.py', '.js', '.java', '.cpp', '.c', '.go', '.rs', '.ts',
            '.jsx', '.tsx', '.h', '.hpp', '.cs', '.rb', '.php'
        }
        
        valid_files = []
        for f in file_list:
            if os.path.isfile(f) and os.path.splitext(f)[1] in code_extensions:
                valid_files.append(f)
        
        print(f"✓ Found {len(valid_files)} valid code files")
        return valid_files

    def _create_balanced_batches(self, files: list, batch_size: int) -> list:
        """Create balanced batches using greedy algorithm.
        
        Files are sorted by size (largest first) and assigned to batches
        to minimize variance in batch sizes.
        
        Args:
            files (list): List of file paths
            batch_size (int): Target number of files per batch
            
        Returns:
            list: List of batches (each batch is a list of file paths)
        """
        print("\nSorting files by size...")
        
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
        print(f"Total size: {total_size / 1024:.2f} KB")
        print(f"Average file size: {total_size / len(file_sizes) / 1024:.2f} KB")
        print(f"Largest file: {file_sizes[0][1] / 1024:.2f} KB")
        print(f"Smallest file: {file_sizes[-1][1] / 1024:.2f} KB")
        
        # Create batches using greedy algorithm
        num_batches = (len(files) + batch_size - 1) // batch_size
        batches = [[] for _ in range(num_batches)]
        batch_sizes = [0] * num_batches
        
        # Assign each file to the batch with smallest current size
        for file_path, file_size in file_sizes:
            min_batch_idx = batch_sizes.index(min(batch_sizes))
            batches[min_batch_idx].append(file_path)
            batch_sizes[min_batch_idx] += file_size
        
        # Remove empty batches
        batches = [b for b in batches if b]
        
        print(f"\nCreated {len(batches)} balanced batches:")
        for i, (batch, size) in enumerate(zip(batches, batch_sizes[:len(batches)]), 1):
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
                    print(f"✓ Batch {batch_num} results merged")
                except Exception as e:
                    print(f"✗ Batch {batch_idx + 1} failed: {e}")
                    # Add error for all files in failed batch
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
        num_batches: int
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
                "batch_size": len(original_files) // num_batches if num_batches > 0 else 0,
            }
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
                "batch_size": 0,
            }
        }
        
        if warning:
            result["warning"] = warning
        
        return result


# ========== CodeAnalysisAgentTest ==========
def CodeAnalysisAgentTest():
    
    repo_path = "./.repos/facebook_zstd"
    
    # Collect test files
    file_list = []
    for root, dirs, files in os.walk(repo_path):
        # Skip common directories
        dirs[:] = [d for d in dirs if d not in ['.git', 'node_modules', '__pycache__', 'build', 'dist']]
        
        for file in files:
            if file.endswith(('.c', '.h')):
                file_path = os.path.join(root, file)
                file_list.append(file_path)
                
                # Limit for testing
                if len(file_list) >= 20:
                    break
        
        if len(file_list) >= 20:
            break
    
    print(f"\nCollected {len(file_list)} files for testing")
    
    # Create agent
    agent = CodeAnalysisAgent(repo_path=repo_path)
    
    # Run analysis
    print("\n" + "="*80)
    print("Running Analysis")
    print("="*80)
    
    results = agent.run(
        file_list=file_list,
        batch_size=5,
        parallel_batches=True,
        max_workers=3
    )
    
    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Processing Mode: {results['processing']['mode']}")
    print(f"Total Files: {results['total_files']}")
    print(f"Analyzed Files: {results['analyzed_files']}")
    print(f"Total Functions: {results['summary']['total_functions']}")
    print(f"Total Classes: {results['summary']['total_classes']}")
    print(f"Average Complexity: {results['summary']['average_complexity']}")
    print(f"Total Lines: {results['summary']['total_lines']}")
    print(f"Languages: {', '.join(results['summary']['languages'])}")
    print(f"Batches: {results['processing']['num_batches']}")
    
    # Show sample results
    print("\n" + "="*80)
    print("SAMPLE ANALYSIS (first 3 files)")
    print("="*80)
    
    for i, (file_path, analysis) in enumerate(list(results['files'].items())[:3], 1):
        print(f"\n{i}. {os.path.basename(file_path)}")
        if "error" in analysis:
            print(f"   ✗ Error: {analysis['error']}")
        else:
            print(f"   Language: {analysis.get('language', 'N/A')}")
            print(f"   Functions: {len(analysis.get('functions', []))}")
            print(f"   Classes: {len(analysis.get('classes', []))}")
            print(f"   Complexity: {analysis.get('complexity_score', 0)}")
            print(f"   Lines: {analysis.get('lines_of_code', 0)}")


if __name__ == "__main__":
    CodeAnalysisAgentTest()