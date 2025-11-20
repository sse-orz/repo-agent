from config import CONFIG
from agents.tools import (
    read_file_tool,
    write_file_tool,
    get_repo_structure_tool,
    get_repo_basic_info_tool,
    get_repo_commit_info_tool,
    code_file_analysis_tool,
)
from .repo_info_agent import RepoInfoAgent
from .code_analysis_agent import CodeAnalysisAgent
from .doc_generation_agent import DocGenerationAgent
from .summary_agent import SummaryAgent
from .base_agent import BaseAgent, AgentState

from langchain_core.messages import SystemMessage, HumanMessage
import os
import json
import traceback
import time
from datetime import datetime
from typing import Dict, Any, List


class SupervisorAgent(BaseAgent):
    """
    Orchestrates the complete Wiki generation pipeline with intelligent
    LLM-driven file selection for optimal code analysis.
    """

    def __init__(
        self, repo_path: str, wiki_path: str, owner: str = None, repo_name: str = None
    ):
        """Initialize the Wiki Supervisor.

        Args:
            repo_path (str): Local path to the repository
            wiki_path (str): Path to save wiki files
            owner (str, optional): Repository owner for remote info
            repo_name (str, optional): Repository name for remote info
        """
        # Initialize base agent with supervisor-specific system prompt
        system_prompt = SystemMessage(
            content="""You are a Wiki generation supervisor specializing in intelligent file selection.
                        
                        Your expertise includes:
                        - Identifying core/critical files that define main functionality
                        - Recognizing architectural patterns and key components
                        - Prioritizing smaller files for efficiency
                        - Avoiding redundant, boilerplate, or generated code
                        - Selecting diverse files covering different aspects
                        
                        Always return your selection in valid JSON format with clear reasoning.
                        Focus on quality over quantity - select files that provide maximum insight into the codebase.
                    """
        )

        # Initialize BaseAgent with no tools (supervisor doesn't use tools directly)
        super().__init__(
            tools=[],
            system_prompt=system_prompt,
            repo_path=repo_path,
            wiki_path=wiki_path,
        )

        self.owner = owner
        self.repo_name = repo_name

        # Initialize agents for each stage
        self.repo_agent = RepoInfoAgent(repo_path=repo_path, wiki_path=wiki_path)
        self.code_agent = CodeAnalysisAgent(repo_path=repo_path, wiki_path=wiki_path)
        self.doc_agent = DocGenerationAgent(repo_path=repo_path, wiki_path=wiki_path)
        self.summary_agent = SummaryAgent(repo_path=repo_path, wiki_path=wiki_path)

        # Store results from each stage
        self.repo_info = None
        self.code_analysis = None
        self.doc_result = None
        self.summary_result = None

        # Time tracking for each stage
        self.stage_times = {
            "repo_info": 0.0,
            "file_selection": 0.0,
            "code_analysis": 0.0,
            "documentation": 0.0,
            "index_generation": 0.0,
            "total": 0.0,
        }

    def generate(
        self, max_files: int = 50, batch_size: int = 10, max_workers: int = 3
    ) -> Dict[str, Any]:
        """Execute the complete wiki generation pipeline with time tracking.

        Args:
            max_files (int): Maximum number of files to analyze (default 50)
            batch_size (int): Files per batch for code analysis (default 10)
            max_workers (int): Maximum parallel workers for batch processing (default 3)

        Returns:
            Dict[str, Any]: Summary of the entire pipeline execution with time statistics
        """
        pipeline_start_time = time.time()

        print("\n" + "=" * 80)
        print("WIKI GENERATION PIPELINE STARTED")
        print("=" * 80)

        try:
            # Stage 1: Collect repository information
            print("\n" + "=" * 80)
            print("Stage 1: Collecting Repository Information")
            print("=" * 80)
            stage_start = time.time()

            self.repo_info = self.repo_agent.run(
                owner=self.owner, repo_name=self.repo_name
            )

            self.stage_times["repo_info"] = time.time() - stage_start
            print(f"✓ Repository: {self.repo_info.get('repo_name', 'Unknown')}")
            print(f"✓ Language: {self.repo_info.get('main_language', 'Unknown')}")
            print(f"✓ Directories: {len(self.repo_info.get('structure', []))}")
            print(f" Time: {self.stage_times['repo_info']:.2f}s")

            # Stage 2: Intelligently select and analyze code files
            print("\n" + "=" * 80)
            print("Stage 2: Intelligent File Selection & Code Analysis")
            print("=" * 80)

            # Sub-stage 2.1: File selection
            selection_start = time.time()
            file_list = self._select_important_files(self.repo_info, max_files)
            self.stage_times["file_selection"] = time.time() - selection_start
            print(f"File Selection Time: {self.stage_times['file_selection']:.2f}s")

            # Sub-stage 2.2: Code analysis
            if not file_list:
                print("⚠ No code files found to analyze")
                self.code_analysis = {
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
                }
                self.stage_times["code_analysis"] = 0.0
            else:
                analysis_start = time.time()
                self.code_analysis = self.code_agent.run(
                    file_list=file_list,
                    batch_size=batch_size,
                    parallel_batches=True,
                    max_workers=max_workers,
                )
                self.stage_times["code_analysis"] = time.time() - analysis_start

                print(
                    f"\n✓ Analyzed {self.code_analysis.get('analyzed_files', 0)} files"
                )
                print(
                    f"✓ Total functions: {self.code_analysis['summary'].get('total_functions', 0)}"
                )
                print(
                    f"✓ Total classes: {self.code_analysis['summary'].get('total_classes', 0)}"
                )
                print(f"Code Analysis Time: {self.stage_times['code_analysis']:.2f}s")

            # Stage 3: Generate documentation
            print("\n" + "=" * 80)
            print("Stage 3: Generating Documentation")
            print("=" * 80)
            doc_start = time.time()

            self.doc_result = self.doc_agent.run_parallel(
                repo_info=self.repo_info,
                code_analysis=self.code_analysis,
                wiki_path=self.wiki_path,
            )

            self.stage_times["documentation"] = time.time() - doc_start
            generated_docs = self.doc_result.get("generated_files", [])
            print(f"✓ Generated {len(generated_docs)} documents")
            print(f"Documentation Time: {self.stage_times['documentation']:.2f}s")

            # Stage 4: Generate index
            print("\n" + "=" * 80)
            print("Stage 4: Generating Index")
            print("=" * 80)
            index_start = time.time()

            self.summary_result = self.summary_agent.run(
                docs=generated_docs,
                wiki_path=self.wiki_path,
                repo_info=self.repo_info,
                code_analysis=self.code_analysis,
            )

            self.stage_times["index_generation"] = time.time() - index_start
            print(f"✓ Index file: {self.summary_result.get('index_file', 'N/A')}")
            print(f"Index Generation Time: {self.stage_times['index_generation']:.2f}s")

            # Calculate total time
            self.stage_times["total"] = time.time() - pipeline_start_time

            # Final summary
            print("\n" + "=" * 80)
            print("WIKI GENERATION PIPELINE COMPLETED")
            print("=" * 80)

            pipeline_summary = self._generate_pipeline_summary()
            self._print_summary(pipeline_summary)

            return pipeline_summary

        except Exception as e:
            self.stage_times["total"] = time.time() - pipeline_start_time
            print(f"\n❌ Error during wiki generation: {e}")
            traceback.print_exc()
            return {
                "status": "failed",
                "error": str(e),
                "stage": self._get_current_stage(),
                "time_statistics": self.stage_times,
            }

    def _select_important_files(self, repo_info: dict, max_files: int) -> list:
        """Intelligently select important files for analysis using LLM.

        Args:
            repo_info (dict): Repository information
            max_files (int): Maximum number of files to select

        Returns:
            list: List of file paths to analyze
        """
        print(
            f"\n Using LLM to intelligently select files for analysis (max: {max_files})..."
        )

        # Step 1: Collect all code files with metadata
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
            ".swift",
            ".kt",
        }

        all_files_metadata = []
        exclude_dirs = {
            ".git",
            "node_modules",
            "__pycache__",
            "build",
            "dist",
            "target",
            "venv",
            "env",
            ".venv",
            ".env",
            "vendor",
            "third_party",
            ".pytest_cache",
            ".mypy_cache",
            ".tox",
            "htmlcov",
            "coverage",
            ".idea",
            ".vscode",
            "eggs",
            ".eggs",
            "lib64",
            "parts",
        }

        print("Scanning repository for code files...")
        for root, dirs, files in os.walk(self.repo_path):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if any(file.endswith(ext) for ext in code_extensions):
                    file_path = os.path.join(root, file)
                    try:
                        file_size = os.path.getsize(file_path)
                        relative_path = os.path.relpath(file_path, self.repo_path)

                        # Categorize file importance based on path
                        importance_hints = []
                        path_lower = relative_path.lower()

                        # Core/important directories
                        if any(
                            d in path_lower
                            for d in [
                                "src/",
                                "lib/",
                                "core/",
                                "api/",
                                "main/",
                                "pkg/",
                                "internal/",
                            ]
                        ):
                            importance_hints.append("core_directory")

                        # Common important file patterns
                        if any(
                            pattern in file.lower()
                            for pattern in [
                                "main",
                                "index",
                                "app",
                                "server",
                                "client",
                                "config",
                            ]
                        ):
                            importance_hints.append("key_file_name")

                        # Entry points
                        if file in [
                            "main.py",
                            "index.js",
                            "app.py",
                            "server.py",
                            "__init__.py",
                            "main.c",
                            "main.cpp",
                            "main.go",
                            "main.rs",
                            "index.ts",
                        ]:
                            importance_hints.append("entry_point")

                        # Test files (lower priority)
                        if any(
                            d in path_lower
                            for d in ["test/", "tests/", "spec/", "__tests__/"]
                        ):
                            importance_hints.append("test_file")

                        # Example/demo files (lower priority)
                        if any(
                            d in path_lower
                            for d in [
                                "example/",
                                "examples/",
                                "demo/",
                                "sample/",
                                "docs/",
                            ]
                        ):
                            importance_hints.append("example_file")

                        all_files_metadata.append(
                            {
                                "path": file_path,
                                "relative_path": relative_path,
                                "filename": file,
                                "size_bytes": file_size,
                                "size_kb": round(file_size / 1024, 2),
                                "extension": os.path.splitext(file)[1],
                                "hints": importance_hints,
                            }
                        )
                    except OSError:
                        continue

        print(f"Found {len(all_files_metadata)} code files in total")

        if not all_files_metadata:
            print("⚠ No code files found")
            return []

        # Step 2: Prepare file list for LLM (limit to avoid token overflow)
        max_files_for_llm = min(500, len(all_files_metadata))

        # Sort by size (smaller first) and take subset for LLM analysis
        all_files_metadata.sort(key=lambda x: x["size_bytes"])
        files_for_selection = all_files_metadata[:max_files_for_llm]

        # Group files by directory for better context
        files_by_dir = {}
        for file_meta in files_for_selection:
            dir_name = os.path.dirname(file_meta["relative_path"]) or "."
            if dir_name not in files_by_dir:
                files_by_dir[dir_name] = []
            files_by_dir[dir_name].append(
                {
                    "file": file_meta["filename"],
                    "size_kb": file_meta["size_kb"],
                    "hints": file_meta["hints"],
                }
            )

        # Limit the JSON size for prompt
        files_by_dir_str = json.dumps(files_by_dir, indent=2)
        if len(files_by_dir_str) > 8000:
            # If too large, sample directories
            sampled_dirs = dict(list(files_by_dir.items())[:50])
            files_by_dir_str = json.dumps(sampled_dirs, indent=2)[:8000]

        # Create structured prompt for LLM
        prompt = f"""
                    You are a code analysis expert. Your task is to select the most important files for code analysis.

                    **Repository Information:**
                    - Name: {repo_info.get('repo_name', 'Unknown')}
                    - Language: {repo_info.get('main_language', 'Unknown')}
                    - Main Directories: {', '.join(repo_info.get('structure', [])[:15])}
                    - Total Files Available: {len(all_files_metadata)}

                    **Selection Criteria (Priority Order):**

                    1. **High Priority Files (Select These First):**
                    - Core library/framework code (src/, lib/, core/, pkg/)
                    - Main entry points (main.py, index.js, app.py, server.py, etc.)
                    - API definitions and interfaces
                    - Important utility modules
                    - Configuration loaders

                    2. **Medium Priority Files:**
                    - Data models and schemas
                    - Common utilities and helpers
                    - Database/storage interfaces
                    - Authentication/authorization code

                    3. **Lower Priority Files (Select Only If Space):**
                    - Test files (only if they define important interfaces)
                    - Example/demo code
                    - Documentation generators
                    - Build scripts

                    4. **File Size Preferences:**
                    - **Strongly prefer** smaller files (< 10 KB) for efficiency
                    - **Acceptable** medium files (10-20 KB) if important
                    - **Avoid** large files (> 50 KB) unless absolutely critical
                    - Balance: Select diverse small files over few large files

                    **Available Files by Directory:**
                    {files_by_dir_str}

                    **Your Task:**
                    Select exactly {max_files} files (or fewer if less available) that:
                    1. Best represent the codebase's core functionality
                    2. Are relatively small in size (prefer < 20 KB)
                    3. Cover different aspects/modules of the system
                    4. Minimize redundancy (don't select similar files)
                    5. Prioritize files with "core_directory" or "entry_point" hints
                    6. Avoid "test_file" and "example_file" unless critical

                    **Output Format:**
                    Return ONLY a JSON object with this exact structure:
                    {{
                        "selected_files": [
                            "relative/path/to/file1.py",
                            "relative/path/to/file2.js",
                            ...
                        ],
                        "reasoning": "Brief 1-2 sentence explanation of your selection strategy"
                    }}

                    **Important:** 
                    - Return ONLY the JSON object, no markdown code blocks or additional text
                    - Use relative paths exactly as shown in the directory structure
                    - Select up to {max_files} files total
                """

        # Run the selection using LLM directly (no need for agent workflow)
        print("LLM is analyzing and selecting files...")

        try:
            # Direct LLM call for faster response (avoid agent workflow overhead)
            system_prompt = SystemMessage(
                content="""You are a code analysis expert specializing in selecting important files for codebase analysis.
                
                Your expertise includes:
                - Identifying core/critical files that define main functionality
                - Recognizing architectural patterns and key components
                - Prioritizing smaller files for efficiency
                - Avoiding redundant, boilerplate, or generated code
                - Selecting diverse files covering different aspects
                
                Always return your selection in valid JSON format with clear reasoning.
                Focus on quality over quantity - select files that provide maximum insight into the codebase.
                """
            )

            response = self.llm.invoke([system_prompt, HumanMessage(content=prompt)])
            content = response.content

            # Extract selected files from LLM response
            selected_files = []

            try:
                import re

                # Extract JSON from response (handle markdown code blocks)
                json_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", content)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_match = re.search(r"\{[\s\S]*\}", content)
                    json_str = json_match.group() if json_match else None

                if json_str:
                    result = json.loads(json_str)
                    selected_relative_paths = result.get("selected_files", [])
                    reasoning = result.get("reasoning", "No reasoning provided")

                    print(f"\n✓ LLM Selection Reasoning: {reasoning}")
                    print(f"✓ LLM selected {len(selected_relative_paths)} files")

                    # Convert relative paths to absolute paths
                    for rel_path in selected_relative_paths:
                        abs_path = os.path.join(self.repo_path, rel_path)
                        if os.path.exists(abs_path):
                            selected_files.append(abs_path)
                        else:
                            # Try to find matching file in metadata
                            for file_meta in all_files_metadata:
                                if file_meta["relative_path"] == rel_path:
                                    selected_files.append(file_meta["path"])
                                    break
                else:
                    print("⚠ No JSON found in LLM response")

            except json.JSONDecodeError as e:
                print(f"⚠ Failed to parse LLM response as JSON: {e}")
                print(f"Response content: {content[:500]}...")
            except Exception as e:
                print(f"⚠ Error processing LLM response: {e}")

        except Exception as e:
            print(f"⚠ LLM selection error: {e}")
            return self._fallback_file_selection(all_files_metadata, max_files)

        # Fallback if LLM selection failed or returned no files
        if not selected_files:
            print(
                "⚠ LLM selection failed or returned no files, using fallback heuristic selection..."
            )
            selected_files = self._fallback_file_selection(
                all_files_metadata, max_files
            )

        # Ensure we don't exceed max_files
        selected_files = selected_files[:max_files]

        print(f"\n✓ Final selection: {len(selected_files)} files")
        if selected_files:
            print("Selected files:")
            for i, f in enumerate(selected_files[:10], 1):
                try:
                    size_kb = os.path.getsize(f) / 1024
                    rel_path = os.path.relpath(f, self.repo_path)
                    print(f"  {i}. {rel_path} ({size_kb:.2f} KB)")
                except:
                    print(f"  {i}. {f}")
            if len(selected_files) > 10:
                print(f"  ... and {len(selected_files) - 10} more files")

        return selected_files

    def _fallback_file_selection(self, all_files: list, max_files: int) -> list:
        """Fallback heuristic-based file selection when LLM fails.

        Args:
            all_files (list): List of all file metadata
            max_files (int): Maximum files to select

        Returns:
            list: Selected file paths
        """
        print("Using heuristic selection algorithm...")

        # Score each file based on heuristics
        scored_files = []

        for file_meta in all_files:
            score = 0
            path = file_meta["relative_path"].lower()
            size_kb = file_meta["size_kb"]

            # Importance scoring
            if "entry_point" in file_meta["hints"]:
                score += 100
            if "core_directory" in file_meta["hints"]:
                score += 50
            if "key_file_name" in file_meta["hints"]:
                score += 30

            # Penalty for test/example files
            if "test_file" in file_meta["hints"]:
                score -= 50
            if "example_file" in file_meta["hints"]:
                score -= 30

            # Size scoring (prefer smaller files)
            if size_kb < 5:
                score += 40
            elif size_kb < 10:
                score += 30
            elif size_kb < 20:
                score += 20
            elif size_kb < 50:
                score += 10
            else:
                score -= 20  # Penalize large files

            # Path-based scoring
            if any(d in path for d in ["src/", "lib/", "core/", "pkg/"]):
                score += 40
            elif any(d in path for d in ["util/", "helper/", "common/", "api/"]):
                score += 20

            # Penalty for test/example paths
            if any(d in path for d in ["test/", "example/", "demo/", "docs/"]):
                score -= 30

            scored_files.append(
                {
                    "path": file_meta["path"],
                    "score": score,
                    "size_kb": size_kb,
                    "relative_path": file_meta["relative_path"],
                }
            )

        # Sort by score (descending) and select top files
        scored_files.sort(key=lambda x: x["score"], reverse=True)
        selected = [f["path"] for f in scored_files[:max_files]]

        print(f"Selected top {len(selected)} files by heuristic scoring")
        if scored_files:
            print(
                f"Top scored file: {scored_files[0]['relative_path']} (score: {scored_files[0]['score']})"
            )

        return selected

    def _get_current_stage(self) -> str:
        """Get the current stage of pipeline execution."""
        if self.summary_result:
            return "completed"
        elif self.doc_result:
            return "stage_4_summary"
        elif self.code_analysis:
            return "stage_3_documentation"
        elif self.repo_info:
            return "stage_2_code_analysis"
        else:
            return "stage_1_repo_info"

    def _generate_pipeline_summary(self) -> Dict[str, Any]:
        """Generate a comprehensive summary of the pipeline execution with time statistics."""
        return {
            "status": "success",
            "wiki_path": self.wiki_path,
            "time_statistics": {
                "total_time": self.stage_times["total"],
                "stages": {
                    "repo_info": {
                        "time_seconds": self.stage_times["repo_info"],
                        "percentage": (
                            (
                                self.stage_times["repo_info"]
                                / self.stage_times["total"]
                                * 100
                            )
                            if self.stage_times["total"] > 0
                            else 0
                        ),
                    },
                    "file_selection": {
                        "time_seconds": self.stage_times["file_selection"],
                        "percentage": (
                            (
                                self.stage_times["file_selection"]
                                / self.stage_times["total"]
                                * 100
                            )
                            if self.stage_times["total"] > 0
                            else 0
                        ),
                    },
                    "code_analysis": {
                        "time_seconds": self.stage_times["code_analysis"],
                        "percentage": (
                            (
                                self.stage_times["code_analysis"]
                                / self.stage_times["total"]
                                * 100
                            )
                            if self.stage_times["total"] > 0
                            else 0
                        ),
                    },
                    "documentation": {
                        "time_seconds": self.stage_times["documentation"],
                        "percentage": (
                            (
                                self.stage_times["documentation"]
                                / self.stage_times["total"]
                                * 100
                            )
                            if self.stage_times["total"] > 0
                            else 0
                        ),
                    },
                    "index_generation": {
                        "time_seconds": self.stage_times["index_generation"],
                        "percentage": (
                            (
                                self.stage_times["index_generation"]
                                / self.stage_times["total"]
                                * 100
                            )
                            if self.stage_times["total"] > 0
                            else 0
                        ),
                    },
                },
            },
            "stages": {
                "repo_info": {
                    "completed": self.repo_info is not None,
                    "repo_name": (
                        self.repo_info.get("repo_name", "N/A")
                        if self.repo_info
                        else "N/A"
                    ),
                    "language": (
                        self.repo_info.get("main_language", "N/A")
                        if self.repo_info
                        else "N/A"
                    ),
                },
                "code_analysis": {
                    "completed": self.code_analysis is not None,
                    "files_analyzed": (
                        self.code_analysis.get("analyzed_files", 0)
                        if self.code_analysis
                        else 0
                    ),
                    "total_functions": (
                        self.code_analysis["summary"].get("total_functions", 0)
                        if self.code_analysis
                        else 0
                    ),
                },
                "documentation": {
                    "completed": self.doc_result is not None,
                    "files_generated": (
                        len(self.doc_result.get("generated_files", []))
                        if self.doc_result
                        else 0
                    ),
                    "verification_status": (
                        self.doc_result.get("verification_status", "N/A")
                        if self.doc_result
                        else "N/A"
                    ),
                },
                "index": {
                    "completed": self.summary_result is not None,
                    "index_file": (
                        self.summary_result.get("index_file", "N/A")
                        if self.summary_result
                        else "N/A"
                    ),
                    "verification_status": (
                        self.summary_result.get("verification_status", "N/A")
                        if self.summary_result
                        else "N/A"
                    ),
                },
            },
            "statistics": {
                "total_documents": (
                    self.summary_result.get("total_documents", 0)
                    if self.summary_result
                    else 0
                ),
                "total_functions": (
                    self.code_analysis["summary"].get("total_functions", 0)
                    if self.code_analysis
                    else 0
                ),
                "total_classes": (
                    self.code_analysis["summary"].get("total_classes", 0)
                    if self.code_analysis
                    else 0
                ),
                "average_complexity": (
                    self.code_analysis["summary"].get("average_complexity", 0)
                    if self.code_analysis
                    else 0
                ),
            },
        }

    def _print_summary(self, summary: Dict[str, Any]):
        """Print a formatted summary of the pipeline execution with time statistics."""
        print(f"\nPipeline Summary:")
        print(f"  Wiki Location: {summary['wiki_path']}")

        print(f"\nTime Statistics:")
        time_stats = summary.get("time_statistics", {})
        total_time = time_stats.get("total_time", 0)
        print(f"  Total Time: {total_time:.2f}s")

        if "stages" in time_stats:
            print(f"\n  Stage Breakdown:")
            for stage_name, stage_data in time_stats["stages"].items():
                time_sec = stage_data.get("time_seconds", 0)
                percentage = stage_data.get("percentage", 0)
                stage_display = stage_name.replace("_", " ").title()
                print(f"    • {stage_display}: {time_sec:.2f}s ({percentage:.1f}%)")

        print(f"\nStatistics:")
        print(f"  Total Documents: {summary['statistics']['total_documents']}")
        print(f"  Functions Analyzed: {summary['statistics']['total_functions']}")
        print(f"  Classes Analyzed: {summary['statistics']['total_classes']}")
        print(
            f"  Average Complexity: {summary['statistics']['average_complexity']:.2f}"
        )

        print(f"\n Stage Completion:")
        for stage_name, stage_info in summary["stages"].items():
            status = "✓" if stage_info.get("completed") else "✗"
            stage_display = stage_name.replace("_", " ").title()
            print(f"  {status} {stage_display}")


# ========== Test Function ==========
def test_wiki_supervisor():
    """Test the complete SupervisorAgent pipeline with LLM-driven file selection."""

    print("\n" + "=" * 80)
    print("SupervisorAgent Test - LLM-Driven File Selection")
    print("=" * 80)

    supervisor = SupervisorAgent(
        repo_path="/mnt/zhongjf25/workspace/repo-agent/.repos/facebook_zstd",
        wiki_path="/mnt/zhongjf25/workspace/repo-agent/.wikis/supervisor_llm_test",
        owner="facebook",
        repo_name="zstd",
    )

    # Run with intelligent file selection
    result = supervisor.generate(
        max_files=30,  # Select up to 30 important files
        batch_size=10,  # Process 10 files per batch
        max_workers=3,  # Use 3 parallel workers
    )

    print("\n" + "=" * 80)
    print("Final Result:")
    print("=" * 80)
    print(json.dumps(result, indent=2))

    return result


if __name__ == "__main__":
    test_wiki_supervisor()
