from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph.state import StateGraph, START
from langchain_core.messages import HumanMessage, SystemMessage
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import asyncio
import json
import os
import time

from utils.file import (
    get_repo_structure,
    write_file,
    read_file,
    resolve_path,
    read_json,
)
from utils.code_analyzer import (
    analyze_file_with_tree_sitter,
    format_tree_sitter_analysis_results,
    format_tree_sitter_analysis_results_to_prompt,
)
from .utils import draw_graph, log_state
from config import CONFIG


class CodeAnalysisSubGraphState(TypedDict):
    # inputs
    basic_info_for_code: dict
    # outputs
    code_analysis: dict


class CodeAnalysisSubGraphBuilder:

    def __init__(self):
        self.graph = self.build()

    @staticmethod
    def _get_system_prompt():
        """Generate a comprehensive system prompt for code analysis."""
        return SystemMessage(
            content="""You are an expert code analyst specializing in identifying relevant code files for static analysis.
Your role is to analyze the provided repository information and file structure to select files that are most pertinent for in-depth examination.
Guidelines:
- Focus on source code files with extensions like .py, .js, .go, .cpp, .java, .rs, .c, .h, .hpp
- Exclude documentation, configuration, and non-code files
- Consider the context provided by the repository information to prioritize files that are central to the project's functionality."""
        )

    @staticmethod
    def _get_code_filter_prompt(
        repo_info, repo_structure, max_num_files=20
    ) -> HumanMessage:
        return HumanMessage(
            content=f"""Based on the following repository information and file structure, identify EXACTLY {max_num_files} code files (or fewer if fewer exist) that are most relevant for static analysis.

CRITICAL CONSTRAINTS:
- Return MAXIMUM {max_num_files} file paths
- Focus on source code files (.py, .js, .go, .cpp, .java, .rs, .c, .h, .hpp, .ts)
- Exclude documentation, config, test, build, and dependency files
- Prioritize files central to the project's core functionality
- Sort by relevance (most important first)

Repository Information:
{repo_info}

File Structure:
{repo_structure}

IMPORTANT: Output ONLY file paths, one per line. No descriptions, headers, numbers, or any other text. If a file path contains special characters, preserve them exactly."""
        )

    @staticmethod
    def _get_code_analysis_prompt_with_content(
        file_path: str, content: str
    ) -> HumanMessage:
        return HumanMessage(
            content=f"""Please provide a concise summary (max 100 words) of the following code file content for file '{file_path}':

{content}

Focus on: key functions/classes, main purpose, and important logic."""
        )

    @staticmethod
    def _get_code_analysis_prompt_with_analysis(
        file_path: str, analysis_result
    ) -> HumanMessage:
        return HumanMessage(
            content=f"""Please provide a concise summary (max 100 words) of the following code analysis content for file '{file_path}':

{analysis_result}

Focus on: key functions/classes, main purpose, and important logic."""
        )

    @staticmethod
    def _get_system_prompt_for_doc():
        """Generate a comprehensive system prompt for code documentation generation."""
        return SystemMessage(
            content="""You are an expert technical documentation writer specializing in generating comprehensive code documentation. 
Your role is to analyze provided code analysis results and create clear, well-structured documentation that helps developers understand the codebase structure, functionality, and architecture.

Guidelines:
- Use Markdown formatting for better readability
- Be concise yet informative
- Highlight key components and their relationships
- Organize information logically with clear sections
- Include code snippets and examples when helpful
- Document complex algorithms and design patterns"""
        )

    @staticmethod
    def _get_code_doc_prompt(
        file_path: str, analysis: dict, summary: str
    ) -> HumanMessage:
        """Generate a prompt for code documentation."""

        # based on the size between analysis and content, adjust the prompt design
        def _compare_size_between_content_and_analysis(
            content: str, formatted_analysis: str
        ) -> str:
            content_size = len(content)
            analysis_size = len(formatted_analysis)

            if content_size <= analysis_size:
                return "content"
            else:
                return "analysis"

        file_path_resolved = resolve_path(file_path)
        content = read_file(file_path_resolved)
        match_choice = _compare_size_between_content_and_analysis(content, analysis)
        match match_choice:
            case "content":
                analysis = f"File Content:\n{content}"
            case "analysis":
                analysis = f"Analysis Result:\n{analysis}"
        if summary:
            summary = f"Analysis Summary:\n{summary}"
        else:
            summary = ""
        return HumanMessage(
            content=f"""Generate comprehensive documentation for the code file '{file_path}'.
Based on the following analysis results and summary, create a detailed documentation that includes:
1. Overview of the file's purpose and functionality
2. Key components (functions, classes, etc.) and their roles
3. Important algorithms or design patterns used
4. Dependencies and relationships with other files
5. Visual representations using Mermaid diagrams where applicable:
   - Class diagram (for OOP code showing class relationships and hierarchies)
   - Flowchart (for complex algorithms or control flow)
   - Sequence diagram (for interactions between components/functions)
   - Call graph (showing function call relationships)
   Use Mermaid markdown syntax (```mermaid ... ```) to create clear, informative diagrams
{analysis}
{summary}
Please structure the documentation with clear sections and make it suitable for both new contributors and project maintainers. Include Mermaid diagrams to visualize code structure, relationships, and logic flow."""
        )

    def code_filter_node(self, state: dict):
        # log_state(state)
        # this node need to filter useful code files for analysis
        # 1. get repo_info and repo_structure from basic_info_for_code
        def get_max_num_files(mode: str) -> int:
            # mode decide the number of code files to analyze
            # if "fast" -> max 20 files
            # if "smart" -> max 100 files
            match mode:
                case "fast":
                    return 20
                case "smart":
                    return 100
                case _:
                    return 20

        print("→ Processing code_filter_node...")
        basic_info_for_code = state.get("basic_info_for_code", {})
        # check code_files_updated to decide whether to re-filter code files
        code_files_updated = basic_info_for_code.get("code_files_updated", False)
        match code_files_updated:
            case True:
                # code files have been updated
                code_files = basic_info_for_code.get("code_files", [])
                print(
                    f"✓ Using updated list of {len(code_files)} code files for analysis"
                )
            case False:
                # need llm to filter code files
                mode = basic_info_for_code.get("mode", "fast")
                max_num_files = get_max_num_files(mode)
                print(f"   → Mode: {mode}, max_num_files: {max_num_files}")
                repo_info = basic_info_for_code.get("repo_info", {})
                repo_structure = basic_info_for_code.get("repo_structure", [])
                # 2. call llm to determine which files to analyze based on repo_info
                # if file_path has exists in basic_info_for_code.code_files, no need to be filtered again
                wiki_root_path = basic_info_for_code.get("wiki_root_path", "./.wikis")
                owner = basic_info_for_code.get("owner", "default")
                repo = basic_info_for_code.get("repo", "default")
                log_path = os.path.join(
                    wiki_root_path,
                    f"{owner}_{repo}",
                    "code_update_log.json",
                )
                log_data = read_json(log_path)
                existing_code_files = log_data.get("code_files", []) if log_data else []
                # remove existing_code_files from repo_structure to avoid re-selecting them
                repo_structure = [
                    file_path
                    for file_path in repo_structure
                    if file_path not in existing_code_files
                ]
                # filter supported code files (.py, .js, .go, .cpp, .java, .rs etc.)
                system_prompt = self._get_system_prompt()
                human_prompt = self._get_code_filter_prompt(
                    repo_info, repo_structure, max_num_files
                )
                llm = CONFIG.get_llm()
                response = llm.invoke([system_prompt, human_prompt])
                # parse the response to get the list of code files
                code_files = [
                    line.strip()
                    for line in response.content.splitlines()
                    if line.strip()
                ]

                # Enforce hard limit on max_num_files
                if len(code_files) > max_num_files:
                    print(
                        f"⚠ LLM returned {len(code_files)} files, limiting to {max_num_files}"
                    )
                    code_files = code_files[:max_num_files]

                # print("DEBUG: code_files =", code_files)
                print(
                    f"✓ Filtered {len(code_files)} code files for analysis (max: {max_num_files})"
                )
        return {
            "code_analysis": {
                "code_files": code_files,
            }
        }

    def code_info_update_log_node(self, state: dict):
        log_state(state)
        # this node is to create code_update_log.json
        print("→ Processing code_info_update_log_node")
        basic_info_for_code = state.get("basic_info_for_code", {})
        owner = basic_info_for_code.get("owner", "default")
        repo = basic_info_for_code.get("repo", "default")
        code_files = state.get("code_analysis", {}).get("code_files", [])
        wiki_root_path = basic_info_for_code.get("wiki_root_path", "./.wikis")
        log_path = os.path.join(
            wiki_root_path,
            f"{owner}_{repo}",
            "code_update_log.json",
        )

        # Read existing log if it exists, otherwise create new one
        existing_data = read_json(log_path)
        if existing_data and isinstance(existing_data, dict):
            # Merge new code files with existing ones
            existing_code_files = existing_data.get("code_files", [])
            merged_code_files = list(set(existing_code_files) | set(code_files))
            update_log_info = {
                "log_date": str(basic_info_for_code.get("date", "N/A")),
                "code_files": merged_code_files,
            }
        else:
            # Create new log
            update_log_info = {
                "log_date": str(basic_info_for_code.get("date", "N/A")),
                "code_files": code_files,
            }

        print("✓ Code update log documentation generated")
        write_file(
            log_path,
            json.dumps(update_log_info, indent=2, ensure_ascii=False),
        )
        print(f"✓ Code update log file written successfully to {log_path}")

    def _analyze_single_file(
        self, file_path: str, idx: int, total: int
    ) -> tuple[str, dict]:
        """
        Analyze a single file and return results.
        This method will be called concurrently for each file.

        Args:
            file_path (str): Path to the code file.
            idx (int): Index of the file in the list.
            total (int): Total number of files.
        Returns:
            tuple[str, dict]: file_path and its analysis results.
        """
        print(f"  → Analyzing {file_path} ({idx}/{total})")

        def _compare_size_between_content_and_analysis(
            content: str, formatted_analysis: str
        ) -> str:
            content_size = len(content)
            analysis_size = len(formatted_analysis)

            if content_size <= analysis_size:
                return "content"
            else:
                return "analysis"

        # Create a new LLM instance for each thread to avoid context accumulation
        llm = CONFIG.get_llm()

        file_path_resolved = resolve_path(file_path)
        content = read_file(file_path_resolved)

        if not content:
            return file_path, {
                "analysis": "File could not be read or is empty.",
                "summary": "File could not be read or is empty.",
            }

        # Perform tree-sitter analysis
        analysis = analyze_file_with_tree_sitter(file_path_resolved)
        formatted_analysis = format_tree_sitter_analysis_results_to_prompt(analysis)

        # choose the one with smaller size to use in the summary prompt
        # match_choice should be "content" or "analysis"
        match_choice = _compare_size_between_content_and_analysis(
            content, formatted_analysis
        )

        # Use LLM to summarize (with fresh context each time)
        # match the summary prompt with the analysis result
        match match_choice:
            case "content":
                summary_prompt = self._get_code_analysis_prompt_with_content(
                    file_path, content
                )
            case "analysis":
                summary_prompt = self._get_code_analysis_prompt_with_analysis(
                    file_path, formatted_analysis
                )
        summary_response = llm.invoke([summary_prompt])

        return file_path, {
            "analysis": formatted_analysis,
            "summary": summary_response.content,
        }

    def _generate_single_file_doc(
        self, file_path: str, analysis: dict, summary: str, idx: int, total: int
    ) -> tuple[str, str]:
        """
        Generate documentation for a single file.
        This method will be called concurrently for each file.

        Returns:
            Tuple of (file_path, documentation_content)
        """
        print(f"  → Generating documentation for {file_path} ({idx}/{total})")

        # Create a new LLM instance for each thread to avoid context accumulation
        llm = CONFIG.get_llm()

        system_prompt = self._get_system_prompt_for_doc()
        human_prompt = self._get_code_doc_prompt(file_path, analysis, summary)

        try:
            response = llm.invoke([system_prompt, human_prompt])
            return file_path, response.content
        except Exception as e:
            print(f"  ✗ Error generating documentation for {file_path}: {str(e)}")
            return file_path, f"# Error\n\nFailed to generate documentation: {str(e)}"

    def _write_single_doc_file(
        self,
        file_path: str,
        doc_content: str,
        wiki_path: str,
        repo_path: str,
        idx: int,
        total: int,
    ) -> tuple[str, bool]:
        """
        Write documentation for a single file.
        This method will be called concurrently for each file.

        Preserves the directory structure of the original file path relative to repo_path.
        E.g., '.repos/facebook_zstd/lib/compress/zstd_lazy.c' with repo_path='.repos/facebook_zstd'
              -> '<wiki_path>/lib/compress/zstd_lazy.c_doc.md'

        Returns:
            Tuple of (file_path, success_status)
        """
        try:
            # Remove the repo_path prefix from file_path to get relative path
            rel_file_path = file_path
            if file_path.startswith(repo_path):
                rel_file_path = file_path[len(repo_path) :].lstrip(os.sep)

            # Preserve the directory structure from the relative file path
            dir_path = os.path.dirname(rel_file_path)
            file_name = os.path.basename(rel_file_path)

            # Create destination directory maintaining the original structure
            dest_dir = os.path.join(wiki_path, dir_path) if dir_path else wiki_path
            os.makedirs(dest_dir, exist_ok=True)

            # Create the documentation file name with _doc.md suffix
            doc_file_name = file_name + "_doc.md"
            output_path = os.path.join(dest_dir, doc_file_name)

            rel_output_path = (
                os.path.join(dir_path, doc_file_name) if dir_path else doc_file_name
            )
            print(f"  → Writing documentation to {rel_output_path} ({idx}/{total})")
            write_file(output_path, doc_content)

            return file_path, True
        except Exception as e:
            print(f"  ✗ Error writing documentation for {file_path}: {str(e)}")
            return file_path, False

    def _single_thread_process(
        self, file_path: str, idx: int, total: int, wiki_path: str, repo_path: str
    ) -> tuple[str, bool]:
        """
        Process a single file through the complete workflow:
        1. Analyze the file
        2. Generate documentation
        3. Write documentation to file

        Args:
            file_path (str): Path to the code file
            idx (int): Index of the file in the list
            total (int): Total number of files
            wiki_path (str): Path to write documentation files
            repo_path (str): Base repository path (for relative path calculation)

        Returns:
            tuple[str, bool]: file_path and success status
        """
        try:
            # Step 1: Analyze the file
            file_path, analysis_result = self._analyze_single_file(
                file_path, idx, total
            )
            analysis = analysis_result.get("analysis", "")
            summary = analysis_result.get("summary", "")

            # Step 2: Generate documentation for the file
            file_path, doc_content = self._generate_single_file_doc(
                file_path, analysis, summary, idx, total
            )

            # Step 3: Write documentation to file
            file_path, write_success = self._write_single_doc_file(
                file_path, doc_content, wiki_path, repo_path, idx, total
            )

            if write_success:
                return file_path, True
            else:
                return file_path, False

        except Exception as e:
            print(f"  ✗ Error in worker process for {file_path}: {str(e)}")
            return file_path, False

    def code_analysis_node(self, state: dict):
        # log_state(state)
        # this node need to analyze the filtered code files
        print("→ Processing code_analysis_node (concurrent)...")
        code_files = state.get("code_analysis", {}).get("code_files", [])

        if not code_files:
            print("⚠ No code files to analyze")
            return

        wiki_path = state.get("basic_info_for_code", {}).get(
            "wiki_path", "./.wikis/default"
        )
        repo_path = state.get("basic_info_for_code", {}).get("repo_path", "./.repos")

        # Determine optimal number of workers
        # Adjust max_workers based on your API rate limits
        max_workers = min(
            state.get("max_workers", 10), len(code_files)
        )  # Max 10 concurrent requests

        print(f"  → Using {max_workers} concurrent workers for {len(code_files)} files")

        # Use ThreadPoolExecutor for concurrent execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks using _single_thread_process
            future_to_file = {
                executor.submit(
                    self._single_thread_process,
                    file_path,
                    idx,
                    len(code_files),
                    wiki_path,
                    repo_path,
                ): file_path
                for idx, file_path in enumerate(code_files, 1)
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result_file_path, result = future.result()
                    if result:
                        print(f"  ✓ Documentation written for {file_path}")
                    else:
                        print(f"  ⚠ Processing failed for {file_path}")
                    completed += 1
                    print(
                        f"  ✓ Completed {completed}/{len(code_files)}: {result_file_path}"
                    )
                except Exception as e:
                    print(f"  ✗ Error processing {file_path}: {str(e)}")

        print(f"✓ Code analysis completed for {len(code_files)} files")

    def build(self):
        analysis_builder = StateGraph(CodeAnalysisSubGraphState)
        analysis_builder.add_node("code_filter_node", self.code_filter_node)
        analysis_builder.add_node("code_analysis_node", self.code_analysis_node)
        analysis_builder.add_node(
            "code_info_update_log_node", self.code_info_update_log_node
        )
        analysis_builder.add_edge(START, "code_filter_node")
        analysis_builder.add_edge("code_filter_node", "code_info_update_log_node")
        analysis_builder.add_edge("code_filter_node", "code_analysis_node")
        return analysis_builder.compile(checkpointer=True)

    def get_graph(self):
        return self.graph
