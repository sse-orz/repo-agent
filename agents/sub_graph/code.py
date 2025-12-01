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
        # this func is to generate a system prompt for code filtering
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
        repo_info: dict, repo_structure: list[str], max_num_files: int
    ) -> HumanMessage:
        # this func is to generate a prompt for code filtering
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
        # this func is to generate a prompt for code analysis with content
        return HumanMessage(
            content=f"""Please provide a concise summary (max 100 words) of the following code file content for file '{file_path}':

{content}

Focus on: key functions/classes, main purpose, and important logic."""
        )

    @staticmethod
    def _get_code_analysis_prompt_with_analysis(
        file_path: str, analysis_result
    ) -> HumanMessage:
        # this func is to generate a prompt for code analysis with analysis result
        return HumanMessage(
            content=f"""Please provide a concise summary (max 100 words) of the following code analysis content for file '{file_path}':

{analysis_result}

Focus on: key functions/classes, main purpose, and important logic."""
        )

    @staticmethod
    def _get_system_prompt_for_doc():
        # this func is to generate a system prompt for code documentation generation
        return SystemMessage(
            content="""You are an expert technical documentation writer specializing in generating comprehensive code documentation. 
Your role is to analyze provided code analysis results and create clear, well-structured documentation that helps developers understand the codebase structure, functionality, and architecture.

Guidelines:
- Use Markdown formatting for better readability
- Be concise yet informative
- Highlight key components and their relationships
- Organize information logically with clear sections
- Include code snippets and examples when helpful
- Document complex algorithms and design patterns

CRITICAL OUTPUT RULES:
- Output ONLY the markdown document content, starting directly with the title (e.g., # Title)
- Do NOT include any introductory phrases, explanations, or closing remarks
- Do NOT add text like "Of course", "Here is", "I'll generate", or any conversational preambles
- The response must begin with the markdown title and end with the last content line
- No meta-commentary, generation notes, or additional context outside the markdown content"""
        )

    @staticmethod
    def _get_code_doc_prompt(
        file_path: str, analysis: dict, summary: str
    ) -> HumanMessage:
        # this func is to generate a prompt for code documentation

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
Please structure the documentation with clear sections and make it suitable for both new contributors and project maintainers. Include Mermaid diagrams to visualize code structure, relationships, and logic flow.

CRITICAL OUTPUT REQUIREMENTS:
- Output ONLY the markdown document content starting with the title (e.g., # Title)
- Do NOT include any introductory text, explanations, or closing remarks before or after the markdown content
- Do NOT include phrases like "Of course", "Here is", "I'll generate", or any similar conversational text
- The output should start directly with the markdown title and end with the last content line
- No meta-commentary or generation information should be included"""
        )

    def code_filter_node(self, state: dict):
        # this node is to filter code files for analysis
        log_state(state)

        def get_max_num_files(mode: str) -> int:
            if mode == "smart":
                return 100
            else:
                return 20

        print("→ [code_filter_node] Processing code_filter_node...")
        basic_info_for_code = state.get("basic_info_for_code", {})
        code_files_updated = basic_info_for_code.get("code_files_updated", False)
        match code_files_updated:
            case True:
                code_files = basic_info_for_code.get("code_files", [])
                print("   → [code_filter_node] using updated code files for analysis")
            case False:
                print("   → [code_filter_node] filtering code files for analysis...")
                mode = basic_info_for_code.get("mode", "fast")
                max_num_files = get_max_num_files(mode)
                print(
                    f"   → [code_filter_node] using mode '{mode}' with max {max_num_files} files for analysis"
                )
                repo_info = basic_info_for_code.get("repo_info", {})
                repo_structure = basic_info_for_code.get("repo_structure", [])
                code_update_log_path = basic_info_for_code.get(
                    "code_update_log_path", ""
                )
                existing_code_files = (
                    read_json(code_update_log_path).get("code_files", [])
                    if read_json(code_update_log_path)
                    else []
                )
                repo_structure = [
                    file_path
                    for file_path in repo_structure
                    if file_path not in existing_code_files
                ]
                system_prompt = self._get_system_prompt()
                human_prompt = self._get_code_filter_prompt(
                    repo_info, repo_structure, max_num_files
                )
                llm = CONFIG.get_llm()
                response = llm.invoke([system_prompt, human_prompt])
                code_files = [
                    line.strip()
                    for line in response.content.splitlines()
                    if line.strip()
                ]

                if len(code_files) > max_num_files:
                    print(
                        f"   → [code_filter_node] LLM returned {len(code_files)} files, limiting to {max_num_files}"
                    )
                    code_files = code_files[:max_num_files]

                print(
                    f"   → [code_filter_node] {len(code_files)} code files generated for analysis (max: {max_num_files})"
                )

        print("✓ [code_filter_node] code_filter_node processed successfully")
        return {
            "code_analysis": {
                "code_files": code_files,
            }
        }

    def code_info_update_log_node(self, state: dict):
        # this node is to process the repository information update log
        log_state(state)
        print("→ [code_info_update_log_node] Processing code_info_update_log_node...")
        basic_info_for_code = state.get("basic_info_for_code", {})
        code_files = state.get("code_analysis", {}).get("code_files", [])
        code_update_log_path = basic_info_for_code.get("code_update_log_path", "")

        existing_data = read_json(code_update_log_path)
        existing_code_files = (
            existing_data.get("code_files", []) if existing_data else []
        )
        merged_code_files = list(set(existing_code_files) | set(code_files))
        update_log_info = {
            "log_date": str(basic_info_for_code.get("date", "N/A")),
            "code_files": merged_code_files,
        }

        write_file(
            code_update_log_path,
            json.dumps(update_log_info, indent=2, ensure_ascii=False),
        )
        print(
            f"✓ [code_info_update_log_node] code_info_update_log_node processed successfully"
        )

    def _analyze_single_file(
        self, file_path: str, idx: int, total: int
    ) -> tuple[str, dict]:
        # this func is to analyze a single code file
        print(f"   → [code_analysis_node] analyzing file {file_path} ({idx}/{total})")

        def _compare_size_between_content_and_analysis(
            content: str, formatted_analysis: str
        ) -> str:
            content_size = len(content)
            analysis_size = len(formatted_analysis)

            if content_size <= analysis_size:
                return "content"
            else:
                return "analysis"

        llm = CONFIG.get_llm()

        file_path_resolved = resolve_path(file_path)
        content = read_file(file_path_resolved)

        if not content:
            return file_path, {
                "analysis": "File could not be read or is empty.",
                "summary": "File could not be read or is empty.",
            }

        analysis = analyze_file_with_tree_sitter(file_path_resolved)
        formatted_analysis = format_tree_sitter_analysis_results_to_prompt(analysis)

        match_choice = _compare_size_between_content_and_analysis(
            content, formatted_analysis
        )

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
        # this func is to generate documentation for a single code file
        print(
            f"   → [code_analysis_node] generating documentation for {file_path} ({idx}/{total})"
        )
        llm = CONFIG.get_llm()

        system_prompt = self._get_system_prompt_for_doc()
        human_prompt = self._get_code_doc_prompt(file_path, analysis, summary)

        try:
            response = llm.invoke([system_prompt, human_prompt])
            return file_path, response.content
        except Exception as e:
            print(
                f"   → [code_analysis_node] Error generating documentation for {file_path}: {str(e)}"
            )
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
        # this func is to write the generated documentation to a file
        try:
            rel_file_path = file_path
            if file_path.startswith(repo_path):
                rel_file_path = file_path[len(repo_path) :].lstrip(os.sep)
            dir_path = os.path.dirname(rel_file_path)
            file_name = os.path.basename(rel_file_path)
            dest_dir = os.path.join(wiki_path, dir_path) if dir_path else wiki_path
            os.makedirs(dest_dir, exist_ok=True)
            doc_file_name = file_name + "_doc.md"
            output_path = os.path.join(dest_dir, doc_file_name)

            rel_output_path = (
                os.path.join(dir_path, doc_file_name) if dir_path else doc_file_name
            )
            print(
                f"   → [code_analysis_node] writing documentation to {rel_output_path} ({idx}/{total})"
            )
            write_file(output_path, doc_content)

            return file_path, True
        except Exception as e:
            print(
                f"   → [code_analysis_node] Error writing documentation for {file_path}: {str(e)}"
            )
            return file_path, False

    def _single_thread_process(
        self, file_path: str, idx: int, total: int, wiki_path: str, repo_path: str
    ) -> tuple[str, bool]:
        # this func is to process a single code file in a single thread
        try:
            file_path, analysis_result = self._analyze_single_file(
                file_path, idx, total
            )
            analysis = analysis_result.get("analysis", "")
            summary = analysis_result.get("summary", "")

            file_path, doc_content = self._generate_single_file_doc(
                file_path, analysis, summary, idx, total
            )

            file_path, write_success = self._write_single_doc_file(
                file_path, doc_content, wiki_path, repo_path, idx, total
            )

            if write_success:
                return file_path, True
            else:
                return file_path, False

        except Exception as e:
            print(
                f"   → [code_analysis_node] Error in worker process for {file_path}: {str(e)}"
            )
            return file_path, False

    def code_analysis_node(self, state: dict):
        # this node is to analyze code files and generate documentation
        log_state(state)
        print("→ [code_analysis_node] Processing code_analysis_node (concurrent)...")
        code_files = state.get("code_analysis", {}).get("code_files", [])

        if not code_files:
            print("⚠ [code_analysis_node] No code files to analyze")
            return

        basic_info_for_code = state.get("basic_info_for_code", {})
        wiki_path = basic_info_for_code.get("wiki_path", "./.wikis/default")
        repo_path = basic_info_for_code.get("repo_path", "./.repos")

        max_workers = min(state.get("max_workers", 10), len(code_files))

        print(
            f"→ [code_analysis_node] Analyzing {len(code_files)} code files with {max_workers} workers"
        )

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
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

            completed = 0
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result_file_path, result = future.result()
                    completed += 1
                    print(
                        f"   → [code_analysis_node] Completed {completed}/{len(code_files)}: {result_file_path}"
                    )
                except Exception as e:
                    print(
                        f"   → [code_analysis_node] Error processing {file_path}: {str(e)}"
                    )

        print(
            f"✓ [code_analysis_node] Code analysis completed for {len(code_files)} files"
        )

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
