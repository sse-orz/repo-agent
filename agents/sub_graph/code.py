from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START
from langchain_core.messages import HumanMessage, SystemMessage
from concurrent.futures import ThreadPoolExecutor, as_completed
from textwrap import dedent
import json
import os
import time

from utils.file import write_file, read_file, resolve_path, read_json
from utils.code_analyzer import (
    analyze_file_with_tree_sitter,
    format_tree_sitter_analysis_results_to_prompt_without_description,
)
from .utils import log_state
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
            content=dedent(
                """
                You are an expert code analyst specializing in identifying relevant code files for static analysis.
                Your role is to analyze the provided repository information and file structure
                to select files that are most pertinent for in-depth examination.

                Guidelines:
                - Focus on source code files with extensions like .py, .js, .go, .cpp, .java, .rs, .c, .h, .hpp
                - Exclude documentation, configuration, and non-code files
                - Use the repository context to prioritize files central to core functionality
                """
            ).strip(),
        )

    @staticmethod
    def _get_code_filter_prompt(
        repo_info: dict, repo_structure: list[str], max_num_files: int
    ) -> HumanMessage:
        # this func is to generate a prompt for code filtering
        return HumanMessage(
            content=dedent(
                f"""
                Based on the repository information and file structure below, identify EXACTLY {max_num_files}
                code files (or fewer if fewer exist) that are most relevant for static analysis.

                Critical constraints:
                - Return at most {max_num_files} file paths
                - Focus on source code files (.py, .js, .go, .cpp, .java, .rs, .c, .h, .hpp, .ts)
                - Exclude documentation, config, test, build, and dependency files
                - Prioritize files central to the project's core functionality
                - Sort by relevance (most important first)
                - Select paths ONLY from the File Structure list below and copy them verbatim
                  (no changes to extensions, directory names, or segments)

                Repository information:
                {repo_info}

                File structure:
                {repo_structure}

                Output format:
                - Output ONLY file paths, one per line
                - No descriptions, headers, numbering, or extra text
                """
            ).strip(),
        )

    @staticmethod
    def _get_code_analysis_prompt_with_content(
        file_path: str, content: str
    ) -> HumanMessage:
        # this func is to generate a prompt for code analysis with content
        return HumanMessage(
            content=dedent(
                f"""
                Provide a concise summary of the following code file content for '{file_path}'.

                {content}

                Focus on: key functions/classes, main purpose, and important logic.
                """
            ).strip(),
        )

    @staticmethod
    def _get_code_analysis_prompt_with_analysis(
        file_path: str, analysis_result
    ) -> HumanMessage:
        # this func is to generate a prompt for code analysis with analysis result
        return HumanMessage(
            content=dedent(
                f"""
                Provide a concise summary of the following tree sitter analysis result for '{file_path}'.

                {analysis_result}

                Focus on: key functions/classes, main purpose, and important logic.
                """
            ).strip(),
        )

    @staticmethod
    def _get_code_analysis_prompt_with_summary(
        file_path: str, summary: str, analysis_result
    ) -> HumanMessage:
        # this func is to generate a prompt for code analysis with summary and analysis result
        return HumanMessage(
            content=dedent(
                f"""
                Provide a concise summary of the following file content summary and tree sitter analysis result for '{file_path}'.

                - File Content Summary:
                {summary}

                - Tree Sitter Analysis Result:
                {analysis_result}

                Focus on: the overall purpose of the file, key functions/classes, and important logic, etc.
                """
            ).strip(),
        )

    @staticmethod
    def _get_system_prompt_for_doc():
        # this func is to generate a system prompt for code documentation generation
        return SystemMessage(
            content=dedent(
                """
                You are an expert technical documentation writer for source code.
                Your job is to turn code and analysis results into clear, structured **Markdown documentation**
                that explains the codebase structure, functionality, and architecture.

                Guidelines:
                - Use Markdown headings, lists, and tables when helpful
                - Be concise yet informative
                - Highlight key components and their relationships
                - Organize content into logical sections with clear titles
                - Include code snippets and examples when they add clarity
                - Document important algorithms and design patterns
                - When diagrams are requested or helpful, use valid Mermaid code blocks (```mermaid ... ```).

                Critical output rules:
                - Output ONLY the final markdown document, starting directly with the title line (e.g., "# Title")
                - Do NOT include introductory phrases, explanations, or closing remarks
                - Do NOT add text like "Of course", "Here is", "I'll generate", etc.
                - The response must begin with the markdown title and end with the last content line
                - No meta-commentary, generation notes, or additional context outside the markdown content
                """
            ).strip(),
        )

    @staticmethod
    def _get_code_doc_prompt(
        file_path: str, analysis: dict, summary: str
    ) -> HumanMessage:
        # this func is to generate a prompt for code documentation
        return HumanMessage(
            content=dedent(
                f"""
                Generate documentation for the code file '{file_path}'.

                Use the data below to write a markdown document that covers:
                - The file's overall purpose and main responsibilities
                - Key functions/classes and what they do
                - Important algorithms or design decisions
                - Dependencies and relationships with other files or modules
                - Any helpful Mermaid diagrams to illustrate structure or flow (when appropriate)

                Data:
                - Tree Sitter Analysis Result:
                {analysis}

                - File Content Summary:
                {summary}
                """
            ).strip(),
        )

    def _filter_single_chunk(
        self, repo_info: dict, chunk_repo_structure: list[str], max_num_files: int
    ) -> list[str]:
        """Filter a single chunk of files using LLM.

        Args:
            repo_info: Repository information dictionary
            chunk_repo_structure: List of file paths in this chunk
            max_num_files: Maximum number of files to return

        Returns:
            List of filtered file paths
        """
        system_prompt = self._get_system_prompt()
        human_prompt = self._get_code_filter_prompt(
            repo_info, chunk_repo_structure, max_num_files
        )
        llm = CONFIG.get_llm()
        response = llm.invoke([system_prompt, human_prompt])
        result_files = [
            line.strip() for line in response.content.splitlines() if line.strip()
        ]
        if len(result_files) > max_num_files:
            result_files = result_files[:max_num_files]
        return result_files

    def _filter_code_files(
        self,
        repo_info: dict,
        repo_structure: list[str],
        max_num_files: int,
        times: int = 2,
        x_num: int = 10,
    ) -> list[str]:
        """Filter code files using LLM, with parallel processing for large file lists.

        Args:
            repo_info: Repository information dictionary
            repo_structure: List of all file paths to filter
            max_num_files: Maximum number of files to return
            times: Number of times to call LLM
            x_num: Number of times to multiply max_num_files to determine if to use parallel processing
        Returns:
            List of filtered file paths
        """
        files_num = len(repo_structure)
        code_files: list[str] = []

        # if num of files is 10x more than max_num_files, call llm times to get max_num_files files
        if files_num > x_num * max_num_files:
            # use thread pool to filter code files in parallel
            single_max_num_files = int(max_num_files / times)

            with ThreadPoolExecutor(max_workers=times) as executor:
                futures = []
                for i in range(times):
                    start_idx = i * files_num // times
                    end_idx = (i + 1) * files_num // times
                    chunk = repo_structure[start_idx:end_idx]
                    if not chunk:
                        continue
                    futures.append(
                        executor.submit(
                            self._filter_single_chunk,
                            repo_info,
                            chunk,
                            single_max_num_files,
                        )
                    )

                for future in as_completed(futures):
                    try:
                        chunk_files = future.result()
                        code_files.extend(chunk_files)
                    except Exception as e:
                        print(
                            f"   → [code_filter_node] Error filtering chunk: {str(e)}"
                        )
        else:
            system_prompt = self._get_system_prompt()
            human_prompt = self._get_code_filter_prompt(
                repo_info, repo_structure, max_num_files
            )
            llm = CONFIG.get_llm()
            response = llm.invoke([system_prompt, human_prompt])
            code_files = [
                line.strip() for line in response.content.splitlines() if line.strip()
            ]

        return code_files

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
                repo_path = basic_info_for_code.get("repo_path", "")
                repo_structure = basic_info_for_code.get("repo_structure", [])
                code_update_log_path = basic_info_for_code.get(
                    "code_update_log_path", ""
                )
                existing_code_files = (
                    read_json(code_update_log_path).get("code_files", [])
                    if read_json(code_update_log_path)
                    else []
                )
                # remove prefix path from existing code files
                existing_code_files = [
                    file_path.replace(repo_path, "")
                    for file_path in existing_code_files
                ]
                repo_structure = [
                    file_path
                    for file_path in repo_structure
                    if file_path not in existing_code_files
                ]

                # Filter code files using LLM
                code_files = self._filter_code_files(
                    repo_info, repo_structure, max_num_files
                )

                # filter code files that are not in repo_structure
                # add prefix path to code files
                repo_files_set = set(repo_structure)
                code_files = [
                    os.path.join(repo_path, f)
                    for f in code_files
                    if f in repo_files_set
                ]

                # deduplicate code files
                seen: set[str] = set()
                deduped_code_files: list[str] = []
                for f in code_files:
                    if f not in seen:
                        seen.add(f)
                        deduped_code_files.append(f)
                code_files = deduped_code_files
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

        llm = CONFIG.get_llm()

        file_path_resolved = resolve_path(file_path)
        content = read_file(file_path_resolved)

        if not content:
            return file_path, {
                "analysis": "File could not be read or is empty.",
                "summary": "File could not be read or is empty.",
            }

        # summarize the content of the file
        summary_prompt = self._get_code_analysis_prompt_with_content(file_path, content)
        summary_response = llm.invoke([summary_prompt])
        summary = summary_response.content

        analysis = analyze_file_with_tree_sitter(file_path_resolved)
        formatted_analysis = (
            format_tree_sitter_analysis_results_to_prompt_without_description(analysis)
        )

        # add summary and analysis to prompt to generate final summary
        final_summary_prompt = self._get_code_analysis_prompt_with_summary(
            file_path, summary, formatted_analysis
        )
        final_summary_response = llm.invoke([final_summary_prompt])
        final_summary = final_summary_response.content

        return file_path, {
            "analysis": formatted_analysis,
            "summary": final_summary,
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
