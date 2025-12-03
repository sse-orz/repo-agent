from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os

from utils.file import write_file, read_file, resolve_path, read_json
from utils.code_analyzer import (
    analyze_file_with_tree_sitter,
    format_tree_sitter_analysis_results_to_prompt_without_description,
)
from .utils import log_state, call_llm, count_tokens, get_llm_max_tokens
from .prompt import CodePrompt
from config import CONFIG


class CodeAnalysisSubGraphState(TypedDict):
    # inputs
    basic_info_for_code: dict
    # outputs
    code_analysis: dict


class CodeAnalysisSubGraphBuilder:

    def __init__(self):
        self.graph = self.build()

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
        content = call_llm(
            [
                CodePrompt._get_system_prompt_for_filter(),  # system prompt
                CodePrompt._get_human_prompt_for_filter(
                    repo_info, chunk_repo_structure, max_num_files
                ),  # human prompt
            ]
        )
        code_files = [line.strip() for line in content.splitlines() if line.strip()]
        return code_files

    def _filter_code_files(
        self,
        repo_info: dict,
        repo_structure: list[str],
        max_num_files: int,
        times: int = 2,
    ) -> list[str]:
        """Filter code files using LLM, with parallel processing for large file lists.

        Args:
            repo_info: Repository information dictionary
            repo_structure: List of all file paths to filter
            max_num_files: Maximum number of files to return
            times: Number of times to call LLM
        Returns:
            List of filtered file paths
        """
        files_num = len(repo_structure)
        code_files: list[str] = []

        repo_structure_tokens = count_tokens(str(repo_structure))
        if repo_structure_tokens > get_llm_max_tokens(compress_ratio=0.90):
            # if the tokens of the repo structure are more than 90% of the max tokens, call llm times to get max_num_files files
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
            content = call_llm(
                [
                    CodePrompt._get_system_prompt_for_filter(),  # system prompt
                    CodePrompt._get_human_prompt_for_filter(
                        repo_info, repo_structure, max_num_files
                    ),  # human prompt
                ]
            )
            code_files = [line.strip() for line in content.splitlines() if line.strip()]

        return code_files

    def _preprocess_code_files(
        self, code_update_log_path: str, repo_path: str, repo_structure: list[str]
    ) -> list[str]:
        """
        Preprocess the code files to:
        1. Get the existing code files from the code update log
        2. Remove prefix path from existing code files
        3. Remove existing code files from repo structure
        4. Return the repo structure
        Args:
            code_update_log_path: Path to the code update log
            repo_path: Path to the repository
            repo_structure: List of all file paths in the repository
        Returns:
            List of repo structure
        """
        data = read_json(code_update_log_path)
        existing_code_files = data.get("code_files", []) if data else []
        # remove prefix path from existing code files
        existing_code_files = [
            file_path.replace(repo_path, "") for file_path in existing_code_files
        ]
        repo_structure = [
            file_path
            for file_path in repo_structure
            if file_path not in existing_code_files
        ]
        return repo_structure

    def _postprocess_code_files(
        self,
        repo_structure: list[str],
        code_files: list[str],
        repo_path: str,
        max_num_files: int,
    ) -> list[str]:
        """
        Postprocess the code files to:
        1. Deduplicate code files
        2. Add prefix path to code files
        3. Limit the number of code files to max_num_files
        4. Return the code files
        Args:
            repo_structure: List of all file paths in the repository
            code_files: List of code files to postprocess
            repo_path: Path to the repository
            max_num_files: Maximum number of code files to return
        Returns:
            List of postprocessed code files
        """
        repo_files_set = set(repo_structure)
        code_files = [
            os.path.join(repo_path, f) for f in code_files if f in repo_files_set
        ]
        seen: set[str] = set()
        deduped_code_files: list[str] = []
        for f in code_files:
            if f not in seen:
                seen.add(f)
                deduped_code_files.append(f)
        code_files = deduped_code_files
        if len(code_files) > max_num_files:
            code_files = code_files[:max_num_files]
        return code_files

    @staticmethod
    def _get_max_num_files(
        mode: str, repo_structure_len: int, ratios: dict[str, float]
    ) -> int:
        if mode == "smart":
            return int(repo_structure_len * ratios.get("smart", 0.75))
        else:
            return int(repo_structure_len * ratios.get("fast", 0.25))

    def code_filter_node(self, state: dict):
        # this node is to filter code files for analysis
        log_state(state)

        print("→ [code_filter_node] Processing code_filter_node...")
        basic_info_for_code = state.get("basic_info_for_code", {})
        code_files_updated = basic_info_for_code.get("code_files_updated", False)
        match code_files_updated:
            case True:
                code_files = basic_info_for_code.get("code_files", [])
                print("   → [code_filter_node] using updated code files for analysis")
            case False:
                print("   → [code_filter_node] filtering code files for analysis...")
                repo_info = basic_info_for_code.get("repo_info", {})
                repo_path = basic_info_for_code.get("repo_path", "")
                repo_structure = basic_info_for_code.get("repo_structure", [])
                code_update_log_path = basic_info_for_code.get(
                    "code_update_log_path", ""
                )

                # Preprocess repo structure: remove existing code files from repo structure
                repo_structure = self._preprocess_code_files(
                    code_update_log_path, repo_path, repo_structure
                )

                mode = basic_info_for_code.get("mode", "fast")
                ratios = basic_info_for_code.get(
                    "ratios",
                    {
                        "fast": 0.25,
                        "smart": 0.75,
                    },
                )
                max_num_files = self._get_max_num_files(
                    mode, len(repo_structure), ratios
                )
                print(
                    f"   → [code_filter_node] using mode '{mode}' with max {max_num_files} files for analysis"
                )

                # Filter code files using LLM
                code_files = self._filter_code_files(
                    repo_info, repo_structure, max_num_files
                )

                # Postprocess code files: add prefix path to code files and deduplicate code files
                code_files = self._postprocess_code_files(
                    repo_structure, code_files, repo_path, max_num_files
                )

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
        """
        Analyze a single code file and return results.
        This method will be called concurrently for each file.

        Args:
            file_path: Path to the code file
            idx: Index of the file in the list
            total: Total number of files
        Returns:
            tuple[str, dict]: file_path and its analysis results.
        """
        print(f"   → [code_analysis_node] analyzing file {file_path} ({idx}/{total})")

        file_path_resolved = resolve_path(file_path)
        content = read_file(file_path_resolved)
        analysis = analyze_file_with_tree_sitter(file_path_resolved)

        # count the tokens in the content
        content_tokens = count_tokens(content)
        # summarize the content of the file if the tokens are more than 90% of the max tokens
        if content_tokens > get_llm_max_tokens(compress_ratio=0.90):
            summary = call_llm(
                [
                    CodePrompt._get_system_prompt_for_analysis(),  # system prompt
                    CodePrompt._get_human_prompt_for_analysis_with_content(
                        file_path, content
                    ),  # human prompt
                ]
            )
        else:
            summary = content

        # format the analysis results
        formatted_analysis = (
            format_tree_sitter_analysis_results_to_prompt_without_description(analysis)
        )

        return file_path, {
            "analysis": formatted_analysis,
            "summary": summary,
        }

    def _generate_single_file_doc(
        self, file_path: str, analysis: dict, summary: str, idx: int, total: int
    ) -> tuple[str, str]:
        """
        Generate documentation for a single code file.
        This method will be called concurrently for each file.

        Args:
            file_path: Path to the code file
            analysis: Analysis results
            summary: Summary of the file
            idx: Index of the file in the list
            total: Total number of files
        Returns:
            tuple[str, str]: file_path and its documentation content.
        """
        print(
            f"   → [code_analysis_node] generating documentation for {file_path} ({idx}/{total})"
        )
        try:
            content = call_llm(
                [
                    CodePrompt._get_system_prompt_for_doc(),  # system prompt
                    CodePrompt._get_human_prompt_for_doc(
                        file_path, analysis, summary
                    ),  # human prompt
                ]
            )
            return file_path, content
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
        """
        Write the generated documentation to a file.
        This method will be called concurrently for each file.

        Args:
            file_path: Path to the code file
            doc_content: Documentation content
            wiki_path: Path to the wiki directory
            repo_path: Path to the repository
            idx: Index of the file in the list
            total: Total number of files
        Returns:
            tuple[str, bool]: file_path and success status.
        """
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
        """
        Process a single code file in a single thread.
        This method will be called concurrently for each file.

        Args:
            file_path: Path to the code file
            idx: Index of the file in the list
            total: Total number of files
            wiki_path: Path to the wiki directory
            repo_path: Path to the repository
        Returns:
            tuple[str, bool]: file_path and success status.
        """
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

        max_workers = min(basic_info_for_code.get("max_workers", 10), len(code_files))

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
