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

from utils.repo import (
    get_repo_info,
    get_repo_commit_info,
    get_commits,
    get_pr,
    get_pr_files,
    get_release_note,
)
from utils.file import get_repo_structure, write_file, read_file, resolve_path
from utils.code_analyzer import (
    analyze_file_with_tree_sitter,
    format_tree_sitter_analysis_results,
    format_tree_sitter_analysis_results_to_prompt,
)
from config import CONFIG


def draw_graph(graph):
    img = graph.get_graph().draw_mermaid_png(
        draw_method=MermaidDrawMethod.API,
    )
    dir_name = "./.graphs"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    graph_file_name = os.path.join(
        dir_name,
        f"test-sub-graph-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png",
    )
    with open(graph_file_name, "wb") as f:
        f.write(img)


def log_state(state: dict):
    for key, value in state.items():
        print(f"{key}: {value}")


class RepoInfoSubGraphState(TypedDict):
    # branch-specific inputs (copied by parent into these namespaced keys)
    basic_info_for_repo: dict
    # outputs
    commit_info: dict
    pr_info: dict
    release_note_info: dict
    overview_info: str
    update_log_info: str


class RepoInfoSubGraphBuilder:

    def __init__(self):
        self.graph = self.build()

    @staticmethod
    def _get_system_prompt():
        """Generate a comprehensive system prompt for documentation generation."""
        return SystemMessage(
            content="""You are an expert technical documentation writer specializing in generating comprehensive repository documentation. 
Your role is to analyze provided repository data and create clear, well-structured documentation that helps developers understand the project's history, development process, and releases.

Guidelines:
- Use Markdown formatting for better readability
- Be concise yet informative
- Highlight key changes and their impact
- Organize information logically with clear sections
- Include relevant metrics and statistics when available"""
        )

    @staticmethod
    def _get_overview_doc_prompt(repo_info: dict, doc_contents: str) -> HumanMessage:
        """Generate a prompt for overall repository documentation."""
        repo_name = repo_info.get("repo", "")
        owner = repo_info.get("owner", "")

        return HumanMessage(
            content=f"""Generate overall repository documentation for the repository '{owner}/{repo_name}'.
Based on the following repository information and documentation source files, create a detailed analysis that includes:
1. Overview of the repository structure and key components
2. Summary of existing documentation and its coverage
3. Identification of documentation gaps and areas for improvement
Repository Information:
{repo_info}
Documentation Source Files Content:
{doc_contents}
Please structure the documentation with clear sections and make it suitable for both new contributors and project maintainers."""
        )

    @staticmethod
    def _generate_commit_doc_prompt(commit_info: dict, repo_info: dict) -> HumanMessage:
        """Generate a prompt for commit documentation."""
        repo_name = repo_info.get("repo", "")
        owner = repo_info.get("owner", "")

        return HumanMessage(
            content=f"""Generate comprehensive commit history documentation for the repository '{owner}/{repo_name}'.

Based on the following commit information, create a detailed analysis that includes:
1. Summary of key commits and their purposes
2. Patterns in development activity (frequency, authors, etc.)
3. Major features or fixes identified in recent commits
4. Development trends and insights

Commit Information:
{commit_info}

Please structure the documentation with clear sections and make it suitable for both new contributors and project maintainers."""
        )

    @staticmethod
    def _generate_pr_doc_prompt(pr_info: dict, repo_info: dict) -> HumanMessage:
        """Generate a prompt for PR documentation."""
        repo_name = repo_info.get("repo", "")
        owner = repo_info.get("owner", "")

        return HumanMessage(
            content=f"""Generate comprehensive pull request (PR) analysis documentation for the repository '{owner}/{repo_name}'.

Based on the following PR information, create a detailed analysis that includes:
1. Overview of active and recent pull requests
2. Common themes or areas of focus in PRs
3. Review and merge patterns
4. Contributor activity and collaboration insights
5. Any pending or long-standing PRs that need attention

PR Information:
{pr_info}

Please structure the documentation with clear sections and highlight any important patterns or trends."""
        )

    @staticmethod
    def _generate_release_note_doc_prompt(
        release_note_info: dict, repo_info: dict
    ) -> HumanMessage:
        """Generate a prompt for release note documentation."""
        repo_name = repo_info.get("repo", "")
        owner = repo_info.get("owner", "")

        return HumanMessage(
            content=f"""Generate comprehensive release history documentation for the repository '{owner}/{repo_name}'.

Based on the following release note information, create a detailed analysis that includes:
1. Timeline of major releases and their key features
2. Breaking changes and migration guides
3. Performance improvements and bug fixes across versions
4. Deprecations and future direction indicators
5. Release frequency and versioning patterns

Release Information:
{release_note_info}

Please structure the documentation with clear sections and make it suitable for users planning upgrades."""
        )

    def repo_info_overview_node(self, state: dict):
        # log_state(state)
        # basic_info_for_repo has repo_structure and repo_info attributes
        # this node need to do these things:
        # 1. based on the repo_info to generate overall repo documentation
        # 2. filter out md files which can be the documentation source files
        print("→ Processing overall_doc_generation_node...")
        owner = state.get("basic_info_for_repo", {}).get("owner", "")
        repo = state.get("basic_info_for_repo", {}).get("repo", "")
        basic_info_for_repo = state.get("basic_info_for_repo", {})
        repo_info = basic_info_for_repo.get("repo_info", {})
        repo_structure = basic_info_for_repo.get("repo_structure", [])
        # filter out md files
        doc_source_files = [
            file_path for file_path in repo_structure if file_path.endswith(".md")
        ]
        doc_contents = []
        for file_path in doc_source_files:
            # file_path -> absolute path
            file_path = resolve_path(file_path)
            content = read_file(file_path)
            if content:
                doc_contents.append(f"# Source File: {file_path}\n\n{content}\n\n")
        combined_doc_content = "\n".join(doc_contents)
        system_prompt = self._get_system_prompt()
        human_prompt = self._get_overview_doc_prompt(repo_info, combined_doc_content)
        llm = CONFIG.get_llm()
        response = llm.invoke([system_prompt, human_prompt])
        print("✓ Overall documentation generated")
        return {
            "overview_info": response.content,
        }

    def overview_doc_generation_node(self, state: dict):
        # log_state(state)
        wiki_path = state.get("basic_info_for_repo", {}).get(
            "wiki_path", "./.wikis/default"
        )
        os.makedirs(wiki_path, exist_ok=True)

        file_name = os.path.join(wiki_path, "overview_documentation.md")
        print(f"→ Writing overview info to {file_name}")
        write_file(
            file_name,
            state.get("overview_info", ""),
        )
        print("✓ Overview documentation written successfully")

    def repo_info_commit_node(self, state: dict):
        # log_state(state)
        # this node need to collect repo commit info and preprocess for doc-generation
        print("→ Processing repo_info_commit_node...")
        commit_info = get_repo_commit_info(
            owner=state.get("basic_info_for_repo", {}).get("owner", ""),
            repo=state.get("basic_info_for_repo", {}).get("repo", ""),
            platform=state.get("basic_info_for_repo", {}).get("platform", "github"),
        )
        print("✓ Commit info retrieved successfully")
        return {"commit_info": commit_info}

    def commit_doc_generation_node(self, state: dict):
        # log_state(state)
        # Generate commit history documentation using LLM
        print("→ Processing commit_doc_generation_node...")
        commit_info = state.get("commit_info", {})
        repo_info = state.get("basic_info_for_repo", {})
        wiki_path = state.get("basic_info_for_repo", {}).get(
            "wiki_path", "./.wikis/default"
        )
        os.makedirs(wiki_path, exist_ok=True)

        system_prompt = self._get_system_prompt()
        human_prompt = self._generate_commit_doc_prompt(commit_info, repo_info)

        llm = CONFIG.get_llm()
        response = llm.invoke([system_prompt, human_prompt])
        print("✓ Commit documentation generated")

        file_name = os.path.join(wiki_path, "commit_documentation.md")
        print(f"→ Writing commit info to {file_name}")
        write_file(
            file_name,
            response.content,
        )
        print("✓ Commit documentation written successfully")

    def repo_info_pr_node(self, state: dict):
        # log_state(state)
        # this node need to collect repo pr info and preprocess for doc-generation
        print("→ Processing repo_info_pr_node...")
        pr = get_pr(
            owner=state.get("basic_info_for_repo", {}).get("owner", ""),
            repo=state.get("basic_info_for_repo", {}).get("repo", ""),
            platform=state.get("basic_info_for_repo", {}).get("platform", "github"),
        )
        print("✓ PR info retrieved successfully")
        return {"pr_info": pr}

    def pr_doc_generation_node(self, state: dict):
        # log_state(state)
        # Generate PR analysis documentation using LLM
        print("→ Processing pr_doc_generation_node...")
        pr_info = state.get("pr_info", {})
        repo_info = state.get("basic_info_for_repo", {})
        wiki_path = state.get("basic_info_for_repo", {}).get(
            "wiki_path", "./.wikis/default"
        )
        os.makedirs(wiki_path, exist_ok=True)

        system_prompt = self._get_system_prompt()
        human_prompt = self._generate_pr_doc_prompt(pr_info, repo_info)

        llm = CONFIG.get_llm()
        response = llm.invoke([system_prompt, human_prompt])
        print("✓ PR documentation generated")

        file_name = os.path.join(wiki_path, "pr_documentation.md")
        print(f"→ Writing PR info to {file_name}")
        write_file(
            file_name,
            response.content,
        )
        print("✓ PR documentation written successfully")

    def repo_info_release_note_node(self, state: dict):
        # log_state(state)
        # this node need to collect repo release note info and preprocess for doc-generation
        print("→ Processing repo_info_release_note_node...")
        release_note = get_release_note(
            owner=state.get("basic_info_for_repo", {}).get("owner", ""),
            repo=state.get("basic_info_for_repo", {}).get("repo", ""),
            platform=state.get("basic_info_for_repo", {}).get("platform", "github"),
        )
        print("✓ Release note info retrieved successfully")
        return {"release_note_info": release_note}

    def release_note_doc_generation_node(self, state: dict):
        # log_state(state)
        # Generate release history documentation using LLM
        print("→ Processing release_note_doc_generation_node...")
        release_note_info = state.get("release_note_info", {})
        repo_info = state.get("basic_info_for_repo", {})
        wiki_path = state.get("basic_info_for_repo", {}).get(
            "wiki_path", "./.wikis/default"
        )
        os.makedirs(wiki_path, exist_ok=True)

        system_prompt = self._get_system_prompt()
        human_prompt = self._generate_release_note_doc_prompt(
            release_note_info, repo_info
        )

        llm = CONFIG.get_llm()
        response = llm.invoke([system_prompt, human_prompt])
        print("✓ Release note documentation generated")

        file_name = os.path.join(wiki_path, "release_note_documentation.md")
        print(f"→ Writing release note info to {file_name}")
        write_file(
            file_name,
            response.content,
        )
        print("✓ Release note documentation written successfully")

    def repo_info_update_log_node(self, state: dict):
        # log_state(state)
        # this node need to generate the doc about the repo commit, pr, and release_note date this time
        # with this doc, next time run agent, can compare date between the date in this doc to update current doc
        print("→ Processing repo_info_update_log_node")
        # 1. get relevant info
        commit_info = state.get("commit_info", {})
        pr_info = state.get("pr_info", {})
        release_note_info = state.get("release_note_info", {})
        # 2. extract the date from info and convert to string
        commit_date = str(commit_info.get("commits", [{}])[0].get("date", "N/A"))
        pr_created_at_date = str(pr_info.get("prs", [{}])[0].get("created_at", "N/A"))
        pr_merged_at_date = str(pr_info.get("prs", [{}])[0].get("merged_at", "N/A"))
        release_note_created_at_date = str(
            release_note_info.get("releases", [{}])[0].get("created_at", "N/A")
        )
        release_note_published_at_date = str(
            release_note_info.get("releases", [{}])[0].get("published_at", "N/A")
        )
        # 3. merge into a json doc
        update_log_info = {
            "log_date": str(state.get("basic_info_for_repo", {}).get("date", "N/A")),
            "commit_date": commit_date,
            "pr_created_at_date": pr_created_at_date,
            "pr_merged_at_date": pr_merged_at_date,
            "release_note_created_at_date": release_note_created_at_date,
            "release_note_published_at_date": release_note_published_at_date,
        }
        print("✓ Repo update log documentation generated")
        return {
            "update_log_info": json.dumps(
                update_log_info, indent=2, ensure_ascii=False
            ),
        }

    def update_log_doc_generation_node(self, state: dict):
        # log_state(state)
        # this node need to write the generated repo update log to file
        print("→ Processing update_log_doc_generation_node...")
        owner = state.get("basic_info_for_repo", {}).get("owner", "")
        repo = state.get("basic_info_for_repo", {}).get("repo", "")
        date = state.get("basic_info_for_repo", {}).get("date", "")
        wiki_path = state.get("basic_info_for_repo", {}).get(
            "wiki_path", "./.wikis/default"
        )
        os.makedirs(wiki_path, exist_ok=True)

        write_file(
            os.path.join(wiki_path, f"repo_update_log_{date}.json"),
            state.get("update_log_info", ""),
        )
        print("✓ Repo update log file written successfully")

    def build(self):
        repo_info_builder = StateGraph(RepoInfoSubGraphState)
        repo_info_builder.add_node(
            "repo_info_overview_node", self.repo_info_overview_node
        )
        repo_info_builder.add_node(
            "overview_doc_generation_node", self.overview_doc_generation_node
        )
        repo_info_builder.add_node("repo_info_commit_node", self.repo_info_commit_node)
        repo_info_builder.add_node(
            "commit_doc_generation_node", self.commit_doc_generation_node
        )
        repo_info_builder.add_node("repo_info_pr_node", self.repo_info_pr_node)
        repo_info_builder.add_node(
            "pr_doc_generation_node", self.pr_doc_generation_node
        )
        repo_info_builder.add_node(
            "repo_info_release_note_node", self.repo_info_release_note_node
        )
        repo_info_builder.add_node(
            "release_note_doc_generation_node", self.release_note_doc_generation_node
        )
        repo_info_builder.add_edge(START, "repo_info_commit_node")
        repo_info_builder.add_edge(START, "repo_info_pr_node")
        repo_info_builder.add_edge(START, "repo_info_release_note_node")
        repo_info_builder.add_edge(START, "repo_info_overview_node")
        repo_info_builder.add_edge(
            "repo_info_overview_node", "overview_doc_generation_node"
        )
        repo_info_builder.add_edge(
            "repo_info_commit_node", "commit_doc_generation_node"
        )
        repo_info_builder.add_edge("repo_info_pr_node", "pr_doc_generation_node")
        repo_info_builder.add_edge(
            "repo_info_release_note_node", "release_note_doc_generation_node"
        )
        return repo_info_builder.compile(checkpointer=True)

    def get_graph(self):
        return self.graph


class CodeAnalysisSubGraphState(TypedDict):
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
{analysis}
{summary}
Please structure the documentation with clear sections and make it suitable for both new contributors and project maintainers."""
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
        mode = state.get("basic_info_for_code", {}).get("mode", "fast")
        max_num_files = get_max_num_files(mode)
        print(f"   → Mode: {mode}, max_num_files: {max_num_files}")
        repo_info = state.get("basic_info_for_code", {}).get("repo_info", {})
        repo_structure = state.get("basic_info_for_code", {}).get("repo_structure", [])
        # 2. call llm to determine which files to analyze based on repo_info
        # filter supported code files (.py, .js, .go, .cpp, .java, .rs etc.)
        system_prompt = self._get_system_prompt()
        human_prompt = self._get_code_filter_prompt(
            repo_info, repo_structure, max_num_files
        )
        llm = CONFIG.get_llm()
        response = llm.invoke([system_prompt, human_prompt])
        # parse the response to get the list of code files
        code_files = [
            line.strip() for line in response.content.splitlines() if line.strip()
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
        analysis_builder.add_edge(START, "code_filter_node")
        analysis_builder.add_edge("code_filter_node", "code_analysis_node")
        return analysis_builder.compile(checkpointer=True)

    def get_graph(self):
        return self.graph


# --- Parent graph ---
class ParentGraphState(TypedDict):
    owner: str
    repo: str
    platform: str
    mode: str
    max_workers: int
    date: str
    # branch-namespaced fields to be consumed by child subgraphs
    basic_info_for_repo: dict | None
    basic_info_for_code: dict | None
    # repo_info_sub_graph outputs
    commit_info: dict | None
    pr_info: dict | None
    release_note_info: dict | None
    # code_analysis_sub_graph outputs
    code_analysis: dict | None
    # repo_doc_generation_sub_graph outputs
    commit_documentation: str | None
    pr_documentation: str | None
    release_note_documentation: str | None
    overall_documentation: str | None
    repo_update_log: str | None
    # code_doc_generation_sub_graph outputs
    code_documentation: dict | None


class ParentGraphBuilder:

    def __init__(self, branch_mode: str = "all"):
        self.repo_info_graph = RepoInfoSubGraphBuilder().get_graph()
        self.code_analysis_graph = CodeAnalysisSubGraphBuilder().get_graph()
        self.checkpointer = MemorySaver()
        self.branch = branch_mode  # "all" or "code" or "repo"
        self.graph = self.build(self.checkpointer)

    def basic_info_node(self, state: ParentGraphState):
        # log_state(state)
        print("\n" + "=" * 80)
        print("🚀 Starting graph execution...")
        print("=" * 80)
        owner = state.get("owner")
        repo = state.get("repo")
        platform = state.get("platform", "github")
        mode = state.get("mode", "fast")
        max_workers = state.get("max_workers", 10)
        date = state.get("date", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        repo_root_path = "./.repos"
        wiki_root_path = "./.wikis"
        repo_path = f"{repo_root_path}/{owner}_{repo}"
        wiki_path = f"{wiki_root_path}/{owner}_{repo}/{date}"

        print(f"📦 Repository: {owner}/{repo}")
        print(f"🔗 Platform: {platform}")
        print(f"⚙️ Mode: {mode}")
        print(f"👷 Max Workers: {max_workers}")
        print(f"📁 Repo Path: {repo_path}")
        print(f"📄 Wiki Path: {wiki_path}")

        repo_info = get_repo_info(owner, repo, platform=platform)
        repo_structure = get_repo_structure(repo_path)

        print(f"✓ Repository initialized")

        # combine repo_info and repo_structure into combined_info
        combined_info = {
            "owner": owner,
            "repo": repo,
            "platform": platform,
            "mode": mode,
            "max_workers": max_workers,
            "date": date,
            "repo_path": repo_path,
            "wiki_path": wiki_path,
            "repo_info": repo_info,
            "repo_structure": repo_structure,
        }

        return {
            "basic_info_for_repo": combined_info,
            "basic_info_for_code": combined_info,
        }

    def build(self, checkpointer):
        builder = StateGraph(ParentGraphState)
        builder.add_node("basic_info_node", self.basic_info_node)
        builder.add_node("repo_info_graph", self.repo_info_graph)
        builder.add_node("code_analysis_graph", self.code_analysis_graph)
        match self.branch:
            case "all":
                builder.add_edge(START, "basic_info_node")
                builder.add_edge("basic_info_node", "repo_info_graph")
                builder.add_edge("basic_info_node", "code_analysis_graph")
            case "code":
                builder.add_edge(START, "basic_info_node")
                builder.add_edge("basic_info_node", "code_analysis_graph")
            case "repo":
                builder.add_edge(START, "basic_info_node")
                builder.add_edge("basic_info_node", "repo_info_graph")
        return builder.compile(checkpointer=checkpointer)

    def get_graph(self):
        return self.graph


def demo():
    CONFIG.display()
    parent_graph_builder = ParentGraphBuilder(branch_mode="code")
    graph = parent_graph_builder.get_graph()
    # draw the graph structure if you want
    # draw_graph(graph)
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    config = {"configurable": {"thread_id": f"wiki-generation-{date}"}}
    start_time = time.time()
    for chunk in graph.stream(
        {
            "owner": "facebook",
            "repo": "zstd",
            "platform": "github",
            "mode": "fast",  # "fast" or "smart"
            "max_workers": 50,  # 20 worker -> 3 - 4 minutes
            "date": date,
        },
        config=config,
        subgraphs=True,
    ):
        pass

    end_time = time.time()
    elapsed_time = end_time - start_time

    print("\n" + "=" * 80)
    print("✅ Graph execution completed successfully!")
    print("=" * 80)
    print(f"Total execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    # use "uv run python -m tests.sub-graph-agent.graph" to run this file
    demo()

    # - draw the repo_info_sub_graph
    # repo_info_sub_graph = RepoInfoSubGraphBuilder().get_graph()
    # draw_graph(repo_info_sub_graph)
