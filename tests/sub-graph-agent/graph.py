from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables.graph import MermaidDrawMethod
from langgraph.graph.state import StateGraph, START
from langchain_core.messages import HumanMessage, SystemMessage
from datetime import datetime
import os

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
    # subgraph output
    commit_info: dict
    pr_info: dict
    release_note_info: dict


class RepoInfoSubGraphBuilder:

    def __init__(self):
        self.graph = self.build()

    def repo_info_commit_node(self, state: dict):
        # log_state(state)
        # this node need to collect repo commit info and preprocess for doc-generation
        print("→ Processing repo_info_commit_node...")
        commit_info = get_repo_commit_info(
            owner=state["basic_info_for_repo"]["owner"],
            repo=state["basic_info_for_repo"]["repo"],
            platform=state["basic_info_for_repo"].get("platform", "github"),
        )
        print("✓ Commit info retrieved successfully")
        return {"commit_info": commit_info}

    def repo_info_pr_node(self, state: dict):
        # log_state(state)
        # this node need to collect repo pr info and preprocess for doc-generation
        print("→ Processing repo_info_pr_node...")
        pr = get_pr(
            owner=state["basic_info_for_repo"]["owner"],
            repo=state["basic_info_for_repo"]["repo"],
            platform=state["basic_info_for_repo"].get("platform", "github"),
        )
        print("✓ PR info retrieved successfully")
        return {"pr_info": pr}

    def repo_info_release_note_node(self, state: dict):
        # log_state(state)
        # this node need to collect repo release note info and preprocess for doc-generation
        print("→ Processing repo_info_release_note_node...")
        release_note = get_release_note(
            owner=state["basic_info_for_repo"]["owner"],
            repo=state["basic_info_for_repo"]["repo"],
            platform=state["basic_info_for_repo"].get("platform", "github"),
        )
        print("✓ Release note info retrieved successfully")
        return {"release_note_info": release_note}

    def build(self):
        repo_info_builder = StateGraph(RepoInfoSubGraphState)
        repo_info_builder.add_node("repo_info_commit_node", self.repo_info_commit_node)
        repo_info_builder.add_node("repo_info_pr_node", self.repo_info_pr_node)
        repo_info_builder.add_node(
            "repo_info_release_note_node", self.repo_info_release_note_node
        )
        repo_info_builder.add_edge(START, "repo_info_commit_node")
        repo_info_builder.add_edge("repo_info_commit_node", "repo_info_pr_node")
        repo_info_builder.add_edge("repo_info_pr_node", "repo_info_release_note_node")
        return repo_info_builder.compile(checkpointer=True)

    def get_graph(self):
        return self.graph


class RepoDocGenerationState(TypedDict):
    basic_info_for_repo: dict
    commit_info: dict
    pr_info: dict
    release_note_info: dict
    repo_documentation: dict


class RepoDocGenerationSubGraphBuilder:

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

    def overall_doc_generation_node(self, state: dict):
        # log_state(state)
        # basic_info_for_repo has repo_structure and repo_info attributes
        # this node need to do these things:
        # 1. based on the repo_info to generate overall repo documentation
        # 2. filter out md files which can be the documentation source files
        print("→ Processing overall_doc_generation_node...")
        owner = state["basic_info_for_repo"]["owner"]
        repo = state["basic_info_for_repo"]["repo"]
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
        human_prompt = HumanMessage(
            content=f"""Generate overall repository documentation for the repository '{owner}/{repo}'.
Based on the following repository information and documentation source files, create a detailed analysis that includes:
1. Overview of the repository structure and key components
2. Summary of existing documentation and its coverage
3. Identification of documentation gaps and areas for improvement
Repository Information:
{repo_info}
Documentation Source Files Content:
{combined_doc_content}
Please structure the documentation with clear sections and make it suitable for both new contributors and project maintainers."""
        )
        llm = CONFIG.get_llm()
        response = llm.invoke([system_prompt, human_prompt])
        print("✓ Overall documentation generated")
        return {
            "repo_documentation": {
                "overall_documentation": response.content,
                "status": "overall repo documentation generated.",
            }
        }

    def commit_doc_generation_node(self, state: dict):
        # log_state(state)
        # Generate commit history documentation using LLM
        print("→ Processing commit_doc_generation_node...")
        commit_info = state.get("commit_info", {})
        repo_info = state.get("basic_info_for_repo", {})

        system_prompt = self._get_system_prompt()
        human_prompt = self._generate_commit_doc_prompt(commit_info, repo_info)

        llm = CONFIG.get_llm()
        response = llm.invoke([system_prompt, human_prompt])
        print("✓ Commit documentation generated")

        return {
            "repo_documentation": {
                **state.get("repo_documentation", {}),
                "commit_documentation": response.content,
                "status": state["repo_documentation"].get("status", "")
                + " repo commit documentation generated.",
            }
        }

    def pr_doc_generation_node(self, state: dict):
        # log_state(state)
        # Generate PR analysis documentation using LLM
        print("→ Processing pr_doc_generation_node...")
        pr_info = state.get("pr_info", {})
        repo_info = state.get("basic_info_for_repo", {})

        system_prompt = self._get_system_prompt()
        human_prompt = self._generate_pr_doc_prompt(pr_info, repo_info)

        llm = CONFIG.get_llm()
        response = llm.invoke([system_prompt, human_prompt])
        print("✓ PR documentation generated")

        return {
            "repo_documentation": {
                **state.get("repo_documentation", {}),
                "pr_documentation": response.content,
                "status": state["repo_documentation"].get("status", "")
                + " repo pr documentation generated.",
            }
        }

    def release_note_doc_generation_node(self, state: dict):
        # log_state(state)
        # Generate release history documentation using LLM
        print("→ Processing release_note_doc_generation_node...")
        release_note_info = state.get("release_note_info", {})
        repo_info = state.get("basic_info_for_repo", {})

        system_prompt = self._get_system_prompt()
        human_prompt = self._generate_release_note_doc_prompt(
            release_note_info, repo_info
        )

        llm = CONFIG.get_llm()
        response = llm.invoke([system_prompt, human_prompt])
        print("✓ Release note documentation generated")

        return {
            "repo_documentation": {
                **state.get("repo_documentation", {}),
                "release_note_documentation": response.content,
                "status": state["repo_documentation"].get("status", "")
                + " repo release note documentation generated.",
            }
        }

    def write_doc_to_file_node(self, state: dict):
        # log_state(state)
        # this node need to write the generated repo documentation to files
        print("→ Processing write_doc_to_file_node...")
        repo_doc = state.get("repo_documentation", {})
        owner = state["basic_info_for_repo"]["owner"]
        repo = state["basic_info_for_repo"]["repo"]
        wiki_path = state["basic_info_for_repo"].get("wiki_path", "./.wikis/default")
        os.makedirs(wiki_path, exist_ok=True)

        write_file(
            os.path.join(wiki_path, "overall_documentation.md"),
            repo_doc.get("overall_documentation", ""),
        )
        write_file(
            os.path.join(wiki_path, "commit_documentation.md"),
            repo_doc.get("commit_documentation", ""),
        )
        write_file(
            os.path.join(wiki_path, "pr_documentation.md"),
            repo_doc.get("pr_documentation", ""),
        )
        write_file(
            os.path.join(wiki_path, "release_note_documentation.md"),
            repo_doc.get("release_note_documentation", ""),
        )
        print("✓ Documentation files written successfully")
        return {
            "repo_documentation": {
                **repo_doc,
                "status": repo_doc.get("status", "")
                + " Documentation written to files.",
            }
        }

    def build(self):
        repo_doc_builder = StateGraph(RepoDocGenerationState)
        repo_doc_builder.add_node(
            "overall_doc_generation_node", self.overall_doc_generation_node
        )
        repo_doc_builder.add_node(
            "commit_doc_generation_node", self.commit_doc_generation_node
        )
        repo_doc_builder.add_node("pr_doc_generation_node", self.pr_doc_generation_node)
        repo_doc_builder.add_node(
            "release_note_doc_generation_node", self.release_note_doc_generation_node
        )
        repo_doc_builder.add_node("write_doc_to_file_node", self.write_doc_to_file_node)
        repo_doc_builder.add_edge(START, "overall_doc_generation_node")
        repo_doc_builder.add_edge(
            "overall_doc_generation_node", "commit_doc_generation_node"
        )
        repo_doc_builder.add_edge(
            "commit_doc_generation_node", "pr_doc_generation_node"
        )
        repo_doc_builder.add_edge(
            "pr_doc_generation_node", "release_note_doc_generation_node"
        )
        repo_doc_builder.add_edge(
            "release_note_doc_generation_node", "write_doc_to_file_node"
        )
        return repo_doc_builder.compile(checkpointer=True)

    def get_graph(self):
        return self.graph


class CodeAnalysisSubGraphState(TypedDict):
    basic_info_for_code: dict
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
            content=f"""Based on the following repository information and file structure, identify up to {max_num_files} code files that are most relevant for static analysis. 
Repository Information:
{repo_info}
File Structure:
{repo_structure}
Please provide ONLY a list of file paths that are relevant for code analysis. Each path should be on a separate line. Do not include any descriptions, section headers, or additional text. Just output the file paths directly."""
        )

    @staticmethod
    def _get_code_analysis_prompt(file_path: str, content: str) -> HumanMessage:
        return HumanMessage(
            content=f"""Please provide a concise summary (max 100 words) of the following code file content for file '{file_path}':

{content}

Focus on: key functions/classes, main purpose, and important logic."""
        )

    def code_filter_node(self, state: dict):
        # log_state(state)
        # this node need to filter useful code files for analysis
        # 1. get repo_info and repo_structure from basic_info_for_code
        def get_max_num_files(mode: str) -> int:
            # mode decide the number of code files to analyze
            # if "fast" -> max 20 files
            # if "smart" -> max 50 files
            match mode:
                case "fast":
                    return 20
                case "smart":
                    return 50
                case _:
                    return 20

        print("→ Processing code_filter_node...")
        mode = state["basic_info_for_code"].get("mode", "fast")
        max_num_files = get_max_num_files(mode)
        print(f"   → Mode: {mode}, max_num_files: {max_num_files}")
        repo_info = state["basic_info_for_code"].get("repo_info", {})
        repo_structure = state["basic_info_for_code"].get("repo_structure", [])
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
        print("DEBUG: code_files =", code_files)
        print(f"✓ Filtered {len(code_files)} code files for analysis")
        return {
            "code_analysis": {
                "code_files": code_files,
                "status": "Code files filtered.",
            }
        }

    def code_analysis_node(self, state: dict):
        # log_state(state)
        # this node need to analyze the filtered code files
        print("→ Processing code_analysis_node...")
        code_files = state["code_analysis"].get("code_files", [])
        analysis_results = {}
        analysis_summary = {}

        for idx, file_path in enumerate(code_files, 1):
            print(f"  → Analyzing {file_path} ({idx}/{len(code_files)})")
            file_path = resolve_path(file_path)
            content = read_file(file_path)
            if content:
                analysis = analyze_file_with_tree_sitter(file_path)
                # formatted_analysis = format_tree_sitter_analysis_results(analysis)
                formatted_analysis = format_tree_sitter_analysis_results_to_prompt(
                    analysis
                )
                analysis_results[file_path] = formatted_analysis

                # Use LLM to summarize the analysis to reduce size
                summary_prompt = self._get_code_analysis_prompt(file_path, content)
                llm = CONFIG.get_llm()
                summary_response = llm.invoke([summary_prompt])
                analysis_summary[file_path] = summary_response.content
            else:
                analysis_results[file_path] = "File could not be read or is empty."
                analysis_summary[file_path] = "File could not be read or is empty."

        print(f"✓ Code analysis completed for {len(analysis_results)} files")
        return {
            "code_analysis": {
                **state["code_analysis"],
                "analysis_results": analysis_results,
                "analysis_summary": analysis_summary,
                "status": state["code_analysis"].get("status", "")
                + " Code analysis completed.",
            }
        }

    def build(self):
        analysis_builder = StateGraph(CodeAnalysisSubGraphState)
        analysis_builder.add_node("code_filter_node", self.code_filter_node)
        analysis_builder.add_node("code_analysis_node", self.code_analysis_node)
        analysis_builder.add_edge(START, "code_filter_node")
        analysis_builder.add_edge("code_filter_node", "code_analysis_node")
        return analysis_builder.compile(checkpointer=True)

    def get_graph(self):
        return self.graph


class CodeDocGenerationState(TypedDict):
    basic_info_for_code: dict
    code_analysis: dict
    code_documentation: dict


class CodeDocGenerationSubGraphBuilder:
    def __init__(self):
        self.graph = self.build()

    @staticmethod
    def _get_system_prompt():
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
        return HumanMessage(
            content=f"""Generate comprehensive documentation for the code file '{file_path}'.
Based on the following analysis results and summary, create a detailed documentation that includes:
1. Overview of the file's purpose and functionality
2. Key components (functions, classes, etc.) and their roles
3. Important algorithms or design patterns used
4. Dependencies and relationships with other files
Analysis Results:
{analysis}
Analysis Summary:
{summary if summary else "No summary available."}
Please structure the documentation with clear sections and make it suitable for both new contributors and project maintainers."""
        )

    def loop_code_doc_node(self, state: dict):
        # log_state(state)
        # Generate documentation for each code file using LLM
        print("→ Processing loop_code_doc_node...")
        analysis_results = state.get("code_analysis", {}).get("analysis_results", {})
        analysis_summary = state.get("code_analysis", {}).get("analysis_summary", {})
        basic_info = state.get("basic_info_for_code", {})
        code_file_docs = {}

        system_prompt = self._get_system_prompt()

        # use analysis_results and analysis_summary to generate documentation
        # analysis_results for func, class, import, dependencies, etc.
        # analysis_summary for file-level summary
        for file_path, analysis in analysis_results.items():
            print(f"  → Generating documentation for {file_path}")
            # create a prompt for the LLM
            human_prompt = self._get_code_doc_prompt(
                file_path, analysis, analysis_summary.get(file_path, "")
            )
            llm = CONFIG.get_llm()
            response = llm.invoke([system_prompt, human_prompt])
            code_file_docs[file_path] = response.content
        print(f"✓ Code documentation generated for {len(code_file_docs)} files")
        return {
            "code_documentation": {
                "code_file_documentation": code_file_docs,
                "status": "Code documentation generated.",
            }
        }

    def write_code_doc_to_file_node(self, state: dict):
        # log_state(state)
        # this node need to write the generated code documentation to files
        print("→ Processing write_code_doc_to_file_node...")
        code_doc = state.get("code_documentation", {})
        basic_info = state.get("basic_info_for_code", {})
        wiki_path = basic_info.get("wiki_path", "./.wikis/default")
        os.makedirs(wiki_path, exist_ok=True)

        # write docs for each code file
        code_file_docs = code_doc.get("code_file_documentation", {})
        for file_path, doc_content in code_file_docs.items():
            # create a safe file name for the documentation
            safe_file_name = file_path.replace("/", "_").replace("\\", "_") + "_doc.md"
            write_file(
                os.path.join(wiki_path, safe_file_name),
                doc_content,
            )

        print(f"✓ Code documentation files written successfully")
        return {
            "code_documentation": {
                **code_doc,
                "status": code_doc.get("status", "")
                + " Code documentation written to files.",
            }
        }

    def build(self):
        code_doc_builder = StateGraph(CodeDocGenerationState)
        code_doc_builder.add_node("loop_code_doc_node", self.loop_code_doc_node)
        code_doc_builder.add_node(
            "write_code_doc_to_file_node", self.write_code_doc_to_file_node
        )
        code_doc_builder.add_edge(START, "loop_code_doc_node")
        code_doc_builder.add_edge("loop_code_doc_node", "write_code_doc_to_file_node")
        return code_doc_builder.compile(checkpointer=True)

    def get_graph(self):
        return self.graph


# --- Parent graph ---
class ParentGraphState(TypedDict):
    owner: str
    repo: str
    platform: str
    mode: str
    basic_info: dict
    # branch-namespaced fields to be consumed by child subgraphs
    basic_info_for_repo: dict | None
    basic_info_for_code: dict | None
    # outputs
    commit_info: dict | None
    pr_info: dict | None
    release_note_info: dict | None
    code_analysis: dict | None
    repo_documentation: dict | None
    code_documentation: dict | None


class ParentGraphBuilder:

    def __init__(self, branch_mode: str = "all"):
        self.repo_info_graph = RepoInfoSubGraphBuilder().get_graph()
        self.code_analysis_graph = CodeAnalysisSubGraphBuilder().get_graph()
        self.repo_doc_generation_graph = RepoDocGenerationSubGraphBuilder().get_graph()
        self.code_doc_generation_graph = CodeDocGenerationSubGraphBuilder().get_graph()
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
        repo_root_path = "./.repos"
        wiki_root_path = "./.wikis"
        repo_path = f"{repo_root_path}/{owner}_{repo}"
        wiki_path = f"{wiki_root_path}/{owner}_{repo}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

        print(f"📦 Repository: {owner}/{repo}")
        print(f"🔗 Platform: {platform}")

        repo_info = get_repo_info(owner, repo, platform=platform)
        repo_structure = get_repo_structure(repo_path)

        print(f"✓ Repository initialized")

        # combine repo_info and repo_structure into combined_info
        combined_info = {
            "owner": owner,
            "repo": repo,
            "platform": platform,
            "mode": mode,
            "repo_path": repo_path,
            "wiki_path": wiki_path,
            "repo_info": repo_info,
            "repo_structure": repo_structure,
        }

        return {
            "basic_info": {
                "owner": owner,
                "repo": repo,
                "platform": platform,
                "mode": mode,
                "message": "Basic info retrieved.",
            },
            "basic_info_for_repo": combined_info,
            "basic_info_for_code": combined_info,
            "owner_for_repo": owner,
            "repo_for_repo": repo,
            "owner_for_code": owner,
            "repo_for_code": repo,
        }

    def build(self, checkpointer):
        builder = StateGraph(ParentGraphState)
        builder.add_node("basic_info_node", self.basic_info_node)
        builder.add_node("repo_info_graph", self.repo_info_graph)
        builder.add_node("code_analysis_graph", self.code_analysis_graph)
        builder.add_node("repo_doc_generation_graph", self.repo_doc_generation_graph)
        builder.add_node("code_doc_generation_graph", self.code_doc_generation_graph)
        match self.branch:
            case "all":
                builder.add_edge(START, "basic_info_node")
                builder.add_edge("basic_info_node", "repo_info_graph")
                builder.add_edge("basic_info_node", "code_analysis_graph")
                builder.add_edge("repo_info_graph", "repo_doc_generation_graph")
                builder.add_edge("code_analysis_graph", "code_doc_generation_graph")
            case "code":
                builder.add_edge(START, "basic_info_node")
                builder.add_edge("basic_info_node", "code_analysis_graph")
                builder.add_edge("code_analysis_graph", "code_doc_generation_graph")
            case "repo":
                builder.add_edge(START, "basic_info_node")
                builder.add_edge("basic_info_node", "repo_info_graph")
                builder.add_edge("repo_info_graph", "repo_doc_generation_graph")
        return builder.compile(checkpointer=checkpointer)

    def get_graph(self):
        return self.graph


if __name__ == "__main__":
    # use "uv run python -m tests.sub-graph-agent.graph" to run this file
    # draw the graph structure if you want
    # draw_graph(graph)
    CONFIG.display()
    parent_graph_builder = ParentGraphBuilder(branch_mode="repo")
    graph = parent_graph_builder.get_graph()
    config = {
        "configurable": {"thread_id": f"wiki-generation-{datetime.now().timestamp()}"}
    }
    for chunk in graph.stream(
        {
            "owner": "facebook",
            "repo": "zstd",
            "platform": "github",
            "mode": "fast",  # or "smart"
        },
        config=config,
        subgraphs=True,
    ):
        pass

    print("\n" + "=" * 80)
    print("✅ Graph execution completed successfully!")
    print("=" * 80)
