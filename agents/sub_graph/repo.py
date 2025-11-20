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
from utils.file import (
    get_repo_structure,
    write_file,
    read_file,
    resolve_path,
    read_json,
)
from .utils import draw_graph, log_state
from config import CONFIG


class RepoInfoSubGraphState(TypedDict):
    # inputs
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
4. Architecture visualization using Mermaid diagrams (include at least one of the following):
   - System architecture diagram (graph/flowchart)
   - Component relationships diagram
   - Module dependency diagram
   - Data flow diagram
   Use Mermaid markdown syntax (```mermaid ... ```) to create clear, informative diagrams

Repository Information:
{repo_info}

Documentation Source Files Content:
{doc_contents}

Please structure the documentation with clear sections and make it suitable for both new contributors and project maintainers. Include Mermaid diagrams to visualize the repository architecture and component relationships."""
        )

    @staticmethod
    def _get_updated_overview_doc_prompt(
        repo_info: dict,
        commit_info: dict,
        pr_info: dict,
        release_note_info: dict,
        overview_doc_path: str,
    ) -> HumanMessage:
        """Generate a prompt for updated overall repository documentation."""
        repo_name = repo_info.get("repo", "")
        owner = repo_info.get("owner", "")
        return HumanMessage(
            content=f"""The overall repository documentation for the repository '{owner}/{repo_name}' needs to be updated based on new commit, PR, and release note information. The previous documentation can be found at '{overview_doc_path}'.
Based on the following updated information, please update the existing documentation to reflect the latest changes. Ensure that the documentation remains comprehensive and well-structured.

Updated Commit Information:
{commit_info}

Updated PR Information:
{pr_info}

Updated Release Note Information:
{release_note_info}

Please structure the documentation with clear sections and make it suitable for both new contributors and project maintainers. 

IMPORTANT: Update or add Mermaid diagrams to reflect any architectural changes:
- Update existing Mermaid diagrams if the architecture has changed
- Add new diagrams (system architecture, component relationships, module dependencies, data flow) if needed
- Use Mermaid markdown syntax (```mermaid ... ```) to visualize the updated repository structure and changes
- Highlight new components or modified relationships in the diagrams"""
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
    def _generate_updated_commit_doc_prompt(
        commit_info: dict, repo_info: dict, commit_doc_path: str
    ) -> HumanMessage:
        """Generate a prompt for updated commit documentation."""
        repo_name = repo_info.get("repo", "")
        owner = repo_info.get("owner", "")
        return HumanMessage(
            content=f"""The commit history documentation for the repository '{owner}/{repo_name}' needs to be updated based on new commit information. The previous documentation can be found at '{commit_doc_path}'.
Based on the following updated commit information, please update the existing documentation to reflect the latest changes. Ensure that the documentation remains comprehensive and well-structured.
Updated Commit Information:
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
    def _generate_updated_pr_doc_prompt(
        pr_info: dict, repo_info: dict, pr_doc_path: str
    ) -> HumanMessage:
        """Generate a prompt for updated PR documentation."""
        repo_name = repo_info.get("repo", "")
        owner = repo_info.get("owner", "")
        return HumanMessage(
            content=f"""The pull request (PR) analysis documentation for the repository '{owner}/{repo_name}' needs to be updated based on new PR information. The previous documentation can be found at '{pr_doc_path}'.
Based on the following updated PR information, please update the existing documentation to reflect the latest changes. Ensure that the documentation remains comprehensive and well-structured.
Updated PR Information:
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

    @staticmethod
    def _generate_updated_release_note_doc_prompt(
        release_note_info: dict, repo_info: dict, release_note_doc_path: str
    ) -> HumanMessage:
        """Generate a prompt for updated release note documentation."""
        repo_name = repo_info.get("repo", "")
        owner = repo_info.get("owner", "")
        return HumanMessage(
            content=f"""The release history documentation for the repository '{owner}/{repo_name}' needs to be updated based on new release note information. The previous documentation can be found at '{release_note_doc_path}'.
Based on the following updated release note information, please update the existing documentation to reflect the latest changes. Ensure that the documentation remains comprehensive and well-structured.
Updated Release Note Information:
{release_note_info}
Please structure the documentation with clear sections and make it suitable for users planning upgrades."""
        )

    def repo_info_overview_node(self, state: dict):
        log_state(state)
        # basic_info_for_repo has repo_structure and repo_info attributes
        # this node need to do these things:
        # 1. based on the repo_info to generate overall repo documentation
        # 2. filter out md files which can be the documentation source files
        print("→ Processing overall_doc_generation_node...")
        basic_info_for_repo = state.get("basic_info_for_repo", {})
        commits_updated = basic_info_for_repo.get("commits_updated", False)
        prs_updated = basic_info_for_repo.get("prs_updated", False)
        releases_updated = basic_info_for_repo.get("releases_updated", False)
        commit_info = state.get("commit_info", {})
        pr_info = state.get("pr_info", {})
        release_note_info = state.get("release_note_info", {})
        repo_info = basic_info_for_repo.get("repo_info", {})
        repo_structure = basic_info_for_repo.get("repo_structure", [])
        wiki_path = basic_info_for_repo.get("wiki_path", "./.wikis/default")
        system_prompt = self._get_system_prompt()
        llm = CONFIG.get_llm()
        file_name = os.path.join(wiki_path, "overview_documentation.md")
        if commits_updated or prs_updated or releases_updated:
            print("   → [update] Generating overall documentation with updates.")
            # get updated prompt
            human_prompt = self._get_updated_overview_doc_prompt(
                repo_info, commit_info, pr_info, release_note_info, file_name
            )
            response = llm.invoke([system_prompt, human_prompt])
            print("✓ Overall documentation updated")
        elif os.path.exists(file_name):
            # if overview_documentation.md exists, show that the relevant documents have been generated,
            # so no need to generate overall doc again
            print("   → [skip] Overall documentation is up-to-date. No updates needed.")
            existing_content = read_file(file_name) or ""
            return {"overview_info": existing_content}
        else:
            print("   → [generate] Generating overall documentation without updates.")
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
            human_prompt = self._get_overview_doc_prompt(
                repo_info, combined_doc_content
            )
            response = llm.invoke([system_prompt, human_prompt])
            print("✓ Overall documentation generated")

        return {
            "overview_info": response.content,
        }

    def overview_doc_generation_node(self, state: dict):
        log_state(state)
        wiki_path = state.get("basic_info_for_repo", {}).get(
            "wiki_path", "./.wikis/default"
        )
        os.makedirs(wiki_path, exist_ok=True)

        file_name = os.path.join(wiki_path, "overview_documentation.md")
        if os.path.exists(file_name):
            print("   → [skip] Overview documentation already exists. Skipping write.")
            return
        print(f"→ Writing overview info to {file_name}")
        write_file(
            file_name,
            state.get("overview_info", ""),
        )
        print("✓ Overview documentation written successfully")

    def commit_doc_generation_node(self, state: dict):
        log_state(state)
        # Generate commit history documentation using LLM
        print("→ Processing commit_doc_generation_node...")
        basic_info_for_repo = state.get("basic_info_for_repo", {})
        repo_info = state.get("repo_info", {})
        commit_info = state.get("commit_info", {})
        commits_updated = basic_info_for_repo.get("commits_updated", False)
        wiki_path = basic_info_for_repo.get("wiki_path", "./.wikis/default")
        os.makedirs(wiki_path, exist_ok=True)

        # check if updated and adjust prompt based on that
        system_prompt = self._get_system_prompt()
        llm = CONFIG.get_llm()
        file_name = os.path.join(wiki_path, "commit_documentation.md")
        if commits_updated:
            # use updated prompt
            human_prompt = self._generate_updated_commit_doc_prompt(
                commit_info, repo_info, file_name
            )
            response = llm.invoke([system_prompt, human_prompt])
            print("✓ [update] Commit documentation updated")
        elif os.path.exists(file_name):
            # if commit_documentation.md exists, show that the commit doc has been generated, so no need to generate again
            print("   → [skip] Commit documentation is up-to-date. No updates needed.")
            return
        else:
            # use normal prompt
            human_prompt = self._generate_commit_doc_prompt(commit_info, repo_info)
            response = llm.invoke([system_prompt, human_prompt])
            print("✓ [generate] Commit documentation generated")

        print(f"→ Writing commit info to {file_name}")
        write_file(
            file_name,
            response.content,
        )
        print("✓ Commit documentation written successfully")

    def pr_doc_generation_node(self, state: dict):
        log_state(state)
        # Generate PR analysis documentation using LLM
        print("→ Processing pr_doc_generation_node...")
        basic_info_for_repo = state.get("basic_info_for_repo", {})
        pr_info = state.get("pr_info", {})
        repo_info = basic_info_for_repo.get("repo_info", {})
        prs_updated = basic_info_for_repo.get("prs_updated", False)
        wiki_path = basic_info_for_repo.get("wiki_path", "./.wikis/default")
        os.makedirs(wiki_path, exist_ok=True)

        system_prompt = self._get_system_prompt()
        llm = CONFIG.get_llm()
        file_name = os.path.join(wiki_path, "pr_documentation.md")
        if prs_updated:
            # use updated prompt
            human_prompt = self._generate_updated_pr_doc_prompt(
                pr_info, repo_info, file_name
            )
            response = llm.invoke([system_prompt, human_prompt])
            print("✓ [update] PR documentation updated")
        elif os.path.exists(file_name):
            # if pr_documentation.md exists, show that the pr doc has been generated, so no need to generate again
            print("   → [skip] PR documentation is up-to-date. No updates needed.")
            return
        else:
            # use normal prompt
            human_prompt = self._generate_pr_doc_prompt(pr_info, repo_info)
            response = llm.invoke([system_prompt, human_prompt])
            print("✓ [generate] PR documentation generated")
        print(f"→ Writing PR info to {file_name}")
        write_file(
            file_name,
            response.content,
        )
        print("✓ PR documentation written successfully")

    def release_note_doc_generation_node(self, state: dict):
        log_state(state)
        # Generate release history documentation using LLM
        print("→ Processing release_note_doc_generation_node...")
        basic_info_for_repo = state.get("basic_info_for_repo", {})
        releases_updated = basic_info_for_repo.get("releases_updated", False)
        release_note_info = state.get("release_note_info", {})
        repo_info = basic_info_for_repo.get("repo_info", {})
        wiki_path = basic_info_for_repo.get("wiki_path", "./.wikis/default")
        os.makedirs(wiki_path, exist_ok=True)

        system_prompt = self._get_system_prompt()
        llm = CONFIG.get_llm()
        file_name = os.path.join(wiki_path, "release_note_documentation.md")
        if releases_updated:
            # use updated prompt
            human_prompt = self._generate_updated_release_note_doc_prompt(
                release_note_info, repo_info, file_name
            )
            response = llm.invoke([system_prompt, human_prompt])
            print("✓ [update] Release note documentation updated")
        elif os.path.exists(file_name):
            # if release_note_documentation.md exists, show that the release note doc has been generated, so no need to generate again
            print(
                "   → [skip] Release note documentation is up-to-date. No updates needed."
            )
            return
        else:
            # use normal prompt
            human_prompt = self._generate_release_note_doc_prompt(
                release_note_info, repo_info
            )
            response = llm.invoke([system_prompt, human_prompt])
            print("✓ [generate] Release note documentation generated")

        print(f"→ Writing release note info to {file_name}")
        write_file(
            file_name,
            response.content,
        )
        print("✓ Release note documentation written successfully")

    def repo_info_update_log_node(self, state: dict):
        log_state(state)
        # this node need to generate the doc about the repo commit, pr, and release_note date this time
        # with this doc, next time run agent, can compare date between the date in this doc to update current doc
        print("→ Processing repo_info_update_log_node")
        # 1. get relevant info
        basic_info_for_repo = state.get("basic_info_for_repo", {})
        commit_info = state.get("commit_info", {})
        pr_info = state.get("pr_info", {})
        release_note_info = state.get("release_note_info", {})
        # 2. extract the date from info and convert to string
        commits = commit_info.get("commits", [])
        commit_date = str(commits[0].get("date", "N/A")) if commits else "N/A"
        commit_sha = str(commits[0].get("sha", "N/A")) if commits else "N/A"

        prs = pr_info.get("prs", [])
        pr_created_at_date = str(prs[0].get("created_at", "N/A")) if prs else "N/A"
        pr_merged_at_date = str(prs[0].get("merged_at", "N/A")) if prs else "N/A"
        pr_number = str(prs[0].get("number", "N/A")) if prs else "N/A"

        releases = release_note_info.get("releases", [])
        release_note_created_at_date = (
            str(releases[0].get("created_at", "N/A")) if releases else "N/A"
        )
        release_note_published_at_date = (
            str(releases[0].get("published_at", "N/A")) if releases else "N/A"
        )
        release_tag_name = (
            str(releases[0].get("tag_name", "N/A")) if releases else "N/A"
        )
        # 3. merge into a json doc
        update_log_info = {
            "log_date": str(basic_info_for_repo.get("date", "N/A")),
            "commit_sha": commit_sha,
            "commit_date": commit_date,
            "pr_number": pr_number,
            "pr_created_at_date": pr_created_at_date,
            "pr_merged_at_date": pr_merged_at_date,
            "release_tag_name": release_tag_name,
            "release_note_created_at_date": release_note_created_at_date,
            "release_note_published_at_date": release_note_published_at_date,
        }
        print("✓ Repo update log documentation generated")
        # write to file
        owner = basic_info_for_repo.get("owner", "")
        repo = basic_info_for_repo.get("repo", "")
        date = basic_info_for_repo.get("date", "")
        wiki_root_path = basic_info_for_repo.get("wiki_root_path", "./.wikis")
        # wiki_path = f"{wiki_root_path}/{owner}_{repo}"
        wiki_path = os.path.join(wiki_root_path, f"{owner}_{repo}")
        os.makedirs(wiki_path, exist_ok=True)

        write_file(
            os.path.join(wiki_path, f"repo_update_log.json"),
            json.dumps(update_log_info, indent=2),
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
        repo_info_builder.add_node(
            "commit_doc_generation_node", self.commit_doc_generation_node
        )
        repo_info_builder.add_node(
            "pr_doc_generation_node", self.pr_doc_generation_node
        )
        repo_info_builder.add_node(
            "release_note_doc_generation_node", self.release_note_doc_generation_node
        )
        repo_info_builder.add_node(
            "repo_info_update_log_node", self.repo_info_update_log_node
        )
        repo_info_builder.add_edge(START, "commit_doc_generation_node")
        repo_info_builder.add_edge(START, "pr_doc_generation_node")
        repo_info_builder.add_edge(START, "release_note_doc_generation_node")
        repo_info_builder.add_edge(START, "repo_info_overview_node")
        repo_info_builder.add_edge(
            "repo_info_overview_node", "overview_doc_generation_node"
        )
        repo_info_builder.add_edge(
            "overview_doc_generation_node", "repo_info_update_log_node"
        )
        repo_info_builder.add_edge(
            "commit_doc_generation_node", "repo_info_update_log_node"
        )
        repo_info_builder.add_edge(
            "pr_doc_generation_node", "repo_info_update_log_node"
        )
        repo_info_builder.add_edge(
            "release_note_doc_generation_node", "repo_info_update_log_node"
        )
        return repo_info_builder.compile(checkpointer=True)

    def get_graph(self):
        return self.graph
