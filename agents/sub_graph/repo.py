from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START
from langchain_core.messages import HumanMessage, SystemMessage
from textwrap import dedent
import json
import os

from utils.file import write_file, read_file, resolve_path
from .utils import log_state
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
        # this func is to get the system prompt for the repo info subgraph
        return SystemMessage(
            content=dedent(
                """
                You are an expert technical documentation writer for software repositories.

                Your job is to turn repository-related data (commits, PRs, releases, existing docs, etc.) into clear, structured **Markdown documentation** for developers.

                Global rules:
                - Use Markdown headings, lists, and tables when helpful
                - Be concise but informative; prefer summaries over raw dumps
                - Highlight important changes, impact, and patterns
                - Organize content into logical sections with clear titles
                - When diagrams are requested, use valid Mermaid code blocks (```mermaid ... ```).

                Critical output rules:
                - Output ONLY the final markdown document, starting directly with the title line
                - Do NOT include explanations, comments, or meta-text about what you are doing
                - Do NOT include phrases like "Here is", "Of course", "I will", etc.
                - Do NOT wrap the result in quotes, JSON, or any extra formatting
                """
            ).strip(),
        )

    @staticmethod
    def _get_overview_doc_prompt(repo_info: dict, doc_contents: str) -> HumanMessage:
        # this func is to generate a prompt for overall repository documentation
        repo_name = repo_info.get("repo", "")
        owner = repo_info.get("owner", "")

        return HumanMessage(
            content=dedent(
                f"""
                Generate overall documentation for repository '{owner}/{repo_name}'.

                Based on:
                - Repository information:
                {repo_info}

                - Documentation source files (content below):
                {doc_contents}

                Write a markdown document that includes:
                1. Overview of repository structure and key components
                2. Summary of existing documentation coverage
                3. Main documentation gaps and areas for improvement
                4. At least one Mermaid diagram for architecture or component relationships
                """
            ).strip(),
        )

    @staticmethod
    def _get_updated_overview_doc_prompt(
        repo_info: dict,
        commit_info: dict,
        pr_info: dict,
        release_note_info: dict,
        overview_doc_path: str,
    ) -> HumanMessage:
        # this func is to generate a prompt for updated overall repository documentation
        repo_name = repo_info.get("repo", "")
        owner = repo_info.get("owner", "")
        return HumanMessage(
            content=dedent(
                f"""
                Update the existing repository overview documentation for '{owner}/{repo_name}'.

                Existing doc path (for reference only, do NOT mention the path in output): {overview_doc_path}

                Use the updated data below:
                - Commit information:
                {commit_info}

                - PR information:
                {pr_info}

                - Release note information:
                {release_note_info}

                Update the markdown so it reflects the latest changes, keeps a clear structure,
                and updates or adds Mermaid diagrams when architecture or relationships change.
                """
            ).strip(),
        )

    @staticmethod
    def _generate_commit_doc_prompt(commit_info: dict, repo_info: dict) -> HumanMessage:
        # this func is to generate a prompt for commit documentation
        repo_name = repo_info.get("repo", "")
        owner = repo_info.get("owner", "")

        return HumanMessage(
            content=dedent(
                f"""
                Generate commit history documentation for repository '{owner}/{repo_name}'.

                From the commit data below, write:
                - A summary of key commits and their purposes
                - Notable patterns in development activity (frequency, authors, areas of change)
                - Major features or fixes from recent commits
                - Any useful trends or insights for maintainers

                Commit data:
                {commit_info}
                """
            ).strip(),
        )

    @staticmethod
    def _generate_updated_commit_doc_prompt(
        commit_info: dict, repo_info: dict, commit_doc_path: str
    ) -> HumanMessage:
        # this func is to generate a prompt for updated commit documentation
        repo_name = repo_info.get("repo", "")
        owner = repo_info.get("owner", "")
        return HumanMessage(
            content=dedent(
                f"""
                Update the commit history documentation for repository '{owner}/{repo_name}'.

                Existing doc path (for reference only, do NOT mention the path in output): {commit_doc_path}

                Use the updated commit data below to adjust or extend the existing markdown:
                {commit_info}
                """
            ).strip(),
        )

    @staticmethod
    def _generate_pr_doc_prompt(pr_info: dict, repo_info: dict) -> HumanMessage:
        # this func is to generate a prompt for PR documentation
        repo_name = repo_info.get("repo", "")
        owner = repo_info.get("owner", "")

        return HumanMessage(
            content=dedent(
                f"""
                Generate pull request (PR) analysis documentation for repository '{owner}/{repo_name}'.

                From the PR data below, write:
                - Overview of recent and important PRs
                - Common themes or focus areas
                - Review and merge patterns
                - Contributor and collaboration insights
                - Any pending or long-lived PRs that need attention

                PR data:
                {pr_info}
                """
            ).strip(),
        )

    @staticmethod
    def _generate_updated_pr_doc_prompt(
        pr_info: dict, repo_info: dict, pr_doc_path: str
    ) -> HumanMessage:
        # this func is to generate a prompt for updated PR documentation
        repo_name = repo_info.get("repo", "")
        owner = repo_info.get("owner", "")
        return HumanMessage(
            content=dedent(
                f"""
                Update the pull request (PR) analysis documentation for repository '{owner}/{repo_name}'.

                Existing doc path (for reference only, do NOT mention the path in output): {pr_doc_path}

                Use the updated PR data below to update or extend the existing markdown:
                {pr_info}
                """
            ).strip(),
        )

    @staticmethod
    def _generate_release_note_doc_prompt(
        release_note_info: dict, repo_info: dict
    ) -> HumanMessage:
        # this func is to generate a prompt for release note documentation
        repo_name = repo_info.get("repo", "")
        owner = repo_info.get("owner", "")

        return HumanMessage(
            content=dedent(
                f"""
                Generate release history documentation for repository '{owner}/{repo_name}'.

                From the release data below, write:
                - Timeline of major releases and key features
                - Breaking changes and any migration guidance
                - Performance improvements and bug fixes across versions
                - Deprecations and hints about future direction
                - Patterns in release frequency and versioning

                Release data:
                {release_note_info}
                """
            ).strip(),
        )

    @staticmethod
    def _generate_updated_release_note_doc_prompt(
        release_note_info: dict, repo_info: dict, release_note_doc_path: str
    ) -> HumanMessage:
        # this func is to generate a prompt for updated release note documentation
        repo_name = repo_info.get("repo", "")
        owner = repo_info.get("owner", "")
        return HumanMessage(
            content=dedent(
                f"""
                Update the release history documentation for repository '{owner}/{repo_name}'.

                Existing doc path (for reference only, do NOT mention the path in output): {release_note_doc_path}

                Use the updated release data below to update or extend the existing markdown:
                {release_note_info}
                """
            ).strip(),
        )

    def repo_info_overview_node(self, state: dict):
        # this node is to process the overall repository documentation
        log_state(state)
        print("→ [repo_info_overview_node] processing overview documentation...")
        basic_info_for_repo = state.get("basic_info_for_repo", {})
        commits_updated = basic_info_for_repo.get("commits_updated", False)
        prs_updated = basic_info_for_repo.get("prs_updated", False)
        releases_updated = basic_info_for_repo.get("releases_updated", False)
        commit_info = state.get("commit_info", {})
        pr_info = state.get("pr_info", {})
        release_note_info = state.get("release_note_info", {})
        repo_info = basic_info_for_repo.get("repo_info", {})
        repo_structure = basic_info_for_repo.get("repo_structure", [])
        overview_doc_path = basic_info_for_repo.get("overview_doc_path", "")
        os.makedirs(os.path.dirname(overview_doc_path), exist_ok=True)

        system_prompt = self._get_system_prompt()
        llm = CONFIG.get_llm()
        overview_info = ""
        if commits_updated or prs_updated or releases_updated:
            print(
                "   → [repo_info_overview_node] generating overall documentation with updates..."
            )
            human_prompt = self._get_updated_overview_doc_prompt(
                repo_info, commit_info, pr_info, release_note_info, overview_doc_path
            )
            response = llm.invoke([system_prompt, human_prompt])
            overview_info = response.content
            print("   → [repo_info_overview_node] overall documentation updated")
        elif os.path.exists(overview_doc_path):
            print(
                "   → [repo_info_overview_node] overall documentation is up-to-date. No updates needed."
            )
            existing_content = read_file(overview_doc_path) or ""
            overview_info = existing_content
        else:
            print(
                "   → [repo_info_overview_node] generating overall documentation without updates..."
            )
            doc_source_files = [
                file_path for file_path in repo_structure if file_path.endswith(".md")
            ]
            doc_contents = []
            for file_path in doc_source_files:
                file_path = resolve_path(file_path)
                content = read_file(file_path)
                if content:
                    doc_contents.append(f"# Source File: {file_path}\n\n{content}\n\n")
            combined_doc_content = "\n".join(doc_contents)
            human_prompt = self._get_overview_doc_prompt(
                repo_info, combined_doc_content
            )
            response = llm.invoke([system_prompt, human_prompt])
            overview_info = response.content
            print("   → [repo_info_overview_node] overall documentation generated")

        print(
            "✓ [repo_info_overview_node] overall documentation processed successfully"
        )
        return {
            "overview_info": overview_info,
        }

    def overview_doc_generation_node(self, state: dict):
        # this node is to generate the overview documentation
        log_state(state)
        print("→ [overview_doc_generation_node] processing overview documentation...")
        basic_info_for_repo = state.get("basic_info_for_repo", {})
        overview_doc_path = basic_info_for_repo.get("overview_doc_path", "")
        overview_info = state.get("overview_info", "")
        if os.path.exists(overview_doc_path):
            print(
                "   → [overview_doc_generation_node] overview documentation is up-to-date. No updates needed."
            )
            return
        print(f"→ Writing overview info to {overview_doc_path}")
        write_file(overview_doc_path, overview_info)
        print(
            "✓ [overview_doc_generation_node] overview documentation processed successfully"
        )

    def commit_doc_generation_node(self, state: dict):
        # this node is to process the commit documentation
        log_state(state)
        print("→ [commit_doc_generation_node] processing commit documentation...")
        basic_info_for_repo = state.get("basic_info_for_repo", {})
        repo_info = state.get("repo_info", {})
        commit_info = state.get("commit_info", {})
        commits_updated = basic_info_for_repo.get("commits_updated", False)
        commit_doc_path = basic_info_for_repo.get("commit_doc_path", "")
        os.makedirs(os.path.dirname(commit_doc_path), exist_ok=True)

        system_prompt = self._get_system_prompt()
        llm = CONFIG.get_llm()
        if commits_updated:
            print(
                "   → [commit_doc_generation_node] generating commit documentation with updates..."
            )
            human_prompt = self._generate_updated_commit_doc_prompt(
                commit_info, repo_info, commit_doc_path
            )
            response = llm.invoke([system_prompt, human_prompt])
            print("   → [commit_doc_generation_node] commit documentation updated")
        elif os.path.exists(commit_doc_path):
            print(
                "   → [commit_doc_generation_node] commit documentation is up-to-date. No updates needed."
            )
            return
        else:
            print(
                "   → [commit_doc_generation_node] generating commit documentation without updates..."
            )
            human_prompt = self._generate_commit_doc_prompt(commit_info, repo_info)
            response = llm.invoke([system_prompt, human_prompt])
            print("   → [commit_doc_generation_node] commit documentation generated")

        print(
            f"   → [commit_doc_generation_node] writing commit info to {commit_doc_path}"
        )
        write_file(
            commit_doc_path,
            response.content,
        )
        print(
            "✓ [commit_doc_generation_node] commit documentation processed successfully"
        )

    def pr_doc_generation_node(self, state: dict):
        # this node is to process the PR documentation
        log_state(state)
        print("→ [pr_doc_generation_node] processing PR documentation...")
        basic_info_for_repo = state.get("basic_info_for_repo", {})
        pr_info = state.get("pr_info", {})
        repo_info = basic_info_for_repo.get("repo_info", {})
        prs_updated = basic_info_for_repo.get("prs_updated", False)
        pr_doc_path = basic_info_for_repo.get("pr_doc_path", "")
        os.makedirs(os.path.dirname(pr_doc_path), exist_ok=True)

        system_prompt = self._get_system_prompt()
        llm = CONFIG.get_llm()
        if prs_updated:
            print(
                "   → [pr_doc_generation_node] generating PR documentation with updates..."
            )
            human_prompt = self._generate_updated_pr_doc_prompt(
                pr_info, repo_info, pr_doc_path
            )
            response = llm.invoke([system_prompt, human_prompt])
            print("   → [pr_doc_generation_node] PR documentation updated")
        elif os.path.exists(pr_doc_path):
            print(
                "   → [pr_doc_generation_node] PR documentation is up-to-date. No updates needed."
            )
            return
        else:
            print(
                "   → [pr_doc_generation_node] generating PR documentation without updates..."
            )
            human_prompt = self._generate_pr_doc_prompt(pr_info, repo_info)
            response = llm.invoke([system_prompt, human_prompt])
            print("   → [pr_doc_generation_node] PR documentation generated")

        print(f"   → [pr_doc_generation_node] writing PR info to {pr_doc_path}")
        write_file(
            pr_doc_path,
            response.content,
        )
        print("✓ [pr_doc_generation_node] PR documentation processed successfully")

    def release_note_doc_generation_node(self, state: dict):
        # this node is to process the release note documentation
        log_state(state)
        print(
            "→ [release_note_doc_generation_node] processing release note documentation..."
        )
        basic_info_for_repo = state.get("basic_info_for_repo", {})
        releases_updated = basic_info_for_repo.get("releases_updated", False)
        release_note_info = state.get("release_note_info", {})
        repo_info = basic_info_for_repo.get("repo_info", {})
        release_note_doc_path = basic_info_for_repo.get("release_note_doc_path", "")
        os.makedirs(os.path.dirname(release_note_doc_path), exist_ok=True)

        system_prompt = self._get_system_prompt()
        llm = CONFIG.get_llm()
        if releases_updated:
            print(
                "   → [release_note_doc_generation_node] generating release note documentation with updates..."
            )
            human_prompt = self._generate_updated_release_note_doc_prompt(
                release_note_info, repo_info, release_note_doc_path
            )
            response = llm.invoke([system_prompt, human_prompt])
            print(
                "   → [release_note_doc_generation_node] release note documentation updated"
            )
        elif os.path.exists(release_note_doc_path):
            print(
                "   → [release_note_doc_generation_node] Release note documentation is up-to-date. No updates needed."
            )
            return
        else:
            print(
                "   → [release_note_doc_generation_node] Generating release note documentation without updates."
            )
            human_prompt = self._generate_release_note_doc_prompt(
                release_note_info, repo_info
            )
            response = llm.invoke([system_prompt, human_prompt])
            print(
                "   → [release_note_doc_generation_node] Release note documentation generated"
            )

        print(
            f"→ [release_note_doc_generation_node] writing release note info to {release_note_doc_path}"
        )
        write_file(
            release_note_doc_path,
            response.content,
        )
        print(
            "✓ [release_note_doc_generation_node] release note documentation processed successfully"
        )

    def repo_info_update_log_node(self, state: dict):
        # this node is to process the repository information update log
        log_state(state)
        print("→ [repo_info_update_log_node] Processing repo_info_update_log_node")
        basic_info_for_repo = state.get("basic_info_for_repo", {})
        commit_info = state.get("commit_info", {})
        pr_info = state.get("pr_info", {})
        release_note_info = state.get("release_note_info", {})

        def get_log_info(
            commit_info: dict, pr_info: dict, release_note_info: dict
        ) -> dict:
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
            return {
                "commit_date": commit_date,
                "commit_sha": commit_sha,
                "pr_number": pr_number,
                "pr_created_at_date": pr_created_at_date,
                "pr_merged_at_date": pr_merged_at_date,
                "release_tag_name": release_tag_name,
                "release_note_created_at_date": release_note_created_at_date,
                "release_note_published_at_date": release_note_published_at_date,
            }

        log_info = get_log_info(commit_info, pr_info, release_note_info)
        update_log_info = {
            "log_date": str(basic_info_for_repo.get("date", "N/A")),
            **log_info,
        }
        print(
            "✓ [repo_info_update_log_node] repo_info_update_log_node processed successfully"
        )
        return {
            "update_log_info": update_log_info,
        }

    def repo_info_update_log_doc_generation_node(self, state: dict):
        # this node is to generate the repository information update log documentation
        log_state(state)
        print(
            "→ [repo_info_update_log_doc_generation_node] Processing repo_info_update_log_doc_generation_node..."
        )
        basic_info_for_repo = state.get("basic_info_for_repo", {})
        update_log_info = state.get("update_log_info", {})
        repo_update_log_path = basic_info_for_repo.get("repo_update_log_path", "")
        os.makedirs(os.path.dirname(repo_update_log_path), exist_ok=True)
        write_file(
            repo_update_log_path,
            json.dumps(update_log_info, indent=2, ensure_ascii=False),
        )
        print(
            "✓ [repo_info_update_log_doc_generation_node] repo_info_update_log_doc_generation_node processed successfully"
        )

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
        repo_info_builder.add_node(
            "repo_info_update_log_doc_generation_node",
            self.repo_info_update_log_doc_generation_node,
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
        repo_info_builder.add_edge(
            "repo_info_update_log_node", "repo_info_update_log_doc_generation_node"
        )
        return repo_info_builder.compile(checkpointer=True)

    def get_graph(self):
        return self.graph
