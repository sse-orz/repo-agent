from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START
from langchain_core.messages import HumanMessage, SystemMessage
import json
import os

from utils.file import write_file, read_file, resolve_path
from .utils import log_state, call_llm
from .prompt import RepoPrompt
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
        repo_path = basic_info_for_repo.get("repo_path", "")
        repo_structure = basic_info_for_repo.get("repo_structure", [])

        overview_doc_path = basic_info_for_repo.get("overview_doc_path", "")
        os.makedirs(os.path.dirname(overview_doc_path), exist_ok=True)

        system_prompt = RepoPrompt._get_system_prompt()
        if commits_updated or prs_updated or releases_updated:
            print(
                "   → [repo_info_overview_node] generating overall documentation with updates..."
            )
            overview_info = call_llm(
                [
                    system_prompt,
                    RepoPrompt._get_updated_overview_doc_prompt(
                        repo_info,
                        commit_info,
                        pr_info,
                        release_note_info,
                        overview_doc_path,
                    ),
                ]
            )
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
            # TODO: need to fix bug here: not all md files are deserving to be the documentation source files
            # add prefix path to doc source files
            doc_source_files = [
                os.path.join(repo_path, file_path)
                for file_path in repo_structure
                if file_path.endswith(".md")
            ]
            doc_contents = []
            for file_path in doc_source_files:
                file_path = resolve_path(file_path)
                content = read_file(file_path)
                if content:
                    doc_contents.append(f"# Source File: {file_path}\n\n{content}\n\n")
            combined_doc_content = "\n".join(doc_contents)
            overview_info = call_llm(
                [
                    system_prompt,
                    RepoPrompt._get_overview_doc_prompt(
                        repo_info, combined_doc_content
                    ),
                ]
            )
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

        system_prompt = RepoPrompt._get_system_prompt()
        if commits_updated:
            print(
                "   → [commit_doc_generation_node] generating commit documentation with updates..."
            )
            commit_info = call_llm(
                [
                    system_prompt,
                    RepoPrompt._get_updated_commit_doc_prompt(
                        commit_info, repo_info, commit_doc_path
                    ),
                ]
            )
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
            commit_info = call_llm(
                [
                    system_prompt,
                    RepoPrompt._get_commit_doc_prompt(commit_info, repo_info),
                ]
            )
            print("   → [commit_doc_generation_node] commit documentation generated")

        print(
            f"   → [commit_doc_generation_node] writing commit info to {commit_doc_path}"
        )
        write_file(
            commit_doc_path,
            commit_info,
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

        system_prompt = RepoPrompt._get_system_prompt()
        if prs_updated:
            print(
                "   → [pr_doc_generation_node] generating PR documentation with updates..."
            )
            pr_info = call_llm(
                [
                    system_prompt,
                    RepoPrompt._get_updated_pr_doc_prompt(
                        pr_info, repo_info, pr_doc_path
                    ),
                ]
            )
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
            pr_info = call_llm(
                [system_prompt, RepoPrompt._get_pr_doc_prompt(pr_info, repo_info)]
            )
            print("   → [pr_doc_generation_node] PR documentation generated")

        print(f"   → [pr_doc_generation_node] writing PR info to {pr_doc_path}")
        write_file(
            pr_doc_path,
            pr_info,
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

        system_prompt = RepoPrompt._get_system_prompt()
        if releases_updated:
            print(
                "   → [release_note_doc_generation_node] generating release note documentation with updates..."
            )
            release_note_info = call_llm(
                [
                    system_prompt,
                    RepoPrompt._get_updated_release_note_doc_prompt(
                        release_note_info, repo_info, release_note_doc_path
                    ),
                ]
            )
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
            release_note_info = call_llm(
                [
                    system_prompt,
                    RepoPrompt._get_release_note_doc_prompt(
                        release_note_info, repo_info
                    ),
                ]
            )
            print(
                "   → [release_note_doc_generation_node] Release note documentation generated"
            )

        print(
            f"→ [release_note_doc_generation_node] writing release note info to {release_note_doc_path}"
        )
        write_file(
            release_note_doc_path,
            release_note_info,
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
