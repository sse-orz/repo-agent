from typing_extensions import TypedDict
from langgraph.graph.state import StateGraph, START
from concurrent.futures import ThreadPoolExecutor
import json
import os

from utils.file import write_file, read_file, resolve_path
from .utils import (
    log_state,
    call_llm,
    curl_content,
    get_md_files,
    count_tokens,
    get_llm_max_tokens,
)
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

    def _select_doc_dirs(
        self,
        repo_info: dict,
        basic_repo_structure: list[str],
        max_num_dirs: int = 100,
    ) -> list[str]:
        # call llm to select the dirs that are deserving to be the documentation source dirs
        if not basic_repo_structure:
            return []

        raw_output = call_llm(
            [
                RepoPrompt._get_system_prompt_for_repo(),
                RepoPrompt._get_human_prompt_for_repo_overview_dir_selection(
                    repo_info, basic_repo_structure, max_num_dirs
                ),
            ]
        )
        chosen_dirs: list[str] = []
        if raw_output:
            for line in str(raw_output).splitlines():
                path = line.strip()
                if not path:
                    continue
                # only accept the dirs that are in the basic_repo_structure and not already chosen
                if path in basic_repo_structure and path not in chosen_dirs:
                    chosen_dirs.append(path)

        return chosen_dirs

    def _generate_single_overview_section(
        self, repo_info: dict, selected_md_file: str
    ) -> str:
        # count the tokens of the selected_md_file
        # if the tokens is greater than 5% of the max tokens, call llm to generate the overview section for the single module
        # otherwise, return the content of the selected_md_file
        doc_contents = read_file(selected_md_file) or ""
        doc_contents_tokens = count_tokens(doc_contents)
        if doc_contents_tokens > get_llm_max_tokens(compress_ratio=0.05):
            result = call_llm(
                [
                    RepoPrompt._get_system_prompt_for_repo(),
                    RepoPrompt._get_human_prompt_for_repo_single_module_overview(
                        repo_info, selected_md_file, doc_contents
                    ),
                ]
            )
        else:
            result = doc_contents

        return result

    def _build_overview_doc(
        self,
        repo_info: dict,
        repo_path: str,
        basic_repo_structure: list[str],
        selected_doc_dirs: list[str],
        max_workers: int,
    ) -> str:
        # use threadpool to generate the overview sections for the selected dirs in parallel
        # combine the overview sections into the overview documentation
        if not selected_doc_dirs:
            return ""

        # collect the md files in the selected dirs (deal with paths correctly)
        selected_md_files = []
        for selected_doc_dir in selected_doc_dirs:
            doc_dir_abs_path = os.path.join(repo_path, selected_doc_dir)
            doc_files = get_md_files(doc_dir_abs_path)
            for doc_file in doc_files:
                selected_md_files.append(os.path.join(doc_dir_abs_path, doc_file))

        # generate the overview sections for the selected md files in parallel
        effective_workers = min(max_workers, len(selected_md_files))
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = [
                executor.submit(
                    self._generate_single_overview_section,
                    repo_info=repo_info,
                    selected_md_file=selected_md_file,
                )
                for selected_md_file in selected_md_files
            ]
            module_sections = [f.result() for f in futures]

        module_sections = [s.strip() for s in module_sections if s and s.strip()]
        if not module_sections:
            return ""

        combined_module_overview = "\n\n".join(module_sections)

        # call llm to generate the overview documentation for the selected dirs
        final_overview = call_llm(
            [
                RepoPrompt._get_system_prompt_for_repo(),
                RepoPrompt._get_human_prompt_for_repo_overview_doc(
                    repo_info,
                    combined_module_overview,
                    basic_repo_structure,
                ),
            ]
        )
        return final_overview or combined_module_overview

    def repo_info_overview_node(self, state: dict):
        # this node is to process the overall repository documentation
        log_state(state)
        print("→ [repo_info_overview_node] processing overview documentation...")
        basic_info_for_repo = state.get("basic_info_for_repo", {})
        commits_updated = basic_info_for_repo.get("commits_updated", False)
        commit_info = state.get("commit_info", {})
        repo_info = basic_info_for_repo.get("repo_info", {})
        repo_path = basic_info_for_repo.get("repo_path", "")
        basic_repo_structure = basic_info_for_repo.get("basic_repo_structure", []) or []
        max_workers = basic_info_for_repo.get("max_workers", 10)

        overview_doc_path = basic_info_for_repo.get("overview_doc_path", "")
        os.makedirs(os.path.dirname(overview_doc_path), exist_ok=True)

        result_info = ""
        if commits_updated:
            print(
                "   → [repo_info_overview_node] generating overall documentation with updates..."
            )
            result_info = call_llm(
                [
                    RepoPrompt._get_system_prompt_for_repo(),
                    RepoPrompt._get_human_prompt_for_repo_updated_overview_doc(
                        repo_info,
                        commit_info,
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
            result_info = existing_content
        else:
            print(
                "   → [repo_info_overview_node] generating overall documentation without updates..."
            )
            selected_doc_dirs = self._select_doc_dirs(
                repo_info=repo_info,
                basic_repo_structure=basic_repo_structure,
                max_num_dirs=100,
            )
            result_info = self._build_overview_doc(
                repo_info=repo_info,
                repo_path=repo_path,
                basic_repo_structure=basic_repo_structure,
                selected_doc_dirs=selected_doc_dirs,
                max_workers=max_workers,
            )
            print("   → [repo_info_overview_node] overall documentation generated")

        print(
            "✓ [repo_info_overview_node] overall documentation processed successfully"
        )
        return {
            "overview_info": result_info,
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

    def _generate_single_commit_section(
        self,
        single_commit: dict,
        commit_repo_info: dict,
    ) -> str:
        return call_llm(
            [
                RepoPrompt._get_system_prompt_for_repo(),
                RepoPrompt._get_human_prompt_for_repo_single_commit_doc(
                    single_commit, commit_repo_info
                ),
            ]
        )

    def _build_commit_doc(
        self,
        commit_info: dict,
        max_workers: int,
        prefix_content: str | None = None,
    ) -> str:
        commits = commit_info.get("commits", []) or []
        commit_repo_info = commit_info.get("repo_info", {})
        if not commits:
            return prefix_content or ""

        effective_workers = min(max_workers, len(commits))
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = [
                executor.submit(
                    self._generate_single_commit_section,
                    single_commit=commit,
                    commit_repo_info=commit_repo_info,
                )
                for commit in commits
            ]
            sections = [f.result() for f in futures]

        combined = "\n\n".join(section.strip() for section in sections if section)

        parts: list[str] = []
        if prefix_content:
            parts.append(prefix_content.rstrip())
        if combined:
            parts.append(combined)
        return "\n\n".join(parts)

    def commit_doc_generation_node(self, state: dict):
        # this node is to process the commit documentation
        log_state(state)
        print("→ [commit_doc_generation_node] processing commit documentation...")
        basic_info_for_repo = state.get("basic_info_for_repo", {})
        commit_info = state.get("commit_info", {})
        commits_updated = basic_info_for_repo.get("commits_updated", False)
        commit_doc_path = basic_info_for_repo.get("commit_doc_path", "")
        max_workers = basic_info_for_repo.get("max_workers", 10)
        os.makedirs(os.path.dirname(commit_doc_path), exist_ok=True)

        result_info = ""

        if commits_updated:
            print(
                "   → [commit_doc_generation_node] generating commit documentation for updated commits in parallel..."
            )
            existing_content = ""
            if os.path.exists(commit_doc_path):
                existing_content = read_file(commit_doc_path) or ""
                print(
                    "   → [commit_doc_generation_node] existing commit documentation found, new sections will be appended."
                )
            result_info = self._build_commit_doc(
                commit_info=commit_info,
                max_workers=max_workers,
                prefix_content=existing_content,
            )
            print("   → [commit_doc_generation_node] commit documentation updated")
        elif os.path.exists(commit_doc_path):
            print(
                "   → [commit_doc_generation_node] commit documentation is up-to-date. No updates needed."
            )
            return
        else:
            print(
                "   → [commit_doc_generation_node] generating commit documentation for all commits in parallel..."
            )
            result_info = self._build_commit_doc(
                commit_info=commit_info,
                max_workers=max_workers,
                prefix_content=None,
            )
            print("   → [commit_doc_generation_node] commit documentation generated")

        print(
            f"   → [commit_doc_generation_node] writing commit info to {commit_doc_path}"
        )
        write_file(
            commit_doc_path,
            result_info,
        )
        print(
            "✓ [commit_doc_generation_node] commit documentation processed successfully"
        )

    def _generate_single_pr_section(
        self,
        single_pr: dict,
        repo_info: dict,
    ) -> str:
        diff_url = single_pr.get("diff_url") or ""
        diff_content = ""
        if diff_url:
            try:
                diff_content = curl_content(diff_url)
            except Exception:
                diff_content = ""
        return call_llm(
            [
                RepoPrompt._get_system_prompt_for_repo(),
                RepoPrompt._get_human_prompt_for_repo_single_pr_doc(
                    single_pr, repo_info, diff_content
                ),
            ]
        )

    def _build_pr_doc(
        self,
        pr_info: dict,
        max_workers: int,
        prefix_content: str | None = None,
    ) -> str:
        prs = pr_info.get("prs", []) or []
        repo_info = pr_info.get("repo_info") or {}
        if not prs:
            return prefix_content or ""

        effective_workers = min(max_workers, len(prs))
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = [
                executor.submit(
                    self._generate_single_pr_section,
                    single_pr=pr,
                    repo_info=repo_info,
                )
                for pr in prs
            ]
            sections = [f.result() for f in futures]

        combined = "\n\n".join(section.strip() for section in sections if section)

        parts: list[str] = []
        if prefix_content:
            parts.append(prefix_content.rstrip())
        if combined:
            parts.append(combined)
        return "\n\n".join(parts)

    def pr_doc_generation_node(self, state: dict):
        # this node is to process the PR documentation
        log_state(state)
        print("→ [pr_doc_generation_node] processing PR documentation...")
        basic_info_for_repo = state.get("basic_info_for_repo", {})
        pr_info = state.get("pr_info", {})
        repo_info = basic_info_for_repo.get("repo_info", {})
        prs_updated = basic_info_for_repo.get("prs_updated", False)
        pr_doc_path = basic_info_for_repo.get("pr_doc_path", "")
        max_workers = basic_info_for_repo.get("max_workers", 10)
        os.makedirs(os.path.dirname(pr_doc_path), exist_ok=True)

        result_info = ""
        if prs_updated:
            print(
                "   → [pr_doc_generation_node] generating PR documentation for updated PRs in parallel..."
            )
            existing_content = ""
            if os.path.exists(pr_doc_path):
                existing_content = read_file(pr_doc_path) or ""
                print(
                    "   → [pr_doc_generation_node] existing PR documentation found, new sections will be appended."
                )
            # Ensure pr_info carries repo_info for prompt context
            enriched_pr_info = {**pr_info, "repo_info": repo_info}
            result_info = self._build_pr_doc(
                pr_info=enriched_pr_info,
                max_workers=max_workers,
                prefix_content=existing_content,
            )
            print("   → [pr_doc_generation_node] PR documentation updated")
        elif os.path.exists(pr_doc_path):
            print(
                "   → [pr_doc_generation_node] PR documentation is up-to-date. No updates needed."
            )
            return
        else:
            print(
                "   → [pr_doc_generation_node] generating PR documentation for all PRs in parallel..."
            )
            enriched_pr_info = {**pr_info, "repo_info": repo_info}
            result_info = self._build_pr_doc(
                pr_info=enriched_pr_info,
                max_workers=max_workers,
                prefix_content=None,
            )
            print("   → [pr_doc_generation_node] PR documentation generated")

        print(f"   → [pr_doc_generation_node] writing PR info to {pr_doc_path}")
        write_file(
            pr_doc_path,
            result_info,
        )
        print("✓ [pr_doc_generation_node] PR documentation processed successfully")

    def _generate_single_release_section(
        self,
        single_release: dict,
        repo_info: dict,
    ) -> str:
        return call_llm(
            [
                RepoPrompt._get_system_prompt_for_repo(),
                RepoPrompt._get_human_prompt_for_repo_single_release_doc(
                    single_release, repo_info
                ),
            ]
        )

    def _build_release_note_doc(
        self,
        release_note_info: dict,
        max_workers: int,
        prefix_content: str | None = None,
    ) -> str:
        releases = release_note_info.get("releases", []) or []
        repo_info = release_note_info.get("repo_info") or {}
        if not releases:
            return prefix_content or ""

        effective_workers = min(max_workers, len(releases))
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = [
                executor.submit(
                    self._generate_single_release_section,
                    single_release=release,
                    repo_info=repo_info,
                )
                for release in releases
            ]
            sections = [f.result() for f in futures]

        combined = "\n\n".join(section.strip() for section in sections if section)

        parts: list[str] = []
        if prefix_content:
            parts.append(prefix_content.rstrip())
        if combined:
            parts.append(combined)
        return "\n\n".join(parts)

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
        max_workers = basic_info_for_repo.get("max_workers", 10)
        os.makedirs(os.path.dirname(release_note_doc_path), exist_ok=True)

        result_info = ""
        if releases_updated:
            print(
                "   → [release_note_doc_generation_node] generating release note documentation for updated releases in parallel..."
            )
            existing_content = ""
            if os.path.exists(release_note_doc_path):
                existing_content = read_file(release_note_doc_path) or ""
                print(
                    "   → [release_note_doc_generation_node] existing release note documentation found, new sections will be appended."
                )
            enriched_release_note_info = {**release_note_info, "repo_info": repo_info}
            result_info = self._build_release_note_doc(
                release_note_info=enriched_release_note_info,
                max_workers=max_workers,
                prefix_content=existing_content,
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
                "   → [release_note_doc_generation_node] generating release note documentation for all releases in parallel..."
            )
            enriched_release_note_info = {**release_note_info, "repo_info": repo_info}
            result_info = self._build_release_note_doc(
                release_note_info=enriched_release_note_info,
                max_workers=max_workers,
                prefix_content=None,
            )
            print(
                "   → [release_note_doc_generation_node] release note documentation generated"
            )

        print(
            f"→ [release_note_doc_generation_node] writing release note info to {release_note_doc_path}"
        )
        write_file(
            release_note_doc_path,
            result_info,
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
