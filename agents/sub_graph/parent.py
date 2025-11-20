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
from utils.code_analyzer import (
    analyze_file_with_tree_sitter,
    format_tree_sitter_analysis_results,
    format_tree_sitter_analysis_results_to_prompt,
)
from .utils import (
    draw_graph,
    log_state,
    get_updated_commit_info,
    get_updated_pr_info,
    get_updated_release_note_info,
    get_updated_code_files,
)
from .repo import RepoInfoSubGraphBuilder
from .code import CodeAnalysisSubGraphBuilder
from config import CONFIG


# --- Parent graph ---
class ParentGraphState(TypedDict):
    owner: str
    repo: str
    platform: str
    mode: str
    max_workers: int
    date: str
    log: bool
    # repo_info_sub_graph inputs
    basic_info_for_repo: dict | None
    # repo_info_sub_graph outputs
    commit_info: dict | None
    pr_info: dict | None
    release_note_info: dict | None
    # code_analysis_sub_graph inputs
    basic_info_for_code: dict | None
    # code_analysis_sub_graph outputs
    code_analysis: dict | None


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
        log = state.get("log", False)
        repo_root_path = "./.repos"
        wiki_root_path = "./.wikis"
        repo_path = f"{repo_root_path}/{owner}_{repo}"
        wiki_path = f"{wiki_root_path}/{owner}_{repo}"

        print(f"📦 Repository: {owner}/{repo}")
        print(f"🔗 Platform: {platform}")
        print(f"⚙️ Mode: {mode}")
        print(f"👷 Max Workers: {max_workers}")
        print(f"📁 Repo Path: {repo_path}")
        print(f"📄 Wiki Path: {wiki_path}")

        repo_info = get_repo_info(owner, repo, platform=platform)
        repo_structure = get_repo_structure(repo_path)

        print(f"✓ Repository initialized")

        basic_info_for_repo = {
            "owner": owner,
            "repo": repo,
            "platform": platform,
            "mode": mode,
            "max_workers": max_workers,
            "date": date,
            "log": log,
            "repo_root_path": repo_root_path,
            "wiki_root_path": wiki_root_path,
            "repo_path": repo_path,
            "wiki_path": wiki_path,
            "repo_info": repo_info,
            "repo_structure": repo_structure,
            "commits_updated": False,
            "prs_updated": False,
            "releases_updated": False,
        }
        basic_info_for_code = {
            "owner": owner,
            "repo": repo,
            "platform": platform,
            "mode": mode,
            "max_workers": max_workers,
            "date": date,
            "log": log,
            "repo_root_path": repo_root_path,
            "wiki_root_path": wiki_root_path,
            "repo_path": repo_path,
            "wiki_path": wiki_path,
            "repo_info": repo_info,
            "repo_structure": repo_structure,
            "code_files_updated": False,
        }

        return {
            "basic_info_for_repo": basic_info_for_repo,
            "basic_info_for_code": basic_info_for_code,
        }

    def check_update_node(self, state: ParentGraphState):
        log_state(state)
        # this node is to check whether the repo has updates since last doc generation
        # p.s. the `relevant_code_update` can be determined from file changes of commits
        print("→ Processing check_update_node...")
        owner = state.get("basic_info_for_repo", {}).get("owner", "")
        repo = state.get("basic_info_for_repo", {}).get("repo", "")
        platform = state.get("basic_info_for_repo", {}).get("platform", "github")
        repo_path = state.get("basic_info_for_repo", {}).get("repo_path", "./.repos")
        wiki_root_path = state.get("basic_info_for_repo", {}).get(
            "wiki_root_path", "./.wikis"
        )
        # get log.json path
        # e.g. f"{wiki_root_path}/{owner}_{repo}/repo_update_log.json"
        log_path = os.path.join(wiki_root_path, f"{owner}_{repo}/repo_update_log.json")

        # Execute get_updated_commit_info, get_updated_pr_info, and get_updated_release_note_info in parallel
        print("→ Fetching updates in parallel (commits, PRs, releases)...")
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all three tasks concurrently
            commit_future = executor.submit(
                get_updated_commit_info, owner, repo, platform, log_path
            )
            pr_future = executor.submit(
                get_updated_pr_info, owner, repo, platform, log_path
            )
            release_future = executor.submit(
                get_updated_release_note_info, owner, repo, platform, log_path
            )

            # Wait for all results
            commits_updated, updated_commit_info = commit_future.result()
            prs_updated, updated_pr_info = pr_future.result()
            releases_updated, updated_release_note_info = release_future.result()

        # Report commit update status
        if commits_updated:
            print(
                f"✓ Detected {updated_commit_info.get('commits_count', 0)} updated commits since last check"
            )
        else:
            print(f"✓ No new commits detected since last check")
        # print(f"DEBUG: Updated commits: {updated_commit_info.get('commits', [])}")

        # Report PR update status
        if prs_updated:
            print(
                f"✓ Detected {updated_pr_info.get('prs_count', 0)} updated PRs since last check"
            )
        else:
            print(f"✓ No new PRs detected since last check")
        # print(f"DEBUG: Updated PRs: {updated_pr_info.get('prs', [])}")

        # Report release update status
        if releases_updated:
            print(
                f"✓ Detected {updated_release_note_info.get('releases_count', 0)} updated release notes since last check"
            )
        else:
            print(f"✓ No new release notes detected since last check")
        # print(
        #     f"DEBUG: Updated release notes: {updated_release_note_info.get('releases', [])}"
        # )

        # call `get_updated_code_files` which depends on commit info
        code_files_updated, updated_code_files = get_updated_code_files(
            repo_path, updated_commit_info.get("commits", []), commits_updated
        )
        if code_files_updated:
            print(
                f"✓ Detected {len(updated_code_files)} updated code files from commits"
            )
        else:
            print(f"✓ No updated code files detected from commits")
        # print(f"DEBUG: Updated code files: {updated_code_files}")

        updated_basic_info_for_repo = {}
        updated_basic_info_for_code = {}

        if commits_updated:
            print(f"✓ New commits detected since last documentation generation")
            updated_basic_info_for_repo = {
                "commits_updated": commits_updated,
            }
        if prs_updated:
            print(f"✓ New PRs detected since last documentation generation")
            updated_basic_info_for_repo = {
                **updated_basic_info_for_repo,
                "prs_updated": prs_updated,
            }
        if releases_updated:
            print(f"✓ New release notes detected since last documentation generation")
            updated_basic_info_for_repo = {
                **updated_basic_info_for_repo,
                "releases_updated": releases_updated,
            }
        if code_files_updated:
            print(f"✓ New code files detected from updated commits")
            updated_basic_info_for_code = {
                "code_files_updated": code_files_updated,
            }

        return {
            "basic_info_for_repo": {
                **state.get("basic_info_for_repo", {}),
                **updated_basic_info_for_repo,
            },
            "basic_info_for_code": {
                **state.get("basic_info_for_code", {}),
                **updated_basic_info_for_code,
            },
            "commit_info": updated_commit_info,
            "pr_info": updated_pr_info,
            "release_note_info": updated_release_note_info,
            "code_analysis": (
                {"code_files": updated_code_files}
                if code_files_updated
                else {"code_files": []}
            ),
        }

    def build(self, checkpointer):
        builder = StateGraph(ParentGraphState)
        builder.add_node("basic_info_node", self.basic_info_node)
        builder.add_node("check_update_node", self.check_update_node)
        builder.add_node("repo_info_graph", self.repo_info_graph)
        builder.add_node("code_analysis_graph", self.code_analysis_graph)
        match self.branch:
            case "all":
                builder.add_edge(START, "basic_info_node")
                builder.add_edge("basic_info_node", "check_update_node")
                builder.add_edge("check_update_node", "repo_info_graph")
                builder.add_edge("check_update_node", "code_analysis_graph")
            case "code":
                builder.add_edge(START, "basic_info_node")
                builder.add_edge("basic_info_node", "check_update_node")
                builder.add_edge("check_update_node", "code_analysis_graph")
            case "repo":
                builder.add_edge(START, "basic_info_node")
                builder.add_edge("basic_info_node", "check_update_node")
                builder.add_edge("check_update_node", "repo_info_graph")
            case "check":
                builder.add_edge(START, "basic_info_node")
                builder.add_edge("basic_info_node", "check_update_node")
            case _:
                builder.add_edge(START, "basic_info_node")
                builder.add_edge("basic_info_node", "check_update_node")
                builder.add_edge("check_update_node", "repo_info_graph")
                builder.add_edge("check_update_node", "code_analysis_graph")
        return builder.compile(checkpointer=checkpointer)

    def get_graph(self):
        return self.graph

    def run(
        self, inputs: ParentGraphState, config: dict = None, count_time: bool = True
    ):
        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if config is None:
            config = {"configurable": {"thread_id": f"wiki-generation-{date}"}}
        print("Running parent graph with inputs:", inputs)
        start_time = time.time()
        for chunk in self.graph.stream(
            inputs,
            config=config,
            subgraphs=True,
        ):
            pass

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("\n" + "=" * 80)
        print("✅ Graph execution completed successfully!")
        print("=" * 80)
        if count_time:
            print(f"Total execution time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    # use "uv run python -m agents.sub_graph.parent" to run this file
    parent_graph_builder = ParentGraphBuilder(branch_mode="all")
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parent_graph_builder.run(
        inputs={
            "owner": "squatting-at-home123",
            "repo": "back-puppet",
            "platform": "gitee",
            "mode": "fast",  # "fast" or "smart"
            "max_workers": 50,  # 20 worker -> 3 - 4 minutes
            "date": date,
            "log": False,
        },
        config={
            "configurable": {
                "thread_id": f"wiki-generation-{date}",
            }
        },
        count_time=True,
    )
