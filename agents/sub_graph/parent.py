from typing_extensions import TypedDict
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.state import StateGraph, START
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time

from utils.repo import get_repo_info
from .utils import (
    log_state,
    get_updated_commit_info,
    get_updated_pr_info,
    get_updated_release_note_info,
    get_updated_code_files,
    get_repo_structure,
    get_basic_repo_structure,
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
    ratios: dict[str, float]
    max_workers: int
    date: str
    log: bool
    repo_root_path: str | None
    wiki_root_path: str | None
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

    @staticmethod
    def _print_repository_overview(
        owner: str,
        repo: str,
        platform: str,
        mode: str,
        ratios: dict[str, float],
        max_workers: int,
        repo_root_path: str,
        wiki_root_path: str,
        repo_path: str,
        wiki_path: str,
        overview_doc_path: str,
        commit_doc_path: str,
        pr_doc_path: str,
        release_note_doc_path: str,
        repo_update_log_path: str,
        code_update_log_path: str,
    ) -> None:
        # this func is to print the repository overview
        print("\n" + "=" * 80)
        print("🚀 Starting graph execution...")
        print("=" * 80)
        print(f"📦 Repository: {owner}/{repo}")
        print(f"🔗 Platform: {platform}")
        print(f"⚙️ Mode: {mode}")
        print(f"🔢 Ratios: {ratios}")
        print(f"👷 Max Workers: {max_workers}")
        print(f"📁 Repo Root Path: {repo_root_path}")
        print(f"📁 Repo Path: {repo_path}")
        print(f"📄 Wiki Root Path: {wiki_root_path}")
        print(f"📄 Wiki Path: {wiki_path}")
        print(f"📄 Overview Doc Path: {overview_doc_path}")
        print(f"📄 Commit Doc Path: {commit_doc_path}")
        print(f"📄 PR Doc Path: {pr_doc_path}")
        print(f"📄 Release Note Doc Path: {release_note_doc_path}")
        print(f"📄 Repo Update Log Path: {repo_update_log_path}")
        print(f"📄 Code Update Log Path: {code_update_log_path}")

    def _build_basic_info_payloads(
        self,
        owner: str,
        repo: str,
        platform: str,
        mode: str,
        ratios: dict[str, float],
        max_workers: int,
        date: str,
        log: bool,
        repo_info: dict | None,
        basic_repo_structure: list[str] | None,
        repo_structure: dict | None,
        repo_root_path: str,
        wiki_root_path: str,
        repo_path: str,
        wiki_path: str,
        overview_doc_path: str,
        commit_doc_path: str,
        pr_doc_path: str,
        release_note_doc_path: str,
        repo_update_log_path: str,
        code_update_log_path: str,
    ) -> tuple[dict, dict]:
        # this func is to build the basic info payloads for repo and code
        # return basic_info_for_repo and basic_info_for_code
        common_payload = {
            "owner": owner,
            "repo": repo,
            "platform": platform,
            "mode": mode,
            "ratios": ratios,
            "max_workers": max_workers,
            "date": date,
            "log": log,
            "repo_root_path": repo_root_path,
            "wiki_root_path": wiki_root_path,
            "repo_path": repo_path,
            "wiki_path": wiki_path,
            "repo_info": repo_info,
            "basic_repo_structure": basic_repo_structure,
            "repo_structure": repo_structure,
        }

        basic_info_for_repo = {
            **common_payload,
            "commits_updated": False,
            "prs_updated": False,
            "releases_updated": False,
            "overview_doc_path": overview_doc_path,
            "commit_doc_path": commit_doc_path,
            "pr_doc_path": pr_doc_path,
            "release_note_doc_path": release_note_doc_path,
            "repo_update_log_path": repo_update_log_path,
        }
        basic_info_for_code = {
            **common_payload,
            "code_files_updated": False,
            "code_update_log_path": code_update_log_path,
        }
        return basic_info_for_repo, basic_info_for_code

    def basic_info_node(self, state: ParentGraphState):
        # this node is to initialize the basic information for the repo and code
        # log_state(state)
        print("→ [basic_info_node] initializing basic information...")
        owner = state.get("owner")
        repo = state.get("repo")
        platform = state.get("platform", "github")
        mode = state.get("mode", "fast")
        ratios = state.get(
            "ratios",
            {
                "fast": 0.25,
                "smart": 0.75,
            },
        )
        max_workers = state.get("max_workers", 10)
        date = state.get("date", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        log = state.get("log", False)
        repo_root_path = state.get("repo_root_path", "./.repos")
        wiki_root_path = state.get("wiki_root_path", "./.wikis")
        repo_dirname = f"{owner}_{repo}"
        repo_path = os.path.join(repo_root_path, repo_dirname)
        wiki_path = os.path.join(wiki_root_path, repo_dirname)
        overview_doc_path = os.path.join(wiki_path, "overview_documentation.md")
        commit_doc_path = os.path.join(wiki_path, "commit_documentation.md")
        pr_doc_path = os.path.join(wiki_path, "pr_documentation.md")
        release_note_doc_path = os.path.join(wiki_path, "release_note_documentation.md")
        repo_update_log_path = os.path.join(wiki_path, "repo_update_log.json")
        code_update_log_path = os.path.join(wiki_path, "code_update_log.json")

        self._print_repository_overview(
            owner,
            repo,
            platform,
            mode,
            ratios,
            max_workers,
            repo_root_path,
            wiki_root_path,
            repo_path,
            wiki_path,
            overview_doc_path,
            commit_doc_path,
            pr_doc_path,
            release_note_doc_path,
            repo_update_log_path,
            code_update_log_path,
        )

        repo_info = get_repo_info(owner, repo, platform=platform)
        basic_repo_structure = get_basic_repo_structure(repo_path)
        repo_structure = get_repo_structure(repo_path)

        print(f"✓ [basic_info_node] repository initialized")

        basic_info_for_repo, basic_info_for_code = self._build_basic_info_payloads(
            owner=owner,
            repo=repo,
            platform=platform,
            mode=mode,
            ratios=ratios,
            max_workers=max_workers,
            date=date,
            log=log,
            repo_info=repo_info,
            basic_repo_structure=basic_repo_structure,
            repo_structure=repo_structure,
            repo_root_path=repo_root_path,
            wiki_root_path=wiki_root_path,
            repo_path=repo_path,
            wiki_path=wiki_path,
            overview_doc_path=overview_doc_path,
            commit_doc_path=commit_doc_path,
            pr_doc_path=pr_doc_path,
            release_note_doc_path=release_note_doc_path,
            repo_update_log_path=repo_update_log_path,
            code_update_log_path=code_update_log_path,
        )

        return {
            "basic_info_for_repo": basic_info_for_repo,
            "basic_info_for_code": basic_info_for_code,
        }

    def _parallel_fetch_updates(
        self, owner: str, repo: str, platform: str, repo_path: str, log_path: str
    ):
        # this func is to fetch updates in parallel
        print("   → [check_update_node] fetching updates in parallel...")
        with ThreadPoolExecutor(max_workers=3) as executor:
            commit_future = executor.submit(
                get_updated_commit_info, owner, repo, platform, log_path
            )
            pr_future = executor.submit(
                get_updated_pr_info, owner, repo, platform, log_path
            )
            release_future = executor.submit(
                get_updated_release_note_info, owner, repo, platform, log_path
            )
            commits_updated, updated_commit_info = commit_future.result()
            prs_updated, updated_pr_info = pr_future.result()
            releases_updated, updated_release_note_info = release_future.result()
        if commits_updated:
            print(
                f"   → [check_update_node] detected {updated_commit_info.get('commits_count', 0)} updated commits since last check"
            )
        else:
            print(f"   → [check_update_node] no new commits detected since last check")
        if prs_updated:
            print(
                f"   → [check_update_node] detected {updated_pr_info.get('prs_count', 0)} updated PRs since last check"
            )
        else:
            print(f"   → [check_update_node] no new PRs detected since last check")
        if releases_updated:
            print(
                f"   → [check_update_node] detected {updated_release_note_info.get('releases_count', 0)} updated release notes since last check"
            )
        else:
            print(
                f"   → [check_update_node] no new release notes detected since last check"
            )

        code_files_updated, updated_code_files = get_updated_code_files(
            repo_path, updated_commit_info.get("commits", []), commits_updated
        )
        if code_files_updated:
            print(
                f"   → [check_update_node] detected {len(updated_code_files)} updated code files from commits"
            )
        else:
            print(
                f"   → [check_update_node] no updated code files detected from commits"
            )
        return (
            commits_updated,
            prs_updated,
            releases_updated,
            code_files_updated,
            updated_commit_info,
            updated_pr_info,
            updated_release_note_info,
            updated_code_files,
        )

    def check_update_node(self, state: ParentGraphState):
        # this node is to check whether the repo has updates since last doc generation
        log_state(state)
        print("→ [check_update_node] processing updates...")
        basic_info_for_repo = state.get("basic_info_for_repo", {})
        owner = basic_info_for_repo.get("owner", "")
        repo = basic_info_for_repo.get("repo", "")
        platform = basic_info_for_repo.get("platform", "github")
        repo_root_path = basic_info_for_repo.get("repo_root_path", "./.repos")
        wiki_root_path = basic_info_for_repo.get("wiki_root_path", "./.wikis")
        repo_dirname = f"{owner}_{repo}"
        repo_path = basic_info_for_repo.get(
            "repo_path", os.path.join(repo_root_path, repo_dirname)
        )
        wiki_path = basic_info_for_repo.get(
            "wiki_path", os.path.join(wiki_root_path, repo_dirname)
        )
        repo_update_log_path = basic_info_for_repo.get(
            "repo_update_log_path", os.path.join(wiki_path, "repo_update_log.json")
        )

        (
            commits_updated,
            prs_updated,
            releases_updated,
            code_files_updated,
            updated_commit_info,
            updated_pr_info,
            updated_release_note_info,
            updated_code_files,
        ) = self._parallel_fetch_updates(
            owner, repo, platform, repo_path, repo_update_log_path
        )

        updated_basic_info_for_repo = {}
        updated_basic_info_for_code = {}

        if commits_updated:
            print(
                f"   → [check_update_node] new commits detected since last documentation generation"
            )
            updated_basic_info_for_repo = {
                "commits_updated": commits_updated,
            }
        if prs_updated:
            print(
                f"   → [check_update_node] new PRs detected since last documentation generation"
            )
            updated_basic_info_for_repo = {
                **updated_basic_info_for_repo,
                "prs_updated": prs_updated,
            }
        if releases_updated:
            print(
                f"   → [check_update_node] new release notes detected since last documentation generation"
            )
            updated_basic_info_for_repo = {
                **updated_basic_info_for_repo,
                "releases_updated": releases_updated,
            }
        if code_files_updated:
            print(
                f"   → [check_update_node] new code files detected from updated commits"
            )
            updated_basic_info_for_code = {
                "code_files_updated": code_files_updated,
            }

        print(f"✓ [check_update_node] updates processed successfully")
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
        print("→ [run] running parent graph with inputs:", inputs)
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
        print("✓ [run] graph execution completed successfully!")
        print("=" * 80)
        if count_time:
            print(f"   → [run] total execution time: {elapsed_time:.2f} seconds")

    def stream(
        self,
        inputs: ParentGraphState,
        progress_callback,
        config: dict = None,
        count_time: bool = True,
    ):
        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if config is None:
            config = {"configurable": {"thread_id": f"wiki-generation-{date}"}}
        print("→ [stream] running parent graph with inputs:", inputs)

        progress_callback(
            {
                "stage": "started",
                "message": "Starting documentation generation",
                "progress": 0,
            }
        )

        start_time = time.time()
        progress_stages = {
            "basic_info_node": 15,
            "check_update_node": 35,
            "repo_info_graph": 65,
            "code_analysis_graph": 85,
        }

        for chunk in self.graph.stream(
            inputs,
            config=config,
            subgraphs=True,
        ):
            # Track progress based on which node is executing
            # Chunk format: (namespace_tuple, {node_name: result_dict})
            if chunk and isinstance(chunk, tuple) and len(chunk) >= 2:
                # chunk[1] is a dict where keys are node names
                if isinstance(chunk[1], dict):
                    for node_name in chunk[1].keys():
                        if node_name in progress_stages and progress_callback:
                            progress_callback(
                                {
                                    "stage": node_name,
                                    "message": f"Processing {node_name}",
                                    "progress": progress_stages[node_name],
                                }
                            )
                            print(
                                f"   → [stream] progress update: {node_name} - {progress_stages[node_name]}%"
                            )

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("\n" + "=" * 80)
        print("✓ [stream] graph execution completed successfully!")
        print("=" * 80)
        if count_time:
            print(f"   → [stream] total execution time: {elapsed_time:.2f} seconds")

        progress_callback(
            {
                "stage": "completed",
                "message": "Documentation generation completed",
                "progress": 100,
                "elapsed_time": elapsed_time,
            }
        )


if __name__ == "__main__":
    # use "uv run python -m agents.sub_graph.parent" to run this file
    parent_graph_builder = ParentGraphBuilder(branch_mode="repo")
    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parent_graph_builder.run(
        inputs={
            "owner": "facebook",
            "repo": "zstd",
            "platform": "github",
            "mode": "fast",  # "fast" or "smart"
            "ratios": {
                "fast": 0.05,
                "smart": 0.75,
            },
            "max_workers": 10,
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
