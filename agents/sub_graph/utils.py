from langchain_core.runnables.graph import MermaidDrawMethod
from datetime import datetime
import json
import os

from utils.repo import (
    get_repo_commit_info,
    get_pr,
    get_release_note,
)
from utils.file import (
    read_json,
)


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
    # for key, value in state.items():
    #     print(f"{key}: {value}")
    # write to a log file

    # check state.get("log") or state.get("basic_info_for_repo", {}).get("log", False)
    # or state.get("basic_info_for_code", {}).get("log", False)
    if not (
        state.get("log", False)
        or state.get("basic_info_for_repo", {}).get("log", False)
        or state.get("basic_info_for_code", {}).get("log", False)
    ):
        return
    log_file = "./.logs/state_log.txt"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("Current State:\n")
        f.write(f"{json.dumps(state, indent=2, default=str)}\n")
        f.write("\n*****************\n\n")


def get_updated_commit_info(
    owner: str, repo: str, platform: str, log_path: str
) -> tuple[bool, dict]:
    # this func is to get updated commits since last doc generation
    # return bool and commit_info dict
    # call `get_repo_commit_info` to get commit info
    commit_info = get_repo_commit_info(
        owner=owner,
        repo=repo,
        platform=platform,
    )
    # commit_info contains `repo_info`, "commits_count`, `commits`
    commits = commit_info.get("commits", [])
    # get `commit_sha`
    log_data = read_json(log_path)
    if log_data is None:
        return False, commit_info
    chosen_sha = log_data.get("commit_sha", "")
    if not chosen_sha:
        return False, commit_info
    commits_updated = False
    commits_count = 0
    updated_commits = []
    # p.s. for example: commits list -> [{"sha": $sha_1}, {"sha": $sha_2}, {"sha": $chosen_sha}, ...]
    # so after chosen_sha commit, there are two new commits -> sha_2 and sha_1
    # add new commits to the update_info
    for commit in commits:
        sha = commit.get("sha", "")
        if sha == chosen_sha:
            break
        updated_commits.append(commit)
        commits_count += 1
        commits_updated = True
    if commits_updated:
        return True, {
            "repo_info": commit_info.get("repo_info", {}),
            "commits_count": commits_count,
            "commits": updated_commits,
        }
    # if false, can return normal `commit_info` as state to avoid re-fetching again
    # and next node can use it without calling again
    return False, commit_info


def get_updated_pr_info(
    owner: str, repo: str, platform: str, log_path: str
) -> tuple[bool, dict]:
    # this func is to get updated prs since last doc generation
    # return bool and pr_info dict
    # call `get_pr` to get pr info
    pr_info = get_pr(
        owner=owner,
        repo=repo,
        platform=platform,
    )
    # pr_info contains `repo`, `prs_count`, `prs`
    prs = pr_info.get("prs", [])
    # get `pr_number`
    log_data = read_json(log_path)
    if log_data is None:
        return False, pr_info
    chosen_pr_number = log_data.get("pr_number", "")
    if not chosen_pr_number:
        return False, pr_info
    prs_updated = False
    prs_count = 0
    updated_prs = []
    # p.s. for example: prs list -> [{"number": $num_1}, {"number": $num_2}, {"number": $chosen_pr_number}, ...]
    # so after chosen_pr_number pr, there are two new prs -> num_2 and num_1
    # add new prs to the update_info
    for pr in prs:
        number = pr.get("number", "")
        if str(number) == str(chosen_pr_number):
            break
        updated_prs.append(pr)
        prs_count += 1
        prs_updated = True
    if prs_updated:
        return True, {
            "repo": pr_info.get("repo", ""),
            "prs_count": prs_count,
            "prs": updated_prs,
        }
    return False, pr_info


def get_updated_release_note_info(
    owner: str, repo: str, platform: str, log_path: str
) -> tuple[bool, dict]:
    # this func is to get updated release notes since last doc generation
    # return bool and release_note_info dict
    # call `get_release_note` to get release note info
    release_note_info = get_release_note(
        owner=owner,
        repo=repo,
        platform=platform,
    )
    # release_note_info contains `repo`, `releases_count`, `releases`
    releases = release_note_info.get("releases", [])
    # get `release_tag_name`
    log_data = read_json(log_path)
    if log_data is None:
        return False, release_note_info
    chosen_tag_name = log_data.get("release_tag_name", "")
    if not chosen_tag_name:
        return False, release_note_info
    releases_updated = False
    releases_count = 0
    updated_releases = []
    # p.s. for example: releases list -> [{"tag_name": $tag_1}, {"tag_name": $tag_2}, {"tag_name": $chosen_tag_name}, ...]
    # so after chosen_tag_name release, there are two new releases -> tag_2 and tag_1
    # add new releases to the update_info
    for release in releases:
        tag_name = release.get("tag_name", "")
        if tag_name == chosen_tag_name:
            break
        updated_releases.append(release)
        releases_count += 1
        releases_updated = True
    if releases_updated:
        return True, {
            "repo": release_note_info.get("repo", ""),
            "releases_count": releases_count,
            "releases": updated_releases,
        }
    return False, release_note_info


def get_updated_code_files(
    repo_path: str, updated_commits: list[dict], commits_updated: bool
) -> tuple[bool, list[str]]:
    # this func is to get updated code files from updated commits
    # extract changed code files from updated_commits
    # and merge into code_file path list
    updated_code_files = set()
    if not commits_updated:
        return False, []
    for commit in updated_commits:
        changed_files = commit.get("files", [])
        for file in changed_files:
            file_path = file.get("filename", "")
            if file_path:
                updated_code_files.add(f"{repo_path}/{file_path}")
    return True, list(updated_code_files)
