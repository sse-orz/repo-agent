from github import Github, Auth
import requests
from typing import Dict, List, Optional, Any
from config import CONFIG
from langchain.tools import tool


@tool(description="Get the latest commits and commit metadata WITHOUT patch data.")
def get_repo_commit_info_tool(
    owner: str,
    repo: str,
    platform: Optional[str] = "github",
    max_num: Optional[int] = 10,
) -> Dict[str, Optional[str]]:
    """Get the latest commit information of a repository WITHOUT patch (diff) data.

    This modified version excludes patch information to reduce token usage.
    Each file in commit will have: filename, status, additions, deletions, changes
    but NOT the patch field.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        platform (Optional[str]): The platform of the repository (default is "github").
        max_num (Optional[int]): The maximum number of commits to retrieve (default is 10).

    Returns:
        Dict[str, Optional[str]]: A dictionary containing the latest commit information of the repository.
    """
    commit_info = get_repo_commit_info(
        owner=owner, repo=repo, platform=platform, max_num=max_num
    )
    return commit_info


def get_github_repo_info(
    owner: str, repo: str, token: Optional[str] = None
) -> Dict[str, Any]:
    """get repository information from GitHub.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        token (str, optional): GitHub personal access token. Defaults to None.

    Returns:
        dict: Repository information.
    """
    g = Github(auth=Auth.Token(token)) if token else Github()
    repo_obj = g.get_repo(f"{owner}/{repo}")
    return {
        "name": repo_obj.name,
        "full_name": repo_obj.full_name,
        "description": repo_obj.description,
        "language": repo_obj.language,
        "stargazers_count": repo_obj.stargazers_count,
        "forks_count": repo_obj.forks_count,
        "open_issues_count": repo_obj.open_issues_count,
        "created_at": repo_obj.created_at,
        "updated_at": repo_obj.updated_at,
    }


def get_gitee_repo_info(
    owner: str, repo: str, token: Optional[str] = None
) -> Dict[str, Any]:
    """get repository information from Gitee.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        token (str, optional): Gitee personal access token. Defaults to None.

    Returns:
        dict: Repository information.
    """
    url = f"https://gitee.com/api/v5/repos/{owner}/{repo}"
    headers = {"Authorization": f"token {token}"} if token else {}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    return {
        "name": data.get("name"),
        "full_name": data.get("full_name"),
        "description": data.get("description"),
        "language": data.get("language"),
        "stargazers_count": data.get("stargazers_count"),
        "forks_count": data.get("forks_count"),
        "open_issues_count": data.get("open_issues_count"),
        "created_at": data.get("created_at"),
        "updated_at": data.get("updated_at"),
    }


def get_repo_info(
    owner: str,
    repo: str,
    platform: Optional[str] = "github",
) -> Dict[str, Any]:
    """get repository information from GitHub or Gitee.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        platform (str, optional): Platform to use ("github" or "gitee"). Defaults to "github".

    Returns:
        dict: Repository information.
    """
    platform = platform.lower()
    token = CONFIG.get_token(platform)
    if platform == "github":
        return get_github_repo_info(owner, repo, token)
    elif platform == "gitee":
        return get_gitee_repo_info(owner, repo, token)
    else:
        raise ValueError("Unsupported platform. Use 'github' or 'gitee'.")


def get_github_commits(
    owner: str, repo: str, token: Optional[str] = None, per_page: Optional[int] = 10
) -> List[Dict[str, Any]]:
    """get recent commits from a GitHub repository.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        token (str, optional): GitHub personal access token. Defaults to None.
        per_page (int, optional): Number of commits to retrieve per page. Defaults to 10.

    Returns:
        list: List of commit information.
    """
    g = Github(auth=Auth.Token(token)) if token else Github()
    repo_obj = g.get_repo(f"{owner}/{repo}")
    commits = repo_obj.get_commits()[:per_page]
    commit_list = []
    for commit in commits:
        commit_list.append(
            {
                "sha": commit.sha,
                "author": commit.author.login if commit.author else None,
                "message": commit.commit.message,
                "date": commit.commit.author.date,
            }
        )
    return commit_list


def get_gitee_commits(
    owner: str, repo: str, token: Optional[str] = None, per_page: Optional[int] = 10
) -> List[Dict[str, Any]]:
    """get recent commits from a Gitee repository.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        token (str, optional): Gitee personal access token. Defaults to None.
        per_page (int, optional): Number of commits to retrieve per page. Defaults to 10.

    Returns:
        list: List of commit information.
    """
    url = f"https://gitee.com/api/v5/repos/{owner}/{repo}/commits"
    headers = {"Authorization": f"token {token}"} if token else {}
    params = {"per_page": per_page}
    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    data = response.json()
    commit_list = []
    for commit in data:
        commit_list.append(
            {
                "sha": commit.get("sha"),
                "author": (
                    commit.get("author", {}).get("login")
                    if commit.get("author")
                    else None
                ),
                "message": commit.get("commit", {}).get("message"),
                "date": commit.get("commit", {}).get("author", {}).get("date"),
            }
        )
    return commit_list


def get_commits(
    owner: str,
    repo: str,
    per_page: Optional[int] = 10,
    platform: Optional[str] = "github",
) -> List[Dict[str, Any]]:
    """get recent commits from a GitHub or Gitee repository.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        per_page (int, optional): Number of commits to retrieve per page. Defaults to 10.
        platform (str, optional): Platform to use ("github" or "gitee"). Defaults to "github".

    Returns:
        list: List of commit information.
    """
    platform = platform.lower()
    token = CONFIG.get_token(platform)
    if platform == "github":
        return get_github_commits(owner, repo, token, per_page)
    elif platform == "gitee":
        return get_gitee_commits(owner, repo, token, per_page)
    else:
        raise ValueError("Unsupported platform. Use 'github' or 'gitee'.")


def get_github_commit_files(
    owner: str, repo: str, sha: str, token: Optional[str] = None
) -> List[Dict[str, Any]]:
    """get modified files in a specific commit WITHOUT patch data.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        sha (str): The commit SHA.
        token (str, optional): GitHub personal access token. Defaults to None.

    Returns:
        list: List of modified files in the commit (without patch data).
    """
    g = Github(auth=Auth.Token(token)) if token else Github()
    repo_obj = g.get_repo(f"{owner}/{repo}")
    commit = repo_obj.get_commit(sha)
    files = []
    for file in commit.files:
        files.append(
            {
                "filename": file.filename,
                "status": file.status,
                "additions": file.additions,
                "deletions": file.deletions,
                "changes": file.changes,
                # "patch": file.patch,  # REMOVED to reduce token usage
            }
        )
    return files


def get_gitee_commit_files(
    owner: str, repo: str, sha: str, token: Optional[str] = None
) -> List[Dict[str, Any]]:
    """get modified files in a specific commit from Gitee WITHOUT patch data.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        sha (str): The commit SHA.
        token (str, optional): Gitee personal access token. Defaults to None.

    Returns:
        list: List of modified files in the commit (without patch data).
    """
    url = f"https://gitee.com/api/v5/repos/{owner}/{repo}/commits/{sha}"
    headers = {"Authorization": f"token {token}"} if token else {}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    files = []
    for file in data.get("files", []):
        # If a commit contains too many changes, the patch field will be very large and consume a large amount of tokens
        # leading to context overlimit in the llm.
        files.append(
            {
                "filename": file.get("filename"),
                "status": file.get("status"),
                "additions": file.get("additions"),
                "deletions": file.get("deletions"),
                "changes": file.get("changes"),
                # "patch": file.get("patch"),  # REMOVED to reduce token usage
            }
        )
    return files


def get_commit_files(
    owner: str,
    repo: str,
    sha: str,
    platform: Optional[str] = "github",
) -> List[Dict[str, Any]]:
    """get modified files in a specific commit from GitHub or Gitee WITHOUT patch data.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        sha (str): The commit SHA.
        platform (str, optional): Platform to use ("github" or "gitee"). Defaults to "github".

    Returns:
        list: List of modified files in the commit (without patch data).
    """
    platform = platform.lower()
    token = CONFIG.get_token(platform)
    if platform == "github":
        return get_github_commit_files(owner, repo, sha, token)
    elif platform == "gitee":
        return get_gitee_commit_files(owner, repo, sha, token)
    else:
        raise ValueError("Unsupported platform. Use 'github' or 'gitee'.")


def get_repo_commit_info(
    owner: str,
    repo: str,
    max_num: Optional[int] = 10,
    platform: Optional[str] = "github",
) -> Dict[str, Any]:
    """Get repository information along with recent commits and their modified files (without patch data).

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        max_num (Optional[int], optional): Maximum number of commits to retrieve. Defaults to 10.
        platform (Optional[str], optional): Platform to use ("github" or "gitee"). Defaults to "github".
    Returns:
        Dict[str, Any]: Repository information along with recent commits and their modified files.
    """
    platform = platform.lower()
    repo_info = get_repo_info(owner, repo, platform)
    commits = get_commits(owner, repo, max_num, platform)
    for commit in commits:
        sha = commit["sha"]
        files = get_commit_files(owner, repo, sha, platform)
        commit["files"] = files
    return {
        "repo_info": repo_info,
        "commits_count": len(commits),
        "commits": commits,
    }


import subprocess
from pathlib import Path


def git_pull_with_retry(repo_path: str, max_retries: int = 3) -> tuple[bool, str]:
    """执行 git pull 并支持自动重试

    Args:
        repo_path: 本地仓库路径
        max_retries: 最大重试次数（默认3次）

    Returns:
        (success: bool, message: str) - 是否成功及状态信息

    Raises:
        Exception: 达到最大重试次数后仍失败时抛出异常
    """
    for attempt in range(1, max_retries + 1):
        try:
            result = subprocess.run(
                ["git", "pull"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
                timeout=60,  # 60秒超时
            )
            return (True, f"Pull succeeded on attempt {attempt}")
        except subprocess.TimeoutExpired:
            if attempt == max_retries:
                raise Exception(f"Git pull timeout after {max_retries} attempts")
            print(f"⚠️  Attempt {attempt} timeout, retrying...")
        except subprocess.CalledProcessError as e:
            if attempt == max_retries:
                raise Exception(
                    f"Git pull failed after {max_retries} attempts: {e.stderr}"
                )
            print(f"⚠️  Attempt {attempt} failed: {e.stderr}, retrying...")

    return (False, "Unexpected error")


def git_diff_name_status(repo_path: str, baseline_sha: str) -> List[Dict[str, str]]:
    """获取从 baseline_sha 到 HEAD 的文件变更状态

    Args:
        repo_path: 本地仓库路径
        baseline_sha: 基线提交 SHA

    Returns:
        List[Dict]: 变更文件列表，每项包含 status (M/A/D) 和 filename

    Examples:
        [
            {"status": "M", "filename": "src/main.py"},
            {"status": "A", "filename": "src/new.py"},
            {"status": "D", "filename": "src/old.py"}
        ]
    """
    try:
        result = subprocess.run(
            ["git", "diff", "--name-status", baseline_sha, "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )

        changes = []
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    status, filename = parts
                    changes.append(
                        {
                            "status": status,  # M=修改, A=新增, D=删除
                            "filename": filename,
                        }
                    )

        return changes

    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to get git diff: {e.stderr}")


def git_diff_file(repo_path: str, baseline_sha: str, filepath: str) -> Dict[str, Any]:
    """获取单个文件从 baseline_sha 到 HEAD 的完整 diff

    Args:
        repo_path: 本地仓库路径
        baseline_sha: 基线提交 SHA
        filepath: 文件相对路径

    Returns:
        Dict: 包含 diff 内容和变更统计
        {
            "filepath": str,
            "diff": str,  # 完整 diff 文本
            "additions": int,
            "deletions": int,
            "changes": int
        }
    """
    try:
        # 获取 diff 内容
        diff_result = subprocess.run(
            ["git", "diff", baseline_sha, "HEAD", "--", filepath],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )

        # 获取统计信息（additions/deletions）
        stat_result = subprocess.run(
            ["git", "diff", "--numstat", baseline_sha, "HEAD", "--", filepath],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )

        # 解析统计信息：格式为 "additions\tdeletions\tfilename"
        additions, deletions = 0, 0
        stat_line = stat_result.stdout.strip()
        if stat_line:
            parts = stat_line.split("\t")
            if len(parts) >= 2:
                additions = int(parts[0]) if parts[0].isdigit() else 0
                deletions = int(parts[1]) if parts[1].isdigit() else 0

        return {
            "filepath": filepath,
            "diff": diff_result.stdout,
            "additions": additions,
            "deletions": deletions,
            "changes": additions + deletions,
        }

    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to get diff for {filepath}: {e.stderr}")


def git_diff_multiple_files(
    repo_path: str, baseline_sha: str, filepaths: List[str]
) -> Dict[str, Dict[str, Any]]:
    """批量获取多个文件的 diff

    Args:
        repo_path: 本地仓库路径
        baseline_sha: 基线提交 SHA
        filepaths: 文件路径列表

    Returns:
        Dict[filename, diff_info]: 文件名到 diff 信息的映射
    """
    result = {}
    for filepath in filepaths:
        try:
            diff_info = git_diff_file(repo_path, baseline_sha, filepath)
            result[filepath] = diff_info
        except Exception as e:
            print(f"⚠️  Failed to get diff for {filepath}: {e}")
            result[filepath] = {
                "filepath": filepath,
                "diff": "",
                "additions": 0,
                "deletions": 0,
                "changes": 0,
                "error": str(e),
            }

    return result


def git_get_current_head_sha(repo_path: str) -> str:
    """获取本地仓库当前 HEAD 的 SHA

    Args:
        repo_path: 本地仓库路径

    Returns:
        str: 当前 HEAD 的 SHA（短格式，7位）
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to get current HEAD: {e.stderr}")
