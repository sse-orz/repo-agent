import json
import subprocess
from github import Github, Auth
import requests
from typing import Dict, Any, List, Optional
from pathlib import Path
from langchain.tools import tool
from utils.file import write_file, read_file, resolve_path


# ========================== Context Tools ==========================


def ensure_context_dir(repo_identifier: str, subdir: str = "") -> str:
    """Ensure context directory exists and return its path.

    Args:
        repo_identifier (str): Unified repository identifier (format: owner_repo_name)
        subdir (str): Subdirectory within the context (e.g., 'module', 'analysis')

    Returns:
        str: Absolute path to the context directory
    """
    base_dir = Path(".cache") / repo_identifier
    if subdir:
        context_dir = base_dir / subdir
    else:
        context_dir = base_dir

    context_dir.mkdir(parents=True, exist_ok=True)
    return str(context_dir.absolute())


@tool(
    description="List all files in a context directory. Use this to discover available context files before reading them."
)
def ls_context_file_tool(context_dir: str) -> Dict[str, Any]:
    """List all files in the specified context directory.

    Args:
        context_dir (str): Path to the context directory (e.g., '.cache/owner_repo/module')

    Returns:
        Dict[str, Any]: Dictionary containing list of files and directory info

    Example:
        ls_context_file_tool('.cache/owner_repo/module')
        # Returns: {
        #     "directory": "/absolute/path/.cache/owner_repo/module",
        #     "files": ["adk.json", "callback.json"],
        #     "count": 2
        # }
    """
    try:
        abs_path = Path(context_dir).absolute()

        if not abs_path.exists():
            return {
                "directory": str(abs_path),
                "files": [],
                "count": 0,
                "message": f"Directory does not exist: {abs_path}",
            }

        if not abs_path.is_dir():
            return {
                "directory": str(abs_path),
                "files": [],
                "count": 0,
                "error": "Path is not a directory",
            }

        # List all files (not directories)
        files = []
        for item in abs_path.iterdir():
            if item.is_file():
                files.append(item.name)

        files.sort()  # Sort alphabetically for consistency

        return {
            "directory": str(abs_path),
            "files": files,
            "count": len(files),
        }

    except Exception as e:
        return {"directory": context_dir, "files": [], "count": 0, "error": str(e)}


def get_context_path(repo_identifier: str, category: str, filename: str = "") -> str:
    """Get the standard context file path.

    Args:
        repo_identifier (str): Unified repository identifier (format: owner_repo_name)
        category (str): Category of context ('module', 'analysis', 'info', etc.)
        filename (str): Optional filename

    Returns:
        str: Absolute path to the context file or directory
    """
    base_dir = Path(".cache") / repo_identifier / category
    base_dir.mkdir(parents=True, exist_ok=True)

    if filename:
        return str((base_dir / filename).absolute())
    else:
        return str(base_dir.absolute())


def save_json_to_context(
    repo_identifier: str, category: str, filename: str, data: dict
) -> bool:
    """Save JSON data to context file.

    Args:
        repo_identifier (str): Unified repository identifier (format: owner_repo_name)
        category (str): Category of context
        filename (str): Filename (should end with .json)
        data (dict): Data to save as JSON

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        file_path = get_context_path(repo_identifier, category, filename)
        content = json.dumps(data, indent=2, ensure_ascii=False)
        return write_file(file_path, content)
    except Exception as e:
        print(f"Error saving JSON to context: {e}")
        return False


@tool(
    description="Read the content of a file in the repository. Supports reading specific line ranges for large files."
)
def read_file_tool(
    file_path: str, start_line: Optional[int] = None, end_line: Optional[int] = None
) -> Dict[str, Any]:
    """Read the content of a file with optional line range support.

    Args:
        file_path (str): The path to the file. It can be a relative or absolute path.
        start_line (Optional[int]): The starting line number (1-indexed). If None, reads from beginning.
        end_line (Optional[int]): The ending line number (1-indexed, inclusive). If None, reads to end.

    Returns:
        Dict[str, Any]: A dictionary containing the file path, content, and line range info.
    """
    abs_path = resolve_path(file_path)
    full_content = read_file(abs_path)

    # If no line range specified, return full content
    if start_line is None and end_line is None:
        return {
            "file_path": abs_path,
            "content": full_content,
            "lines": "all",
        }

    # Split content into lines
    lines = full_content.split("\n")
    total_lines = len(lines)

    # Adjust line numbers (convert from 1-indexed to 0-indexed)
    start_idx = (start_line - 1) if start_line else 0
    end_idx = end_line if end_line else total_lines

    # Validate range
    start_idx = max(0, start_idx)
    end_idx = min(total_lines, end_idx)

    # Extract specified lines
    selected_lines = lines[start_idx:end_idx]
    content = "\n".join(selected_lines)

    return {
        "file_path": abs_path,
        "content": content,
        "lines": f"{start_idx + 1}-{end_idx}",
        "total_lines": total_lines,
    }


# ========================== Repository Utils ==========================


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
    from config import CONFIG

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
    from config import CONFIG

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
    from config import CONFIG

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


# ========================== Git Operations ==========================


def git_pull_with_retry(repo_path: str, max_retries: int = 3) -> tuple[bool, str]:
    """Execute git pull with automatic retry support.

    Args:
        repo_path: Local repository path.
        max_retries: Maximum number of retries (default: 3).

    Returns:
        (success: bool, message: str) - Whether successful and status message.

    Raises:
        Exception: Raised when max retries reached and still failing.
    """
    for attempt in range(1, max_retries + 1):
        try:
            result = subprocess.run(
                ["git", "pull"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=True,
                timeout=60,  # 60 second timeout
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
    """Get file change status from baseline_sha to HEAD.

    Args:
        repo_path: Local repository path.
        baseline_sha: Baseline commit SHA.

    Returns:
        List[Dict]: List of changed files, each containing status (M/A/D) and filename.

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
                            "status": status,  # M=Modified, A=Added, D=Deleted
                            "filename": filename,
                        }
                    )

        return changes

    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to get git diff: {e.stderr}")


def git_diff_file(repo_path: str, baseline_sha: str, filepath: str) -> Dict[str, Any]:
    """Get the complete diff of a single file from baseline_sha to HEAD.

    Args:
        repo_path: Local repository path.
        baseline_sha: Baseline commit SHA.
        filepath: File relative path.

    Returns:
        Dict: Contains diff content and change statistics.
        {
            "filepath": str,
            "diff": str,  # Complete diff text
            "additions": int,
            "deletions": int,
            "changes": int
        }
    """
    try:
        # Get diff content
        diff_result = subprocess.run(
            ["git", "diff", baseline_sha, "HEAD", "--", filepath],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )

        # Get statistics (additions/deletions)
        stat_result = subprocess.run(
            ["git", "diff", "--numstat", baseline_sha, "HEAD", "--", filepath],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True,
        )

        # Parse statistics: format is "additions\tdeletions\tfilename"
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
    """Get diffs for multiple files in batch.

    Args:
        repo_path: Local repository path.
        baseline_sha: Baseline commit SHA.
        filepaths: List of file paths.

    Returns:
        Dict[filename, diff_info]: Mapping from filename to diff information.
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
    """Get the current HEAD SHA of the local repository.

    Args:
        repo_path: Local repository path.

    Returns:
        str: Current HEAD SHA (short format, 7 characters).
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
