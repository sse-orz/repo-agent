from github import Github, Auth
import requests
import os
from typing import Dict, List, Optional, Any
from config import CONFIG


def clone_repo(
    platform: str,
    owner: str,
    repo: str,
    dest: str,
) -> bool:
    """clone a repository from GitHub or Gitee to a local destination.

    Args:
        platform (str): Platform to use ("github" or "gitee").
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        dest (str): The local destination path to clone the repository to.
    Returns:
        bool: True if the repository was cloned, False if it already exists.
    """
    import subprocess

    platform = platform.lower()
    if platform == "github":
        url = f"https://github.com/{owner}/{repo}.git"
    elif platform == "gitee":
        url = f"https://gitee.com/{owner}/{repo}.git"
    else:
        raise ValueError("Unsupported platform. Use 'github' or 'gitee'.")

    # Check if the repo has already been cloned
    if os.path.exists(dest):
        return False

    # Clone the repository
    subprocess.run(["git", "clone", url, dest], check=True)
    return True


def pull_repo(
    platform: str,
    owner: str,
    repo: str,
    dest: str,
) -> bool:
    """Pull the latest changes from a repository on GitHub or Gitee.

    Args:
        platform (str): Platform to use ("github" or "gitee").
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
    Returns:
        bool: True if the pull was successful, False otherwise.
    """
    import subprocess
    import os

    platform = platform.lower()
    if platform not in ["github", "gitee"]:
        raise ValueError("Unsupported platform. Use 'github' or 'gitee'.")

    # Check if the repo has already been cloned
    if not os.path.exists(dest):
        raise FileNotFoundError(
            f"The repository {owner}/{repo} does not exist locally."
        )

    # Pull the latest changes
    subprocess.run(["git", "-C", dest, "pull"], check=True)
    return True


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
        token (str, optional): GitHub personal access token. Defaults to None.
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
    """get modified files in a specific commit.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        sha (str): The commit SHA.
        token (str, optional): GitHub personal access token. Defaults to None.

    Returns:
        list: List of modified files in the commit.
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
                "patch": file.patch,
            }
        )
    return files


def get_gitee_commit_files(
    owner: str, repo: str, sha: str, token: Optional[str] = None
) -> List[Dict[str, Any]]:
    """get modified files in a specific commit from Gitee.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        sha (str): The commit SHA.
        token (str, optional): Gitee personal access token. Defaults to None.

    Returns:
        list: List of modified files in the commit.
    """
    url = f"https://gitee.com/api/v5/repos/{owner}/{repo}/commits/{sha}"
    headers = {"Authorization": f"token {token}"} if token else {}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()
    files = []
    for file in data.get("files", []):
        files.append(
            {
                "filename": file.get("filename"),
                "status": file.get("status"),
                "additions": file.get("additions"),
                "deletions": file.get("deletions"),
                "changes": file.get("changes"),
                "patch": file.get("patch"),
            }
        )
    return files


def get_commit_files(
    owner: str,
    repo: str,
    sha: str,
    platform: Optional[str] = "github",
) -> List[Dict[str, Any]]:
    """get modified files in a specific commit from GitHub or Gitee.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        sha (str): The commit SHA.
        platform (str, optional): Platform to use ("github" or "gitee"). Defaults to "github".

    Returns:
        list: List of modified files in the commit.
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
    """Get repository information along with recent commits and their modified files.

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


def get_github_release_note(
    owner: str,
    repo: str,
    release_tag: Optional[str] = None,
    token: Optional[str] = None,
    limit: Optional[int] = 2,
) -> Dict[str, Any]:
    """Get release notes for a specific release from a GitHub repository.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        release_tag (str, optional): The tag name of the release. If None, get the latest 2 releases.
        token (str, optional): GitHub personal access token. Defaults to None.
        limit (int, optional): Number of releases to retrieve. Defaults to 2.

    Returns:
        Dict[str, Any]: Release notes for the specified release or list of latest releases.
    """
    g = Github(auth=Auth.Token(token)) if token else Github()
    repo_obj = g.get_repo(f"{owner}/{repo}")
    release_list = {
        "repo": f"{owner}/{repo}",
        "releases_count": 0,
        "releases": [],
    }
    if release_tag:
        release = repo_obj.get_release(release_tag)
        release_list["releases"].append(
            {
                "tag_name": release.tag_name,
                "name": release.name,
                "body": release.body,
                "created_at": release.created_at,
                "published_at": release.published_at,
            }
        )
        release_list["releases_count"] += 1
    else:
        for i, release in enumerate(repo_obj.get_releases()):
            if i >= limit:
                break
            release_list["releases"].append(
                {
                    "tag_name": release.tag_name,
                    "name": release.name,
                    "body": release.body,
                    "created_at": release.created_at,
                    "published_at": release.published_at,
                }
            )
            release_list["releases_count"] += 1
    return release_list


def get_gitee_release_note(
    owner: str,
    repo: str,
    release_tag: Optional[str] = None,
    token: Optional[str] = None,
    limit: Optional[int] = 2,
) -> Dict[str, Any]:
    """Get the latest releases from a Gitee repository.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        release_tag (str, optional): The tag name of the release. If None, get the latest 2 releases.
        token (str, optional): Gitee personal access token. Defaults to None.
        limit (int, optional): Number of releases to retrieve. Defaults to 2.

    Returns:
        Dict[str, Any]: Release notes for the specified release or list of latest releases.
    """
    url = f"https://gitee.com/api/v5/repos/{owner}/{repo}/releases"
    headers = {"Authorization": f"token {token}"} if token else {}
    release_list = {
        "repo": f"{owner}/{repo}",
        "releases_count": 0,
        "releases": [],
    }
    if release_tag:
        url += f"/tags/{release_tag}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        release_list["releases"].append(
            {
                "tag_name": data.get("tag_name"),
                "name": data.get("name"),
                "body": data.get("body"),
                "created_at": data.get("created_at"),
                "published_at": data.get("published_at"),
            }
        )
        release_list["releases_count"] += 1
    else:
        params = {"per_page": limit}
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        for release in data:
            release_list["releases"].append(
                {
                    "tag_name": release.get("tag_name"),
                    "name": release.get("name"),
                    "body": release.get("body"),
                    "created_at": release.get("created_at"),
                    "published_at": release.get("published_at"),
                }
            )
            release_list["releases_count"] += 1
    return release_list


def get_release_note(
    owner: str,
    repo: str,
    release_tag: Optional[str] = None,
    platform: Optional[str] = "github",
    limit: Optional[int] = 2,
) -> Dict[str, Any]:
    """Get release notes for a specific release or the latest releases from GitHub or Gitee repository.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        release_tag (Optional[str]): The tag name of the release. If None, get the latest 2 releases.
        platform (str, optional): Platform to use ("github" or "gitee"). Defaults to "github".
        limit (int, optional): Number of releases to retrieve. Defaults to 2.

    Returns:
        Dict[str, Any]: Release notes for the specified release or list of latest releases.
    """
    platform = platform.lower()
    token = CONFIG.get_token(platform)
    if platform == "github":
        return get_github_release_note(owner, repo, release_tag, token, limit=limit)
    elif platform == "gitee":
        return get_gitee_release_note(owner, repo, release_tag, token, limit=limit)
    else:
        raise ValueError("Unsupported platform. Use 'github' or 'gitee'.")


def get_github_pr(
    owner: str,
    repo: str,
    pr_tag: Optional[str] = None,
    token: Optional[str] = None,
    limit: Optional[int] = 2,
) -> Dict[str, Any]:
    """Get pull request details from a GitHub repository.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        pr_tag (str, optional): The pull request number.
        token (str, optional): GitHub personal access token. Defaults to None.
        limit (int, optional): Number of pull requests to retrieve. Defaults to 2.

    Returns:
        Dict[str, Any]: Pull request information.
    """
    g = Github(auth=Auth.Token(token)) if token else Github()
    repo_obj = g.get_repo(f"{owner}/{repo}")
    pr_list = {
        "repo": f"{owner}/{repo}",
        "prs_count": 0,
        "prs": [],
    }
    if pr_tag:
        pr = repo_obj.get_pull(int(pr_tag))
        pr_list["prs"].append(
            {
                "number": pr.number,
                "title": pr.title,
                "body": pr.body,
                "user": pr.user.login,
                "labels": [label.name for label in pr.labels],
                "issue_url": pr.issue_url,
                "diff_url": pr.diff_url,
                "patch_url": pr.patch_url,
                "state": pr.state,
                "created_at": pr.created_at,
                "merged_at": pr.merged_at,
            }
        )
        pr_list["prs_count"] += 1
    else:
        prs = repo_obj.get_pulls(state="all")[:limit]
        for pr in prs:
            pr_list["prs"].append(
                {
                    "number": pr.number,
                    "title": pr.title,
                    "body": pr.body,
                    "user": pr.user.login,
                    "labels": [label.name for label in pr.labels],
                    "issue_url": pr.issue_url,
                    "diff_url": pr.diff_url,
                    "patch_url": pr.patch_url,
                    "state": pr.state,
                    "created_at": pr.created_at,
                    "merged_at": pr.merged_at,
                }
            )
            pr_list["prs_count"] += 1
    return pr_list


def get_gitee_pr(
    owner: str,
    repo: str,
    pr_tag: Optional[str] = None,
    token: Optional[str] = None,
    limit: Optional[int] = 2,
) -> Dict[str, Any]:
    """Get pull request details from a Gitee repository.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        pr_tag (str, optional): The pull request number.
        token (str, optional): Gitee personal access token. Defaults to None.
        limit (int, optional): Number of pull requests to retrieve. Defaults to 2.

    Returns:
        Dict[str, Any]: Pull request information.
    """
    url = f"https://gitee.com/api/v5/repos/{owner}/{repo}/pulls"
    headers = {"Authorization": f"token {token}"} if token else {}
    pr_list = {
        "repo": f"{owner}/{repo}",
        "prs_count": 0,
        "prs": [],
    }
    if pr_tag:
        url += f"/{pr_tag}"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        pr_list["prs"].append(
            {
                "number": data.get("number"),
                "title": data.get("title"),
                "body": data.get("body"),
                "user": data.get("user", {}).get("login"),
                "labels": [label.get("name") for label in data.get("labels", [])],
                "issue_url": data.get("issue_url"),
                "diff_url": data.get("diff_url"),
                "patch_url": data.get("patch_url"),
                "state": data.get("state"),
                "created_at": data.get("created_at"),
                "merged_at": data.get("merged_at"),
            }
        )
        pr_list["prs_count"] += 1
    else:
        params = {"per_page": limit}
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        for pr in data:
            pr_list["prs"].append(
                {
                    "number": pr.get("number"),
                    "title": pr.get("title"),
                    "body": pr.get("body"),
                    "user": pr.get("user", {}).get("login"),
                    "labels": [label.get("name") for label in pr.get("labels", [])],
                    "issue_url": pr.get("issue_url"),
                    "diff_url": pr.get("diff_url"),
                    "patch_url": pr.get("patch_url"),
                    "state": pr.get("state"),
                    "created_at": pr.get("created_at"),
                    "merged_at": pr.get("merged_at"),
                }
            )
            pr_list["prs_count"] += 1
    return pr_list


def get_pr(
    owner: str,
    repo: str,
    pr_tag: Optional[str] = None,
    platform: Optional[str] = "github",
    limit: Optional[int] = 2,
) -> Dict[str, Any]:
    """Get pull request details from GitHub or Gitee repository.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        pr_tag (Optional[str]): The pull request number.
        platform (str, optional): Platform to use ("github" or "gitee"). Defaults to "github".
        limit (int, optional): Number of pull requests to retrieve. Defaults to 2.

    Returns:
        Dict[str, Any]: Pull request information.
    """
    platform = platform.lower()
    token = CONFIG.get_token(platform)
    if platform == "github":
        return get_github_pr(owner, repo, pr_tag, token, limit=limit)
    elif platform == "gitee":
        return get_gitee_pr(owner, repo, pr_tag, token, limit=limit)
    else:
        raise ValueError("Unsupported platform. Use 'github' or 'gitee'.")


def get_github_pr_files(
    owner: str,
    repo: str,
    pr_tag: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Get modified files in a specific pull request from GitHub.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        pr_tag (str, optional): The pull request number.
        token (str, optional): GitHub personal access token. Defaults to None.

    Returns:
        Dict[str, Any]: List of modified files in the pull request.
    """
    g = Github(auth=Auth.Token(token)) if token else Github()
    repo_obj = g.get_repo(f"{owner}/{repo}")
    files = {
        "repo": f"{owner}/{repo}",
        "pr_tag": pr_tag,
        "files": [],
    }
    if pr_tag:
        pr = repo_obj.get_pull(int(pr_tag))
        for file in pr.get_files():
            files["files"].append(
                {
                    "filename": file.filename,
                    "status": file.status,
                    "additions": file.additions,
                    "deletions": file.deletions,
                    "changes": file.changes,
                    "patch": file.patch,
                }
            )
    else:
        pulls = list(repo_obj.get_pulls(state="all"))
        if pulls:
            pr = pulls[0]
            for file in pr.get_files():
                files["files"].append(
                    {
                        "filename": file.filename,
                        "status": file.status,
                        "additions": file.additions,
                        "deletions": file.deletions,
                        "changes": file.changes,
                        "patch": file.patch,
                    }
                )
            files["pr_tag"] = pr.number
        # If no pull requests exist, files["files"] remains empty and pr_tag is unchanged
    return files


def get_gitee_pr_files(
    owner: str,
    repo: str,
    pr_tag: Optional[str] = None,
    token: Optional[str] = None,
) -> Dict[str, Any]:
    """Get modified files in a specific pull request from Gitee.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        pr_tag (str, optional): The pull request number.
        token (str, optional): Gitee personal access token. Defaults to None.

    Returns:
        Dict[str, Any]: List of modified files in the pull request.
    """
    headers = {"Authorization": f"token {token}"} if token else {}
    files = {
        "repo": f"{owner}/{repo}",
        "pr_tag": pr_tag,
        "files": [],
    }
    if pr_tag:
        url = f"https://gitee.com/api/v5/repos/{owner}/{repo}/pulls/{pr_tag}/files"
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        for file in data:
            files["files"].append(
                {
                    "filename": file.get("filename"),
                    "status": file.get("status"),
                    "additions": file.get("additions"),
                    "deletions": file.get("deletions"),
                    "changes": file.get("changes"),
                    "patch": file.get("patch"),
                }
            )
    else:
        # If no pr_tag is provided, get the latest pull request
        prs_url = f"https://gitee.com/api/v5/repos/{owner}/{repo}/pulls"
        params = {"per_page": 1}
        response = requests.get(prs_url, headers=headers, params=params)
        response.raise_for_status()
        prs_data = response.json()
        if prs_data:
            latest_pr = prs_data[0]
            pr_number = latest_pr.get("number")
            files["pr_tag"] = pr_number
            files_url = (
                f"https://gitee.com/api/v5/repos/{owner}/{repo}/pulls/{pr_number}/files"
            )
            response = requests.get(files_url, headers=headers)
            response.raise_for_status()
            data = response.json()
            for file in data:
                files["files"].append(
                    {
                        "filename": file.get("filename"),
                        "status": file.get("status"),
                        "additions": file.get("additions"),
                        "deletions": file.get("deletions"),
                        "changes": file.get("changes"),
                        "patch": file.get("patch"),
                    }
                )
    return files


def get_pr_files(
    owner: str,
    repo: str,
    pr_tag: Optional[str] = None,
    platform: Optional[str] = "github",
) -> Dict[str, Any]:
    """Get modified files in a specific pull request from GitHub or Gitee.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        pr_tag (str, optional): The pull request number.
        platform (str, optional): Platform to use ("github" or "gitee"). Defaults to "github".

    Returns:
        Dict[str, Any]: List of modified files in the pull request.
    """
    platform = platform.lower()
    token = CONFIG.get_token(platform)
    if platform == "github":
        return get_github_pr_files(owner, repo, pr_tag, token)
    elif platform == "gitee":
        return get_gitee_pr_files(owner, repo, pr_tag, token)
    else:
        raise ValueError("Unsupported platform. Use 'github' or 'gitee'.")


if __name__ == "__main__":
    # Example usage
    # use "uv run python -m utils.repo" to run this file
    owner = "facebook"
    repo = "zstd"
    platform = "github"

    # owner = "openharmony"
    # repo = "arkui_ace_engine"
    # platform = "gitee"

    # To get a specific release note
    # release_tag = "v1.0.0"  # Specify the release number you want to retrieve
    # release_note = get_release_note(owner, repo, release_tag, platform=platform)
    # print("Release Note:", release_note)

    # To get the latest 2 releases
    # latest_releases = get_release_note(owner, repo, platform=platform)
    # print("Latest Releases:", latest_releases)

    # pr_tag = "1"  # Specify the pull request number you want to retrieve
    # pr_info = get_pr(owner, repo, pr_tag, platform=platform)
    # print("Pull Request Info:", pr_info)

    # pr_files = get_pr_files(owner, repo, pr_tag, platform=platform)
    # print("Pull Request Files:", pr_files)

    # commit = get_commits(owner, repo, platform=platform)
    # print("Commits:", commit)

    # commit_file = get_commit_files(owner, repo, commit[0]["sha"], platform=platform)
    # print("\n\nCommit Files:", commit_file)

    # commit_info = get_repo_commit_info(owner, repo, platform=platform)
    # print("\n\nCommit Info:", commit_info)

    pr = get_pr(owner, repo, platform=platform)
    print("Pull Requests:", pr)

    pr_files = get_pr_files(owner, repo, platform=platform)
    print("\n\nPull Request Files:", pr_files)
