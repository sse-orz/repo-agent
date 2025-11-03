from utils.file import resolve_path, write_file, read_file, get_repo_structure
from utils.repo import (
    get_repo_info,
    get_repo_commit_info,
    get_release_note,
    get_pr,
    get_pr_files,
)
from utils.code_analyzer import (
    analyze_file_with_tree_sitter,
    format_tree_sitter_analysis_results,
)

from typing import Optional, Dict, Any
from langchain.tools import tool


@tool(description="Read the content of a file in the repository. ")
def read_file_tool(file_path: str) -> Dict[str, str]:
    """Read the content of a file.

    Args:
        file_path (str): The path to the file. It can be a relative or absolute path.

    Returns:
        Dict[str, str]: A dictionary containing the file path and its content.
    """
    abs_path = resolve_path(file_path)
    content = read_file(abs_path)
    return {
        "file_path": abs_path,
        "content": content,
    }


@tool(description="Write content to a file in the repository or wiki.")
def write_file_tool(
    file_path: str,
    content: str,
) -> Dict[str, Any]:
    """Write content to a file.

    Args:
        file_path (str): The path to the file. It can be a relative or absolute path.
        content (str): The content to write to the file.

    Returns:
        Dict[str, Any]: A dictionary containing the file path and a success flag.
    """
    abs_path = resolve_path(file_path)
    success = write_file(abs_path, content)
    return {
        "file_path": abs_path,
        "success": success,
    }


@tool(description="Retrieve the directory and file structure of the repository.")
def get_repo_structure_tool(repo_path: str) -> Dict[str, Any]:
    """Get the structure of a repository.

    Args:
        repo_path (str): The path to the repository. It can be a relative or absolute path.

    Returns:
        Dict[str, Any]: A dictionary containing the repository structure.
    """
    abs_path = resolve_path(repo_path)
    structure = get_repo_structure(abs_path)
    return {
        "repo_path": abs_path,
        "structure": structure,
    }


@tool(description="Get basic repository information (name, description, topics, etc.).")
def get_repo_basic_info_tool(
    owner: str, repo: str, platform: Optional[str] = "github"
) -> Dict[str, Optional[str]]:
    """Get basic information about a repository.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        platform (Optional[str]): The platform of the repository (default is "github").

    Returns:
        Dict[str, Optional[str]]: A dictionary containing basic information about the repository.
    """
    repo_info = get_repo_info(owner=owner, repo=repo, platform=platform)
    return repo_info


@tool(description="Get the latest commits and commit metadata.")
def get_repo_commit_info_tool(
    owner: str,
    repo: str,
    platform: Optional[str] = "github",
    max_num: Optional[int] = 10,
) -> Dict[str, Optional[str]]:
    """Get the latest commit information of a repository.

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


@tool
def get_repo_release_note_tool(
    owner: str,
    repo: str,
    release_tag: Optional[str] = None,
    platform: Optional[str] = "github",
    limit: Optional[int] = 2,
) -> Dict[str, Any]:
    """Get the latest release notes of a repository.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        release_tag (Optional[str]): The release tag to filter the release notes (default is None).
        platform (Optional[str]): The platform of the repository (default is "github").
        limit (Optional[int]): The maximum number of release notes to retrieve (default is 2).

    Returns:
        Dict[str, Any]: A dictionary containing the latest release notes or certain release note of the repository.
    """
    release_notes = get_release_note(
        owner=owner,
        repo=repo,
        release_tag=release_tag,
        platform=platform,
        limit=limit,
    )
    return release_notes


@tool
def get_repo_pr_tool(
    owner: str,
    repo: str,
    pr_tag: Optional[str] = None,
    platform: Optional[str] = "github",
    limit: Optional[int] = 2,
) -> Dict[str, Any]:
    """Get pull request information from a repository.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        pr_tag (Optional[str]): The pull request tag to filter the pull requests (default is None).
        platform (Optional[str]): The platform of the repository (default is "github").
        limit (Optional[int]): The maximum number of pull requests to retrieve (default is 2).

    Returns:
        Dict[str, Any]: pull request information from the repository.
    """
    pr_info = get_pr(
        owner=owner,
        repo=repo,
        pr_tag=pr_tag,
        platform=platform,
        limit=limit,
    )
    return pr_info


@tool
def get_repo_pr_files_tool(
    owner: str,
    repo: str,
    pr_tag: Optional[str] = None,
    platform: Optional[str] = "github",
) -> Dict[str, Any]:
    """Get modified files in a specific pull request.

    Args:
        owner (str): The owner of the repository.
        repo (str): The name of the repository.
        pr_tag (Optional[str]): The pull request tag to filter the pull requests (default is None).
        platform (Optional[str]): The platform of the repository (default is "github").

    Returns:
        Dict[str, Any]: List of modified files in the pull request.
    """
    pr_files = get_pr_files(
        owner=owner,
        repo=repo,
        pr_tag=pr_tag,
        platform=platform,
    )
    return pr_files


@tool
def code_file_analysis_tool(file_path: str) -> Dict[str, any]:
    """Analyze a code file using Tree-sitter.

    Args:
        file_path (str): The path to the code file. It can be a relative or absolute path.

    Returns:
        Dict[str, any]: A dictionary containing the analysis results.
    """
    abs_path = resolve_path(file_path)
    analysis_results = analyze_file_with_tree_sitter(abs_path)
    formatted_results = format_tree_sitter_analysis_results(analysis_results)
    return formatted_results


if __name__ == "__main__":
    # Example usage of the tools
    # commit_info = get_repo_commit_info_tool(
    #     owner="octocat", repo="Hello-World", platform="github", max_num=5
    # )
    # print(commit_info)
    # release_notes = get_repo_release_note_tool(
    #     owner="facebook", repo="zstd", platform="github", limit=2
    # )
    # print(release_notes)
    # structure = get_repo_structure_tool(repo_path="./utils")
    # print(structure)
    # pr_info = get_repo_pr_tool(
    #     owner="facebook", repo="zstd", platform="github", limit=2
    # )
    # print(pr_info)
    pr_files = get_repo_pr_files_tool(owner="facebook", repo="zstd", platform="github")
    print(pr_files)
