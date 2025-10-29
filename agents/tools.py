from utils.file import resolve_path, write_file, read_file, get_repo_structure
from utils.repo import get_repo_info, get_repo_commit_info
from utils.code_analyzer import (
    analyze_file_with_tree_sitter,
    format_tree_sitter_analysis_results,
)

import os
from typing import List, Optional, Dict
import datetime
import json
from langchain.tools import tool


@tool
def read_file_tool(file_path: str) -> str:
    """Read the content of a file.

    Args:
        file_path (str): The path to the file. It can be a relative or absolute path.

    Returns:
        str: The content of the file.
    """
    abs_path = resolve_path(file_path)
    content = read_file(abs_path)
    return content


@tool
def write_file_tool(
    file_path: str,
    content: str,
) -> str:
    """Write content to a file.

    Args:
        file_path (str): The path to the file. It can be a relative or absolute path.
        content (str): The content to write to the file.

    Returns:
        str: A message indicating the result of the write operation.
    """
    abs_path = resolve_path(file_path)
    success = write_file(abs_path, content)
    if success:
        return f"Successfully wrote to file {abs_path}."
    else:
        return f"Failed to write to file {abs_path}."


@tool
def get_repo_structure_tool(repo_path: str) -> List[str]:
    """Get the structure of a repository.

    Args:
        repo_path (str): The path to the repository. It can be a relative or absolute path.

    Returns:
        List[str]: A list of file paths in the repository.
    """
    abs_path = resolve_path(repo_path)
    structure = get_repo_structure(abs_path)
    return structure


@tool
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


@tool
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
