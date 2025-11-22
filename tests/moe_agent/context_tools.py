import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from langchain.tools import tool
from utils.file import write_file, read_file, resolve_path


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
