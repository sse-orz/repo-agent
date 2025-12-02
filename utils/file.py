import os
import json
from typing import List, Optional


def resolve_path(file_path: str) -> str:
    """Resolve a file path to an absolute path.

    Args:
        file_path (str): The file path to resolve.

    Returns:
        str: The absolute file path.
    """
    return os.path.abspath(file_path)


def get_gitignore_dirs(gitignore_path: str) -> List[str]:
    ignore_dirs = []
    try:
        with open(gitignore_path, "r") as f:
            gitignore_content = f.readlines()
        for line in gitignore_content:
            line = line.strip()
            if line and not line.startswith("#") and not line.startswith("/"):
                ignore_dirs.append(line)
    except Exception as e:
        print(f"Error reading .gitignore file at {gitignore_path}: {e}")
    return ignore_dirs


def get_ignore_dirs(dir_path: str) -> List[str]:
    """Get a list of directories to ignore within a given directory.

    Args:
        dir_path (str): The directory path to check for ignore directories.

    Returns:
        List[str]: A list of directory names to ignore.
    """
    default_ignore_dirs = [
        ".git",
        ".github",
        "__pycache__",
        "node_modules",
        ".vscode",
        ".idea",
        "build",
        "dist",
        "Include",
        "Lib",
        "Scripts",
        "bin",
        "obj",
        "out",
        "target",
    ]
    gitignore_path = os.path.join(dir_path, ".gitignore")
    if os.path.exists(gitignore_path):
        gitignore_ignore_dirs = get_gitignore_dirs(gitignore_path)
        ignore_dirs = list(set(default_ignore_dirs + gitignore_ignore_dirs))
        return ignore_dirs
    return default_ignore_dirs


def get_ignore_extensions() -> List[str]:
    """Get a list of file extensions to ignore.

    Returns:
        List[str]: A list of file extensions to ignore.
    """
    return [
        ".exe",
        ".dll",
        ".bin",
        ".class",
        ".so",
        ".o",
        ".a",
        ".pyc",
        ".pyo",
        ".jar",
        ".war",
        ".ear",
        ".lib",
        ".obj",
        ".dylib",
        ".lib",
    ]


def get_repo_structure(repo_path: str) -> List[str]:
    """Get the structure of a repository, returning a list of file paths.

    Args:
        repo_path (str): The path to the repository.
    Returns:
        List[str]: A list of file paths in the repository.
    """
    ignore_dirs = get_ignore_dirs(repo_path)
    ignore_extensions = get_ignore_extensions()
    file_paths = []
    for root, dirs, files in os.walk(repo_path):
        # Modify dirs in-place to skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for file in files:
            if not any(file.endswith(ext) for ext in ignore_extensions):
                file_paths.append(os.path.join(root, file))
    return file_paths


def read_file(file_path: str) -> str:
    """Read the content of a file.

    Args:
        file_path (str): The path to the file.
    Returns:
        str: The content of the file.
    """
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return ""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""


def write_file(file_path: str, content: str) -> bool:
    """write content to a file, creating directories if needed.

    Args:
        file_path (str): The path to the file.
        content (str): The content to write to the file.
    Returns:
        bool: True if the write was successful.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Error writing to file {file_path}: {e}")
        return False


def read_json(file_path: str) -> Optional[dict]:
    """Read a JSON file and return its content as a dictionary.

    Args:
        file_path (str): The path to the JSON file.
    Returns:
        Optional[dict]: The content of the JSON file as a dictionary, or None if an error occurs.
    """

    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist.")
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error reading JSON file {file_path}: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    repo_path = "./.repos/facebook_react"
    structure = get_repo_structure(repo_path)
    print(structure)
