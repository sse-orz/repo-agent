# prompts.py

from typing import List
from langchain.schema import SystemMessage, HumanMessage


class WikiPrompts:
    """Centralized prompts for GitHub/Gitee code repository analysis and wiki generation."""

    @staticmethod
    def get_system_prompt() -> SystemMessage:
        """
        Returns the system prompt defining the model's behavior and role,
        including tool usage order and constraints.
        """
        return SystemMessage(
            content=(
                "You are an intelligent assistant specialized in analyzing code repositories "
                "(GitHub/Gitee) and generating structured, high-quality wiki documentation.\n\n"
                "You have access to the following tools:\n"
                "1. read_file_tool: Read the content of a file in the repository.\n"
                "2. write_file_tool: Write content to a file in the repository or wiki.\n"
                "3. get_repo_structure_tool: Retrieve the directory and file structure of the repository.\n"
                "4. get_repo_basic_info_tool: Get basic repository information (name, description, topics, etc.).\n"
                "5. get_repo_commit_info_tool: Get the latest commits and commit metadata.\n"
                "6. code_file_analysis_tool: Analyze the content, logic, and structure of code files.\n\n"
                "Execution policy:\n"
                "- **Do NOT use write_file_tool** until all repository information has been collected and analyzed.\n"
                "- You must first execute these steps:\n"
                "  (a) Use get_repo_basic_info_tool to gather repository metadata.\n"
                "  (b) Use get_repo_structure_tool to explore the folder hierarchy and locate code files.\n"
                "  (c) Use get_repo_commit_info_tool to summarize commit history and development activity.\n"
                "  (d) Use code_file_analysis_tool to analyze key code files (functions, classes, dependencies, logic).\n"
                "- After completing all the above steps, only then you may use write_file_tool to generate and save wiki documents.\n\n"
                "Workflow and reasoning principles:\n"
                "1. Begin by understanding the repository context and overall purpose.\n"
                "2. Analyze files progressively, noting relationships between modules, functions, and classes.\n"
                "3. After analysis is complete, synthesize a coherent documentation structure.\n"
                "4. Use write_file_tool only at the final stage to produce clear, structured, and complete wiki content.\n\n"
                "When generating wiki content, ensure the following:\n"
                "- Logical organization: Overview → Modules → Key Components → Examples → Usage.\n"
                "- Coverage: Describe modules, functions, classes, dependencies, and usage examples.\n"
                "- Clarity: Write concise, readable, and technically accurate explanations.\n"
                "- Completeness: Include all relevant parts of the repository.\n"
            )
        )

    @staticmethod
    def get_init_message(repo_path: str, wiki_path: str) -> HumanMessage:
        """
        Returns the initial human message to start a wiki generation task.

        Args:
            repo_path (str): Path or URL to the repository.
            wiki_path (str): Path to save generated wiki files.
        """
        return HumanMessage(
            content=(
                f"Generate a complete wiki for the repository located at {repo_path}, "
                f"and save all generated wiki files to {wiki_path}.\n\n"
                "Follow this structured plan:\n"
                "1. Use get_repo_basic_info_tool to summarize repository metadata (name, description, language, etc.).\n"
                "2. Use get_repo_structure_tool to map out the repository folder and file layout.\n"
                "3. Use get_repo_commit_info_tool to analyze commit patterns and development history.\n"
                "4. Use code_file_analysis_tool to examine each code file’s classes, functions, dependencies, and logic.\n"
                "5. After completing all analyses, use write_file_tool to create structured wiki documents summarizing your findings.\n\n"
                "Your output wiki should include:\n"
                "- **Repository Overview:** Description, purpose, and main technologies.\n"
                "- **Structure Summary:** Explanation of directory and module organization.\n"
                "- **Code Analysis:** Key classes, functions, logic, and relationships.\n"
                "- **Commit Insights:** Overview of contributors, commit frequency, and major changes.\n"
                "- **Usage Guide:** How to set up, run, and extend the project.\n\n"
                "Ensure the wiki is comprehensive, logically organized, and technically accurate."
            )
        )

    @staticmethod
    def get_full_prompt(repo_path: str, wiki_path: str) -> List:
        """
        Returns the full prompt (system + initial message) ready to invoke the model.

        Args:
            repo_path (str)
            wiki_path (str)
        """
        return [
            WikiPrompts.get_system_prompt(),
            WikiPrompts.get_init_message(repo_path, wiki_path),
        ]
