from config import CONFIG
from .tools import (
    read_file_tool,
    write_file_tool,
    get_repo_structure_tool,
    get_repo_basic_info_tool,
    get_repo_commit_info_tool,
    code_file_analysis_tool,
)
from .base_agent import BaseAgent, AgentState

from langchain_core.messages import HumanMessage
from datetime import datetime


class WikiAgent(BaseAgent):
    def __init__(self, repo_path: str, wiki_path: str):
        tools = [
            read_file_tool,
            write_file_tool,
            get_repo_structure_tool,
            get_repo_basic_info_tool,
            get_repo_commit_info_tool,
            code_file_analysis_tool,
        ]
        system_prompt = """
You are a Wiki Generation Agent. Your task is to generate comprehensive wiki documentation for a given code repository. You have access to various tools that allow you to read and write files, analyze code files, and gather information about the repository structure and commit history.
"""
        super().__init__(tools, system_prompt, repo_path, wiki_path)

    def run(self):
        # Start the wiki generation process
        init_message = HumanMessage(
            content=f"""
Generate wiki files for the repository located at {self.repo_path} and save them to {self.wiki_path}. Use the tools available to gather information about the repository, analyze code files, and write the necessary wiki documentation. Make sure to structure the wiki files appropriately and cover all relevant aspects of the repository.
            """
        )
        initial_state = AgentState(
            messages=[init_message],
            repo_path=self.repo_path,
            wiki_path=self.wiki_path,
        )
        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._print_stream(
            file_name=time,
            stream=self.app.stream(
                initial_state,
                stream_mode="values",
                config={
                    "configurable": {"thread_id": "wiki-generation-thread"},
                    "recursion_limit": 100,
                },
            ),
        )


if __name__ == "__main__":
    # use "uv run -m agents.wiki_agent" to run this file
    CONFIG.display()
    from utils.repo import clone_repo, pull_repo

    repo_path = "./.repos/facebook_zstd"
    clone_repo(
        platform="github",
        owner="facebook",
        repo="zstd",
        dest=repo_path,
    )
    pull_repo(
        platform="github",
        owner="facebook",
        repo="zstd",
        dest=repo_path,
    )
    wiki_path = "./.wikis/facebook_zstd"
    agent = WikiAgent(repo_path, wiki_path)
    agent.run()
    # agent._draw_graph()
