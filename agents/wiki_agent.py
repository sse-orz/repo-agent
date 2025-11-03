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

from prompt import WikiPrompts


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
        system_prompt = WikiPrompts.get_system_prompt()
        super().__init__(tools, system_prompt, repo_path, wiki_path)

    def run(self):
        # Start the wiki generation process
        init_message = WikiPrompts.get_init_message(self.repo_path, self.wiki_path)
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
