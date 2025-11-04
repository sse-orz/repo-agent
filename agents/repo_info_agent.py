from config import CONFIG
from .tools import (
    get_repo_structure_tool,
    get_repo_basic_info_tool,
    get_repo_commit_info_tool,
)
from .base_agent import BaseAgent, AgentState

from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, END
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    ToolMessage,
    HumanMessage,
)
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Sequence, Literal, Dict, Any
from datetime import datetime
import json
import os
import io


class RepoInfoAgent:
    def __init__(self, llm, tools):
        self.llm = llm.bind_tools(tools, parallel_tool_calls=False)
        self.tools = tools
        self.tool_executor = ToolNode(tools)
        self.memory = InMemorySaver()
        self.app = self._build_app()

    def _build_app(self):
        """Build the agent workflow graph."""
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self._agent_node)
        workflow.add_node("tools", self.tool_executor)
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "end": END,
            },
        )
        workflow.add_edge("tools", "agent")
        return workflow.compile(checkpointer=self.memory)

    def _should_continue(self, state: AgentState) -> str:
        """Determine whether to continue or end."""
        messages = state["messages"]
        last_message = messages[-1]
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return "end"
        else:
            return "continue"

    def _agent_node(self, state: AgentState) -> AgentState:
        """Call LLM with current state."""
        system_prompt = SystemMessage(
            content="""You are a repository information collector. Your task is to:
                    1. Extract repository basic information (name, description, main language)
                    2. Get directory structure (main directories only, skip common directories like node_modules, .git, __pycache__, etc.)
                    3. Retrieve the latest 10 commit records

                    Use the available tools to gather this information and format the output as JSON with the following structure:
                    {
                        "repo_name": "string",
                        "description": "string",
                        "main_language": "string",
                        "structure": ["list", "of", "main", "directories"],
                        "commits": [
                            {
                                "sha": "commit_hash",
                                "message": "commit_message",
                                "author": "author_name",
                                "date": "commit_date"
                            }
                        ]
                    }
                    """
        )
        response = self.llm.invoke([system_prompt] + state["messages"])
        return {"messages": [response]}

    def run(self, repo_path: str) -> dict:
        """Collect repository information.

        Args:
            repo_path (str): Local path to the repository

        Returns:
            dict: Repository information in structured format
        """
        prompt = f"""Collect comprehensive information about the local repository at: {repo_path}

                    Tasks:
                    1. Use get_repo_structure_tool to get the directory structure (filter out common build/cache directories)
                    2. Extract the repository name from the path
                    3. If possible, infer the main programming language from file extensions

                    Provide the result in JSON format. For commits, if remote info is not available, set commits to an empty array.
                    """

        initial_state = AgentState(
            messages=[HumanMessage(content=prompt)],
            repo_path=repo_path,
            wiki_path="",  # Not needed for this agent
        )

        # Run the agent workflow
        final_state = None
        for state in self.app.stream(
            initial_state,
            stream_mode="values",
            config={
                "configurable": {
                    "thread_id": f"repo-info-{datetime.now().timestamp()}"
                },
                "recursion_limit": 100,
            },
        ):
            final_state = state

        # Extract the result from the final message
        if final_state:
            last_message = final_state["messages"][-1]
            if hasattr(last_message, "content"):
                try:
                    # Try to parse JSON from the content
                    content = last_message.content
                    # Find JSON block in the content
                    import re

                    json_match = re.search(r"\{[\s\S]*\}", content)
                    if json_match:
                        result = json.loads(json_match.group())
                        return result
                    else:
                        # Return a default structure if parsing fails
                        return {
                            "repo_name": os.path.basename(repo_path),
                            "description": "Unable to extract description",
                            "main_language": "Unknown",
                            "structure": [],
                            "commits": [],
                            "raw_output": content,
                        }
                except json.JSONDecodeError as e:
                    print(f"Failed to parse JSON from agent output: {e}")
                    return {
                        "repo_name": os.path.basename(repo_path),
                        "description": "JSON parse error",
                        "main_language": "Unknown",
                        "structure": [],
                        "commits": [],
                        "error": str(e),
                        "raw_output": last_message.content,
                    }

        # Fallback if no result
        return {
            "repo_name": os.path.basename(repo_path),
            "description": "No information collected",
            "main_language": "Unknown",
            "structure": [],
            "commits": [],
        }


# ========== RepoInfoAgentTest ==========
def RepoInfoAgentTest():
    # 本地仓库
    llm = CONFIG.get_llm()
    tools = [
        get_repo_structure_tool,
        get_repo_basic_info_tool,
        get_repo_commit_info_tool,
    ]
    agent = RepoInfoAgent(llm, tools)

    repo_info = agent.run(
        repo_path="./.repos/facebook_zstd",
    )

    print(json.dumps(repo_info, indent=2))


if __name__ == "__main__":
    RepoInfoAgentTest()
    # Expected output format:
    """
    {
        "repo_name": "zstd",
        "description": "Zstandard - Fast real-time compression algorithm",
        "main_language": "C",
        "structure": [
            "lib",
            "doc",
            "zlibWrapper",
            "tests",
            "examples",
            "contrib",
            ".github",
            "programs"
        ],
        "commits": [
            {
            "sha": "448cd340879adc0ffe36ed1e26823ee2dcb3217b",
            "message": "Merge pull request #4522 from facebook/dependabot/github_actions/github/codeql-action-4.31.2\n\nBump github/codeql-action from 3.30.1 to 4.31.2",
            "author": "Cyan4973",
            "date": "2025-11-03T14:51:17+00:00"
            },
            {
            "sha": "273ab1bbdf479f68a25290d25e540cd41f9dacf5",
            "message": "Bump github/codeql-action from 3.30.1 to 4.31.2\n\nBumps [github/codeql-action](https://github.com/github/codeql-action) from 3.30.1 to 4.31.2.\n- [Release notes](https://github.com/
        github/codeql-action/releases)\n- [Changelog](https://github.com/github/codeql-action/blob/main/CHANGELOG.md)\n- [Commits](https://github.com/github/codeql-action/compare/f1f6e5f6af878fb37288ce1c627459e94dbf7d01...0499de31b99561a6d14a36a5f662c2a54f91beee)\n\n---\nupdated-dependencies:\n- dependency-name: github/codeql-action\n  dependency-version: 4.31.2\n  dependency-type: direct:production\n  update-type: version-update:semver-major\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>",                                                                                                          "author": "dependabot[bot]",
            "date": "2025-11-03T05:08:56+00:00"
            },
            {
            "sha": "a25c1fc96f431e69abea38f52cb31e6bc074e9f1",
            "message": "Merge pull request #4519 from facebook/dependabot/github_actions/actions/upload-artifact-5\n\nBump actions/upload-artifact from 4 to 5",
            "author": "Cyan4973",
            "date": "2025-10-27T14:15:00+00:00"
            },
            {
            "sha": "129769d04c459866697391d0e21f0121a614052d",
            "message": "Bump actions/upload-artifact from 4 to 5\n\nBumps [actions/upload-artifact](https://github.com/actions/upload-artifact) from 4 to 5.\n- [Release notes](https://github.com/actions/upl
        oad-artifact/releases)\n- [Commits](https://github.com/actions/upload-artifact/compare/v4...v5)\n\n---\nupdated-dependencies:\n- dependency-name: actions/upload-artifact\n  dependency-version: '5'\n  dependency-type: direct:production\n  update-type: version-update:semver-major\n...\n\nSigned-off-by: dependabot[bot] <support@github.com>",                                                                  "author": "dependabot[bot]",
            "date": "2025-10-27T05:18:47+00:00"
            },
            {
            "sha": "711e17da98510a3567bf47f85a08a76f64811474",
            "message": "Merge pull request #4491 from facebook/cmake_root\n\n[cmake] propose a root wrapper",
            "author": "Cyan4973",
            "date": "2025-10-26T21:40:00+00:00"
            },
            {
            "sha": "57d494909858d19fdcb42f215850a5e04b83de8f",
            "message": "Merge pull request #4517 from Cyan4973/asyncio_revisit\n\nRemove asyncio from the compression path",
            "author": "Cyan4973",
            "date": "2025-10-26T21:39:25+00:00"
            },
            {
            "sha": "7a3c940e7f6df29124f31f7e8c64d4f4dc59786e",
            "message": "syncio interface only enabled when compression is enabled",
            "author": "Cyan4973",
            "date": "2025-10-26T16:48:45+00:00"
            },
            {
            "sha": "41f2673acda6543c6e8f3780926733439030e797",
            "message": "changed name to syncIO for clarity",
            "author": "Cyan4973",
            "date": "2025-10-25T18:23:16+00:00"
            },
            {
            "sha": "ccadc33a599d61f1a8cbeee7a96cd07a7cc8de24",
            "message": "minor: use init/destroy pair naming convention",
            "author": "Cyan4973",
            "date": "2025-10-25T18:11:48+00:00"
            },
            {
            "sha": "d1dd7e1481d76db7f0c94f90ba57a607c4e5df9e",
            "message": "removed asyncio completely for compression path\n\nthis does not provide speed benefits,\nsince most of the leverage happens internally within the library,\nand can even become detri
        mental in certain scenario, due to complex and wasteful memory management.\nAt a minimum, it makes the logic simpler, easier to debug, at essentially the same performance.",                                 "author": "Cyan4973",
            "date": "2025-10-25T18:02:51+00:00"
            }
        ]
    }

    """
