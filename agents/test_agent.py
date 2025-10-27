from config import CONFIG
from .tools import (
    read_file_tool,
    write_file_tool,
    get_repo_structure_tool,
    get_repo_basic_info_tool,
    get_repo_commit_info_tool,
    code_file_analysis_tool,
)

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
from typing import TypedDict, Annotated, Sequence, Literal
from contextlib import redirect_stdout
from datetime import datetime
import json
import os
import io


class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    repo_path: str
    wiki_path: str


class WikiAgent:
    def __init__(self, repo_path: str, wiki_path: str):
        self.llm = CONFIG.get_llm()
        self.repo_path = repo_path
        self.wiki_path = wiki_path
        self.memory = InMemorySaver()
        tools = [
            read_file_tool,
            write_file_tool,
            get_repo_structure_tool,
            get_repo_basic_info_tool,
            get_repo_commit_info_tool,
            code_file_analysis_tool,
        ]
        self.llm_with_tools = self.llm.bind_tools(tools, parallel_tool_calls=False)
        self.tools = tools
        self.tool_executor = ToolNode(tools)
        self.app = self._build_app()

    def _build_app(self):
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
        """Determine whether the agent should continue or end.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            str: "continue" if the agent should continue, "end" if it should stop.
        """
        messages = state["messages"]
        last_message = messages[-1]
        if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
            return "end"
        else:
            return "continue"

    def _agent_node(self, state: AgentState) -> AgentState:
        """Call the language model with the current state.

        Args:
            state (AgentState): The current state of the agent.

        Returns:
            AgentState: The updated state of the agent after calling the model.
        """
        system_prompt = SystemMessage(
            content="You are a helpful assistant that generates wiki files for a code repository using available tools."
        )
        response = self.llm_with_tools.invoke([system_prompt] + state["messages"])
        return {"messages": [response]}

    def _write_log(self, file_name: str, message):
        """Write a log message to a file.

        Args:
            message: The log message to write (can be a message object or string).
        """
        f = io.StringIO()
        with redirect_stdout(f):
            if hasattr(message, "pretty_print"):
                message.pretty_print()
            else:
                print(message)
        pretty_output = f.getvalue()
        dir_name = "./.logs"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        log_file_name = os.path.join(dir_name, f"{file_name}.log")
        with open(log_file_name, "a") as log_file:
            log_file.write(pretty_output + "\n\n")

    def _print_stream(self, stream):
        file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        for s in stream:
            message = s["messages"][-1]
            self._write_log(file_name, message)
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()

    def generate(self):
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
        self._print_stream(
            self.app.stream(
                initial_state,
                stream_mode="values",
                config={
                    "configurable": {"thread_id": "wiki-generation-thread"},
                    "recursion_limit": 100,
                },
            )
        )

    def save(self):
        # Save the generated wiki files to the vector database
        pass

    def ask(self, query: str):
        # Start the question answering process
        pass


# ========== 1. RepoInfoAgent ==========
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

    def run(self, repo_path: str, owner: str = None, repo_name: str = None) -> dict:
        """Collect repository information.
        
        Args:
            repo_path (str): Local path to the repository
            owner (str, optional): Repository owner (for remote info)
            repo_name (str, optional): Repository name (for remote info)
            
        Returns:
            dict: Repository information in structured format
        """
        # Construct the initial message
        if owner and repo_name:
            prompt = f"""
                    Collect comprehensive information about the repository:
                    - Local path: {repo_path}
                    - Remote: {owner}/{repo_name}

                    Tasks:
                    1. Use get_repo_basic_info_tool to get basic repository information (name, description, language)
                    2. Use get_repo_structure_tool to get the directory structure (filter out unnecessary directories)
                    3. Use get_repo_commit_info_tool to get the latest 10 commits

                    Please execute these tasks step by step and provide the final result in the JSON format specified in the system prompt.
                    """
        else:
            prompt = f"""
                    Collect comprehensive information about the local repository at: {repo_path}

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
                "configurable": {"thread_id": f"repo-info-{datetime.now().timestamp()}"},
                "recursion_limit": 50,
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
                    json_match = re.search(r'\{[\s\S]*\}', content)
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
                            "raw_output": content
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
                        "raw_output": last_message.content
                    }

        # Fallback if no result
        return {
            "repo_name": os.path.basename(repo_path),
            "description": "No information collected",
            "main_language": "Unknown",
            "structure": [],
            "commits": []
        }

# ========== RepoInfoAgentTest ==========

# 本地仓库
llm = CONFIG.get_llm()
tools = [get_repo_structure_tool, get_repo_basic_info_tool, get_repo_commit_info_tool]
agent = RepoInfoAgent(llm, tools)

repo_info = agent.run(
    repo_path="/mnt/zhongjf25/workspace/repo-agent/.repos/facebook_zstd",
    owner="facebook",
    repo_name="zstd"
)

# repo_info = agent.run(repo_path="/mnt/zhongjf25/workspace/repo-agent/.repos/facebook_zstd")


print(json.dumps(repo_info, indent=2))

# ========== RepoInfoAgentTest ==========

