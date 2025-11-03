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
from langchain_core.runnables.graph import MermaidDrawMethod
from typing import TypedDict, Annotated, Sequence
from contextlib import redirect_stdout
from datetime import datetime
import json
import os
import io

from prompt import WikiPrompts

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
        system_prompt = WikiPrompts.get_system_prompt()
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

    def _print_stream(self, file_name: str, stream):
        for s in stream:
            message = s["messages"][-1]
            self._write_log(file_name, message)
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()

    def _draw_graph(self):
        img = self.app.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
        dir_name = "./.graphs"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        graph_file_name = os.path.join(
            dir_name,
            f"wiki_agent_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png",
        )
        with open(graph_file_name, "wb") as f:
            f.write(img)

    def generate(self):
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

    def save(self):
        # Save the generated wiki files to the vector database
        pass

    def ask(self, query: str):
        # Start the question answering process
        pass


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
    agent.generate()
    # agent._draw_graph()
