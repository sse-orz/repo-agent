from config import CONFIG

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


class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    repo_path: str
    wiki_path: str


class BaseAgent:
    def __init__(
        self, tools, system_prompt: SystemMessage, repo_path: str, wiki_path: str
    ):
        self.llm = CONFIG.get_llm()
        self.repo_path = repo_path
        self.wiki_path = wiki_path
        self.memory = InMemorySaver()
        self.tools = tools
        self.llm_with_tools = self.llm.bind_tools(self.tools, parallel_tool_calls=False)
        self.tool_executor = ToolNode(self.tools)
        self.app = self._build_app(system_prompt)

    def _build_app(self, system_prompt: SystemMessage):
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", lambda state: self._agent_node(system_prompt, state))
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

    def _agent_node(
        self, system_prompt: SystemMessage, state: AgentState
    ) -> AgentState:
        """Call the language model with the current state.

        Args:
            system_prompt (SystemMessage): The system prompt to use.
            state (AgentState): The current state of the agent.

        Returns:
            AgentState: The updated state of the agent after calling the model.
        """
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
