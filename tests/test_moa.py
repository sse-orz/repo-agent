from config import CONFIG
from agents.tools import (
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
from datetime import datetime
import json
import os
import io


class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    repo_path: str
    wiki_path: str


class MoAgent:

    def __init__(self, repo_path: str, wiki_path: str) -> None:
        self.repo_path = repo_path
        self.wiki_path = wiki_path
        self.memory = InMemorySaver()
        self.app = self._build_app()

    def _build_app(self):
        workflow = StateGraph(AgentState)
        # supervisor_agent is the entry point agent to supervise the whole process
        workflow.add_node("supervisor_agent", self.supervisor_agent_node)
        # super_router_node routes the workflow to repo_router_node and code_router_node
        workflow.add_node("super_router", self.super_router_node)
        workflow.add_node("repo_router", self.repo_router_node)
        workflow.add_node("code_router", self.code_router_node)
        # repo_info_agent, repo_commit_agent, repo_pr_agent, repo_release_agent are parallel agents
        workflow.add_node(
            "repo_info_agent", self.repo_info_agent_node
        )  # repo basic info, repo structure
        workflow.add_node(
            "repo_commit_agent", self.repo_commit_agent_node
        )  # repo commit info
        workflow.add_node("repo_pr_agent", self.repo_pr_agent_node)  # repo pr info
        workflow.add_node(
            "repo_release_agent", self.repo_release_agent_node
        )  # repo release info
        # code_analysis_plan_agent, code_analysis_exec_agent are sequential agents
        workflow.add_node(
            "code_analysis_plan_agent", self.code_analysis_plan_agent_node
        )  # code analysis plan: plan code file analysis consequently
        workflow.add_node(
            "code_analysis_exec_agent", self.code_analysis_exec_agent_node
        )  # code analysis execution: execute code file analysis plan
        # wiki_generate_agent is the final agent to generate wiki based on repo info and code analysis result
        workflow.add_node(
            "wiki_generate_agent", self.wiki_generate_agent_node
        )  # wiki generation agent: generate wiki based on repo info or code analysis result
        workflow.add_node(
            "evaluate_agent", self.evaluate_agent_node
        )  # evaluate the generated wiki
        workflow.set_entry_point("supervisor_agent")
        workflow.add_conditional_edges(
            "supervisor_agent",
            self._should_continue,
            {
                "continue": "super_router",
                "end": END,
            },
        )
        # workflow.add_edge("supervisor_agent", "super_router")  # Removed redundant edge per CodeQL warning
        workflow.add_edge("super_router", "repo_router")
        workflow.add_edge("super_router", "code_router")
        # parallel edge
        workflow.add_edge("repo_router", "repo_info_agent")
        workflow.add_edge("repo_router", "repo_commit_agent")
        workflow.add_edge("repo_router", "repo_pr_agent")
        workflow.add_edge("repo_router", "repo_release_agent")
        # sequential edge
        workflow.add_edge("code_router", "code_analysis_plan_agent")
        workflow.add_edge("code_analysis_plan_agent", "code_analysis_exec_agent")
        # edge to wiki_generate_agent from repo_info_agent, repo_commit_agent, repo_pr_agent, repo_release_agent, code_analysis_exec_agent
        workflow.add_edge("repo_info_agent", "wiki_generate_agent")
        workflow.add_edge("repo_commit_agent", "wiki_generate_agent")
        workflow.add_edge("repo_pr_agent", "wiki_generate_agent")
        workflow.add_edge("repo_release_agent", "wiki_generate_agent")
        workflow.add_edge("code_analysis_exec_agent", "wiki_generate_agent")
        workflow.add_edge("wiki_generate_agent", "evaluate_agent")
        workflow.add_edge("evaluate_agent", "supervisor_agent")
        return workflow.compile(checkpointer=self.memory)

    def supervisor_agent_node(self, state: AgentState) -> AgentState:
        """The supervisor agent node."""
        pass

    def super_router_node(self, state: AgentState) -> AgentState:
        """The super router node."""
        pass

    def repo_router_node(self, state: AgentState) -> AgentState:
        """The repo router node."""
        pass

    def code_router_node(self, state: AgentState) -> AgentState:
        """The code router node."""
        pass

    def repo_info_agent_node(self, state: AgentState) -> AgentState:
        """The repo info agent node."""
        pass

    def repo_commit_agent_node(self, state: AgentState) -> AgentState:
        """The repo commit agent node."""
        pass

    def repo_pr_agent_node(self, state: AgentState) -> AgentState:
        """The repo pr agent node."""
        pass

    def repo_release_agent_node(self, state: AgentState) -> AgentState:
        """The repo release agent node."""
        pass

    def code_analysis_plan_agent_node(self, state: AgentState) -> AgentState:
        """The code analysis plan agent node."""
        pass

    def code_analysis_exec_agent_node(self, state: AgentState) -> AgentState:
        """The code analysis execution agent node."""
        pass

    def wiki_generate_agent_node(self, state: AgentState) -> AgentState:
        """The wiki generate agent node."""
        pass

    def evaluate_agent_node(self, state: AgentState) -> AgentState:
        """The evaluate agent node."""
        pass

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


if __name__ == "__main__":
    moagent = MoAgent(
        repo_path="./.repos/TheAlgorithms/Python",
        wiki_path="./.wikis/TheAlgorithms_Python",
    )
    moagent._draw_graph()
