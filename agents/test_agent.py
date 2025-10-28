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
# llm = CONFIG.get_llm()
# tools = [get_repo_structure_tool, get_repo_basic_info_tool, get_repo_commit_info_tool]
# agent = RepoInfoAgent(llm, tools)

# repo_info = agent.run(
#     repo_path="/mnt/zhongjf25/workspace/repo-agent/.repos/facebook_zstd",
#     owner="facebook",
#     repo_name="zstd"
# )

# # repo_info = agent.run(repo_path="/mnt/zhongjf25/workspace/repo-agent/.repos/facebook_zstd")


# print(json.dumps(repo_info, indent=2))

# ========== RepoInfoAgentTest ==========


# ========== 2. CodeAnalysisAgent ==========

class CodeAnalysisAgent:
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
            content="""You are a code analysis expert. Your task is to analyze code files efficiently.

                        IMPORTANT RULES:
                        1. Call code_file_analysis_tool ONLY ONCE per file
                        2. After analyzing all files, return results immediately in JSON format
                        3. Do NOT repeat tool calls if you already have the results

                        For each file, extract:
                        - Main functions/classes with signatures
                        - Dependencies and imports
                        - Complexity score (1-10)
                        - Lines of code

                        Return results in this exact JSON structure:
                        {
                            "files": {
                                "file_path": {
                                    "language": "string",
                                    "functions": [{"name": "...", "signature": "...", "line_start": 0, "line_end": 0}],
                                    "classes": [{"name": "...", "methods": [], "line_start": 0, "line_end": 0}],
                                    "imports": [],
                                    "complexity_score": 5,
                                    "lines_of_code": 100,
                                    "summary": "Brief description"
                                }
                            }
                        }

                        Do NOT include any text outside the JSON structure.
                    """
        )
        response = self.llm.invoke([system_prompt] + state["messages"])
        return {"messages": [response]}

    def run(self, repo_path: str, file_list: list, batch_size: int = 5) -> dict:
        """Analyze core code files.
        
        Args:
            repo_path (str): Local path to the repository
            file_list (list): List of file paths to analyze (relative to repo_path)
            batch_size (int): Number of files to analyze in one batch (default 5)
            
        Returns:
            dict: Analysis results for all files
        """
        if not file_list:
            return {
                "total_files": 0,
                "analyzed_files": 0,
                "files": {},
                "summary": {
                    "total_functions": 0,
                    "total_classes": 0,
                    "average_complexity": 0,
                    "total_lines": 0
                }
            }

        # Filter valid code files (skip non-code files)
        code_extensions = {'.py', '.js', '.java', '.cpp', '.c', '.go', '.rs', '.ts', '.jsx', '.tsx'}
        valid_files = []
        for f in file_list:
            # 如果是相对路径，拼接 repo_path
            if not os.path.isabs(f):
                full_path = os.path.join(repo_path, f)
            else:
                full_path = f
            
            if os.path.exists(full_path):
                if any(full_path.endswith(ext) for ext in code_extensions):
                    valid_files.append(full_path)
            else:
                print(f"Warning: File {full_path} does not exist, skipping...")
        
        if not valid_files:
            return {
                "total_files": len(file_list),
                "analyzed_files": 0,
                "files": {},
                "summary": {
                    "total_functions": 0,
                    "total_classes": 0,
                    "average_complexity": 0,
                    "total_lines": 0
                },
                "warning": "No valid code files found to analyze"
            }

        # Limit to first N files to avoid context overflow
        max_files = 20
        if len(valid_files) > max_files:
            valid_files = valid_files[:max_files]
            print(f"Warning: Analyzing only first {max_files} files to avoid context overflow")

        # Process files in batches
        all_results = {}
        
        for i in range(0, len(valid_files), batch_size):
            batch = valid_files[i:i + batch_size]
            
            prompt = f"""
                        Analyze the following code files:

                        Files to analyze (absolute paths):
                        {json.dumps(batch, indent=2)}

                        For each file:
                        1. Use code_file_analysis_tool with the file path AS IS (already absolute)
                        2. Extract the analysis information as specified in the system prompt
                        3. Calculate a complexity score based on the analysis results

                        After analyzing all files, provide results in this exact JSON format:
                        {{
                            "files": {{
                                "file_path_1": {{
                                    "language": "...",
                                    "functions": [...],
                                    "classes": [...],
                                    "complexity_score": 5,
                                    "lines_of_code": 100,
                                    "summary": "..."
                                }},
                                "file_path_2": {{...}}
                            }}
                        }}

                        Important: 
                        - Call code_file_analysis_tool for EACH file separately
                        - Return results in the exact JSON format above
                        - Do NOT add extra text outside the JSON structure
                    """

            initial_state = AgentState(
                messages=[HumanMessage(content=prompt)],
                repo_path=repo_path,
                wiki_path="",
            )

            # Run the agent workflow
            final_state = None
            for state in self.app.stream(
                initial_state,
                stream_mode="values",
                config={
                    "configurable": {"thread_id": f"code-analysis-{datetime.now().timestamp()}"},
                    "recursion_limit": 100,  # Allow more recursion for multiple file analysis
                },
            ):
                final_state = state
                # Optional: print progress
                last_msg = state["messages"][-1]
                if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                    print(f"  Analyzing files in batch {i//batch_size + 1}...")

            # Extract results from final state
            if final_state:
                last_message = final_state["messages"][-1]
                if hasattr(last_message, "content"):
                    try:
                        content = last_message.content
                        import re
                        json_match = re.search(r'\{[\s\S]*\}', content)
                        if json_match:
                            batch_results = json.loads(json_match.group())
                            if "files" in batch_results:
                                all_results.update(batch_results["files"])
                            else:
                                # If direct file results
                                all_results.update(batch_results)
                    except json.JSONDecodeError as e:
                        print(f"Failed to parse JSON from batch analysis: {e}")
                        # Store raw output for debugging
                        for file_path in batch:
                            all_results[file_path] = {
                                "error": "JSON parse error",
                                "raw_output": last_message.content[:500]
                            }

        # Calculate summary statistics
        total_functions = 0
        total_classes = 0
        total_complexity = 0
        total_lines = 0
        analyzed_count = 0

        for file_path, analysis in all_results.items():
            if "error" not in analysis:
                analyzed_count += 1
                total_functions += len(analysis.get("functions", []))
                total_classes += len(analysis.get("classes", []))
                total_complexity += analysis.get("complexity_score", 0)
                total_lines += analysis.get("lines_of_code", 0)

        avg_complexity = total_complexity / analyzed_count if analyzed_count > 0 else 0

        return {
            "total_files": len(file_list),
            "analyzed_files": analyzed_count,
            "files": all_results,
            "summary": {
                "total_functions": total_functions,
                "total_classes": total_classes,
                "average_complexity": round(avg_complexity, 2),
                "total_lines": total_lines,
                "languages": list(set(
                    analysis.get("language", "unknown") 
                    for analysis in all_results.values() 
                    if "error" not in analysis
                ))
            }
        }


# ========== CodeAnalysisAgentTest ==========
llm = CONFIG.get_llm()
tools = [code_file_analysis_tool, read_file_tool]
code_agent = CodeAnalysisAgent(llm, tools)

repo_path = "/mnt/zhongjf25/workspace/repo-agent/.repos/facebook_zstd"
import os

file_list = []
for root, dirs, files in os.walk(repo_path):
    dirs[:] = [d for d in dirs if d not in ['.git', 'node_modules', '__pycache__']]
    for file in files:
        if file.endswith(('.c', '.h')) and len(file_list) < 4:
            file_list.append(os.path.join(root, file))

print(f"Found files: {file_list}")

code_analysis = code_agent.run(
    repo_path=repo_path,
    file_list=file_list,
    batch_size=4
)

print(json.dumps(code_analysis, indent=2))

# ========== CodeAnalysisAgentTest ==========


# # ========== 3. DocGenerationAgent ==========
# class DocGenerationAgent:
#     def __init__(self, llm):
#         self.llm = llm

#     def run(self, repo_info: dict, code_analysis: dict, wiki_path: str) -> list:
#         """生成 Markdown 文档"""
#         prompt = f"""
#         Generate Wiki documentation based on the following information:
#         Repository info: {json.dumps(repo_info)}
#         Code analysis: {json.dumps(code_analysis)}

#         Generate the following documents:
#         1. README.md (Overview)
#         2. STRUCTURE.md (Directory structure)
#         3. API.md (API documentation)

#         Return a JSON array of filenames and content.
#         """
#         response = self.llm.invoke([HumanMessage(content=prompt)])
#         # 解析并写入文件
#         docs = [
#             {"file": "README.md", "content": "..."},
#             {"file": "STRUCTURE.md", "content": "..."}
#         ]
#         for doc in docs:
#             with open(f"{wiki_path}/{doc['file']}", "w") as f:
#                 f.write(doc["content"])
#         return docs


# # ========== 4. SummaryAgent ==========
# class SummaryAgent:
#     def __init__(self, llm):
#         self.llm = llm

#     def run(self, docs: list, wiki_path: str):
#         """生成索引和目录"""
#         prompt = f"""
#         Generate an index table of contents (INDEX.md) for the following documents:
#         {[doc['file'] for doc in docs]}
#         """
#         response = self.llm.invoke([HumanMessage(content=prompt)])
#         with open(f"{wiki_path}/INDEX.md", "w") as f:
#             f.write(response.content)


# # ========== 5. WikiSupervisor ==========
# class WikiSupervisor:
#     def __init__(self, repo_path: str, wiki_path: str):
#         self.repo_path = repo_path
#         self.wiki_path = wiki_path
#         self.llm = CONFIG.get_llm()

#         # 初始化各阶段 Agent
#         self.repo_agent = RepoInfoAgent(self.llm, [get_repo_structure_tool, ...])
#         self.code_agent = CodeAnalysisAgent(self.llm, [code_file_analysis_tool])
#         self.doc_agent = DocGenerationAgent(self.llm)
#         self.summary_agent = SummaryAgent(self.llm)

#     def generate(self):
#         """分阶段执行"""
#         print("=== 阶段 1: 收集仓库信息 ===")
#         repo_info = self.repo_agent.run(self.repo_path)

#         print("=== 阶段 2: 分析代码 ===")
#         file_list = repo_info.get("key_files", [])
#         code_analysis = self.code_agent.run(self.repo_path, file_list)

#         print("=== 阶段 3: 生成文档 ===")
#         docs = self.doc_agent.run(repo_info, code_analysis, self.wiki_path)

#         print("=== 阶段 4: 生成索引 ===")
#         self.summary_agent.run(docs, self.wiki_path)

#         print("Wiki 生成完成！")


# class SupervisorState(TypedDict):
#     repo_path: str
#     wiki_path: str
#     repo_info: dict
#     code_analysis: dict
#     docs: list


# def build_supervisor_graph():
#     workflow = StateGraph(SupervisorState)

#     workflow.add_node("repo_info", lambda s: {"repo_info": repo_agent.run(s["repo_path"])})
#     workflow.add_node("code_analysis", lambda s: {"code_analysis": code_agent.run(s["repo_path"], s["repo_info"]["files"])})
#     workflow.add_node("doc_gen", lambda s: {"docs": doc_agent.run(s["repo_info"], s["code_analysis"], s["wiki_path"])})
#     workflow.add_node("summary", lambda s: summary_agent.run(s["docs"], s["wiki_path"]))

#     workflow.set_entry_point("repo_info")
#     workflow.add_edge("repo_info", "code_analysis")
#     workflow.add_edge("code_analysis", "doc_gen")
#     workflow.add_edge("doc_gen", "summary")
#     workflow.add_edge("summary", END)

#     return workflow.compile()
