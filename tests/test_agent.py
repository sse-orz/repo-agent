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
from typing import TypedDict, Annotated, Sequence, Literal, Dict, Any
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
                "configurable": {
                    "thread_id": f"repo-info-{datetime.now().timestamp()}"
                },
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
    tools = [get_repo_structure_tool, get_repo_basic_info_tool, get_repo_commit_info_tool]
    agent = RepoInfoAgent(llm, tools)

    repo_info = agent.run(
        repo_path="/mnt/zhongjf25/workspace/repo-agent/.repos/facebook_zstd",
        owner="facebook",
        repo_name="zstd"
    )

    # repo_info = agent.run(repo_path="/mnt/zhongjf25/workspace/repo-agent/.repos/facebook_zstd")


    print(json.dumps(repo_info, indent=2))
    
# RepoInfoAgentTest()
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
                    "total_lines": 0,
                },
            }

        # Filter valid code files (skip non-code files)
        code_extensions = {
            ".py",
            ".js",
            ".java",
            ".cpp",
            ".c",
            ".go",
            ".rs",
            ".ts",
            ".jsx",
            ".tsx",
            ".h",
            ".hpp",
        }
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
                    "total_lines": 0,
                },
                "warning": "No valid code files found to analyze",
            }

        # Limit to first N files to avoid context overflow
        max_files = 20
        if len(valid_files) > max_files:
            valid_files = valid_files[:max_files]
            print(
                f"Warning: Analyzing only first {max_files} files to avoid context overflow"
            )

        # Process files in batches
        all_results = {}

        for i in range(0, len(valid_files), batch_size):
            batch = valid_files[i : i + batch_size]

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
                    "configurable": {
                        "thread_id": f"code-analysis-{datetime.now().timestamp()}"
                    },
                    "recursion_limit": 100,  # Allow more recursion for multiple file analysis
                },
            ):
                final_state = state
                # Optional: print progress
                last_msg = state["messages"][-1]
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    print(f"  Analyzing files in batch {i//batch_size + 1}...")

            # Extract results from final state
            if final_state:
                last_message = final_state["messages"][-1]
                if hasattr(last_message, "content"):
                    try:
                        content = last_message.content
                        import re

                        json_match = re.search(r"\{[\s\S]*\}", content)
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
                                "raw_output": last_message.content[:500],
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
                "languages": list(
                    set(
                        analysis.get("language", "unknown")
                        for analysis in all_results.values()
                        if "error" not in analysis
                    )
                ),
            },
        }


# ========== CodeAnalysisAgentTest ==========
def CodeAnalysisAgentTest():
    llm = CONFIG.get_llm()
    tools = [code_file_analysis_tool, read_file_tool]
    code_agent = CodeAnalysisAgent(llm, tools)

    repo_path = "/mnt/zhongjf25/workspace/repo-agent/.repos/facebook_zstd"
    import os

    file_list = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in [".git", "node_modules", "__pycache__"]]
        for file in files:
            if file.endswith((".c", ".h")) and len(file_list) < 4:
                file_list.append(os.path.join(root, file))

    print(f"Found files: {file_list}")

    code_analysis = code_agent.run(repo_path=repo_path, file_list=file_list, batch_size=4)

    print(json.dumps(code_analysis, indent=2))

# CodeAnalysisAgentTest()
# ========== CodeAnalysisAgentTest ==========


# ========== 3. DocGenerationAgent ==========
class DocGenerationAgent:
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
        workflow.add_node("tools", self._tool_executor_node)
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

    def _tool_executor_node(self, state: AgentState) -> AgentState:
        """Execute tools and return results."""
        return self.tool_executor.invoke(state)

    def _agent_node(self, state: AgentState) -> AgentState:
        """Call LLM with current state."""
        system_prompt = SystemMessage(
            content="""You are a technical documentation writer. Your task is to generate comprehensive Wiki documentation.

                        IMPORTANT RULES:
                        1. Generate well-structured Markdown documents
                        2. Use appropriate headers, lists, code blocks, and formatting
                        3. Include examples and explanations where necessary
                        4. Use write_file_tool to save each document
                        5. After writing all files, return a summary in JSON format

                        Generate the following documents:
                        1. **README.md**: Overview of the repository
                        - Project name and description
                        - Main features
                        - Technology stack
                        - Quick start guide
                        
                        2. **ARCHITECTURE.md**: System architecture
                        - Directory structure explanation
                        - Main components and their responsibilities
                        - Data flow and interactions
                        
                        3. **API.md**: API documentation (if applicable)
                        - Main functions/classes
                        - Parameters and return values
                        - Usage examples
                        
                        4. **DEVELOPMENT.md**: Development guide
                        - Setup instructions
                        - Build and test commands
                        - Contribution guidelines

                        Return final summary in this JSON format:
                        {
                            "generated_files": ["list", "of", "file", "paths"],
                            "total_files": 4,
                            "status": "success"
                        }

                        Do NOT include text outside the JSON structure in your final response.
                    """
        )
        response = self.llm.invoke([system_prompt] + state["messages"])
        return {"messages": [response]}

    def run(self, repo_info: dict, code_analysis: dict, wiki_path: str) -> dict:
        """Generate Markdown documentation.

        Args:
            repo_info (dict): Repository information from RepoInfoAgent
            code_analysis (dict): Code analysis results from CodeAnalysisAgent
            wiki_path (str): Path to save wiki files

        Returns:
            dict: Summary of generated documents
        """
        # Ensure wiki directory exists
        if not os.path.exists(wiki_path):
            os.makedirs(wiki_path)
            print(f"Created wiki directory: {wiki_path}")

        # Prepare structured information for the agent
        repo_summary = {
            "name": repo_info.get("repo_name", "Unknown"),
            "description": repo_info.get("description", "No description"),
            "language": repo_info.get("main_language", "Unknown"),
            "structure": repo_info.get("structure", []),
            "total_commits": len(repo_info.get("commits", [])),
        }

        code_summary = {
            "total_files_analyzed": code_analysis.get("analyzed_files", 0),
            "total_functions": code_analysis["summary"].get("total_functions", 0),
            "total_classes": code_analysis["summary"].get("total_classes", 0),
            "average_complexity": code_analysis["summary"].get("average_complexity", 0),
            "languages": code_analysis["summary"].get("languages", []),
        }

        # Get top 5 most important files for detailed documentation
        analyzed_files = code_analysis.get("files", {})
        important_files = []
        for file_path, analysis in list(analyzed_files.items())[:5]:
            if "error" not in analysis:
                important_files.append({
                    "path": file_path,
                    "functions": [f["name"] for f in analysis.get("functions", [])[:3]],
                    "classes": [c["name"] for c in analysis.get("classes", [])[:3]],
                    "summary": analysis.get("summary", "No summary"),
                })

        prompt = f"""
                    Generate comprehensive Wiki documentation for the repository.

                    **Repository Information:**
                    {json.dumps(repo_summary, indent=2)}

                    **Code Analysis Summary:**
                    {json.dumps(code_summary, indent=2)}

                    **Important Files (for API documentation):**
                    {json.dumps(important_files, indent=2)}

                    **Target Wiki Directory:** {wiki_path}

                    **Tasks:**
                    1. Generate README.md with project overview, features, and quick start
                    2. Generate ARCHITECTURE.md with directory structure and component explanations
                    3. Generate API.md with documentation for the important files listed above
                    4. Generate DEVELOPMENT.md with setup and development instructions

                    Use write_file_tool to save each document to the wiki directory.
                    After completing all documents, provide a summary in the JSON format specified.

                    **Important Guidelines:**
                    - Use clear headings and subheadings
                    - Include code examples in appropriate language-specific code blocks
                    - Add tables for structured information where applicable
                    - Keep language professional and concise
                    - Ensure all file paths are correct (use absolute paths)
                """

        initial_state = AgentState(
            messages=[HumanMessage(content=prompt)],
            repo_path="",
            wiki_path=wiki_path,
        )

        # Run the agent workflow
        print("=== Generating Wiki Documentation ===")
        final_state = None
        generated_files = []
        
        for state in self.app.stream(
            initial_state,
            stream_mode="values",
            config={
                "configurable": {
                    "thread_id": f"doc-gen-{datetime.now().timestamp()}"
                },
                "recursion_limit": 50,
            },
        ):
            final_state = state
            last_msg = state["messages"][-1]
            
            # Track file generation
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                for tool_call in last_msg.tool_calls:
                    if tool_call["name"] == "write_file_tool":
                        file_path = tool_call["args"].get("file_path", "")
                        if file_path and file_path not in generated_files:
                            generated_files.append(file_path)
                            print(f"  ✓ Generated: {os.path.basename(file_path)}")

        # Extract final result
        result = {
            "generated_files": generated_files,
            "total_files": len(generated_files),
            "status": "success" if generated_files else "partial",
            "wiki_path": wiki_path,
        }

        if final_state:
            last_message = final_state["messages"][-1]
            if hasattr(last_message, "content"):
                try:
                    content = last_message.content
                    import re
                    json_match = re.search(r"\{[\s\S]*\}", content)
                    if json_match:
                        parsed_result = json.loads(json_match.group())
                        # Merge with tracked results
                        result.update(parsed_result)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse final JSON: {e}")
                    result["warning"] = "Could not parse final summary"

        # Verify generated files
        verified_files = [f for f in generated_files if os.path.exists(f)]
        result["verified_files"] = verified_files
        result["verification_status"] = "complete" if len(verified_files) == len(generated_files) else "incomplete"

        print(f"\n=== Documentation Generation Complete ===")
        print(f"Total files: {result['total_files']}")
        print(f"Verified files: {len(verified_files)}")
        
        return result


# ========== DocGenerationAgentTest ==========
def DocGenerationAgentTest():
    llm = CONFIG.get_llm()
    tools = [write_file_tool, read_file_tool]
    doc_agent = DocGenerationAgent(llm, tools)

    # Mock data for testing
    repo_info = {
        "repo_name": "facebook_zstd",
        "description": "Zstandard - Fast real-time compression algorithm",
        "main_language": "C",
        "structure": [".github", "contrib", "doc", "examples", "lib", "programs", "tests"],
        "commits": [{"sha": "abc123", "message": "Update API"}],
    }

    code_analysis = {
        "total_files": 4,
        "analyzed_files": 4,
        "files": {
            "/path/to/file1.c": {
                "language": "C",
                "functions": [
                    {"name": "compress_data", "signature": "int compress_data(void* src, size_t size)"}
                ],
                "classes": [],
                "complexity_score": 6,
                "lines_of_code": 350,
                "summary": "Main compression function implementation"
            },
        },
        "summary": {
            "total_functions": 25,
            "total_classes": 0,
            "average_complexity": 5.5,
            "total_lines": 1500,
            "languages": ["C"],
        },
    }

    wiki_path = "/mnt/zhongjf25/workspace/repo-agent/.wikis/test_output"

    result = doc_agent.run(repo_info, code_analysis, wiki_path)
    print(json.dumps(result, indent=2))

# DocGenerationAgentTest()
# ========== DocGenerationAgentTest ==========


# ========== 4. SummaryAgent ==========
class SummaryAgent:
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
            content="""You are a documentation organizer. Your task is to create a comprehensive index and summary for Wiki documentation.

                        IMPORTANT RULES:
                        1. Generate a well-structured INDEX.md file
                        2. Include links to all generated documents
                        3. Provide brief descriptions for each document
                        4. Create a logical navigation structure
                        5. Use write_file_tool to save the index file
                        6. Return a summary in JSON format

                        The INDEX.md should include:
                        1. **Table of Contents**: Links to all wiki documents
                        2. **Document Descriptions**: Brief overview of each document's purpose
                        3. **Quick Links**: Navigation to important sections
                        4. **Repository Statistics**: Summary of analyzed data

                        Return final result in this JSON format:
                        {
                            "index_file": "path/to/INDEX.md",
                            "total_documents": 5,
                            "status": "success"
                        }

                        Do NOT include text outside the JSON structure in your final response.
                    """
        )
        response = self.llm.invoke([system_prompt] + state["messages"])
        return {"messages": [response]}

    def run(self, docs: list, wiki_path: str, repo_info: dict = None, code_analysis: dict = None) -> dict:
        """Generate index and table of contents.

        Args:
            docs (list): List of generated documents with metadata
            wiki_path (str): Path to the wiki directory
            repo_info (dict, optional): Repository information for statistics
            code_analysis (dict, optional): Code analysis summary for statistics

        Returns:
            dict: Summary of index generation
        """
        # Prepare document information
        doc_descriptions = {
            "README.md": "Project overview, features, and quick start guide",
            "ARCHITECTURE.md": "System architecture and component explanations",
            "API.md": "API documentation with usage examples",
            "DEVELOPMENT.md": "Setup instructions and development guidelines",
        }

        doc_list = []
        for doc_path in docs:
            filename = os.path.basename(doc_path)
            doc_list.append({
                "filename": filename,
                "path": doc_path,
                "description": doc_descriptions.get(filename, "Documentation file"),
            })

        # Prepare statistics
        statistics = {}
        if repo_info:
            statistics["repository"] = {
                "name": repo_info.get("repo_name", "Unknown"),
                "language": repo_info.get("main_language", "Unknown"),
                "directories": len(repo_info.get("structure", [])),
                "commits": len(repo_info.get("commits", [])),
            }
        
        if code_analysis:
            statistics["code_analysis"] = {
                "files_analyzed": code_analysis.get("analyzed_files", 0),
                "total_functions": code_analysis.get("summary", {}).get("total_functions", 0),
                "total_classes": code_analysis.get("summary", {}).get("total_classes", 0),
                "average_complexity": code_analysis.get("summary", {}).get("average_complexity", 0),
            }

        prompt = f"""
                    Generate a comprehensive INDEX.md file for the Wiki documentation.

                    **Generated Documents:**
                    {json.dumps(doc_list, indent=2)}

                    **Repository Statistics:**
                    {json.dumps(statistics, indent=2)}

                    **Target Wiki Directory:** {wiki_path}

                    **Tasks:**
                    1. Create INDEX.md with:
                    - Project title and description
                    - Table of contents with links to all documents
                    - Brief description for each document
                    - Repository statistics section
                    - Quick navigation links

                    2. Use Markdown formatting:
                    - Use headers (##, ###) for sections
                    - Use bullet points for lists
                    - Use links: [Document Name](./filename.md)
                    - Use tables for statistics

                    3. Save the file using write_file_tool to {wiki_path}/INDEX.md

                    **Example Structure:**
                    ```markdown
                    # Project Documentation Index

                    ## Table of Contents
                    - [README](./README.md) - Project overview
                    - [Architecture](./ARCHITECTURE.md) - System design
                    ...

                    ## Repository Statistics
                    | Metric | Value |
                    |--------|-------|
                    | Language | C |
                    | Functions | 25 |
                    ...

                    ## Quick Links
                    - [Getting Started](./README.md#quick-start)
                    - [API Reference](./API.md)
                    ```

                    After saving the file, return the summary in JSON format.
                """

        initial_state = AgentState(
            messages=[HumanMessage(content=prompt)],
            repo_path="",
            wiki_path=wiki_path,
        )

        # Run the agent workflow
        print("=== Generating Wiki Index ===")
        final_state = None
        index_file = None

        for state in self.app.stream(
            initial_state,
            stream_mode="values",
            config={
                "configurable": {
                    "thread_id": f"summary-{datetime.now().timestamp()}"
                },
                "recursion_limit": 30,
            },
        ):
            final_state = state
            last_msg = state["messages"][-1]

            # Track index file creation
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                for tool_call in last_msg.tool_calls:
                    if tool_call["name"] == "write_file_tool":
                        file_path = tool_call["args"].get("file_path", "")
                        if "INDEX.md" in file_path:
                            index_file = file_path
                            print(f"  ✓ Generated: INDEX.md")

        # Extract final result
        result = {
            "index_file": index_file or f"{wiki_path}/INDEX.md",
            "total_documents": len(docs) + 1,  # +1 for INDEX.md
            "status": "success" if index_file else "partial",
            "wiki_path": wiki_path,
        }

        if final_state:
            last_message = final_state["messages"][-1]
            if hasattr(last_message, "content"):
                try:
                    content = last_message.content
                    import re
                    json_match = re.search(r"\{[\s\S]*\}", content)
                    if json_match:
                        parsed_result = json.loads(json_match.group())
                        result.update(parsed_result)
                except json.JSONDecodeError as e:
                    print(f"Warning: Failed to parse final JSON: {e}")
                    result["warning"] = "Could not parse final summary"

        # Verify index file exists
        if index_file and os.path.exists(index_file):
            result["verification_status"] = "success"
            print(f"\n=== Index Generation Complete ===")
            print(f"Index file: {index_file}")
        else:
            result["verification_status"] = "failed"
            print(f"\nWarning: Index file may not have been created")

        return result


# ========== SummaryAgentTest ==========
def SummaryAgentTest():
    llm = CONFIG.get_llm()
    tools = [write_file_tool, read_file_tool]
    summary_agent = SummaryAgent(llm, tools)

    # Mock data for testing
    docs = [
        "/mnt/zhongjf25/workspace/repo-agent/.wikis/test_output/README.md",
        "/mnt/zhongjf25/workspace/repo-agent/.wikis/test_output/ARCHITECTURE.md",
        "/mnt/zhongjf25/workspace/repo-agent/.wikis/test_output/API.md",
        "/mnt/zhongjf25/workspace/repo-agent/.wikis/test_output/DEVELOPMENT.md",
    ]

    repo_info = {
        "repo_name": "facebook_zstd",
        "description": "Zstandard - Fast real-time compression algorithm",
        "main_language": "C",
        "structure": [".github", "contrib", "doc", "examples", "lib", "programs", "tests"],
        "commits": [{"sha": "abc123", "message": "Update API"}],
    }

    code_analysis = {
        "analyzed_files": 4,
        "summary": {
            "total_functions": 25,
            "total_classes": 0,
            "average_complexity": 5.5,
            "total_lines": 1500,
        },
    }

    wiki_path = "/mnt/zhongjf25/workspace/repo-agent/.wikis/test_output"

    result = summary_agent.run(docs, wiki_path, repo_info, code_analysis)
    print(json.dumps(result, indent=2))

# SummaryAgentTest()
# ========== SummaryAgentTest ==========
