from config import CONFIG
from agents.tools import (
    read_file_tool,
    write_file_tool,
)
from .base_agent import BaseAgent, AgentState

from langchain_core.messages import SystemMessage, HumanMessage
from datetime import datetime
import json
import os
import re


class SummaryAgent(BaseAgent):
    """Agent for generating Wiki index and summary.
    
    Inherits from BaseAgent to leverage common workflow patterns.
    """
    
    def __init__(self, repo_path: str = "", wiki_path: str = ""):
        """Initialize the SummaryAgent.
        
        Args:
            repo_path (str): Local path to the repository (optional)
            wiki_path (str): Path to wiki directory
        """
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
        
        tools = [
            write_file_tool,
            read_file_tool,
        ]
        
        super().__init__(
            tools=tools,
            system_prompt=system_prompt,
            repo_path=repo_path,
            wiki_path=wiki_path
        )

    def run(
        self,
        docs: list,
        wiki_path: str,
        repo_info: dict = None,
        code_analysis: dict = None,
    ) -> dict:
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
        doc_list = self._prepare_document_list(docs)
        
        # Prepare statistics
        statistics = self._prepare_statistics(repo_info, code_analysis)

        # Build prompt
        prompt = self._build_prompt(doc_list, statistics, wiki_path)

        # Create initial state
        initial_state = AgentState(
            messages=[HumanMessage(content=prompt)],
            repo_path=self.repo_path,
            wiki_path=wiki_path,
        )

        # Run the agent workflow
        print("\n=== Generating Wiki Index ===")
        final_state = None
        index_file = None

        for state in self.app.stream(
            initial_state,
            stream_mode="values",
            config={
                "configurable": {"thread_id": f"summary-{datetime.now().timestamp()}"},
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

        # Build and return result
        result = self._build_result(final_state, index_file, docs, wiki_path)
        
        print(f"\n=== Index Generation Complete ===")
        print(f"Index file: {result['index_file']}")
        print(f"Status: {result['verification_status']}")

        return result

    def _prepare_document_list(self, docs: list) -> list:
        """Prepare document list with descriptions.
        
        Args:
            docs (list): List of document paths
            
        Returns:
            list: List of documents with metadata
        """
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
        
        return doc_list

    def _prepare_statistics(self, repo_info: dict = None, 
                           code_analysis: dict = None) -> dict:
        """Prepare repository and code statistics.
        
        Args:
            repo_info (dict, optional): Repository information
            code_analysis (dict, optional): Code analysis results
            
        Returns:
            dict: Structured statistics
        """
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
                "total_functions": code_analysis.get("summary", {}).get(
                    "total_functions", 0
                ),
                "total_classes": code_analysis.get("summary", {}).get(
                    "total_classes", 0
                ),
                "average_complexity": code_analysis.get("summary", {}).get(
                    "average_complexity", 0
                ),
            }
        
        return statistics

    def _build_prompt(self, doc_list: list, statistics: dict, 
                      wiki_path: str) -> str:
        """Build the prompt for index generation.
        
        Args:
            doc_list (list): List of documents
            statistics (dict): Repository statistics
            wiki_path (str): Wiki path
            
        Returns:
            str: Complete prompt
        """
        return f"""
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

    def _build_result(self, final_state: AgentState, index_file: str,
                      docs: list, wiki_path: str) -> dict:
        """Build the final result dictionary.
        
        Args:
            final_state (AgentState): Final agent state
            index_file (str): Path to index file
            docs (list): List of generated documents
            wiki_path (str): Wiki path
            
        Returns:
            dict: Complete result with verification
        """
        result = {
            "index_file": index_file or f"{wiki_path}/INDEX.md",
            "total_documents": len(docs) + 1,  # +1 for INDEX.md
            "status": "success" if index_file else "partial",
            "wiki_path": wiki_path,
        }

        # Try to extract JSON summary from LLM response
        if final_state:
            last_message = final_state["messages"][-1]
            if hasattr(last_message, "content"):
                try:
                    content = last_message.content
                    json_match = re.search(r"\{[\s\S]*\}", content)
                    if json_match:
                        parsed_result = json.loads(json_match.group())
                        result.update(parsed_result)
                except json.JSONDecodeError as e:
                    print(f"   Warning: Failed to parse final JSON: {e}")
                    result["warning"] = "Could not parse final summary"

        # Verify index file exists
        if index_file and os.path.exists(index_file):
            result["verification_status"] = "success"
        else:
            result["verification_status"] = "failed"
            print(f"   Warning: Index file may not have been created")

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
        "structure": [
            ".github",
            "contrib",
            "doc",
            "examples",
            "lib",
            "programs",
            "tests",
        ],
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


if __name__ == "__main__":
    SummaryAgentTest()
