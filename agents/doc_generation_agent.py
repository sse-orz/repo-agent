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


class DocGenerationAgent(BaseAgent):
    """Agent for generating Wiki documentation.
    
    Inherits from BaseAgent to leverage common workflow patterns.
    """
    
    def __init__(self, repo_path: str = "", wiki_path: str = ""):
        """Initialize the DocGenerationAgent.
        
        Args:
            repo_path (str): Local path to the repository (optional)
            wiki_path (str): Path to save wiki files
        """
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

        # Prepare structured information
        repo_summary = self._prepare_repo_summary(repo_info)
        code_summary = self._prepare_code_summary(code_analysis)
        important_files = self._extract_important_files(code_analysis)

        # Build prompt
        prompt = self._build_prompt(repo_summary, code_summary, important_files, wiki_path)

        # Create initial state
        initial_state = AgentState(
            messages=[HumanMessage(content=prompt)],
            repo_path=self.repo_path,
            wiki_path=wiki_path,
        )

        # Run the agent workflow
        print("\n=== Generating Wiki Documentation ===")
        final_state = None
        generated_files = []

        for state in self.app.stream(
            initial_state,
            stream_mode="values",
            config={
                "configurable": {"thread_id": f"doc-gen-{datetime.now().timestamp()}"},
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

        # Build result
        result = self._build_result(final_state, generated_files, wiki_path)
        
        print(f"\n=== Documentation Generation Complete ===")
        print(f"Total files: {result['total_files']}")
        print(f"Verified files: {len(result['verified_files'])}")

        return result

    def _prepare_repo_summary(self, repo_info: dict) -> dict:
        """Prepare repository summary for documentation.
        
        Args:
            repo_info (dict): Repository information
            
        Returns:
            dict: Structured repository summary
        """
        return {
            "name": repo_info.get("repo_name", "Unknown"),
            "description": repo_info.get("description", "No description"),
            "language": repo_info.get("main_language", "Unknown"),
            "structure": repo_info.get("structure", []),
            "total_commits": len(repo_info.get("commits", [])),
        }

    def _prepare_code_summary(self, code_analysis: dict) -> dict:
        """Prepare code analysis summary for documentation.
        
        Args:
            code_analysis (dict): Code analysis results
            
        Returns:
            dict: Structured code summary
        """
        return {
            "total_files_analyzed": code_analysis.get("analyzed_files", 0),
            "total_functions": code_analysis["summary"].get("total_functions", 0),
            "total_classes": code_analysis["summary"].get("total_classes", 0),
            "average_complexity": code_analysis["summary"].get("average_complexity", 0),
            "languages": code_analysis["summary"].get("languages", []),
        }

    def _extract_important_files(self, code_analysis: dict, max_files: int = 5) -> list:
        """Extract important files for detailed documentation.
        
        Args:
            code_analysis (dict): Code analysis results
            max_files (int): Maximum number of files to extract
            
        Returns:
            list: List of important files with their details
        """
        analyzed_files = code_analysis.get("files", {})
        important_files = []
        
        for file_path, analysis in list(analyzed_files.items())[:max_files]:
            if "error" not in analysis:
                important_files.append({
                    "path": file_path,
                    "functions": [f["name"] for f in analysis.get("functions", [])[:3]],
                    "classes": [c["name"] for c in analysis.get("classes", [])[:3]],
                    "summary": analysis.get("summary", "No summary"),
                })
        
        return important_files

    def _build_prompt(self, repo_summary: dict, code_summary: dict, 
                      important_files: list, wiki_path: str) -> str:
        """Build the prompt for documentation generation.
        
        Args:
            repo_summary (dict): Repository summary
            code_summary (dict): Code analysis summary
            important_files (list): Important files list
            wiki_path (str): Wiki path
            
        Returns:
            str: Complete prompt
        """
        return f"""
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
                    - Ensure all file paths are correct (use absolute paths: {os.path.abspath(wiki_path)}/filename.md)
                """

    def _build_result(self, final_state: AgentState, generated_files: list, 
                      wiki_path: str) -> dict:
        """Build the final result dictionary.
        
        Args:
            final_state (AgentState): Final agent state
            generated_files (list): List of generated file paths
            wiki_path (str): Wiki path
            
        Returns:
            dict: Complete result with verification
        """
        result = {
            "generated_files": generated_files,
            "total_files": len(generated_files),
            "status": "success" if generated_files else "partial",
            "wiki_path": wiki_path,
        }

        # Try to extract JSON summary from LLM response
        if final_state:
            last_message = final_state["messages"][-1]
            if hasattr(last_message, "content"):
                try:
                    content = last_message.content
                    json_match = re.search(r'\{[\s\S]*\}', content)
                    if json_match:
                        parsed_result = json.loads(json_match.group())
                        # Merge with tracked results
                        result.update(parsed_result)
                except json.JSONDecodeError as e:
                    print(f"  ⚠ Warning: Failed to parse final JSON: {e}")
                    result["warning"] = "Could not parse final summary"

        # Verify generated files exist
        verified_files = [f for f in generated_files if os.path.exists(f)]
        result["verified_files"] = verified_files
        result["verification_status"] = (
            "complete" if len(verified_files) == len(generated_files) else "incomplete"
        )

        return result


# ========== DocGenerationAgentTest ==========
def DocGenerationAgentTest():
    doc_agent = DocGenerationAgent(repo_path="", wiki_path="test_output")

    # Mock data for testing
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
        "total_files": 4,
        "analyzed_files": 4,
        "files": {
            "/path/to/file1.c": {
                "language": "C",
                "functions": [
                    {
                        "name": "compress_data",
                        "signature": "int compress_data(void* src, size_t size)",
                    }
                ],
                "classes": [],
                "complexity_score": 6,
                "lines_of_code": 350,
                "summary": "Main compression function implementation",
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

    wiki_path = "test_output"

    result = doc_agent.run(repo_info, code_analysis, wiki_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    DocGenerationAgentTest()