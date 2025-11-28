"""Modified RepoInfoAgent that uses patch-free commit utilities.

This is a modified version of agents/repo_info_agent.py that uses local
tools which exclude patch data from commits to avoid context window overflow.
"""

from config import CONFIG
from agents.tools import get_repo_structure_tool, get_repo_basic_info_tool
from .utils import get_repo_commit_info_tool  # Use local patch-free version
from agents.base_agent import BaseAgent, AgentState

from langchain_core.messages import SystemMessage, HumanMessage
from datetime import datetime
import json
import os
import re


class RepoInfoAgent(BaseAgent):
    """Agent for collecting repository information (with patch-free commits).

    Inherits from BaseAgent to leverage common workflow patterns.
    This version uses modified tools that exclude patch data to reduce token usage.
    """

    def __init__(self, repo_path: str, wiki_path: str = "", llm=None):
        """Initialize the RepoInfoAgent.

        Args:
            repo_path (str): Local path to the repository
            wiki_path (str): Path to wiki (optional, not used by this agent)
        """
        system_prompt = SystemMessage(
            content=""" You are a repository information collector. Your task is to:
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

                        IMPORTANT:
                        - Return ONLY the JSON object, no additional text
                        - If remote repository information is not available, set commits to an empty array
                        - Filter out common build/cache directories from structure
                    """
        )

        tools = [
            get_repo_structure_tool,
            get_repo_basic_info_tool,
            get_repo_commit_info_tool,  # Using patch-free version
        ]

        super().__init__(
            tools=tools,
            system_prompt=system_prompt,
            repo_path=repo_path,
            wiki_path=wiki_path,
            llm=llm,
        )

    def run(self, owner: str = None, repo_name: str = None) -> dict:
        """Collect repository information.

        Args:
            owner (str, optional): Repository owner (for remote info)
            repo_name (str, optional): Repository name (for remote info)

        Returns:
            dict: Repository information in structured format
        """
        # Build the prompt
        prompt_parts = [
            f"Collect comprehensive information about the local repository at: {self.repo_path}",
            "",
            "Tasks:",
            "1. Use get_repo_structure_tool to get the directory structure",
            "2. Extract the repository name from the path",
            "3. Infer the main programming language from file extensions",
        ]

        # Add remote repository info if available
        if owner and repo_name:
            prompt_parts.extend(
                [
                    f"4. Use get_repo_basic_info_tool with owner='{owner}' and repo='{repo_name}' for description",
                    f"5. Use get_repo_commit_info_tool with owner='{owner}' and repo='{repo_name}' for commit history",
                ]
            )
        else:
            prompt_parts.append(
                "4. Set commits to empty array since remote info is not available"
            )

        prompt_parts.append(
            "\nProvide the result in JSON format as specified in the system prompt."
        )

        prompt = "\n".join(prompt_parts)

        # Create initial state
        initial_state = AgentState(
            messages=[HumanMessage(content=prompt)],
            repo_path=self.repo_path,
            wiki_path=self.wiki_path,
        )

        # Run the agent workflow
        print("\n=== Collecting Repository Information ===")
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
            # Optional: log progress
            last_msg = state["messages"][-1]
            if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                for tool_call in last_msg.tool_calls:
                    print(f"  Calling tool: {tool_call['name']}")

        # Extract and parse the result
        return self._extract_result(final_state)

    def _extract_result(self, final_state: AgentState) -> dict:
        """Extract and parse JSON result from final state.

        Args:
            final_state (AgentState): The final state from agent execution

        Returns:
            dict: Parsed repository information
        """
        if not final_state:
            return self._get_fallback_result("No final state")

        last_message = final_state["messages"][-1]

        if not hasattr(last_message, "content"):
            return self._get_fallback_result("No content in final message")

        try:
            content = last_message.content

            # Try to extract JSON from the content
            json_match = re.search(r"\{[\s\S]*\}", content)

            if json_match:
                result = json.loads(json_match.group())

                # Validate required fields
                required_fields = ["repo_name", "main_language", "structure"]
                if all(field in result for field in required_fields):
                    print("  ✓ Repository information collected successfully")
                    return result
                else:
                    print("  ⚠ Missing required fields in result")
                    return self._get_fallback_result(
                        "Missing required fields", partial_data=result
                    )
            else:
                print("  ⚠ No JSON found in response")
                return self._get_fallback_result(
                    "No JSON in response", raw_output=content
                )

        except json.JSONDecodeError as e:
            print(f"  ✗ JSON parse error: {e}")
            return self._get_fallback_result(
                f"JSON parse error: {e}", raw_output=last_message.content
            )
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            return self._get_fallback_result(
                f"Unexpected error: {e}",
                raw_output=(
                    str(last_message.content)
                    if hasattr(last_message, "content")
                    else ""
                ),
            )

    def _get_fallback_result(
        self, error_msg: str, partial_data: dict = None, raw_output: str = None
    ) -> dict:
        """Generate fallback result when parsing fails.

        Args:
            error_msg (str): Error message
            partial_data (dict, optional): Partially parsed data
            raw_output (str, optional): Raw output from LLM

        Returns:
            dict: Fallback repository information
        """
        result = {
            "repo_name": os.path.basename(self.repo_path),
            "description": f"Unable to extract description: {error_msg}",
            "main_language": "Unknown",
            "structure": [],
            "commits": [],
            "error": error_msg,
        }

        if partial_data:
            result.update(partial_data)

        if raw_output:
            result["raw_output"] = raw_output[:1000]  # Limit size

        return result
