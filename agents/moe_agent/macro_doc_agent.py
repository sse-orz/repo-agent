import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from config import CONFIG
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from agents.base_agent import BaseAgent, AgentState

from agents.tools import write_file_tool, edit_file_tool
from .utils import ls_context_file_tool, read_file_tool
from .summarization import ConversationSummarizer
from .prompt import MacroDocPrompt


class MacroDocAgent(BaseAgent):
    """Agent for generating macro-level project documentation."""

    # Document types and their specifications (from centralized prompt module)
    DOC_SPECS = MacroDocPrompt.DOC_SPECS
    DIAGRAM_REQUIREMENTS = MacroDocPrompt.DIAGRAM_REQUIREMENTS

    def __init__(
        self,
        owner: str,
        repo_name: str,
        llm=None,
        max_workers: int = 5,
        wiki_base_path: str = None,
    ):
        """Initialize the MacroDocAgent.

        Args:
            owner (str): Repository owner
            repo_name (str): Repository name
            llm: Language model instance. If None, uses CONFIG.get_llm()
            max_workers (int): Maximum number of parallel workers for doc generation
            wiki_base_path (str): Base wiki path (optional, defaults to .wikis/{owner}_{repo_name})
        """
        self.owner = owner
        self.repo_name = repo_name
        self.llm = llm if llm else CONFIG.get_llm()
        self.max_workers = max_workers

        # Derive paths from owner and repo_name
        self.repo_identifier = f"{owner}_{repo_name}"
        if wiki_base_path:
            self.wiki_base_path = Path(wiki_base_path)
        else:
            self.wiki_base_path = Path(f".wikis/{self.repo_identifier}")
        self.cache_base_path = Path(f".cache/{self.repo_identifier}")

        # Ensure directories exist
        self.wiki_base_path.mkdir(parents=True, exist_ok=True)

        # Use centralized prompt module for system prompt
        system_prompt = MacroDocPrompt.get_system_prompt()

        tools = [
            read_file_tool,
            write_file_tool,
            ls_context_file_tool,
            edit_file_tool,
        ]

        self.summarizer = ConversationSummarizer(
            model=self.llm,
            max_tokens_before_summary=8000,
            messages_to_keep=20,
        )

        super().__init__(
            tools=tools,
            system_prompt=system_prompt,
            repo_path="",
            wiki_path=str(self.wiki_base_path),
            llm=self.llm,
        )

    def _agent_node(
        self, system_prompt: SystemMessage, state: AgentState
    ) -> AgentState:
        messages = list(state["messages"])
        summary_update = self.summarizer.build_messages_update(messages)
        if summary_update:
            summarized_messages = [
                m
                for m in summary_update["messages"]
                if not isinstance(m, RemoveMessage)
            ]
        else:
            summarized_messages = messages

        response = self.llm_with_tools.invoke([system_prompt] + summarized_messages)

        extra_messages = summary_update["messages"] if summary_update else []
        return {"messages": extra_messages + [response]}

    def generate_single_doc(
        self, doc_type: str, repo_info: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate a single macro document.

        Args:
            doc_type (str): Type of document to generate
            repo_info (Dict[str, Any]): Optional repository information

        Returns:
            Dict[str, Any]: Generation result
        """
        print(f"\n📄 Generating {doc_type}...")

        doc_path = self.wiki_base_path / doc_type

        # Use centralized prompt module
        prompt_message = MacroDocPrompt.get_generation_prompt(
            doc_type=doc_type,
            repo_identifier=self.repo_identifier,
            wiki_base_path=str(self.wiki_base_path),
            repo_info=repo_info,
        )
        prompt = prompt_message.content

        try:
            # Invoke the agent workflow
            initial_state = AgentState(
                messages=[HumanMessage(content=prompt)],
                repo_path="",
                wiki_path=str(self.wiki_base_path),
            )

            _ = self.app.invoke(
                initial_state,
                config={
                    "configurable": {
                        "thread_id": f"macro-doc-{self.repo_identifier}-{doc_type}"
                    },
                    "recursion_limit": 60,
                },
            )

            # Check if document was created
            if doc_path.exists():
                file_size = doc_path.stat().st_size
                return {
                    "doc_type": doc_type,
                    "path": str(doc_path),
                    "status": "success",
                    "size": file_size,
                }
            else:
                print(f"⚠️  {doc_type} not created by agent, creating fallback...")
                return self._create_fallback_doc(doc_type, doc_path)

        except Exception as e:
            print(f"❌ Error generating {doc_type}: {e}")
            return {
                "doc_type": doc_type,
                "path": None,
                "status": "error",
                "error": str(e),
            }

    def _create_fallback_doc(self, doc_type: str, doc_path: Path) -> Dict[str, Any]:
        """Create a fallback document when agent fails.

        Args:
            doc_type (str): Type of document
            doc_path (Path): Path to save document

        Returns:
            Dict[str, Any]: Result information
        """
        spec = self.DOC_SPECS[doc_type]

        content = f"""# {spec['title']}

> Documentation for {self.repo_identifier}

## Overview
*Documentation generation in progress...*

"""

        for section in spec["sections"]:
            # Convert section to title case
            section_title = section.capitalize()
            content += f"\n## {section_title}\n\n*To be documented...*\n"

        # Save fallback document
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(content)

        return {
            "doc_type": doc_type,
            "path": str(doc_path),
            "status": "fallback",
            "size": len(content),
        }

    def generate_all_docs(
        self, repo_info: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """Generate all macro documentation files (Always Parallel).

        Args:
            repo_info (Dict[str, Any]): Optional repository information

        Returns:
            List[Dict[str, Any]]: List of results for each document
        """
        doc_types = list(self.DOC_SPECS.keys())

        print(
            f"\n🚀 Starting macro documentation generation for {len(doc_types)} documents..."
        )
        print(f"   Parallel processing: Enabled (max_workers={self.max_workers})")

        results = []

        # Parallel generation using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_doc = {
                executor.submit(self.generate_single_doc, doc_type, repo_info): doc_type
                for doc_type in doc_types
            }

            # Collect results as they complete
            for future in as_completed(future_to_doc):
                doc_type = future_to_doc[future]
                try:
                    result = future.result()
                    results.append(result)

                    if result["status"] == "success":
                        print(f"✅ {doc_type}: Success ({result.get('size', 0)} bytes)")
                    elif result["status"] == "fallback":
                        print(f"⚠️  {doc_type}: Fallback created")
                    else:
                        print(f"❌ {doc_type}: Failed")
                except Exception as e:
                    print(f"❌ {doc_type}: Exception - {e}")
                    results.append(
                        {
                            "doc_type": doc_type,
                            "path": None,
                            "status": "error",
                            "error": str(e),
                        }
                    )

        print(f"\n✨ Macro documentation generation complete!")
        print(f"   Success: {sum(1 for r in results if r['status'] == 'success')}")
        print(f"   Fallback: {sum(1 for r in results if r['status'] == 'fallback')}")
        print(f"   Failed: {sum(1 for r in results if r['status'] == 'error')}")

        return results

    def update_docs_incremental(
        self,
        doc_types: Optional[List[str]] = None,
        change_summary: Optional[str] = None,
        repo_info: Optional[Dict[str, Any]] = None,
        full_change_info: Optional[Dict[str, Any]] = None,  # NEW
    ) -> List[Dict[str, Any]]:
        """Incrementally update existing macro docs using edit operations."""
        doc_types = doc_types or list(self.DOC_SPECS.keys())
        summary_text = change_summary or "No detailed summary was provided."
        results: List[Dict[str, Any]] = []

        print(
            f"\n✏️  Incrementally updating macro docs ({len(doc_types)} files) using edit workflow..."
        )

        # Create temporary change cache with FULL change details for LLM
        change_cache_path = self.cache_base_path / "incremental_changes_temp.json"
        if full_change_info:
            full_change_info["_metadata"] = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "text_summary": summary_text,
            }
            with open(change_cache_path, "w", encoding="utf-8") as f:
                json.dump(full_change_info, f, indent=2, ensure_ascii=False)
            print(f"   💾 Change cache saved: {change_cache_path}")

        for doc_type in doc_types:
            doc_path = self.wiki_base_path / doc_type
            if not doc_path.exists():
                print(f"   ⚠️  {doc_type} missing. Generating full document instead.")
                results.append(self.generate_single_doc(doc_type, repo_info))
                continue

            # Use centralized prompt module
            prompt_message = MacroDocPrompt.get_incremental_update_prompt(
                doc_type=doc_type,
                repo_identifier=self.repo_identifier,
                doc_path=str(doc_path),
                change_cache_path=str(change_cache_path),
                repo_info=repo_info,
            )
            prompt = prompt_message.content

            try:
                initial_state = AgentState(
                    messages=[HumanMessage(content=prompt)],
                    repo_path="",
                    wiki_path=str(self.wiki_base_path),
                )

                _ = self.app.invoke(
                    initial_state,
                    config={
                        "configurable": {
                            "thread_id": f"macro-doc-incremental-{self.repo_identifier}-{doc_type}"
                        },
                        "recursion_limit": 60,
                    },
                )

                results.append(
                    {
                        "doc_type": doc_type,
                        "path": str(doc_path),
                        "status": "updated",
                        "size": doc_path.stat().st_size,
                    }
                )
                print(f"   ✅ {doc_type}: incrementally updated")
            except Exception as e:
                print(f"   ❌ {doc_type}: Incremental update failed - {e}")
                results.append(
                    {
                        "doc_type": doc_type,
                        "path": str(doc_path),
                        "status": "error",
                        "error": str(e),
                    }
                )

        # Cleanup temporary change cache
        if change_cache_path.exists():
            try:
                change_cache_path.unlink()
                print(f"   🗑️  Cleaned up temporary change cache")
            except Exception as e:
                print(f"   ⚠️  Failed to cleanup change cache: {e}")

        return results
