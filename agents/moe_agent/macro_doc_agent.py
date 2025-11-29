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


class MacroDocAgent(BaseAgent):
    """Agent for generating macro-level project documentation."""

    # Document types and their specifications
    DOC_SPECS = {
        "README.md": {
            "title": "README",
            "sections": [
                "Project name and description",
                "Main features and capabilities",
                "Technology stack",
                "Quick start guide",
                "Installation instructions",
                "Basic usage examples",
            ],
            "audience": "new users and developers",
        },
        "ARCHITECTURE.md": {
            "title": "Architecture",
            "sections": [
                "System architecture overview",
                "Directory structure explanation",
                "Main components and their responsibilities",
                "Data flow and interactions",
                "Design patterns and principles",
                "Module dependencies",
            ],
            "audience": "developers and architects",
        },
        "DEVELOPMENT.md": {
            "title": "Development Guide",
            "sections": [
                "Development environment setup",
                "Build and compilation instructions",
                "Testing guidelines and commands",
                "Debugging tips",
                "Contribution guidelines",
                "Code style and standards",
            ],
            "audience": "contributors and maintainers",
        },
        "API.md": {
            "title": "API Reference",
            "sections": [
                "Main APIs and interfaces",
                "Function/method signatures",
                "Parameters and return values",
                "Usage examples for each API",
                "Error handling",
                "Best practices",
            ],
            "audience": "API users and integrators",
        },
    }
    DIAGRAM_REQUIREMENTS = {
        "README.md": "Include a high-level system architecture diagram using ```mermaid graph LR``` that shows main components, services, and data stores plus their relationships.",
        "ARCHITECTURE.md": "Include at least two diagrams: (1) an overall system architecture diagram (modules/services and their relationships) and (2) a key data flow or call chain flowchart.",
        "DEVELOPMENT.md": "Include a development/build/test process flowchart (e.g., CI/CD) using ```mermaid flowchart```.",
        "API.md": "Include a typical request sequence diagram using ```mermaid sequenceDiagram``` (e.g., Client → API → Service → DB).",
    }

    def __init__(
        self,
        owner: str,
        repo_name: str,
        llm=None,
        max_workers: int = 5,
    ):
        """Initialize the MacroDocAgent.

        Args:
            owner (str): Repository owner
            repo_name (str): Repository name
            llm: Language model instance. If None, uses CONFIG.get_llm()
            max_workers (int): Maximum number of parallel workers for doc generation
        """
        self.owner = owner
        self.repo_name = repo_name
        self.llm = llm if llm else CONFIG.get_llm()
        self.max_workers = max_workers

        # Derive paths from owner and repo_name
        self.repo_identifier = f"{owner}_{repo_name}"
        self.wiki_base_path = Path(f".wikis/{self.repo_identifier}")
        self.cache_base_path = Path(f".cache/{self.repo_identifier}")

        # Ensure directories exist
        self.wiki_base_path.mkdir(parents=True, exist_ok=True)

        system_prompt = SystemMessage(
            content="""You are a technical documentation expert specializing in macro-level project documentation (README, ARCHITECTURE, DEVELOPMENT, API) for software repositories.

## Available Tools
- `ls_context_file_tool`: Discover available context files in a directory
- `read_file_tool`: Read context files (JSON analysis data)
- `write_file_tool`: Write final documentation (call EXACTLY ONCE per task)
- `edit_file_tool`: Edit existing documentation files

## Directory Convention
| Directory | Purpose | Action |
|-----------|---------|--------|
| `.cache/{repo}/` | Pre-analyzed JSON context (repo_info, module_analysis) | READ |
| `.repos/{repo}/` | Original source code | DO NOT READ (use .cache instead) |
| `.wikis/{repo}/` | Output documentation | WRITE (your target) |

## Core Principles
1. **Gather First, Write Once**: Collect all needed information before writing; call `write_file_tool` exactly once with complete content
2. **Path Discipline**: Always verify write path starts with `.wikis/` and matches the assigned filename
3. **Stop After Write**: After successful write, task is complete—no further tool calls

## Quality Standards
- Follow Markdown best practices with proper heading hierarchy
- Include concrete examples and code snippets where appropriate
- Tailor content to the specified target audience
- Use mermaid diagrams for architecture visualization
"""
        )

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

        spec = self.DOC_SPECS[doc_type]

        # Build prompt (doc-type specific instructions moved here)
        target_file_path = str(self.wiki_base_path / doc_type)

        prompt = f"""# Task: Generate {doc_type}

**Repository**: {self.repo_identifier}
**Output Path**: `{target_file_path}`
**Target Audience**: {spec['audience']}

## Required Sections
{chr(10).join(f"- {section}" for section in spec['sections'])}

## Context Resources
Start by exploring available context:
```
ls_context_file_tool('.cache/{self.repo_identifier}/')
```

Key context files:
- `repo_info.json` — Repository metadata and structure
- `module_structure.json` — Module organization and dependencies
- `module_analysis/*.json` — Static analysis per module

## Execution Steps
1. **Explore**: List context files in `.cache/{self.repo_identifier}/`
2. **Gather**: Read relevant JSON context files to understand the project
3. **Generate**: Create comprehensive {spec['title']} documentation covering all required sections
4. **Write**: Call `write_file_tool` with path `{target_file_path}` and complete content
5. **Stop**: Task complete after successful write

"""

        diagram_req = self.DIAGRAM_REQUIREMENTS.get(doc_type, "")
        if diagram_req:
            prompt += f"\n\n## Diagram Requirements\n{diagram_req}\n"
            prompt += "All diagrams must use ```mermaid code blocks; no images are required.\n"

        if repo_info:
            prompt += f"\n**Repository Info**:\n```json\n{json.dumps(repo_info, indent=2)}\n```\n"

        prompt += "\nBegin documentation generation now."

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

            diagram_req = self.DIAGRAM_REQUIREMENTS.get(doc_type, "")
            prompt = f"""# Task: Evaluate and Update {doc_type}

**Repository**: {self.repo_identifier}
**Target Document**: `{doc_path}`
**Change Details**: `{change_cache_path}`

## Phase 1: Analyze Changes
1. Read `{change_cache_path}` to understand all code changes (files, diffs, statistics)
2. Read the current `{doc_path}` document
3. Determine if updates are needed:
   - Do changes affect documented content?
   - Are there new features, bug fixes, or architectural changes?
   - Would current documentation mislead readers?

## Phase 2: Take Action

**If NO update needed:**
- Explain why (e.g., "Changes only affect internal implementation not covered in {doc_type}")

**If UPDATE needed:**
- Use `edit_file_tool` for surgical edits to affected sections
- Update mermaid diagrams if architecture/flows changed
- Document new behaviors, fixed bugs, or deprecated APIs
- Preserve existing structure and headings

## Additional Context
- `.cache/{self.repo_identifier}/repo_info.json` — Repository metadata
- `.cache/{self.repo_identifier}/module_structure.json` — Module organization
- `.cache/{self.repo_identifier}/module_analysis/` — Per-module static analysis

## Standards
- Prefer `edit_file_tool` over full rewrites
- Keep headings stable for link consistency
- Highlight breaking changes and security fixes
- {diagram_req or "Keep diagrams consistent with current system state"}
"""

            if repo_info:
                prompt += f"\n### Repo Info Snapshot\n```json\n{json.dumps(repo_info, indent=2)}\n```\n"

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
