import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import CONFIG
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from agents.base_agent import BaseAgent, AgentState

from agents.tools import write_file_tool, edit_file_tool
from .utils import ls_context_file_tool, read_file_tool
from .summarization import ConversationSummarizer
from utils.code_analyzer import (
    analyze_file_with_tree_sitter,
    format_tree_sitter_analysis_results,
)


class ModuleDocAgent(BaseAgent):
    """Agent for generating module-level documentation."""

    def __init__(
        self,
        owner: str,
        repo_name: str,
        llm=None,
    ):
        """Initialize the ModuleDocAgent.

        Args:
            owner (str): Repository owner
            repo_name (str): Repository name
            llm: Language model instance. If None, uses CONFIG.get_llm()
        """
        self.owner = owner
        self.repo_name = repo_name
        self.llm = llm if llm else CONFIG.get_llm()
        
        # Derive paths from owner and repo_name
        self.repo_identifier = f"{owner}_{repo_name}"
        self.repo_root = Path(f".repos/{self.repo_identifier}").absolute()
        
        # Set up wiki and cache paths
        self.wiki_base_path = Path(f".wikis/{self.repo_identifier}/modules")
        self.module_analysis_base_path = Path(f".cache/{self.repo_identifier}/module_analysis")

        # Ensure directories exist
        self.wiki_base_path.mkdir(parents=True, exist_ok=True)
        self.module_analysis_base_path.mkdir(parents=True, exist_ok=True)

        # System prompt for module documentation
        self.system_prompt = """You are a technical documentation expert specializing in module-level code analysis and documentation.

## Available Tools
- `read_file_tool`: Read files (supports `start_line`/`end_line` for large files)
- `write_file_tool`: Write complete documentation (use for new docs or full rewrites)
- `edit_file_tool`: Apply surgical edits to existing documentation
- `ls_context_file_tool`: List available context files

## Directory Convention
| Directory | Purpose | Action |
|-----------|---------|--------|
| `.cache/{repo}/module_analysis/` | Pre-generated static analysis (JSON) | READ FIRST |
| Source files (listed in task) | Original code | READ selectively |
| `.wikis/{repo}/modules/` | Output documentation | WRITE |

## Core Principles
1. **Analysis First**: Always read the static analysis JSON before source files; it contains pre-parsed structure
2. **Selective Reading**: Only read source files when analysis JSON lacks needed detail; use line ranges for large files
3. **Fail Gracefully**: If a file read fails, do not retry—continue with available information
4. **Quality Focus**: Generate clear, professional documentation that helps developers understand quickly

## Diagram Standards
Use mermaid code blocks for architecture visualization:
- `graph TD/LR` for module structure and dependencies
- `sequenceDiagram` for key call chains (optional)
"""

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

        system_message = SystemMessage(content=self.system_prompt)

        super().__init__(
            tools=tools,
            system_prompt=system_message,
            repo_path="",
            wiki_path=str(self.wiki_base_path),
            llm=self.llm,
        )

    def _module_slug(self, module_name: str) -> str:
        return module_name.replace("/", "_").replace("\\", "_")

    def _get_module_analysis_path(self, module_name: str) -> Path:
        return self.module_analysis_base_path / f"{self._module_slug(module_name)}.json"

    def _get_module_doc_path(self, module_name: str) -> Path:
        doc_filename = f"{module_name.replace('/', '_')}.md"
        return self.wiki_base_path / doc_filename

    def get_module_doc_path(self, module_name: str) -> Path:
        """Public helper for retrieving module doc path."""
        return self._get_module_doc_path(module_name)

    def _resolve_file_path(self, file_path: str) -> Path:
        candidate = Path(file_path)
        if candidate.is_absolute():
            return candidate
        if candidate.exists():
            return candidate.resolve()
        repo_candidate = (self.repo_root / file_path).resolve()
        return repo_candidate

    def _load_cached_module_analysis(
        self, module_name: str, files: List[str]
    ) -> Optional[Dict[str, Any]]:
        analysis_path = self._get_module_analysis_path(module_name)
        if not analysis_path.exists():
            return None
        try:
            with open(analysis_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("files") == files:
                return data
        except Exception:
            return None
        return None

    def _build_module_analysis(
        self, module_name: str, files: List[str]
    ) -> Dict[str, Any]:
        analysis: Dict[str, Any] = {
            "module_name": module_name,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "files": files,
            "existing_files": [],
            "missing_files": [],
            "file_summaries": {},
        }

        for file_path in files:
            resolved_path = self._resolve_file_path(file_path)
            record: Dict[str, Any] = {
                "requested_path": file_path,
                "absolute_path": str(resolved_path),
            }
            if not resolved_path.exists():
                record["status"] = "missing"
                analysis["missing_files"].append(file_path)
                analysis["file_summaries"][file_path] = record
                continue

            try:
                stats = analyze_file_with_tree_sitter(str(resolved_path))
                formatted = (
                    format_tree_sitter_analysis_results(stats) if stats else None
                )
                # Remove redundant summary
                if formatted and "summary" in formatted:
                    del formatted["summary"]
                record["status"] = "analyzed" if formatted else "unsupported"
                record["analysis"] = formatted or {
                    "summary": {
                        "info": "Unsupported file type or analysis unavailable."
                    }
                }
            except Exception as exc:
                record["status"] = "error"
                record["analysis"] = {
                    "summary": {
                        "error": f"Analysis failed: {exc}",
                    }
                }

            analysis["existing_files"].append(file_path)
            analysis["file_summaries"][file_path] = record

        return analysis

    def _ensure_module_analysis(
        self, module_name: str, files: List[str]
    ) -> Tuple[Dict[str, Any], Path]:
        """Ensure module analysis is available. If not, build it."""
        cached = self._load_cached_module_analysis(module_name, files)
        if cached:
            return cached, self._get_module_analysis_path(module_name)

        analysis = self._build_module_analysis(module_name, files)
        analysis_path = self._get_module_analysis_path(module_name)
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        return analysis, analysis_path

    def _rebuild_module_analysis(
        self, module_name: str, files: List[str]
    ) -> Tuple[Dict[str, Any], Path]:
        """Force rebuild of module analysis regardless of cache."""
        analysis = self._build_module_analysis(module_name, files)
        analysis_path = self._get_module_analysis_path(module_name)
        with open(analysis_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        return analysis, analysis_path

    def refresh_module_analysis(
        self, module_name: str, files: List[str], force: bool = False
    ) -> Tuple[Dict[str, Any], Path]:
        """Refresh analysis cache for a module."""
        if force:
            return self._rebuild_module_analysis(module_name, files)
        return self._ensure_module_analysis(module_name, files)

    def _format_diff_context(
        self, changed_details: List[Dict[str, Any]], max_chars: int = 6000
    ) -> str:
        """Format diff snippets for prompting."""
        if not changed_details:
            return "No detailed diffs were supplied for this module."

        sections: List[str] = []
        remaining = max_chars
        for detail in changed_details:
            filename = detail.get("filename", "unknown")
            status = detail.get("status", "M")
            changes = detail.get("changes", 0)
            raw_diff = detail.get("diff") or ""
            snippet = raw_diff.strip() or "(diff unavailable)"

            per_section = min(len(snippet), max(500, max_chars // 6))
            trimmed_snippet = snippet[:per_section]
            if len(snippet) > per_section:
                trimmed_snippet += "\n... (truncated)"

            block = (
                f"### {filename} ({status}, ±{changes} lines)\n"
                f"```\n{trimmed_snippet}\n```"
            )

            if remaining - len(block) <= 0:
                sections.append("... (additional changes omitted)")
                break

            sections.append(block)
            remaining -= len(block)

        return "\n\n".join(sections)

    def _agent_node(
        self, system_prompt: SystemMessage, state: AgentState
    ) -> AgentState:
        """Call LLM with current state."""
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

    def generate_module_doc(self, module: Dict[str, Any]) -> Dict[str, Any]:
        """Generate documentation for a single module.

        Args:
            module (Dict[str, Any]): Module information including:
                - name: Module name
                - files: List of file paths
                - priority: Module priority
                - estimated_size: Size estimate

        Returns:
            Dict[str, Any]: Result containing:
                - module_name: Name of the module
                - doc_path: Path to generated documentation
                - summary: Brief summary of the module
                - status: 'success' or 'error'
        """
        module_name = module["name"]
        files = module["files"]

        print(f"\n📝 Generating documentation for module: {module_name}")
        print(f"   Files: {len(files)}")

        # Prepare documentation path
        doc_path = self._get_module_doc_path(module_name)
        doc_filename = doc_path.name

        analysis_data, analysis_path = self._ensure_module_analysis(module_name, files)
        print(f"   Static analysis cache: {analysis_path}")
        confirmed_files = analysis_data.get("existing_files", [])

        confirmed_files_text = (
            json.dumps(confirmed_files, indent=2, ensure_ascii=False)
            if confirmed_files
            else "[]"
        )

        prompt = f"""# Task: Generate Module Documentation

**Module**: {module_name}
**Output Path**: `{self.wiki_base_path / doc_filename}`
**Priority**: {module.get('priority', 'unknown')} | **Estimated Size**: {module.get('estimated_size', 'unknown')}

## Module Files
```json
{json.dumps(files, indent=2, ensure_ascii=False)}
```

## Static Analysis
- **Analysis JSON**: `{analysis_path}` (read this first)
- **Readable source files**: {confirmed_files_text}

## Required Documentation Sections
1. **Module Overview** — Purpose and responsibilities
2. **File List** — All files with brief descriptions
3. **Core Components** — Classes, functions, constants with signatures
4. **Usage Examples** — How to use this module
5. **Dependencies** — External and internal dependencies
6. **Architecture & Diagrams** — Mermaid diagram(s) showing structure/interactions
7. **Notes** — Important considerations or warnings

## Execution Steps
1. Read the static analysis JSON at `{analysis_path}`
2. Selectively read key source files if more detail needed (use line ranges for large files)
3. Generate comprehensive documentation covering all required sections
4. Write to `{self.wiki_base_path / doc_filename}`

## Constraints
- Only read files from the "Readable source files" list or cache JSONs
- Do not retry failed file reads—continue with available information
"""

        try:
            # Invoke the agent
            initial_state = AgentState(
                messages=[HumanMessage(content=prompt)],
                repo_path="",
                wiki_path=str(self.wiki_base_path),
            )

            _ = self.app.invoke(
                initial_state,
                config={
                    "configurable": {
                        "thread_id": f"module-doc-{self.repo_identifier}-{module_name}"
                    },
                    "recursion_limit": 60,
                },
            )

            # Check if documentation was generated
            if doc_path.exists():
                return {
                    "module_name": module_name,
                    "doc_path": str(doc_path),
                    "summary": f"Documentation generated for {module_name}",
                    "status": "success",
                }
            else:
                # Fallback: create basic documentation if agent didn't complete
                print(
                    f"⚠️  Agent didn't complete all tasks. Creating fallback documentation..."
                )
                return self._create_fallback_doc(module, doc_path)

        except Exception as e:
            print(f"❌ Error generating documentation for {module_name}: {e}")
            return {
                "module_name": module_name,
                "doc_path": None,
                "summary": f"Error: {str(e)}",
                "status": "error",
            }

    def update_module_doc_incremental(
        self,
        module: Dict[str, Any],
        changed_details: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Incrementally update an existing module document."""
        module_name = module["name"]
        files = module["files"]
        changed_details = changed_details or []

        doc_path = self._get_module_doc_path(module_name)
        if not doc_path.exists():
            print(
                f"⚠️  Existing doc for {module_name} not found. Falling back to full generation."
            )
            return self.generate_module_doc(module)

        print(f"\n✏️  Incrementally updating module doc: {module_name}")
        print(f"   Target doc: {doc_path}")
        print(f"   Changed files: {len(changed_details) or len(files)}")

        _, analysis_path = self.refresh_module_analysis(module_name, files, force=True)
        diff_context = self._format_diff_context(changed_details)
        changed_files_meta = [
            {
                "filename": detail.get("filename"),
                "status": detail.get("status"),
                "changes": detail.get("changes"),
            }
            for detail in changed_details
        ]

        prompt = f"""# Task: Incrementally Update Module Documentation

**Module**: {module_name}
**Target Document**: `{doc_path}`
**Static Analysis**: `{analysis_path}` (freshly rebuilt)

## Changed Files
```json
{json.dumps(changed_files_meta, indent=2, ensure_ascii=False)}
```

## Diff Snippets
{diff_context}

## Execution Steps
1. Read the updated static analysis JSON at `{analysis_path}`
2. Read the current documentation at `{doc_path}`
3. Identify sections affected by the changes (Overview, Core Components, Usage, Dependencies, Architecture, Notes)
4. Apply targeted edits using `edit_file_tool`:
   - Update affected descriptions and signatures
   - Add/update mermaid diagrams if structure changed
   - Document removed files or deprecated APIs
5. Only use `write_file_tool` if a near-total rewrite is necessary

## Constraints
- Preserve existing headings and formatting
- Maintain terminology consistency with other wiki pages
- Ensure summaries reflect the new behavior from the diffs
"""

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
                        "thread_id": f"module-doc-incremental-{self.repo_identifier}-{module_name}"
                    },
                    "recursion_limit": 60,
                },
            )

            return {
                "module_name": module_name,
                "doc_path": str(doc_path),
                "summary": f"Incrementally updated documentation for {module_name}",
                "status": "updated",
            }
        except Exception as e:
            print(f"❌ Error updating documentation for {module_name}: {e}")
            return {
                "module_name": module_name,
                "doc_path": str(doc_path),
                "summary": f"Incremental update error: {str(e)}",
                "status": "error",
            }

    def _create_fallback_doc(
        self, module: Dict[str, Any], doc_path: Path
    ) -> Dict[str, Any]:
        """Create fallback documentation when agent fails.

        Args:
            module: Module information
            doc_path: Path to save documentation

        Returns:
            Dict[str, Any]: Result information
        """
        module_name = module["name"]
        files = module["files"]

        # Create basic documentation
        doc_content = f"""# {module_name}

## Overview
This module contains {len(files)} file(s).

## Files
{chr(10).join(f'- `{f}`' for f in files)}

## Description
*Documentation generation in progress...*

## Components
*To be analyzed...*

## Usage
*To be documented...*
"""

        # Save documentation
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(doc_content)

        return {
            "module_name": module_name,
            "doc_path": str(doc_path),
            "summary": f"Module containing {len(files)} files",
            "status": "fallback",
        }

    def generate_all_modules(
        self, modules: List[Dict[str, Any]], max_workers: int = 5
    ) -> List[Dict[str, Any]]:
        """Generate documentation for all modules in parallel.

        Args:
            modules (List[Dict[str, Any]]): List of module information
            max_workers (int): Maximum number of worker threads. Default is 5.

        Returns:
            List[Dict[str, Any]]: List of results for each module
        """
        # Handle empty modules list
        if not modules:
            print("⚠️  No modules to generate documentation for, skipping module documentation")
            return []
        
        print(
            f"\n🚀 Starting parallel module documentation generation for {len(modules)} modules..."
        )
        print(f"   Max workers: {max_workers}")

        results = []
        total = len(modules)
        completed_count = 0

        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_module = {
                executor.submit(self.generate_module_doc, module): (i + 1, module)
                for i, module in enumerate(modules)
            }

            # Process completed tasks as they finish
            for future in as_completed(future_to_module):
                module_idx, module = future_to_module[future]
                completed_count += 1

                try:
                    result = future.result()
                    results.append(result)

                    print(f"\n[{completed_count}/{total}] Module: {module['name']}")
                    if result["status"] == "success":
                        print(f"✅ Success: {result['summary']}")
                    elif result["status"] == "fallback":
                        print(f"⚠️  Fallback documentation created")
                    else:
                        print(f"❌ Failed: {result.get('summary', 'Unknown error')}")

                except Exception as e:
                    print(f"\n[{completed_count}/{total}] Module: {module['name']}")
                    print(f"❌ Exception occurred: {str(e)}")
                    results.append(
                        {
                            "module_name": module["name"],
                            "doc_path": None,
                            "cache_path": None,
                            "summary": f"Exception: {str(e)}",
                            "status": "error",
                        }
                    )

        print(f"\n✨ Module documentation generation complete!")
        print(f"   Success: {sum(1 for r in results if r['status'] == 'success')}")
        print(f"   Fallback: {sum(1 for r in results if r['status'] == 'fallback')}")
        print(f"   Failed: {sum(1 for r in results if r['status'] == 'error')}")

        return results
