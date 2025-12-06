"""Centralized prompt definitions for MoeAgent.

This module contains all prompt templates used by MoeAgent components,
following the pattern from sub_graph/prompt.py for better maintainability.
"""

from textwrap import dedent
from typing import List, Dict, Any
from langchain_core.messages import SystemMessage, HumanMessage


class FileFilterPrompt:
    """Prompts for intelligent file filtering."""

    @staticmethod
    def get_system_prompt() -> SystemMessage:
        """Generate system prompt for file filtering."""
        return SystemMessage(
            content=dedent(
                """
                You are an expert code analyst specializing in identifying relevant code files for documentation generation.
                Your role is to analyze the provided repository information and file list to select files that are most important for understanding the project.

                Guidelines:
                - Prioritize core source code files that define main functionality
                - Focus on entry points, APIs, and core business logic
                - Include important utility and helper modules
                - Exclude test files, mock files, and example files
                - Consider file paths to understand module structure
                - Prioritize files in src/, lib/, core/, api/, pkg/, internal/, cmd/, app/, services/, handlers/, models/ directories
                - Give highest priority to main entry files (e.g., main.go, index.ts, app.py, server.go, main.py)
                - Base ALL decisions ONLY on the provided file list
                - NEVER invent or guess file paths that are not in the provided list
                """
            ).strip()
        )

    @staticmethod
    def get_human_prompt(
        repo_name: str, file_list: List[str], max_files: int
    ) -> HumanMessage:
        """Generate user prompt for file filtering.

        Args:
            repo_name: Name of the repository
            file_list: List of file paths to filter
            max_files: Maximum number of files to select

        Returns:
            HumanMessage: User prompt for LLM
        """
        files_str = "\n".join(file_list)
        return HumanMessage(
            content=dedent(
                f"""
                Based on the following file list from repository '{repo_name}', select EXACTLY {max_files} most important code files for documentation.

                CRITICAL CONSTRAINTS:
                - Return MAXIMUM {max_files} file paths
                - Prioritize files central to the project's core functionality
                - Sort by importance (most important first)
                - Each file path must be from the provided list
                - Select paths ONLY from the File List below and copy them verbatim

                File List ({len(file_list)} files):
                {files_str}

                IMPORTANT: Output ONLY file paths, one per line. No descriptions, headers, numbers, or any other text.
                """
            ).strip()
        )


class ModuleDocPrompt:
    """Prompts for module documentation generation."""

    @staticmethod
    def get_system_prompt() -> SystemMessage:
        """Generate system prompt for module documentation."""
        return SystemMessage(
            content=dedent(
                """
                You are a technical documentation expert specializing in module-level code analysis and documentation.

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

                ## Documentation Style Rules
                - Explain code snippets with accompanying text; avoid listing code blocks without explanation
                - Balance code and prose to ensure readability

                ## Diagram Standards
                Use mermaid code blocks for architecture visualization:
                - `graph TD/LR` for module structure and dependencies
                - `sequenceDiagram` for key call chains (optional)
                """
            ).strip()
        )

    @staticmethod
    def get_generation_prompt(
        module_name: str,
        files: List[str],
        wiki_base_path: str,
        analysis_path: str,
        confirmed_files: List[str],
        priority: str = "unknown",
        estimated_size: str = "unknown",
    ) -> HumanMessage:
        """Generate prompt for module documentation generation.

        Args:
            module_name: Name of the module
            files: List of files in the module
            wiki_base_path: Base path for wiki output
            analysis_path: Path to the analysis JSON file
            confirmed_files: List of confirmed existing files
            priority: Module priority
            estimated_size: Estimated module size

        Returns:
            HumanMessage: Generation prompt
        """
        import json
        import os

        doc_filename = f"{module_name.replace('/', '_')}.md"
        confirmed_files_text = (
            json.dumps(confirmed_files, indent=2, ensure_ascii=False)
            if confirmed_files
            else "[]"
        )

        return HumanMessage(
            content=dedent(
                f"""
                # Task: Generate Module Documentation

                **Module**: {module_name}
                **Output Path**: `{os.path.join(wiki_base_path, doc_filename)}`
                **Priority**: {priority} | **Estimated Size**: {estimated_size}

                ## Module Files
                ```json
                {json.dumps(files, indent=2, ensure_ascii=False)}
                ```

                ## Static Analysis
                - **Analysis JSON**: `{analysis_path}` (read this first)
                - **Readable source files**: {confirmed_files_text}

                ## Required Documentation Sections
                1. **Module Overview** — Purpose, responsibilities, and design rationale
                2. **File List** — All files with brief descriptions
                3. **Core Components** — Classes/functions with explanations of their purpose, then show signatures
                4. **Usage Examples** — Scenarios with explanatory text alongside code
                5. **Dependencies** — List with brief explanation of each dependency's role
                6. **Architecture & Diagrams** — Mermaid diagrams with descriptive text
                7. **Notes** — Important considerations and caveats

                ## Execution Steps
                1. Read the static analysis JSON at `{analysis_path}`
                2. Selectively read key source files if more detail needed (use line ranges for large files)
                3. Generate comprehensive documentation covering all required sections
                4. Write to `{os.path.join(wiki_base_path, doc_filename)}`

                ## Constraints
                - Only read files from the "Readable source files" list or cache JSONs
                - Do not retry failed file reads—continue with available information
                """
            ).strip()
        )

    @staticmethod
    def get_incremental_update_prompt(
        module_name: str,
        doc_path: str,
        analysis_path: str,
        changed_files_meta: List[Dict[str, Any]],
        diff_context: str,
    ) -> HumanMessage:
        """Generate prompt for incremental module documentation update.

        Args:
            module_name: Name of the module
            doc_path: Path to existing documentation
            analysis_path: Path to the analysis JSON file
            changed_files_meta: Metadata about changed files
            diff_context: Formatted diff snippets

        Returns:
            HumanMessage: Update prompt
        """
        import json

        return HumanMessage(
            content=dedent(
                f"""
                # Task: Incrementally Update Module Documentation

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
            ).strip()
        )


class MacroDocPrompt:
    """Prompts for macro-level project documentation."""

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

    @staticmethod
    def get_system_prompt() -> SystemMessage:
        """Generate system prompt for macro documentation."""
        return SystemMessage(
            content=dedent(
                """
                You are a technical documentation expert specializing in macro-level project documentation (README, ARCHITECTURE, DEVELOPMENT, API) for software repositories.

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

                ## Documentation Style Rules
                - Explain code snippets with accompanying text; avoid listing code blocks without explanation
                - Balance code and prose to ensure readability

                ## Quality Standards
                - Follow Markdown best practices with proper heading hierarchy
                - Include concrete examples and code snippets where appropriate
                - Tailor content to the specified target audience
                - Use mermaid diagrams for architecture visualization
                """
            ).strip()
        )

    @staticmethod
    def get_generation_prompt(
        doc_type: str,
        repo_identifier: str,
        wiki_base_path: str,
        repo_info: Dict[str, Any] = None,
    ) -> HumanMessage:
        """Generate prompt for macro documentation generation.

        Args:
            doc_type: Type of document (README.md, ARCHITECTURE.md, etc.)
            repo_identifier: Repository identifier (owner_repo)
            wiki_base_path: Base path for wiki output
            repo_info: Optional repository information

        Returns:
            HumanMessage: Generation prompt
        """
        import json
        import os

        spec = MacroDocPrompt.DOC_SPECS[doc_type]
        target_file_path = os.path.join(wiki_base_path, doc_type)

        sections_text = "\n".join(f"- {section}" for section in spec["sections"])

        prompt = dedent(
            f"""
            # Task: Generate {doc_type}

            **Repository**: {repo_identifier}
            **Output Path**: `{target_file_path}`
            **Target Audience**: {spec['audience']}

            ## Required Sections
            {sections_text}

            ## Context Resources
            Start by exploring available context:
            ```
            ls_context_file_tool('.cache/{repo_identifier}/')
            ```

            Key context files:
            - `repo_info.json` — Repository metadata and structure
            - `module_structure.json` — Module organization and dependencies
            - `module_analysis/*.json` — Static analysis per module

            ## Execution Steps
            1. **Explore**: List context files in `.cache/{repo_identifier}/`
            2. **Gather**: Read relevant JSON context files to understand the project
            3. **Generate**: Create comprehensive {spec['title']} documentation covering all required sections
            4. **Write**: Call `write_file_tool` with path `{target_file_path}` and complete content
            5. **Stop**: Task complete after successful write
            """
        ).strip()

        diagram_req = MacroDocPrompt.DIAGRAM_REQUIREMENTS.get(doc_type, "")
        if diagram_req:
            prompt += f"\n\n## Diagram Requirements\n{diagram_req}\n"
            prompt += "All diagrams must use ```mermaid code blocks; no images are required.\n"

        if repo_info:
            prompt += f"\n**Repository Info**:\n```json\n{json.dumps(repo_info, indent=2)}\n```\n"

        prompt += "\nBegin documentation generation now."

        return HumanMessage(content=prompt)

    @staticmethod
    def get_incremental_update_prompt(
        doc_type: str,
        repo_identifier: str,
        doc_path: str,
        change_cache_path: str,
        repo_info: Dict[str, Any] = None,
    ) -> HumanMessage:
        """Generate prompt for incremental macro documentation update.

        Args:
            doc_type: Type of document
            repo_identifier: Repository identifier
            doc_path: Path to existing documentation
            change_cache_path: Path to the change cache file
            repo_info: Optional repository information

        Returns:
            HumanMessage: Update prompt
        """
        import json

        diagram_req = MacroDocPrompt.DIAGRAM_REQUIREMENTS.get(doc_type, "")

        prompt = dedent(
            f"""
            # Task: Evaluate and Update {doc_type}

            **Repository**: {repo_identifier}
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
            - `.cache/{repo_identifier}/repo_info.json` — Repository metadata
            - `.cache/{repo_identifier}/module_structure.json` — Module organization
            - `.cache/{repo_identifier}/module_analysis/` — Per-module static analysis

            ## Standards
            - Prefer `edit_file_tool` over full rewrites
            - Keep headings stable for link consistency
            - Highlight breaking changes and security fixes
            - {diagram_req or "Keep diagrams consistent with current system state"}
            """
        ).strip()

        if repo_info:
            prompt += f"\n### Repo Info Snapshot\n```json\n{json.dumps(repo_info, indent=2)}\n```\n"

        return HumanMessage(content=prompt)


class CodeAnalysisPrompt:
    """Prompts for code analysis and summarization."""

    @staticmethod
    def get_system_prompt() -> SystemMessage:
        """Generate system prompt for code analysis."""
        return SystemMessage(
            content=dedent(
                """
                You are an expert code analyst.

                Rules for ALL analysis tasks:
                - Base all reasoning strictly on the provided inputs (code, summaries, Tree-sitter results); do not use outside knowledge.
                - Never invent functions, classes, modules, behaviors, or external systems; if something cannot be determined, omit it instead of guessing.
                - Output ONLY the requested summary or documentation content; do not include meta commentary or reasoning about missing information.
                """
            ).strip()
        )

    @staticmethod
    def get_summarization_prompt(file_path: str, content: str) -> HumanMessage:
        """Generate prompt for summarizing raw code content.

        Args:
            file_path: Path to the file being summarized
            content: Raw file content

        Returns:
            HumanMessage: Summarization prompt
        """
        return HumanMessage(
            content=dedent(
                f"""
                You are given the **raw source code** of the file `{file_path}`.
                Your task is to produce a **concise but comprehensive textual summary** that captures:
                1) The file's main purpose within the overall project, and
                2) A high-level, structured description of its internal logic.

                Output requirements:
                - Start with 1–2 short sentences describing the **overall purpose** of the file (what responsibility it has in the project or system).
                - Then provide 4–10 bullet points that summarize the **internal logic**, grouped by responsibility when possible. Prioritize:
                  - Main modules/classes/functions and what each is responsible for (use their actual names when helpful).
                  - How major functions/methods/components interact or depend on each other in typical execution flows or usage patterns.
                  - Important control flows (loops, branches, retries, background tasks, event handling, error-handling strategies).
                  - Key side effects (I/O, data storage, network communication, logging, global state or cache updates).
                  - Non-trivial algorithms, data structures, or transformations (what problem they solve and how at a high level).
                  - Any important invariants, validation rules, security checks, or business constraints enforced in the code.
                  - Significant integrations with external systems, frameworks, libraries, or services used by this file.
                - Cover all **major** behaviors and responsibilities in the file; avoid omitting significant functionality.
                - Stay concise and high-level: do NOT describe every line, small helpers, or trivial boilerplate.
                - When possible, connect behaviors to concrete entrypoints (e.g., public APIs, exported functions/classes, UI event handlers, CLI commands, scheduled jobs, HTTP endpoints) so that the file's role is easy to understand.
                - Do NOT paste large code fragments; always describe behavior in natural language instead.

                Raw file content:

                {content}
                """
            ).strip()
        )

