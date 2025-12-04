from langchain_core.messages import SystemMessage, HumanMessage
from textwrap import dedent


class CodePrompt:

    @staticmethod
    def _get_system_prompt_for_filter():
        # this func is to generate a system prompt for code filtering
        return SystemMessage(
            content=dedent(
                """
                You are an expert code analyst specializing in identifying relevant code files for static analysis.
                Your role is to analyze the provided repository information and file structure
                to select files that are most pertinent for in-depth examination.

                Guidelines:
                - Focus on source code files with extensions like .py, .js, .go, .cpp, .java, .rs, .c, .h, .hpp
                - Exclude documentation, configuration, and non-code files
                - Use the repository context to prioritize files central to core functionality
                - Base ALL decisions ONLY on the repository information and file structure explicitly provided in the input
                - NEVER invent, guess, or assume the existence of files or paths that are not present in the given file structure
                - If relevance is uncertain from the given data, choose the most conservative option and do not fabricate paths or file roles
                """
            ).strip(),
        )

    @staticmethod
    def _get_human_prompt_for_filter(
        repo_info: dict, repo_structure: list[str], max_num_files: int
    ) -> HumanMessage:
        # this func is to generate a prompt for code filtering
        return HumanMessage(
            content=dedent(
                f"""
                Based on the repository information and file structure below, identify EXACTLY {max_num_files}
                code files (or fewer if fewer exist) that are most relevant for static analysis.

                Critical constraints:
                - Return at most {max_num_files} file paths
                - Focus on source code files (.py, .js, .go, .cpp, .java, .rs, .c, .h, .hpp, .ts)
                - Exclude documentation, config, test, build, and dependency files
                - Prioritize files central to the project's core functionality
                - Sort by relevance (most important first)
                - Select paths ONLY from the File Structure list below and copy them verbatim
                (no changes to extensions, directory names, or segments)
                - If you cannot confidently identify {max_num_files} relevant files from the given data, return fewer files instead of guessing

                Repository information:
                {repo_info}

                File structure:
                {repo_structure}

                Output format:
                - Output ONLY file paths, one per line
                - No descriptions, headers, numbering, or extra text
                """
            ).strip(),
        )

    @staticmethod
    def _get_system_prompt_for_analysis() -> SystemMessage:
        # this func is to generate a shared system prompt for all code analysis tasks
        return SystemMessage(
            content=dedent(
                """
                You are an expert code analyst.

                Rules for ALL analysis tasks:
                - Base all reasoning strictly on the provided inputs (code, summaries, Tree-sitter results); do not use outside knowledge.
                - Never invent functions, classes, modules, behaviors, or external systems; if something cannot be determined, omit it instead of guessing.
                - Output ONLY the requested summary or documentation content; do not include meta commentary or reasoning about missing information.
                """
            ).strip(),
        )

    @staticmethod
    def _get_human_prompt_for_analysis_with_content(
        file_path: str, content: str
    ) -> HumanMessage:
        # this func is to generate a prompt for summarizing raw code content
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
            ).strip(),
        )

    @staticmethod
    def _get_system_prompt_for_doc() -> SystemMessage:
        # this func is to generate a system prompt for code documentation generation
        return SystemMessage(
            content=dedent(
                """
                You are an expert technical documentation writer for source code.
                Your job is to turn code and analysis results into clear, structured **Markdown documentation**
                that explains the codebase structure, functionality, and architecture.

                Guidelines:
                - Use Markdown headings, lists, and tables when helpful
                - Be concise yet informative
                - Highlight key components and their relationships
                - Organize content into logical sections with clear titles
                - Include code snippets and examples when they add clarity
                - Document important algorithms and design patterns
                - When diagrams are requested or helpful, use valid Mermaid code blocks (```mermaid ... ```).
                - Base ALL descriptions strictly on the provided analysis and summaries; do NOT invent modules, functions, or behaviors that are not supported by the input data.
                - If some information (e.g., exact data types, full behavior, performance characteristics) is not present in the inputs, simply omit it instead of speculating.

                Critical output rules:
                - Output ONLY the final markdown document, starting directly with the title line (e.g., "# Title")
                - Do NOT include introductory phrases, explanations, or closing remarks
                - Do NOT add text like "Of course", "Here is", "I'll generate", etc.
                - The response must begin with the markdown title and end with the last content line
                - No meta-commentary, generation notes, or additional context outside the markdown content
                """
            ).strip(),
        )

    @staticmethod
    def _get_human_prompt_for_doc(
        file_path: str, analysis: dict, summary: str
    ) -> HumanMessage:
        # this func is to generate a prompt for code documentation
        return HumanMessage(
            content=dedent(
                f"""
                Generate documentation for the code file '{file_path}'.

                Use the data below to write a markdown document that covers:
                - The file's overall purpose and main responsibilities
                - Key functions/classes and what they do
                - Important algorithms or design decisions
                - Dependencies and relationships with other files or modules
                - Any helpful Mermaid diagrams to illustrate structure or flow (when appropriate)

                Important correctness rules:
                - Base ALL statements strictly on the "Tree Sitter Analysis Result" and "File Content Summary" provided below.
                - Do NOT invent undocumented APIs, behaviors, modules, or external systems.
                - Do NOT guess about performance characteristics, security properties, or side effects unless they are clearly indicated in the inputs.

                Data:
                - Tree Sitter Analysis Result:
                {analysis}

                - File Content Summary:
                {summary}
                """
            ).strip(),
        )


class RepoPrompt:
    @staticmethod
    def _get_system_prompt_for_repo() -> SystemMessage:
        # this func is to get the system prompt for the repo info subgraph
        return SystemMessage(
            content=dedent(
                """
                You are an expert technical documentation writer for software repositories.

                Your job is to turn repository-related data (commits, PRs, releases, existing docs, etc.) into clear, structured **Markdown documentation** for developers.

                Global rules:
                - Use Markdown headings, lists, and tables when helpful
                - Be concise but informative; prefer summaries over raw dumps
                - Highlight important changes, impact, and patterns
                - Organize content into logical sections with clear titles
                - When diagrams are requested, use valid Mermaid code blocks (```mermaid ... ```).

                Critical output rules:
                - Output ONLY the final markdown document, starting directly with the title line
                - Do NOT include explanations, comments, or meta-text about what you are doing
                - Do NOT include phrases like "Here is", "Of course", "I will", etc.
                - Do NOT wrap the result in quotes, JSON, or any extra formatting
                """
            ).strip(),
        )

    @staticmethod
    def _get_human_prompt_for_repo_overview_doc(
        repo_info: dict, doc_contents: str
    ) -> HumanMessage:
        # this func is to generate a prompt for overall repository documentation
        repo_name = repo_info.get("repo", "")
        owner = repo_info.get("owner", "")

        return HumanMessage(
            content=dedent(
                f"""
                Generate overall documentation for repository '{owner}/{repo_name}'.

                Based on:
                - Repository information:
                {repo_info}

                - Documentation source files (content below):
                {doc_contents}

                Write a markdown document that includes:
                1. Overview of repository structure and key components
                2. Summary of existing documentation coverage
                3. Main documentation gaps and areas for improvement
                4. At least one Mermaid diagram for architecture or component relationships
                """
            ).strip(),
        )

    @staticmethod
    def _get_human_prompt_for_repo_updated_overview_doc(
        repo_info: dict,
        commit_info: dict,
        pr_info: dict,
        release_note_info: dict,
        overview_doc_path: str,
    ) -> HumanMessage:
        # this func is to generate a prompt for updated overall repository documentation
        repo_name = repo_info.get("repo", "")
        owner = repo_info.get("owner", "")
        return HumanMessage(
            content=dedent(
                f"""
                Update the existing repository overview documentation for '{owner}/{repo_name}'.

                Existing doc path (for reference only, do NOT mention the path in output): {overview_doc_path}

                Use the updated data below:
                - Commit information:
                {commit_info}

                - PR information:
                {pr_info}

                - Release note information:
                {release_note_info}

                Update the markdown so it reflects the latest changes, keeps a clear structure,
                and updates or adds Mermaid diagrams when architecture or relationships change.
                """
            ).strip(),
        )

    @staticmethod
    def _get_human_prompt_for_repo_single_commit_doc(
        commit: dict, repo_info: dict
    ) -> HumanMessage:
        # this func is to generate a prompt for a single commit documentation section
        repo_name = repo_info.get("repo", "")
        owner = repo_info.get("owner", "")
        return HumanMessage(
            content=dedent(
                f"""
                You are documenting a single commit for repository '{owner}/{repo_name}'.

                Commit data:
                {commit}

                Write a self-contained **markdown section** for this commit that includes:
                - A level-2 heading combining commit message and short SHA
                - "Author" and "Date" lines
                - A "**Type:** ..." line that classifies the commit into one of:
                  - "Feature", "Bugfix", "Perf", "Refactor", "Dependency", "CI/Infra", or "Docs"
                - A "Summary" subheading explaining in 1–2 short paragraphs what changed and why it matters
                - A "Modified Files" subheading with a concise bullet list of key files and their roles, based ONLY on the given data

                Classification & brevity rules:
                - If the commit message starts with "Bump" or only touches CI/workflow or tooling files
                  (e.g., paths under .github/workflows, CodeQL, actions/checkout, actions/upload-artifact, etc.),
                  classify it as "Dependency" or "CI/Infra" and:
                  - Keep the entire section VERY short (at most about 6–8 lines of markdown excluding bullets)
                  - Use 1 short summary sentence
                  - Use at most 1–3 bullets under "Modified Files"
                  - Avoid repeating generic rationale like "keep dependencies up-to-date" more than once.
                - For non-maintenance commits (real features / bugfixes / perf work), focus on:
                  - Impacted components or subsystems
                  - Why the change was needed
                  - Risks or notable side effects (only when clearly indicated in the data)

                Formatting rules:
                - Always use the following structure:
                  - "## ..." for the commit title
                  - "Author", "Date", and "Type" lines
                  - "### Summary" followed by 1–2 short paragraphs
                  - "### Modified Files" followed by a few bullets (omit this section only if there are no files listed)

                Output only the markdown section for this commit, without any surrounding introduction or conclusion.
                """
            ).strip(),
        )

    @staticmethod
    def _get_human_prompt_for_repo_single_pr_doc(
        pr: dict, repo_info: dict, diff_content: str | None
    ) -> HumanMessage:
        # this func is to generate a prompt for a single PR documentation section
        repo_name = repo_info.get("repo", "")
        owner = repo_info.get("owner", "")
        return HumanMessage(
            content=dedent(
                f"""
                You are documenting a single pull request for repository '{owner}/{repo_name}'.

                PR metadata:
                {pr}

                Unified diff for this PR (may be truncated or empty if unavailable):
                {diff_content or "<no diff content available>"}

                Write a self-contained **markdown section** for this PR that includes:
                - A level-2 heading combining PR title and number
                - Basic metadata (author, state, created_at, merged_at)
                - A "**Type:** ..." line that classifies the PR into one of:
                  - "Feature", "Bugfix", "Perf", "Refactor", "Dependency", "CI/Infra", "Docs", or "Release"
                - A "Summary" subheading explaining in 1–2 short paragraphs what this PR does and why it matters
                - A "Changes" subheading with a concise bullet list of the main technical changes,
                  using the PR body and diff content when helpful (but never pasting raw diffs)
                - Optionally a "Impact / Risk" subheading when the change is notable for maintainers

                Rules:
                - Prefer the PR body for intent and high-level description
                - Use the diff content ONLY to infer types of files and changes, never to dump line-level patches
                - Do not mention that you saw a diff or that content might be truncated
                - When the PR is an automated dependency bump or CI-only change (Dependabot author, title starts with "Bump",
                  or changes limited to workflows / actions / code scanning):
                  - Classify it as "Dependency" or "CI/Infra"
                  - Keep the whole section short:
                    - 1–2 summary sentences
                    - 1–3 bullets under "Changes"
                  - Do NOT restate upstream release notes in detail; only highlight the most relevant effect on this repository.

                Output only the markdown section for this PR.
                """
            ).strip(),
        )

    @staticmethod
    def _get_human_prompt_for_repo_single_release_doc(
        release: dict, repo_info: dict
    ) -> HumanMessage:
        # this func is to generate a prompt for a single release documentation section
        repo_name = repo_info.get("repo", "")
        owner = repo_info.get("owner", "")
        return HumanMessage(
            content=dedent(
                f"""
                You are documenting a single release for repository '{owner}/{repo_name}'.

                Release metadata:
                {release}

                Write a self-contained **markdown section** for this release that includes:
                - A level-2 heading with the release name and tag (for example: "## v1.2.3 - 2025-01-01")
                - Basic metadata (tag_name, created_at, published_at)
                - A "Highlights" subheading summarizing in a few bullets (at most 5) the key features / changes from the body
                - Optionally other subheadings like "Breaking Changes", "Bug Fixes", or "Performance" if they are clearly present in the body

                Rules:
                - Base your content strictly on the provided release body and metadata.
                - Focus on changes that are most relevant for downstream users of this repository:
                  - critical bug fixes
                  - security or behavior changes
                  - major performance or compatibility updates
                  - new APIs / flags that may require code or deployment changes
                - Do NOT copy long upstream release notes verbatim:
                  - Summarize long lists into a few representative bullets
                  - Avoid reproducing large tables, benchmarks, or exhaustive change logs
                - Do NOT invent features or changes that are not reflected in the input.
                - Output only the markdown section for this release.
                """
            ).strip(),
        )
