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
    pass
