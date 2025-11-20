import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import CONFIG
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from agents.base_agent import BaseAgent, AgentState

from agents.tools import write_file_tool
from .context_tools import ls_context_file_tool, read_file_tool
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
            "audience": "new users and developers"
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
            "audience": "developers and architects"
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
            "audience": "contributors and maintainers"
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
            "audience": "API users and integrators"
        }
    }
    
    def __init__(
        self, 
        repo_identifier: str, 
        wiki_path: Optional[str] = None, 
        cache_path: Optional[str] = None, 
        llm=None, 
        max_workers: int = 5
    ):
        """Initialize the MacroDocAgent.
        
        Args:
            repo_identifier (str): Unified repository identifier (format: owner_repo_name)
            wiki_path (Optional[str]): Base wiki path. If provided, macro docs will be saved directly to this path
            cache_path (Optional[str]): Base cache path for reading context files
            llm: Language model instance. If None, uses CONFIG.get_llm()
            max_workers (int): Maximum number of parallel workers for doc generation
        """
        self.repo_identifier = repo_identifier
        self.llm = llm if llm else CONFIG.get_llm()
        self.max_workers = max_workers
        
        # Use provided wiki_path or default to .wikis/{repo_identifier}
        if wiki_path:
            self.wiki_base_path = Path(wiki_path)
        else:
            self.wiki_base_path = Path(f".wikis/{repo_identifier}")
        
        # Use provided cache_path or default to .cache/{repo_identifier}
        if cache_path:
            self.cache_base_path = Path(cache_path)
        else:
            self.cache_base_path = Path(f".cache/{repo_identifier}")

        # Ensure directories exist
        self.wiki_base_path.mkdir(parents=True, exist_ok=True)

        system_prompt = SystemMessage(
            content="""You are a technical documentation expert specializing in high-level project documentation
(README, ARCHITECTURE, DEVELOPMENT, API, etc.) for software repositories.

## Your Workflow
1. **Explore**: Use `ls_context_file_tool` to discover available context files
2. **Gather**: Use `read_file_tool` to collect necessary information from context files
3. **Generate**: Create comprehensive documentation based on the gathered information
4. **Write Once**: Call `write_file_tool` EXACTLY ONCE with the specified file path and complete content
5. **Stop**: After writing the file, your task is complete - do NOT call any more tools

## Critical Constraints
- You MUST write to ONLY the specific file path provided in the task
- You MUST call `write_file_tool` EXACTLY ONE TIME with complete, final content
- After calling `write_file_tool`, STOP immediately - do not call any other tools
- DO NOT write to any other file paths beyond what is explicitly specified
- DO NOT make multiple write_file_tool calls - gather all information first, then write once

## CRITICAL PATH RULES - READ CAREFULLY
**UNDERSTAND THE THREE DIRECTORY TYPES:**

1. **`.cache/` directory** = Context files (JSON format, analysis results)
   - Purpose: Read-only context and analysis data
   - Format: JSON files containing code analysis, module info, repo structure
   - Example: `.cache/cloudwego_eino/repo_info.json`, `.cache/cloudwego_eino/module_analysis/root.json`
   - Action: READ these files to gather information

2. **`.repos/` directory** = Source code repository files  
   - Purpose: Original source code files (Go, Python, JavaScript, etc.)
   - Format: Actual code files (.go, .py, .js, etc.)
   - Example: `.repos/cloudwego_eino/adk/interface.go`, `.repos/cloudwego_eino/doc.go`
   - Action: You generally should NOT read these directly (use context files instead)
   - IMPORTANT: These files are referenced in context but you read the JSON summaries in .cache/

3. **`.wikis/` directory** = Output documentation (YOUR TARGET)
   - Purpose: Where you write generated documentation
   - Format: Markdown files (.md)
   - Example: `.wikis/cloudwego_eino/README.md`, `.wikis/cloudwego_eino/API.md`
   - Action: WRITE your documentation here using the EXACT path provided

**PATH VERIFICATION BEFORE WRITING:**
- ✅ CORRECT: `.wikis/cloudwego_eino/README.md` (writing documentation)
- ❌ WRONG: `.cache/cloudwego_eino/README.md` (cache is for reading context only!)
- ❌ WRONG: `.repos/cloudwego_eino/README.md` (repos is source code only!)
- ❌ WRONG: `README.md` (missing directory!)

**REMEMBER:**
- READ from `.cache/` (JSON context files)
- DO NOT try to read from `.repos/` (source code - use context instead)
- WRITE to `.wikis/` (output documentation) using the EXACT path in your task

## Quality Standards
Produce clear, professional, and comprehensive documentation that:
- Follows Markdown best practices
- Is well-structured with proper headings and sections
- Includes concrete examples where appropriate
- Is tailored to the specified target audience
"""
        )

        tools = [
            read_file_tool,
            write_file_tool,
            ls_context_file_tool,
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

        response = self.llm_with_tools.invoke(
            [system_prompt] + summarized_messages
        )

        extra_messages = summary_update["messages"] if summary_update else []
        return {"messages": extra_messages + [response]}
    
    def generate_single_doc(self, doc_type: str, repo_info: Dict[str, Any] = None) -> Dict[str, Any]:
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
        
        prompt = f"""
# Task: Generate {doc_type} ONLY

**CURRENT TASK**: Generate {doc_type} ({spec['title']})
**Repository**: {self.repo_identifier}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 YOUR SINGLE OUTPUT FILE - USE THIS EXACT PATH
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ WRITE TO THIS PATH (and ONLY this path):
   `{target_file_path}`

This is a {doc_type} file. Do NOT write to any other filename!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 DIRECTORY STRUCTURE - UNDERSTAND BEFORE PROCEEDING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**Three types of directories in this system:**

1.  `.cache/{self.repo_identifier}/` - READ context files from here
   - Contains: JSON analysis files (repo_info.json, module_analysis/*.json, etc.)
   - Purpose: Pre-analyzed repository information for your reference
   - Action: READ these JSON files to gather information
   - Examples you can read:
     * `.cache/{self.repo_identifier}/repo_info.json`
     * `.cache/{self.repo_identifier}/module_analysis/root.json`
     * `.cache/{self.repo_identifier}/module_structure.json`

2.  `.repos/{self.repo_identifier}/` - Source code (DO NOT READ)
   - Contains: Actual source code files (.go, .py, .js, etc.)
   - Purpose: Original repository code
   - Action: DO NOT try to read these - use context JSON files instead
   - Note: These paths may appear in context files as references

3.  `.wikis/{self.repo_identifier}/` - WRITE documentation here (YOUR TARGET)
   - Contains: Generated documentation (README.md, API.md, etc.)
   - Purpose: Your output destination
   - Action: WRITE your {doc_type} here
   - Your exact target: `{target_file_path}`

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
❌ WRONG PATHS - DO NOT USE THESE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

❌ `.cache/{self.repo_identifier}/{doc_type}` (Cache is for reading, not writing!)
❌ `.repos/{self.repo_identifier}/{doc_type}` (Repos is source code, not docs!)
❌ `{doc_type}` (Missing directory path!)
❌ `.wikis/{self.repo_identifier}/README.md` (Only if task is NOT README.md!)
   
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

## Target Audience
{spec['audience']}

## Required Sections
{chr(10).join(f"{i+1}. {section}" for i, section in enumerate(spec['sections']))}

## Available Context
Explore these context directories to gather information:
- `ls_context_file_tool('.cache/{self.repo_identifier}/')`
- `ls_context_file_tool('.cache/{self.repo_identifier}/module_analysis/')`

Context files contain:
- Repository structure and metadata (repo_info.json)
- Static code analysis results (module_analysis/*.json)
- Code components and dependencies (module_structure.json)

## Your Process
1. **Explore**: Use ls_context_file_tool('.cache/{self.repo_identifier}/') to discover available context
2. **Read**: Use read_file_tool to gather information from `.cache/{self.repo_identifier}/` directory (read as many as needed)
3. **Generate**: Create a complete, comprehensive {doc_type} with all required sections
4. **Write ONCE**: Call write_file_tool EXACTLY ONCE with:
   - file_path = `{target_file_path}` (EXACT PATH - this is your {doc_type} file)
   - content = your complete markdown documentation for {doc_type}
5. **STOP**: After write_file_tool returns success - do NOT call any more tools

## Completion Criteria - Verify Before Writing
✅ You have called `write_file_tool` with:
   - file_path argument = `{target_file_path}` (MUST match exactly)
   - content argument = complete markdown content for {doc_type}
✅ The path contains `.wikis/{self.repo_identifier}/` (NOT `.cache/` or `.repos/`)
✅ The filename is `{doc_type}` (not another document name!)
✅ After successful write, you STOP immediately - no more tool calls

## Final Path Verification Checklist
Before calling write_file_tool, triple-check:
- file_path = `{target_file_path}` (exact match)
- Path does NOT contain `.cache/` (that's for reading!)
- Path does NOT contain `.repos/` (that's source code!)
- Path DOES contain `.wikis/{self.repo_identifier}/`
- Filename ends with `{doc_type}` (your assigned document)
- Content is complete and ready to write

"""
        
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
        
        for section in spec['sections']:
            # Convert section to title case
            section_title = section.capitalize()
            content += f"\n## {section_title}\n\n*To be documented...*\n"
        
        # Save fallback document
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "doc_type": doc_type,
            "path": str(doc_path),
            "status": "fallback",
            "size": len(content),
        }
    
    def generate_all_docs(self, repo_info: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Generate all macro documentation files (Always Parallel).
        
        Args:
            repo_info (Dict[str, Any]): Optional repository information
            
        Returns:
            List[Dict[str, Any]]: List of results for each document
        """
        doc_types = list(self.DOC_SPECS.keys())
        
        print(f"\n🚀 Starting macro documentation generation for {len(doc_types)} documents...")
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
                    results.append({
                        "doc_type": doc_type,
                        "path": None,
                        "status": "error",
                        "error": str(e),
                    })
        
        print(f"\n✨ Macro documentation generation complete!")
        print(f"   Success: {sum(1 for r in results if r['status'] == 'success')}")
        print(f"   Fallback: {sum(1 for r in results if r['status'] == 'fallback')}")
        print(f"   Failed: {sum(1 for r in results if r['status'] == 'error')}")
        
        return results