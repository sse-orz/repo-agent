import json
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import CONFIG
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from agents.base_agent import BaseAgent, AgentState

from agents.tools import write_file_tool
from .context_tools import ls_context_file_tool, read_file_tool
from .summarization import ConversationSummarizer
from utils.code_analyzer import (
    analyze_file_with_tree_sitter,
    format_tree_sitter_analysis_results,
)


class ModuleDocAgent(BaseAgent):
    """Agent for generating module-level documentation."""
    
    def __init__(
        self, 
        repo_identifier: str, 
        repo_root: Optional[str] = None, 
        wiki_path: Optional[str] = None, 
        cache_path: Optional[str] = None, 
        llm=None
    ):
        """Initialize the ModuleDocAgent.
        
        Args:
            repo_identifier (str): Unified repository identifier (format: owner_repo_name)
            repo_root (Optional[str]): Absolute path to repository root for file resolution
            wiki_path (Optional[str]): Base wiki path. Module docs will be saved to {wiki_path}/modules/
            cache_path (Optional[str]): Base cache path. Module analysis will be saved to {cache_path}/module_analysis/
            llm: Language model instance. If None, uses CONFIG.get_llm()
        """
        self.repo_identifier = repo_identifier
        self.llm = llm if llm else CONFIG.get_llm()
        self.repo_root = Path(repo_root).absolute() if repo_root else Path(".").absolute()
        
        # Use provided wiki_path or default to .wikis/{repo_identifier}
        if wiki_path:
            self.wiki_base_path = Path(wiki_path) / "modules"
        else:
            self.wiki_base_path = Path(f".wikis/{repo_identifier}/modules")
        
        # Use provided cache_path or default to .cache/{repo_identifier}
        if cache_path:
            self.module_analysis_base_path = Path(cache_path) / "module_analysis"
        else:
            self.module_analysis_base_path = Path(f".cache/{repo_identifier}/module_analysis")
        
        # Ensure directories exist
        self.wiki_base_path.mkdir(parents=True, exist_ok=True)
        self.module_analysis_base_path.mkdir(parents=True, exist_ok=True)
        
        # System prompt for module documentation
        self.system_prompt = """You are a technical documentation expert specializing in module-level code analysis and documentation.

Your task is to:
1. Analyze the provided module files in depth
2. Generate comprehensive module documentation in Markdown format

## Available Tools
- `read_file_tool`: Read source code files (supports line ranges for large files)
- `write_file_tool`: Save generated documentation
- `ls_context_file_tool`: List available context files from previous analysis

## Local Analysis Cache
- Before reading source files, load the pre-generated static analysis JSON stored under `.cache/{repo_identifier}/module_analysis/`.
- Use that summarized data to understand structure and only open source files when additional detail is required.

## Documentation Requirements
Your documentation should include:
1. **Module Overview**: Purpose and responsibilities
2. **File List**: All files in this module
3. **Core Components**: 
   - Classes with their methods and attributes
   - Important functions with parameters and return values
   - Key constants and configurations
4. **Usage Examples**: How to use the module
5. **Dependencies**: What other modules/libraries this depends on
6. **Notes**: Important considerations, best practices, or warnings

## Important Guidelines
- Read files incrementally if they are large (use start_line/end_line parameters)
- Use context files from previous stages if available
- Generate clear, professional technical documentation
- Focus on helping developers understand the module quickly
- You may only read files explicitly listed for the module or the permitted cache files. If a file read fails, do not attempt to read it again. It is acceptable to focus on a subset of files you deem most important.
"""
        
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
    
    def _build_module_analysis(self, module_name: str, files: List[str]) -> Dict[str, Any]:
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
        
        response = self.llm_with_tools.invoke(
            [system_prompt] + summarized_messages
        )
        
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
        doc_filename = f"{module_name.replace('/', '_')}.md"
        doc_path = self.wiki_base_path / doc_filename
        
        analysis_data, analysis_path = self._ensure_module_analysis(module_name, files)
        print(f"   Static analysis cache: {analysis_path}")
        confirmed_files = analysis_data.get("existing_files", [])
        
        confirmed_files_text = (
            json.dumps(confirmed_files, indent=2, ensure_ascii=False)
            if confirmed_files
            else "[]"
        )
        
        prompt = f"""
Please generate documentation for the module based on the following information:

**Module Name**: {module_name}
**All Declared Files**: {json.dumps(files, indent=2, ensure_ascii=False)}
**Estimated Size**: {module.get('estimated_size', 'unknown')}
**Priority**: {module.get('priority', 'unknown')}

## Static Analysis
- Pre-generated Tree-sitter analysis file: `{analysis_path}`
- Readable files (limited to the following list): {confirmed_files_text}

## Tasks
1. First use `read_file_tool` to read the above static analysis JSON to understand the module overview.
2. Based on the analysis results, selectively read a small number of key source files (not required to read all), use `start_line`/`end_line` to control range when necessary.
3. Generate complete Markdown module documentation and save to `{self.wiki_base_path / doc_filename}`.

## Strict Constraints
- Only read source files in the "readable files" list or the above static analysis JSON and context cache files.
- After a path read fails, do not attempt the same path again, record the failure reason and continue with other work.
- Any files not in the list are forbidden to access.
- If static analysis already provides sufficient information, you can directly reference the content within it.

Please complete the analysis and documentation writing while adhering to the above restrictions.
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
                print(f"⚠️  Agent didn't complete all tasks. Creating fallback documentation...")
                return self._create_fallback_doc(module, doc_path)
                
        except Exception as e:
            print(f"❌ Error generating documentation for {module_name}: {e}")
            return {
                "module_name": module_name,
                "doc_path": None,
                "summary": f"Error: {str(e)}",
                "status": "error",
            }
    
    def _create_fallback_doc(self, module: Dict[str, Any], doc_path: Path) -> Dict[str, Any]:
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
        with open(doc_path, 'w', encoding='utf-8') as f:
            f.write(doc_content)
        
        return {
            "module_name": module_name,
            "doc_path": str(doc_path),
            "summary": f"Module containing {len(files)} files",
            "status": "fallback",
        }
    
    def generate_all_modules(self, modules: List[Dict[str, Any]], max_workers: int = 5) -> List[Dict[str, Any]]:
        """Generate documentation for all modules in parallel.
        
        Args:
            modules (List[Dict[str, Any]]): List of module information
            max_workers (int): Maximum number of worker threads. Default is 5.
            
        Returns:
            List[Dict[str, Any]]: List of results for each module
        """
        print(f"\n🚀 Starting parallel module documentation generation for {len(modules)} modules...")
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
                    results.append({
                        "module_name": module["name"],
                        "doc_path": None,
                        "cache_path": None,
                        "summary": f"Exception: {str(e)}",
                        "status": "error",
                    })
        
        print(f"\n✨ Module documentation generation complete!")
        print(f"   Success: {sum(1 for r in results if r['status'] == 'success')}")
        print(f"   Fallback: {sum(1 for r in results if r['status'] == 'fallback')}")
        print(f"   Failed: {sum(1 for r in results if r['status'] == 'error')}")
        
        return results
