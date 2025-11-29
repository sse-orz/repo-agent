import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from collections import defaultdict
from config import CONFIG
from langchain_core.messages import SystemMessage, HumanMessage
from utils.code_analyzer import (
    analyze_file_with_tree_sitter,
    format_tree_sitter_analysis_results,
)


class ModuleClusterer:
    """Cluster code files into logical modules."""

    def __init__(self, llm=None, repo_root: Optional[str] = None):
        """Initialize the ModuleClusterer.

        Args:
            llm: Language model instance. If None, uses CONFIG.get_llm()
            repo_root: Optional[str] = None
        """
        self.llm = llm if llm else CONFIG.get_llm()
        try:
            self.project_root = Path(__file__).resolve().parents[2]
        except IndexError:
            self.project_root = Path.cwd()

        if repo_root:
            self.repo_root = Path(repo_root).resolve()
        else:
            self.repo_root = self.project_root

    def _resolve_file_path(self, file_path: str) -> Path:
        """Resolve a file path relative to the repository root or CWD."""
        path = Path(file_path)

        if path.is_absolute():
            return path

        repo_candidate = self.repo_root / path
        if repo_candidate.exists():
            return repo_candidate

        project_candidate = self.project_root / path
        if project_candidate.exists():
            return project_candidate

        cwd_candidate = Path.cwd() / path
        if cwd_candidate.exists():
            return cwd_candidate

        return repo_candidate

    def _map_files_to_modules(self, modules: Dict[str, List[str]]) -> Dict[str, str]:
        """Build a quick lookup from file path to initial module."""
        file_to_module = {}
        for module_name, files in modules.items():
            for file_path in files:
                file_to_module[file_path] = module_name
        return file_to_module

    def _analyze_single_file(self, file_path: str) -> Dict[str, Any]:
        """Run tree-sitter analysis for a single file (best-effort)."""
        abs_path = self._resolve_file_path(file_path)
        result: Dict[str, Any] = {
            "path": file_path,
            "absolute_path": str(abs_path),
            "analysis": None,
            "error": None,
        }

        if not abs_path.exists():
            result["error"] = "file_not_found"
            return result

        try:
            stats = analyze_file_with_tree_sitter(str(abs_path))
            if not stats:
                result["error"] = "unsupported_language_or_parse_failure"
                return result
            formatted = format_tree_sitter_analysis_results(stats)
            # Remove redundant summary
            if "summary" in formatted:
                del formatted["summary"]
            result["analysis"] = formatted
        except Exception as exc:
            result["error"] = f"analysis_failed: {exc}"

        return result

    def _extract_dependency_strings(self, analysis: Dict[str, Any]) -> List[str]:
        """Flatten outer dependency metadata into human-friendly strings."""
        if not analysis:
            return []

        dependencies = []
        for dep in analysis.get("outer_dependencies", []):
            module = dep.get("module", "").strip()
            name = dep.get("name", "").strip()
            if module and name:
                dependencies.append(f"{module}.{name}")
            elif module:
                dependencies.append(module)
            elif name:
                dependencies.append(name)
        return dependencies

    def _build_module_metadata(
        self, modules: Dict[str, List[str]]
    ) -> Dict[str, Dict[str, Any]]:
        """Prepare metadata for each initial module."""
        metadata = {}
        for module_name, files in modules.items():
            metadata[module_name] = {
                "file_count": len(files),
                "priority": self.assign_priority(module_name, files),
                "estimated_size": self.estimate_module_size(files),
            }
        return metadata

    def _build_dependency_graph(
        self,
        file_list: List[str],
        initial_modules: Dict[str, List[str]],
        file_summaries: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Build a lightweight dependency graph leveraging tree-sitter metadata."""
        file_to_module = self._map_files_to_modules(initial_modules)
        module_dependency_counter = defaultdict(lambda: defaultdict(int))

        files_info = []
        edges = []

        for file_path in file_list:
            analysis_result = self._analyze_single_file(file_path)
            formatted = analysis_result.get("analysis") or {}
            summary = formatted.get("summary", {})

            classes = [
                cls.get("name") for cls in summary.get("classes", []) if cls.get("name")
            ]
            functions = [
                func.get("name")
                for func in summary.get("functions", [])
                if func.get("name")
            ]
            structs = [
                struct.get("name")
                for struct in summary.get("structs", [])
                if struct.get("name")
            ]
            interfaces = [
                interface.get("name")
                for interface in summary.get("interfaces", [])
                if interface.get("name")
            ]

            dependencies = self._extract_dependency_strings(formatted)
            manual_summary = (
                file_summaries[file_path]
                if file_summaries and file_path in file_summaries
                else ""
            )

            file_entry = {
                "path": file_path,
                "initial_module": file_to_module.get(file_path, "unmapped"),
                "classes": classes,
                "functions": functions,
                "structs": structs,
                "interfaces": interfaces,
                "dependencies": dependencies,
            }
            if manual_summary:
                file_entry["manual_summary"] = manual_summary
            if analysis_result.get("error"):
                file_entry["analysis_error"] = analysis_result["error"]
            files_info.append(file_entry)
            edges.append({"source": file_path, "targets": dependencies})

            module_name = file_entry["initial_module"]
            for dep in dependencies:
                module_dependency_counter[module_name][dep] += 1

        module_dependency_summary = {}
        for module_name, dep_counts in module_dependency_counter.items():
            sorted_deps = sorted(
                dep_counts.items(), key=lambda item: item[1], reverse=True
            )
            module_dependency_summary[module_name] = {
                "file_count": len(initial_modules.get(module_name, [])),
                "top_dependencies": [
                    {"target": dep, "count": count} for dep, count in sorted_deps[:10]
                ],
            }

        return {
            "files": files_info,
            "edges": edges,
            "module_dependency_summary": module_dependency_summary,
        }

    def cluster_by_directory(self, file_list: List[str]) -> Dict[str, List[str]]:
        """Cluster files based on directory structure.

        This is a simple heuristic approach that groups files by their parent directory.

        Args:
            file_list (List[str]): List of file paths

        Returns:
            Dict[str, List[str]]: Dictionary mapping module names to file lists
        """
        modules = defaultdict(list)

        for file_path in file_list:
            path = Path(file_path)

            # Get the parent directory path
            if len(path.parts) > 1:
                # Use the first directory as module name
                module_name = path.parts[0]

                # For nested structures, use more specific naming
                if len(path.parts) > 2:
                    # e.g., "adk/prebuilt" -> "adk-prebuilt"
                    module_name = "/".join(path.parts[:-1])
            else:
                # Root level files
                module_name = "root"

            modules[module_name].append(file_path)

        return dict(modules)

    def estimate_module_size(self, files: List[str]) -> str:
        """Estimate module size based on number of files.

        Args:
            files (List[str]): List of files in the module

        Returns:
            str: Size estimate ('small', 'medium', 'large')
        """
        num_files = len(files)

        if num_files <= 3:
            return "small"
        elif num_files <= 8:
            return "medium"
        else:
            return "large"

    def assign_priority(self, module_name: str, files: List[str]) -> int:
        """Assign priority to a module based on heuristics.

        Higher priority modules should be processed first (lower number = higher priority).

        Args:
            module_name (str): Name of the module
            files (List[str]): List of files in the module

        Returns:
            int: Priority level (1 = highest, 3 = lowest)
        """
        # Core modules get higher priority
        core_keywords = ["core", "base", "main", "interface", "api", "schema"]

        module_lower = module_name.lower()

        # Priority 1: Core modules or root level
        if (
            any(keyword in module_lower for keyword in core_keywords)
            or module_name == "root"
        ):
            return 1

        # Priority 2: Medium-sized important modules
        if len(files) >= 2:
            return 2

        # Priority 3: Small or utility modules
        return 3

    def cluster_by_llm(
        self,
        file_list: List[str],
        file_summaries: Optional[Dict[str, str]] = None,
        initial_modules: Optional[Dict[str, List[str]]] = None,
        dependency_graph: Optional[Dict[str, Any]] = None,
        module_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Use an LLM to validate/refine modules based on directory and dependency info."""
        initial_modules = initial_modules or self.cluster_by_directory(file_list)
        module_metadata = module_metadata or self._build_module_metadata(
            initial_modules
        )

        initial_snapshot = []
        for name, files in initial_modules.items():
            snapshot = {
                "name": name,
                "files": files,
                "file_count": module_metadata.get(name, {}).get(
                    "file_count", len(files)
                ),
                "priority": module_metadata.get(name, {}).get("priority"),
                "estimated_size": module_metadata.get(name, {}).get("estimated_size"),
            }
            initial_snapshot.append(snapshot)

        payload = {
            "initial_modules": initial_snapshot,
            "dependency_graph": dependency_graph or {},
        }
        if file_summaries:
            payload["file_notes"] = [
                {"path": path, "summary": summary}
                for path, summary in file_summaries.items()
            ]

        system_content = """You are a code organization expert specializing in module boundary detection and codebase architecture.

## Workflow Context
You are the final validation step in a three-phase module clustering pipeline:
1. **Directory Heuristic** — Initial grouping by file path structure (already completed)
2. **Dependency Analysis** — Tree-sitter based call graph and import analysis (already completed)
3. **LLM Validation** — Your task: refine boundaries, assign meaningful names, ensure coherence

## Core Principles
- **Dependency Cohesion**: Files with strong mutual dependencies belong together
- **Single Responsibility**: Each module should have a clear, focused purpose
- **Complete Coverage**: Every file must appear in exactly one module
- **Meaningful Naming**: Use clear, human-friendly module names that reflect functionality

## Output Format
Return ONLY valid JSON (no markdown, no explanation):
```
{
  "modules": [
    {
      "name": "module_name",
      "files": ["file1.py", "file2.py"],
      "description": "One sentence summary"
    }
  ]
}
```
"""

        instructions = f"""# Task: Validate and Refine Module Clustering

## Input Data
```json
{json.dumps(payload, ensure_ascii=False, indent=2)}
```

## Actions
- Review initial modules and dependency graph
- Keep, merge, or split modules based on dependency cohesion
- Assign clear, descriptive module names
- Ensure every file is assigned to exactly one module

Return the final module structure as JSON."""

        try:
            system_message = SystemMessage(content=system_content)

            response = self.llm.invoke(
                [
                    system_message,
                    HumanMessage(content=instructions),
                ]
            )

            content = (
                response.content if hasattr(response, "content") else str(response)
            )

            content = content.strip()
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                content = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                content = content[start:end].strip()

            parsed_result = json.loads(content)
            modules = parsed_result.get("modules", [])

            refined_modules: List[Dict[str, Any]] = []
            module_lookup: Dict[str, Dict[str, Any]] = {}
            assigned_files = set()
            file_set = set(file_list)
            file_to_module = self._map_files_to_modules(initial_modules)

            for module in modules:
                name = module.get("name")
                module_files = module.get("files", [])
                if not name or not isinstance(module_files, list):
                    continue

                sanitized_files = []
                for file_path in module_files:
                    if file_path in file_set and file_path not in assigned_files:
                        sanitized_files.append(file_path)
                        assigned_files.add(file_path)

                if not sanitized_files:
                    continue

                entry = {
                    "name": name,
                    "files": sanitized_files,
                    "description": module.get("description", ""),
                }
                refined_modules.append(entry)
                module_lookup[name] = entry

            leftovers = sorted(file_set - assigned_files)
            for file_path in leftovers:
                origin_module = file_to_module.get(file_path, "unassigned")
                target_name = origin_module or "unassigned"

                if target_name not in module_lookup:
                    module_lookup[target_name] = {
                        "name": target_name,
                        "files": [],
                        "description": "Auto-assigned from initial directory grouping",
                    }
                    refined_modules.append(module_lookup[target_name])

                if file_path not in module_lookup[target_name]["files"]:
                    module_lookup[target_name]["files"].append(file_path)

            if refined_modules:
                return refined_modules
            raise ValueError("LLM response did not contain valid modules")

        except Exception as e:
            print(f"LLM clustering failed: {e}")
            print("Falling back to directory-based clustering")

        return [
            {"name": name, "files": files, "description": ""}
            for name, files in initial_modules.items()
        ]

    def cluster(
        self, file_list: List[str], file_summaries: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """Perform complete clustering of files into modules.

        Uses LLM-based semantic clustering with dependency graph analysis.
        Falls back to directory-based clustering if LLM fails.

        Args:
            file_list (List[str]): List of file paths to cluster
            file_summaries (Dict[str, str]): Optional summaries of each file for LLM analysis

        Returns:
            Dict[str, Any]: Complete module structure with metadata
        """
        # Handle empty file list
        if not file_list:
            print("⚠️  No files to cluster, returning empty module structure")
            return {
                "modules": [],
                "total_modules": 0,
                "total_files": 0,
                "initial_module_snapshot": {},
                "dependency_graph": {"files": [], "edges": [], "module_dependency_summary": {}},
            }

        initial_modules = self.cluster_by_directory(file_list)
        module_metadata = self._build_module_metadata(initial_modules)

        # use LLM clustering with dependency graph analysis
        dependency_graph = self._build_dependency_graph(
            file_list, initial_modules, file_summaries
        )
        module_entries_raw = self.cluster_by_llm(
            file_list=file_list,
            file_summaries=file_summaries,
            initial_modules=initial_modules,
            dependency_graph=dependency_graph,
            module_metadata=module_metadata,
        )

        modules = []
        for module_data in module_entries_raw:
            module_name = module_data.get("name")
            original_files = [f for f in module_data.get("files", [])]

            resolved_files = []
            for file_path in original_files:
                abs_path = self._resolve_file_path(file_path)
                try:
                    rel_path = abs_path.relative_to(self.project_root)
                    resolved_files.append(str(rel_path))
                except ValueError:
                    resolved_files.append(str(abs_path))

            files = sorted(set(resolved_files))
            if not module_name or not files:
                continue

            module = {
                "name": module_name,
                "files": files,
                "priority": self.assign_priority(module_name, files),
                "estimated_size": self.estimate_module_size(files),
                "file_count": len(files),
            }
            description = module_data.get("description", "")
            if description:
                module["description"] = description
            modules.append(module)

        # Sort modules by priority (lower number = higher priority)
        modules.sort(key=lambda m: (m["priority"], m["name"]))

        result = {
            "modules": modules,
            "total_modules": len(modules),
            "total_files": len(file_list),
            "initial_module_snapshot": module_metadata,
            "dependency_graph": dependency_graph,
        }
        return result
