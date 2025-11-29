import json
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from config import CONFIG
from .utils import (
    get_commits,
    git_pull_with_retry,
    git_diff_name_status,
    git_diff_multiple_files,
    git_get_current_head_sha,
)
from .module_doc_agent import ModuleDocAgent
from .macro_doc_agent import MacroDocAgent
from .module_clusterer import ModuleClusterer


class IncrementalUpdateAgent:
    """
    Agent responsible for incremental documentation updates.

    Responsibilities:
    1. Detect repository changes (baseline commit vs current HEAD)
    2. Analyze affected files and modules
    3. Coordinate partial updates (via ModuleDocAgent, MacroDocAgent)
    4. Generate an incremental update summary
    """

    def __init__(
        self,
        owner: str,
        repo_name: str,
        llm=None,
        wiki_path: str = None,
    ):
        """Initialize the incremental update agent.

        Args:
            owner: Repository owner.
            repo_name: Repository name.
            llm: Optional LLM instance to use for reasoning.
        """
        self.owner = owner
        self.repo_name = repo_name
        self.llm = llm or CONFIG.get_llm()

        # Derive paths from owner and repo_name
        self.repo_identifier = f"{owner}_{repo_name}"
        self.repo_path = Path(f".repos/{self.repo_identifier}").absolute()
        if wiki_path:
            self.wiki_path = Path(wiki_path).absolute()
        else:
            self.wiki_path = Path(f".wikis/{self.repo_identifier}").absolute()
        self.cache_path = Path(f".cache/{self.repo_identifier}").absolute()

        # Threshold of changed lines below which we prefer incremental updates
        self.incremental_change_threshold = 600

        self.module_doc_agent = ModuleDocAgent(
            owner=owner,
            repo_name=repo_name,
            llm=self.llm,
            wiki_base_path=str(self.wiki_path),
        )
        self.macro_doc_agent = MacroDocAgent(
            owner=owner,
            repo_name=repo_name,
            llm=self.llm,
            wiki_base_path=str(self.wiki_path),
        )

    def can_update_incrementally(self) -> Tuple[bool, Dict[str, Any]]:
        """Check whether incremental update is possible.

        Returns:
            Tuple[bool, Dict[str, Any]]: ``(can_update, update_info)`` where
                - ``can_update`` indicates whether there are new commits to process
                - ``update_info`` contains details such as ``baseline_sha``, ``new_sha`` and ``reason``
        """
        # 1. Check if cache exists
        if not self.cache_path.exists():
            return (False, {"reason": "no_cache"})

        # 2. Load baseline commit information from cached repo info
        baseline_info = self._load_baseline()
        if not baseline_info:
            return (False, {"reason": "no_baseline"})

        # 3. Fetch latest remote commit
        try:
            current_commits = get_commits(self.owner, self.repo_name, per_page=1)
            if not current_commits:
                return (False, {"reason": "cannot_fetch_commits"})

            current_sha = current_commits[0]["sha"]
        except Exception as e:
            print(f"⚠️  Failed to fetch remote commits: {e}")
            return (False, {"reason": "fetch_error", "error": str(e)})

        # 4. Compare baseline with current
        if baseline_info["sha"] == current_sha:
            return (False, {"reason": "no_new_commits"})

        return (
            True,
            {
                "old_sha": baseline_info["sha"],
                "old_date": baseline_info.get("date"),
                "new_sha": current_sha,
                "new_date": current_commits[0].get("date"),
            },
        )

    def update(self, max_workers: int = 5) -> Dict[str, Any]:
        """Run the end‑to‑end incremental update workflow.

        Pipeline steps:
        1. Run ``git pull`` to update the local repository
        2. Compute changed files via ``git diff``
        3. Map changes to affected modules
        4. Partially update documentation for dirty modules
        5. Decide whether macro‑level docs need updates
        6. Compile and persist an incremental update summary

        Args:
            max_workers: Number of worker threads for module updates.

        Returns:
            Dict[str, Any]: Aggregate incremental update summary.
        """
        start_time = time.time()

        print(f"\n{'='*60}")
        print(f"🔄 Starting Incremental Documentation Update")
        print(f"{'='*60}")

        # Step 1: Git pull
        print(f"\n📥 Step 1: Updating local repository...")
        success, message = self._git_pull_with_retry()
        if not success:
            return {
                "status": "error",
                "error": message,
                "total_time": time.time() - start_time,
            }
        print(f"✅ {message}")

        # Step 2: Collect changed files between baseline and HEAD
        print(f"\n📊 Step 2: Analyzing file changes...")
        baseline_info = self._load_baseline()
        baseline_sha = baseline_info["sha"]
        changed_files_info = self._get_changed_files(baseline_sha)

        if not changed_files_info["changed_files"]:
            print(f"ℹ️  No file changes detected")
            return {
                "status": "no_changes",
                "baseline_sha": baseline_sha,
                "total_time": time.time() - start_time,
            }

        print(f"✅ Found {len(changed_files_info['changed_files'])} changed files")
        print(f"   Modified: {changed_files_info['stats']['modified']}")
        print(f"   Deleted: {changed_files_info['stats']['deleted']}")

        # Step 3: Analyze affected modules based on changed files
        print(f"\n🗂️  Step 3: Analyzing affected modules...")
        affected_modules = self._analyze_affected_modules(changed_files_info)

        if not affected_modules["dirty_modules"]:
            print(f"ℹ️  No modules affected (changes outside selected files)")
            # Still need to refresh repo_info.json baseline and summary cache
            self._update_baseline()
            return {
                "status": "no_module_changes",
                "baseline_sha": baseline_sha,
                "changed_files": len(changed_files_info["changed_files"]),
                "total_time": time.time() - start_time,
            }

        print(f"✅ {len(affected_modules['dirty_modules'])} modules affected:")
        for mod_info in affected_modules["dirty_modules"][:5]:
            print(
                f"   • {mod_info['module_name']} ({len(mod_info['changed_files'])} files)"
            )
        if len(affected_modules["dirty_modules"]) > 5:
            print(f"   ... and {len(affected_modules['dirty_modules']) - 5} more")

        # Step 4: Update module‑level documentation
        print(f"\n📝 Step 4: Updating module documentation...")
        module_results = self._update_module_docs(
            affected_modules["dirty_modules"], max_workers
        )

        # Step 4.5: Regenerate module structure for future runs
        print(f"\n🔄 Step 4.5: Regenerating module structure...")
        self._regenerate_module_structure()

        # Step 5: Update macro docs (LLM decides per document if changes matter)
        print(f"\n📄 Step 5: Updating macro documentation...")
        macro_results = self._update_macro_docs(affected_modules, changed_files_info)

        # Step 6: Save incremental update summary to cache
        print(f"\n💾 Step 6: Saving update summary...")
        total_time = time.time() - start_time
        summary = self._compile_update_summary(
            baseline_info,
            changed_files_info,
            affected_modules,
            module_results,
            macro_results,
            total_time,
        )
        self._save_update_summary(summary)

        # Update baseline
        self._update_baseline()

        print(f"\n{'='*60}")
        print(f"✨ Incremental Update Complete!")
        print(f"{'='*60}")
        print(f"⏱️  Total Time: {total_time:.2f}s")
        print(f"📊 Updated: {len(module_results)} modules")
        if macro_results:
            print(f"📄 Macro docs: {len(macro_results)} updated")
        print(f"{'='*60}\n")

        return summary

    def _load_baseline(self) -> Optional[Dict[str, Any]]:
        """Load baseline commit information from ``repo_info.json``.

        Returns:
            Optional[Dict[str, Any]]: Baseline commit metadata or ``None`` if unavailable.
        """
        repo_info_path = self.cache_path / "repo_info.json"
        if not repo_info_path.exists():
            return None

        try:
            with open(repo_info_path, "r", encoding="utf-8") as f:
                repo_info = json.load(f)

            commits = repo_info.get("commits", [])
            if not commits:
                return None

            # Use commits[0] as baseline
            return {
                "sha": commits[0]["sha"],
                "date": commits[0].get("date"),
                "author": commits[0].get("author"),
            }
        except Exception as e:
            print(f"⚠️  Failed to load baseline: {e}")
            return None

    def _git_pull_with_retry(self) -> Tuple[bool, str]:
        """Run ``git pull`` with automatic retry handling."""
        try:
            success, message = git_pull_with_retry(str(self.repo_path), max_retries=3)
            return (success, message)
        except Exception as e:
            return (False, str(e))

    def _get_changed_files(self, baseline_sha: str) -> Dict[str, Any]:
        """Collect file changes between a baseline commit and HEAD.

        Args:
            baseline_sha: Baseline commit SHA.

        Returns:
            Dict[str, Any]: Contains ``changed_files``, ``file_diffs`` and aggregate ``stats``.
        """
        # First, get raw change list for all files
        all_changes = git_diff_name_status(str(self.repo_path), baseline_sha)

        # Load ``selected_files.json`` so we only process files in the selected set
        selected_files_path = self.cache_path / "selected_files.json"
        if selected_files_path.exists():
            with open(selected_files_path, "r", encoding="utf-8") as f:
                selected_data = json.load(f)
                selected_files = set(selected_data.get("files", []))
        else:
            selected_files = set()

        # Filter: only handle modified (M) and deleted (D) files, skip additions (A) for now
        filtered_changes = []
        for change in all_changes:
            status = change["status"]
            filename = change["filename"]

            # Only consider files that were selected during initial scan
            if filename not in selected_files:
                continue

            # Skip newly added files for this incremental path
            if status == "A":
                continue

            if status in ["M", "D"]:
                filtered_changes.append(change)

        # Compute detailed diff only for modified files
        modified_files = [c["filename"] for c in filtered_changes if c["status"] == "M"]
        file_diffs = {}
        if modified_files:
            file_diffs = git_diff_multiple_files(
                str(self.repo_path), baseline_sha, modified_files
            )

        # Basic statistics about filtering results
        stats = {
            "total": len(all_changes),
            "filtered": len(filtered_changes),
            "modified": sum(1 for c in filtered_changes if c["status"] == "M"),
            "deleted": sum(1 for c in filtered_changes if c["status"] == "D"),
        }

        return {
            "changed_files": filtered_changes,
            "file_diffs": file_diffs,
            "stats": stats,
        }

    def _analyze_affected_modules(
        self, changed_files_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Map file‑level changes to module‑level impact.

        Args:
            changed_files_info: File‑level change information from ``_get_changed_files``.

        Returns:
            Dict[str, Any]: Contains ``dirty_modules``, ``clean_modules`` and ``total_modules``.
        """
        # Load existing module structure from cache
        module_structure_path = self.cache_path / "module_structure.json"
        if not module_structure_path.exists():
            return {
                "dirty_modules": [],
                "clean_modules": [],
                "total_modules": 0,
            }

        with open(module_structure_path, "r", encoding="utf-8") as f:
            module_structure = json.load(f)

        modules = module_structure.get("modules", [])
        changed_filenames = {c["filename"] for c in changed_files_info["changed_files"]}
        file_diffs = changed_files_info["file_diffs"]
        change_status_map = {
            c["filename"]: c.get("status", "M")
            for c in changed_files_info["changed_files"]
        }

        dirty_modules = []
        clean_modules = []

        for module in modules:
            module_files = set(module.get("files", []))
            # Check intersection between module files and changed files
            affected_files = module_files & changed_filenames

            if affected_files:
                changed_details = []
                total_changes = 0
                for filepath in affected_files:
                    diff_info = file_diffs.get(filepath, {})
                    file_changes = diff_info.get("changes", 0)
                    total_changes += file_changes
                    changed_details.append(
                        {
                            "filename": filepath,
                            "status": change_status_map.get(filepath, "M"),
                            "changes": file_changes,
                            "diff": diff_info.get("diff", ""),
                        }
                    )

                # Decide priority based on whether module is core and total change size
                is_core = module.get("is_core", False)
                if is_core or total_changes > 500:
                    priority = "high"
                elif total_changes >= 100:
                    priority = "medium"
                else:
                    priority = "low"

                dirty_modules.append(
                    {
                        "module_name": module["name"],
                        "module": module,
                        "changed_files": list(affected_files),
                        "changed_details": changed_details,
                        "total_changes": total_changes,
                        "priority": priority,
                    }
                )
            else:
                clean_modules.append(module["name"])

        # Sort dirty modules by priority (high → low)
        priority_order = {"high": 0, "medium": 1, "low": 2}
        dirty_modules.sort(key=lambda x: priority_order.get(x["priority"], 3))

        return {
            "dirty_modules": dirty_modules,
            "clean_modules": clean_modules,
            "total_modules": len(modules),
        }

    def _update_module_docs(
        self, dirty_modules: List[Dict[str, Any]], max_workers: int
    ) -> List[Dict[str, Any]]:
        """Update documentation for all affected ("dirty") modules.

        Args:
            dirty_modules: Modules that were impacted by recent code changes.
            max_workers: Maximum number of worker threads.

        Returns:
            List[Dict[str, Any]]: Per‑module update results.
        """
        results = []
        total = len(dirty_modules)
        completed_count = 0

        # Refresh analysis cache for each dirty module before updating docs
        print(f"   Pre-refreshing {total} module analyses...")
        for mod_info in dirty_modules:
            module = mod_info["module"]
            module_name = mod_info["module_name"]
            self.module_doc_agent.refresh_module_analysis(
                module_name, module.get("files", []), force=True
            )
        print(f"   ✅ All {total} analyses refreshed\n")

        print(f"   Updating {total} modules with max_workers={max_workers}...")

        def process_module(mod_info: Dict[str, Any]) -> Dict[str, Any]:
            module = mod_info["module"]
            module_name = mod_info["module_name"]
            changed_details = mod_info.get("changed_details", [])

            doc_path = self.module_doc_agent.get_module_doc_path(module_name)
            doc_exists = doc_path.exists()
            prefer_incremental = (
                doc_exists
                and changed_details
                and mod_info.get("total_changes", 0)
                <= self.incremental_change_threshold
            )

            if prefer_incremental:
                print(f"   • {module_name}: applying incremental edit workflow")
                result = self.module_doc_agent.update_module_doc_incremental(
                    module=module,
                    changed_details=changed_details,
                )
                if result.get("status") == "error":
                    print(
                        f"   • {module_name}: incremental update failed, falling back to full regeneration"
                    )
                    result = self.module_doc_agent.generate_module_doc(module)
                return result

            print(f"   • {module_name}: regenerating full document")
            return self.module_doc_agent.generate_module_doc(module)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_module = {
                executor.submit(process_module, mod_info): (i + 1, mod_info)
                for i, mod_info in enumerate(dirty_modules)
            }

            # Consume futures as they complete to report progress
            for future in as_completed(future_to_module):
                module_idx, mod_info = future_to_module[future]
                completed_count += 1

                try:
                    result = future.result()
                    results.append(result)

                    status_icon = "✅" if result["status"] == "success" else "⚠️"
                    print(
                        f"   [{completed_count}/{total}] {status_icon} {mod_info['module_name']}"
                    )
                except Exception as e:
                    print(
                        f"   [{completed_count}/{total}] ❌ {mod_info['module_name']}: {e}"
                    )
                    results.append(
                        {
                            "module_name": mod_info["module_name"],
                            "status": "error",
                            "error": str(e),
                        }
                    )

        return results

    def _update_macro_docs(
        self,
        affected_modules: Dict[str, Any],
        changed_files_info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Update macro documentation using the full change context.

        Returns:
            List[Dict[str, Any]]: Macro‑level documentation update results.
        """
        # Load cached ``repo_info.json`` for high‑level repository metadata
        repo_info_path = self.cache_path / "repo_info.json"
        repo_info = None
        if repo_info_path.exists():
            with open(repo_info_path, "r", encoding="utf-8") as f:
                repo_info = json.load(f)

        # Build text summary for display
        change_summary = self._build_macro_change_summary(
            affected_modules, changed_files_info
        )

        # Prepare FULL change info for LLM cache
        full_change_info = {
            "changed_files": changed_files_info.get("changed_files", []),
            "file_diffs": changed_files_info.get("file_diffs", {}),
            "stats": changed_files_info.get("stats", {}),
            "affected_modules": [
                {
                    "module_name": m["module_name"],
                    "priority": m.get("priority"),
                    "total_changes": m.get("total_changes", 0),
                    "changed_files": m.get("changed_files", []),
                    "changed_details": m.get("changed_details", []),
                }
                for m in affected_modules.get("dirty_modules", [])
            ],
            "clean_modules": affected_modules.get("clean_modules", []),
            "total_modules": affected_modules.get("total_modules", 0),
        }

        return self.macro_doc_agent.update_docs_incremental(
            change_summary=change_summary,
            repo_info=repo_info,
            full_change_info=full_change_info,
        )

    def _build_macro_change_summary(
        self,
        affected_modules: Dict[str, Any],
        changed_files_info: Dict[str, Any],
    ) -> str:
        """Build a human‑readable change summary for macro docs."""
        lines = []
        stats = changed_files_info.get("stats", {})
        lines.append(
            f"- Changed files (selected set): {stats.get('filtered', 0)} "
            f"(modified: {stats.get('modified', 0)}, deleted: {stats.get('deleted', 0)})"
        )

        dirty_modules = affected_modules.get("dirty_modules", [])
        if not dirty_modules:
            lines.append("- No modules were flagged as dirty.")
            return "\n".join(lines)

        lines.append(
            f"- Affected modules: {len(dirty_modules)} / {affected_modules.get('total_modules', 0)}"
        )
        for module_info in dirty_modules[:8]:
            module_line = (
                f"  • {module_info['module_name']} "
                f"({module_info['priority']} priority, "
                f"{len(module_info.get('changed_files', []))} files, "
                f"{module_info.get('total_changes', 0)} changed lines)"
            )
            lines.append(module_line)

            detail_lines = []
            for detail in module_info.get("changed_details", [])[:3]:
                detail_lines.append(
                    f"      - {detail.get('filename')} ({detail.get('status')}, ±{detail.get('changes', 0)} lines)"
                )
            lines.extend(detail_lines)

        if len(dirty_modules) > 8:
            lines.append("  • ... additional modules omitted")

        return "\n".join(lines)

    def _compile_update_summary(
        self,
        baseline_info: Dict[str, Any],
        changed_files_info: Dict[str, Any],
        affected_modules: Dict[str, Any],
        module_results: List[Dict[str, Any]],
        macro_results: List[Dict[str, Any]],
        total_time: float,
    ) -> Dict[str, Any]:
        """Assemble a compact summary payload for the incremental run.

        Returns:
            Dict[str, Any]: Complete incremental update summary.
        """
        # Roughly estimate token usage for reporting/monitoring
        total_tokens = 0
        for mod_result in module_results:
            # Rough estimate: ~5000 tokens per module doc
            total_tokens += 5000

        for macro_result in macro_results:
            # Rough estimate: ~8000 tokens per macro‑level doc
            total_tokens += 8000

        # Fetch current local HEAD SHA to record new baseline
        try:
            new_head_sha = git_get_current_head_sha(str(self.repo_path))
        except:
            new_head_sha = "unknown"

        return {
            "repo_name": self.repo_name,
            "repo_path": str(self.repo_path),
            "wiki_path": str(self.wiki_path),
            "cache_path": str(self.cache_path),
            "update_type": "incremental",
            "total_time": total_time,
            "baseline_sha": baseline_info["sha"],
            "baseline_date": baseline_info.get("date"),
            "new_head_sha": new_head_sha,
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "incremental_stats": {
                "changed_files": len(changed_files_info["changed_files"]),
                "affected_modules": len(affected_modules["dirty_modules"]),
                "cached_modules": len(affected_modules["clean_modules"]),
                "total_modules": affected_modules["total_modules"],
                "total_llm_calls": len(module_results) + len(macro_results),
                "total_tokens_used_estimate": total_tokens,
                "modules_updated": {
                    "success": sum(
                        1 for r in module_results if r["status"] == "success"
                    ),
                    "failed": sum(1 for r in module_results if r["status"] == "error"),
                    "total": len(module_results),
                },
                "macro_docs_updated": len(macro_results) > 0,
            },
            "module_results": module_results,
            "macro_results": macro_results,
        }

    def _save_update_summary(self, summary: Dict[str, Any]):
        """Persist incremental update summary to ``generation_summary.json``.

        Args:
            summary: Summary dictionary produced by ``_compile_update_summary``.
        """
        summary_path = self.cache_path / "generation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"✅ Summary saved to: {summary_path}")

    def _update_baseline(self):
        """Refresh baseline commits in ``repo_info.json`` by refetching latest history."""
        try:
            # Re-fetch the latest commits
            commits = get_commits(self.owner, self.repo_name, per_page=10)

            # Load existing repo_info.json
            repo_info_path = self.cache_path / "repo_info.json"
            if repo_info_path.exists():
                with open(repo_info_path, "r", encoding="utf-8") as f:
                    repo_info = json.load(f)

                # Update commits
                repo_info["commits"] = commits

                # Save back to file
                with open(repo_info_path, "w", encoding="utf-8") as f:
                    json.dump(repo_info, f, indent=2, ensure_ascii=False)

                print(f"✅ Baseline updated in repo_info.json")
        except Exception as e:
            print(f"⚠️  Failed to update baseline: {e}")

    def _regenerate_module_structure(self):
        """Regenerate ``module_structure.json`` using ``ModuleClusterer``.

        This keeps the cached module layout in sync with any new/removed files
        after an incremental update.
        """
        try:
            # Load selected files that were originally chosen for documentation
            selected_files_path = self.cache_path / "selected_files.json"
            if not selected_files_path.exists():
                print(
                    "⚠️  selected_files.json not found, skipping module structure regeneration"
                )
                return

            with open(selected_files_path, "r", encoding="utf-8") as f:
                selected_data = json.load(f)
                file_list = selected_data.get("files", [])

            # Regenerate module clustering using the latest file set
            clusterer = ModuleClusterer(llm=self.llm, repo_root=str(self.repo_path))
            module_structure = clusterer.cluster(file_list)

            # Save updated structure back to cache
            module_structure_path = self.cache_path / "module_structure.json"
            with open(module_structure_path, "w", encoding="utf-8") as f:
                json.dump(module_structure, f, indent=2, ensure_ascii=False)

            print(
                f"✅ Module structure regenerated: {module_structure['total_modules']} modules"
            )
        except Exception as e:
            print(f"⚠️  Failed to regenerate module structure: {e}")
