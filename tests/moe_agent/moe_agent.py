import os
import json
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future
from langchain_openai import ChatOpenAI

from config import CONFIG
from agents.summary_agent import SummaryAgent

from .repo_info_agent import RepoInfoAgent
from .module_clusterer import ModuleClusterer
from .module_doc_agent import ModuleDocAgent
from .macro_doc_agent import MacroDocAgent
from .incremental_update_agent import IncrementalUpdateAgent
from .context_tools import save_json_to_context


class MoeAgent:
    """
    Main controller for MoeAgent documentation generation system.

    Orchestrates a 6-stage pipeline:
    1. Repository information collection
    2. Intelligent file selection
    3. Module clustering
    4. Incremental module documentation generation
    5. Macro documentation generation
    6. Index/summary generation
    """

    def __init__(
        self,
        repo_path: str,
        wiki_path: str = None,
        owner: str = None,
        repo_name: str = None,
    ):
        """Initialize MoeAgent.

        Args:
            repo_path (str): Local path to the repository
            wiki_path (str): Path to save wiki files (default: .wikis/{owner}_{repo_name})
            owner (str): Repository owner (required for path management)
            repo_name (str): Repository name (optional, inferred from path if not provided)
        """
        self.repo_path = Path(repo_path).absolute()

        # Infer repo_name from path if not provided
        if not repo_name:
            repo_name = self.repo_path.name

        self.repo_name = repo_name
        self.owner = owner
        self.llm = CONFIG.get_llm()

        # Create unified repo identifier using owner_repo_name format
        self.repo_identifier = f"{owner}_{repo_name}"

        # Setup paths using repo_identifier
        if wiki_path:
            self.wiki_path = Path(wiki_path).absolute()
        else:
            self.wiki_path = Path(f".wikis/{self.repo_identifier}").absolute()

        self.cache_path = Path(f".cache/{self.repo_identifier}").absolute()

        # Create directories
        self.wiki_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        repo_llm = ChatOpenAI(
            model="Minimax-M2",
            openai_api_key=CONFIG.MINIMAX_API_KEY,
            openai_api_base="https://api.minimaxi.com/v1",
        )
        self.repo_agent = RepoInfoAgent(
            repo_path=str(self.repo_path), wiki_path=str(self.wiki_path), llm=repo_llm
        )

        self.clusterer = ModuleClusterer(llm=self.llm, repo_root=str(self.repo_path))
        self.module_doc_agent = ModuleDocAgent(
            repo_identifier=self.repo_identifier,
            repo_root=str(self.repo_path),
            wiki_path=str(self.wiki_path),
            cache_path=str(self.cache_path),
            llm=self.llm,
        )
        self.macro_doc_agent = MacroDocAgent(
            repo_identifier=self.repo_identifier,
            wiki_path=str(self.wiki_path),
            cache_path=str(self.cache_path),
            llm=self.llm,
        )
        self.summary_agent = SummaryAgent(
            repo_path=str(self.repo_path), wiki_path=str(self.wiki_path)
        )

        # Store results from each stage
        self.repo_info = None
        self.selected_files = []
        self.module_structure = None
        self.module_results = []
        self.macro_results = []
        self.summary_result = None

        # Store timing information for each stage
        self.stage_timings = {}

        print(f"\n{'='*60}")
        print(f"🎯 MoeAgent Initialized")
        print(f"{'='*60}")
        print(f"Repository: {self.repo_name}")
        print(f"Path: {self.repo_path}")
        print(f"Wiki: {self.wiki_path}")
        print(f"Cache: {self.cache_path}")
        print(f"{'='*60}\n")

    def generate(
        self, max_files: int = 50, max_workers: int = 5, allow_incremental: bool = True
    ) -> Dict[str, Any]:
        """Execute the complete documentation generation pipeline.

        Args:
            max_files (int): Maximum number of files to analyze
            max_workers (int): Number of parallel workers
            allow_incremental (bool): Whether to allow incremental update (default: True)

        Returns:
            Dict[str, Any]: Complete generation results
        """
        # Decide whether to run full generation or incremental update
        if allow_incremental:
            incremental_agent = IncrementalUpdateAgent(
                repo_path=str(self.repo_path),
                wiki_path=str(self.wiki_path),
                cache_path=str(self.cache_path),
                repo_identifier=self.repo_identifier,
                owner=self.owner,
                repo_name=self.repo_name,
                llm=self.llm,
            )

            can_update, info = incremental_agent.can_update_incrementally()

            if can_update:
                print("✅ Detected new commits, performing incremental update...")
                return incremental_agent.update(max_workers)
            elif info["reason"] == "no_new_commits":
                print("✅ No new commits, using cached documentation")
                return self._load_cached_summary()

        # Fallback to full documentation generation
        print("🚀 Starting full documentation generation...")
        return self._full_generate(max_files, max_workers)

    def _repo_info(self):
        """Stage 1: Repository Information Collection."""
        stage_start = time.time()

        print(f"\n{'='*60}")
        print(f"📊 STAGE 1: Repository Information Collection")
        print(f"{'='*60}")

        try:
            self.repo_info = self.repo_agent.run(
                owner=self.owner, repo_name=self.repo_name
            )

            # Save to cache
            save_json_to_context(
                self.repo_identifier, "", "repo_info.json", self.repo_info
            )

            print(f"\n✅ Repository information collected:")
            print(f"   Name: {self.repo_info.get('repo_name', 'N/A')}")
            print(f"   Language: {self.repo_info.get('main_language', 'N/A')}")
            print(f"   Directories: {len(self.repo_info.get('structure', []))}")
            print(f"   Commits: {len(self.repo_info.get('commits', []))}")

        except Exception as e:
            print(f"❌ Stage 1 failed: {e}")
            # Create minimal repo info
            self.repo_info = {
                "repo_name": self.repo_name,
                "description": "",
                "main_language": "Unknown",
                "structure": [],
                "commits": [],
            }

        stage_time = time.time() - stage_start
        self.stage_timings["Stage 1: Repo Info"] = stage_time
        print(f"⏱️  Stage 1 completed in {stage_time:.2f}s")

    def _file_selection(self, max_files: int):
        """Stage 2: Intelligent File Selection."""
        stage_start = time.time()

        print(f"\n{'='*60}")
        print(f"📂 STAGE 2: Intelligent File Selection")
        print(f"{'='*60}")
        print(f"Target: Select up to {max_files} important files")

        # Collect all code files
        code_extensions = {
            ".py",
            ".js",
            ".java",
            ".cpp",
            ".c",
            ".go",
            ".rs",
            ".ts",
            ".jsx",
            ".tsx",
        }

        # Directories to exclude
        exclude_dirs = {
            ".git",
            "node_modules",
            "__pycache__",
            "build",
            "dist",
            "target",
            "venv",
            "env",
            ".venv",
            ".env",
            "vendor",
            "third_party",
            ".pytest_cache",
            ".mypy_cache",
            ".tox",
            "htmlcov",
            "coverage",
            ".idea",
            ".vscode",
            "eggs",
            ".eggs",
            "lib64",
            "parts",
            ".cache",
        }

        all_files = []
        file_metadata = []

        print("Scanning repository...")
        for root, dirs, files in os.walk(self.repo_path):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if any(file.endswith(ext) for ext in code_extensions):
                    file_path = Path(root) / file
                    try:
                        rel_path = file_path.relative_to(self.repo_path)
                        size = file_path.stat().st_size

                        # Score based on heuristics
                        score = self._score_file_importance(str(rel_path), size)

                        file_metadata.append(
                            {"path": str(rel_path), "size": size, "score": score}
                        )
                        all_files.append(str(rel_path))
                    except Exception:
                        continue

        print(f"Found {len(all_files)} code files")

        # Select top files by score
        file_metadata.sort(key=lambda x: x["score"], reverse=True)
        self.selected_files = [f["path"] for f in file_metadata[:max_files]]

        # Save selection
        save_json_to_context(
            self.repo_identifier,
            "",
            "selected_files.json",
            {
                "files": self.selected_files,
                "count": len(self.selected_files),
                "total_scanned": len(all_files),
            },
        )

        print(f"\n✅ Selected {len(self.selected_files)} files for analysis")

        stage_time = time.time() - stage_start
        self.stage_timings["Stage 2: File Selection"] = stage_time
        print(f"⏱️  Stage 2 completed in {stage_time:.2f}s")

    def _score_file_importance(self, rel_path: str, size: int) -> float:
        """Score file importance for selection.

        Args:
            rel_path (str): Relative file path
            size (int): File size in bytes

        Returns:
            float: Importance score (higher = more important)
        """
        score = 0.0
        path_lower = rel_path.lower()
        filename = Path(rel_path).name.lower()

        # Core directories (high priority)
        if any(
            d in path_lower
            for d in ["src/", "lib/", "core/", "api/", "pkg/", "internal/"]
        ):
            score += 50

        # Entry points (highest priority)
        if filename in [
            "main.py",
            "index.js",
            "app.py",
            "server.py",
            "__init__.py",
            "main.go",
            "main.rs",
            "index.ts",
            "main.c",
            "main.cpp",
        ]:
            score += 100

        # Important files
        if any(
            keyword in filename
            for keyword in ["main", "index", "app", "server", "config", "interface"]
        ):
            score += 30

        # Test files (low priority)
        # Check if path contains test directories
        if any(d in path_lower for d in ["test/", "tests/", "spec/", "__tests__/"]):
            score -= 30
        # Check if filename contains test keywords (e.g. test_xxx.py, xxx_test.py, xxx.test.js etc.)
        elif any(keyword in filename for keyword in ["test", "spec"]):
            score -= 30

        # Example files (low priority)
        if any(d in path_lower for d in ["example/", "examples/", "demo/", "sample/"]):
            score -= 20

        # Size penalty (prefer smaller files)
        size_kb = size / 1024
        if size_kb < 10:
            score += 20
        elif size_kb < 50:
            score += 10
        elif size_kb > 100:
            score -= 20

        return score

    def _module_clustering(self):
        """Stage 3: Module Clustering."""
        stage_start = time.time()

        print(f"\n{'='*60}")
        print(f"🗂️  STAGE 3: Module Clustering")
        print(f"{'='*60}")
        print(f"Method: LLM-based semantic clustering")

        self.module_structure = self.clusterer.cluster(self.selected_files)

        # Save module structure
        save_json_to_context(
            self.repo_identifier, "", "module_structure.json", self.module_structure
        )

        print(f"\n✅ Clustered into {self.module_structure['total_modules']} modules:")
        for module in self.module_structure["modules"][:10]:
            print(
                f"   • {module['name']}: {module['file_count']} files (Priority: {module['priority']})"
            )

        if self.module_structure["total_modules"] > 10:
            print(
                f"   ... and {self.module_structure['total_modules'] - 10} more modules"
            )

        stage_time = time.time() - stage_start
        self.stage_timings["Stage 3: Module Clustering"] = stage_time
        print(f"⏱️  Stage 3 completed in {stage_time:.2f}s")

    def _module_documentation(self, max_workers: int = 5):
        """Stage 4: Incremental Module Documentation Generation."""
        stage_start = time.time()

        print(f"\n{'='*60}")
        print(f"📝 STAGE 4: Module Documentation Generation")
        print(f"{'='*60}")
        print(
            f"Generating documentation for {self.module_structure['total_modules']} modules..."
        )

        self.module_results = self.module_doc_agent.generate_all_modules(
            modules=self.module_structure["modules"], max_workers=max_workers
        )

        # Save module results summary
        summary = {
            "total_modules": len(self.module_results),
            "successful": sum(
                1 for r in self.module_results if r["status"] == "success"
            ),
            "failed": sum(1 for r in self.module_results if r["status"] == "error"),
            "results": self.module_results,
        }

        save_json_to_context(self.repo_identifier, "", "module_results.json", summary)

        print(f"\n✅ Module documentation complete")

        stage_time = time.time() - stage_start
        self.stage_timings["Stage 4: Module Documentation"] = stage_time
        print(f"⏱️  Stage 4 completed in {stage_time:.2f}s")

    def _macro_documentation(self):
        """Stage 5: Macro Documentation Generation (Always Parallel)."""
        stage_start = time.time()

        print(f"\n{'='*60}")
        print(f"📄 STAGE 5: Macro Documentation Generation")
        print(f"{'='*60}")

        self.macro_results = self.macro_doc_agent.generate_all_docs(
            repo_info=self.repo_info,
        )

        # Save macro results
        save_json_to_context(
            self.repo_identifier,
            "",
            "macro_results.json",
            {"results": self.macro_results, "total_docs": len(self.macro_results)},
        )

        print(f"\n✅ Macro documentation complete")

        stage_time = time.time() - stage_start
        self.stage_timings["Stage 5: Macro Documentation"] = stage_time
        print(f"⏱️  Stage 5 completed in {stage_time:.2f}s")

    def _index_generation(self):
        """Stage 6: Index/Summary Generation."""
        stage_start = time.time()

        print(f"\n{'='*60}")
        print(f"📑 STAGE 6: Index Generation")
        print(f"{'='*60}")

        # Prepare document list for summary agent
        docs = []

        # Add macro docs
        for result in self.macro_results:
            if result["status"] in ["success", "fallback"] and result.get("path"):
                docs.append(result["path"])

        # Add module docs
        for result in self.module_results:
            if result["status"] in ["success", "fallback"] and result.get("doc_path"):
                docs.append(result["doc_path"])

        try:
            self.summary_result = self.summary_agent.run(
                docs=docs, wiki_path=str(self.wiki_path), repo_info=self.repo_info
            )

            print(
                f"\n✅ Index generated: {self.summary_result.get('index_file', 'INDEX.md')}"
            )

        except Exception as e:
            print(f"⚠️  Index generation failed: {e}")
            self.summary_result = {"status": "error", "error": str(e)}

        stage_time = time.time() - stage_start
        self.stage_timings["Stage 6: Index Generation"] = stage_time
        print(f"⏱️  Stage 6 completed in {stage_time:.2f}s")

    def _full_generate(self, max_files: int, max_workers: int) -> Dict[str, Any]:
        """Run the full multi-stage documentation generation pipeline.

        Args:
            max_files: Maximum number of files to analyze during selection.
            max_workers: Maximum number of parallel workers for module docs.

        Returns:
            Dict[str, Any]: Aggregated results from all pipeline stages.
        """
        start_time = time.time()
        print(f"Starting MoeAgent Documentation Generation Pipeline")

        try:
            # Stage 1: Repository Information Collection (Async - Non-blocking)
            executor = ThreadPoolExecutor(max_workers=1)
            stage1_future = executor.submit(self._repo_info)

            # Stage 2: Intelligent File Selection
            self._file_selection(max_files)

            # Stage 3: Module Clustering
            self._module_clustering()

            # Stage 4: Incremental Module Documentation Generation
            self._module_documentation(max_workers)

            # Wait for Stage 1 to complete before Stage 5
            print(f"\n{'='*60}")
            print(f"⏳ Waiting for Stage 1 (repo info collection) to complete...")
            print(f"{'='*60}")
            stage1_future.result()  # Block here until Stage 1 completes
            executor.shutdown(wait=True)
            print(f"✅ Stage 1 completed, continuing with Stage 5...\n")

            # Stage 5: Macro Documentation Generation
            self._macro_documentation()

            # Stage 6: Index Generation
            self._index_generation()

            # Calculate total time
            total_time = time.time() - start_time

            # Compile final results
            results = self._compile_results(total_time)

            # Save summary
            self._save_generation_summary(results)

            # Print summary with stage timings
            print(f"\n{'='*60}")
            print(f"✨ Documentation Generation Complete!")
            print(f"{'='*60}")
            print(f"⏱️  Total Time: {total_time:.2f}s")
            print(f"\n📊 Stage Timings:")
            for stage_name, stage_time in self.stage_timings.items():
                print(f"   • {stage_name}: {stage_time:.2f}s")
            print(f"\n📁 Wiki Location: {self.wiki_path}")
            print(f"💾 Cache Location: {self.cache_path}")
            print(f"{'='*60}\n")

            return results

        except Exception as e:
            print(f"\n{'='*60}")
            print(f"❌ Error in documentation generation pipeline:")
            print(f"   {str(e)}")
            print(f"{'='*60}\n")
            raise

    def _load_cached_summary(self) -> Dict[str, Any]:
        """Load a cached generation summary from ``generation_summary.json``.

        Returns:
            Dict[str, Any]: Cached generation summary or an error payload.
        """
        summary_path = self.cache_path / "generation_summary.json"

        if not summary_path.exists():
            return {
                "status": "error",
                "error": "No cached summary found",
                "cache_path": str(self.cache_path),
            }

        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)

            print(f"\n{'='*60}")
            print(f"📋 Using Cached Documentation")
            print(f"{'='*60}")
            print(f"Cache Path: {self.cache_path}")
            print(f"Wiki Path: {self.wiki_path}")
            if "total_time" in summary:
                print(f"Original Generation Time: {summary['total_time']:.2f}s")
            print(f"{'='*60}\n")

            return summary
        except Exception as e:
            return {
                "status": "error",
                "error": f"Failed to load cached summary: {str(e)}",
                "cache_path": str(self.cache_path),
            }

    def _compile_results(self, total_time: float) -> Dict[str, Any]:
        """Compile final results from all stages.

        Args:
            total_time (float): Total execution time in seconds

        Returns:
            Dict[str, Any]: Complete results
        """
        return {
            "repo_name": self.repo_name,
            "repo_path": str(self.repo_path),
            "wiki_path": str(self.wiki_path),
            "cache_path": str(self.cache_path),
            "total_time": total_time,
            "stage_timings": self.stage_timings,  # Add stage timings
            "stages": {
                "1_repo_info": {
                    "status": "success" if self.repo_info else "error",
                    "time": self.stage_timings.get("Stage 1: Repo Info", 0),
                    "result": self.repo_info,
                },
                "2_file_selection": {
                    "status": "success",
                    "time": self.stage_timings.get("Stage 2: File Selection", 0),
                    "files_selected": len(self.selected_files),
                    "files": self.selected_files[:20],  # First 20 for brevity
                },
                "3_module_clustering": {
                    "status": "success" if self.module_structure else "error",
                    "time": self.stage_timings.get("Stage 3: Module Clustering", 0),
                    "total_modules": (
                        self.module_structure.get("total_modules", 0)
                        if self.module_structure
                        else 0
                    ),
                },
                "4_module_docs": {
                    "status": "success",
                    "time": self.stage_timings.get("Stage 4: Module Documentation", 0),
                    "total": len(self.module_results),
                    "successful": sum(
                        1 for r in self.module_results if r["status"] == "success"
                    ),
                    "failed": sum(
                        1 for r in self.module_results if r["status"] == "error"
                    ),
                },
                "5_macro_docs": {
                    "status": "success",
                    "time": self.stage_timings.get("Stage 5: Macro Documentation", 0),
                    "total": len(self.macro_results),
                    "successful": sum(
                        1 for r in self.macro_results if r["status"] == "success"
                    ),
                    "failed": sum(
                        1 for r in self.macro_results if r["status"] == "error"
                    ),
                },
                "6_index": {
                    "status": (
                        self.summary_result.get("status", "unknown")
                        if self.summary_result
                        else "error"
                    ),
                    "time": self.stage_timings.get("Stage 6: Index Generation", 0),
                },
            },
        }

    def _save_generation_summary(self, results: Dict[str, Any]):
        """Save generation summary to cache.

        Args:
            results (Dict[str, Any]): Complete results
        """
        # 添加 baseline 信息（从 repo_info 的 commits[0] 获取）
        if self.repo_info and self.repo_info.get("commits"):
            commits = self.repo_info["commits"]
            if commits:
                results["baseline_sha"] = commits[0].get("sha")
                results["baseline_date"] = commits[0].get("date")

        summary_path = self.cache_path / "generation_summary.json"

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Summary saved to: {summary_path}")


# ========================== Test MoeAgent ==========================
def test_moe_agent():
    """Main function for testing MoeAgent."""
    print("\n" + "=" * 60)
    print("MoeAgent - Intelligent Documentation Generation System")
    print("=" * 60 + "\n")

    # Create MoeAgent instance
    agent = MoeAgent(
        repo_path=".repos/cloudwego_eino",
        wiki_path=".wikis/cloudwego_eino",
        owner="cloudwego",
        repo_name="eino",
    )

    # Generate documentation
    results = agent.generate(max_files=50, max_workers=10)

    # Print summary
    print("\n" + "=" * 60)
    print("Generation Summary:")
    print("=" * 60)
    print(f"Repository: {results['repo_name']}")
    print(f"Total Time: {results['total_time']:.2f}s")
    print(f"\nStages:")
    for stage_name, stage_result in results["stages"].items():
        status_icon = "✅" if stage_result.get("status") == "success" else "❌"
        print(f"{status_icon} {stage_name}: {stage_result.get('status', 'unknown')}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    test_moe_agent()
