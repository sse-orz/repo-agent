import os
import json
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future, as_completed

from config import CONFIG

from .repo_info_agent import RepoInfoAgent
from .module_clusterer import ModuleClusterer
from .module_doc_agent import ModuleDocAgent
from .macro_doc_agent import MacroDocAgent
from .incremental_update_agent import IncrementalUpdateAgent
from .utils import save_json_to_context, count_tokens, get_llm_max_tokens
from .prompt import FileFilterPrompt
from utils.repo import clone_repo


class MoeAgent:
    """
    Main controller for MoeAgent documentation generation system.

    Orchestrates a 5-stage pipeline:
    1. Repository information collection
    2. Intelligent file selection
    3. Module clustering
    4. Module documentation generation
    5. Macro documentation generation
    """

    # Default ratios for different modes
    # These control what percentage of files to select based on repository size
    DEFAULT_RATIOS = {
        "fast": 0.25,   # Fast mode: select 25% of files for quick analysis
        "smart": 0.75,  # Smart mode: select 75% of files for thorough analysis
        "full": 1.0,    # Full mode: select all files (100%)
    }

    def __init__(
        self,
        owner: str = None,
        repo_name: str = None,
        wiki_path: str = None,
        ratios: Dict[str, float] = None,
    ):
        """Initialize MoeAgent.

        Args:
            owner (str): Repository owner (required for path management)
            repo_name (str): Repository name (optional, inferred from path if not provided)
            wiki_path (str): Custom wiki path (optional, defaults to .wikis/{owner}_{repo_name})
            ratios (Dict[str, float]): Custom ratios for file selection modes
                (defaults to DEFAULT_RATIOS if not provided)
        """
        self.repo_name = repo_name
        self.owner = owner
        self.llm = CONFIG.get_llm()
        self.ratios = ratios or self.DEFAULT_RATIOS.copy()

        # Create unified repo identifier using owner_repo_name format
        self.repo_identifier = f"{owner}_{repo_name}"

        # Setup paths using repo_identifier
        self.repo_path = Path(f".repos/{self.repo_identifier}").absolute()
        if wiki_path:
            self.wiki_path = Path(wiki_path).absolute()
        else:
            self.wiki_path = Path(f".wikis/{self.repo_identifier}").absolute()
        self.cache_path = Path(f".cache/{self.repo_identifier}").absolute()

        # Create directories
        self.wiki_path.mkdir(parents=True, exist_ok=True)
        self.cache_path.mkdir(parents=True, exist_ok=True)

        # Clone repository if it does not exist
        self._ensure_repo_cloned()

        # Initialize components
        self.repo_agent = RepoInfoAgent(
            repo_path=str(self.repo_path), wiki_path=str(self.wiki_path), llm=self.llm
        )

        self.clusterer = ModuleClusterer(llm=self.llm, repo_root=str(self.repo_path))
        self.module_doc_agent = ModuleDocAgent(
            owner=self.owner,
            repo_name=self.repo_name,
            llm=self.llm,
            wiki_base_path=str(self.wiki_path),
        )
        self.macro_doc_agent = MacroDocAgent(
            owner=self.owner,
            repo_name=self.repo_name,
            llm=self.llm,
            wiki_base_path=str(self.wiki_path),
        )

        # Store results from each stage
        self.repo_info = None
        self.selected_files = []
        self.module_structure = None
        self.module_results = []
        self.macro_results = []

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


    def _get_dynamic_max_files(self, total_files: int, mode: str) -> int:
        """Calculate dynamic max files based on mode and total file count.

        Args:
            total_files: Total number of discovered files
            mode: Selection mode (fast, smart, full)

        Returns:
            int: Calculated maximum number of files to select
        """
        ratio = self.ratios.get(mode, self.ratios.get("fast", 0.25))
        return max(1, int(total_files * ratio))

    def generate(
        self,
        max_workers: int = 5,
        allow_incremental: bool = True,
        mode: str = "fast",
    ) -> Dict[str, Any]:
        """Execute the complete documentation generation pipeline.

        Args:
            max_workers (int): Number of parallel workers
            allow_incremental (bool): Whether to allow incremental update (default: True)
            mode (str): File selection mode - "fast" (25%), "smart" (75%), or "full" (100%)
                The actual file limit is calculated from ratios configuration.

        Returns:
            Dict[str, Any]: Complete generation results
        """
        # Decide whether to run full generation or incremental update
        if allow_incremental:
            incremental_agent = IncrementalUpdateAgent(
                owner=self.owner,
                repo_name=self.repo_name,
                llm=self.llm,
                wiki_path=str(self.wiki_path),
            )

            can_update, info = incremental_agent.can_update_incrementally()

            if can_update:
                print("✅ Detected new commits, performing incremental update...")
                return incremental_agent.update(max_workers)
            elif info["reason"] == "no_new_commits":
                print("✅ No new commits, using cached documentation")
                return self._load_cached_summary()

        # Fallback to full documentation generation
        print(f"🚀 Starting full documentation generation (mode: {mode})...")
        return self._full_generate(max_workers, mode=mode)

    def stream(
        self,
        max_workers: int = 5,
        allow_incremental: bool = True,
        progress_callback=None,
        mode: str = "fast",
    ) -> Dict[str, Any]:
        """Execute documentation generation pipeline with streaming progress updates.

        Args:
            max_workers (int): Number of parallel workers
            allow_incremental (bool): Whether to allow incremental update (default: True)
            progress_callback (callable): Callback function for progress updates
            mode (str): File selection mode - "fast" (25%), "smart" (75%), or "full" (100%)
                The actual file limit is calculated from ratios configuration.

        Returns:
            Dict[str, Any]: Complete generation results
        """
        # Send initial progress
        if progress_callback:
            progress_callback(
                {
                    "stage": "started",
                    "message": f"Starting MoeAgent documentation generation (mode: {mode})",
                    "progress": 0,
                }
            )

        # Decide whether to run full generation or incremental update
        if allow_incremental:
            incremental_agent = IncrementalUpdateAgent(
                owner=self.owner,
                repo_name=self.repo_name,
                llm=self.llm,
                wiki_path=str(self.wiki_path),
            )

            can_update, info = incremental_agent.can_update_incrementally()

            if can_update:
                print("✅ Detected new commits, performing incremental update...")
                if progress_callback:
                    progress_callback(
                        {
                            "stage": "incremental_update",
                            "message": "Performing incremental update",
                            "progress": 50,
                        }
                    )
                result = incremental_agent.update(max_workers)
                if progress_callback:
                    progress_callback(
                        {
                            "stage": "completed",
                            "message": "Incremental update completed",
                            "progress": 100,
                        }
                    )
                return result
            elif info["reason"] == "no_new_commits":
                print("✅ No new commits, using cached documentation")
                if progress_callback:
                    progress_callback(
                        {
                            "stage": "cached",
                            "message": "Using cached documentation",
                            "progress": 100,
                        }
                    )
                return self._load_cached_summary()

        # Fallback to full documentation generation with streaming
        print(f"🚀 Starting full documentation generation with streaming (mode: {mode})...")
        return self._full_generate(max_workers, progress_callback, mode=mode)

    def _ensure_repo_cloned(self):
        """Ensure the repository is cloned locally."""
        if not self.repo_path.exists():
            print(f"Cloning repository {self.owner}/{self.repo_name}...")
            clone_repo(
                platform=CONFIG.GIT_PLATFORM if CONFIG.GIT_PLATFORM else "github",
                owner=self.owner,
                repo=self.repo_name,
                dest=str(self.repo_path),
            )
            print(f"Repository cloned to {self.repo_path}")

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

    def _preprocess_files(self, all_files: List[str]) -> List[str]:
        """Preprocess file list by removing already processed files.

        Args:
            all_files: List of all discovered file paths

        Returns:
            List[str]: Filtered file list with already processed files removed
        """
        # Try to load existing selection from cache
        cache_file = self.cache_path / "selected_files.json"
        if not cache_file.exists():
            return all_files

        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)

            existing_files = set(cached_data.get("files", []))
            if not existing_files:
                return all_files

            # Remove already processed files
            filtered_files = [f for f in all_files if f not in existing_files]
            removed_count = len(all_files) - len(filtered_files)

            if removed_count > 0:
                print(f"   Preprocessor: removed {removed_count} already processed files")

            return filtered_files

        except Exception as e:
            print(f"   ⚠️ Preprocessor warning: {e}")
            return all_files

    def _postprocess_files(
        self, selected_files: List[str], all_files: List[str], effective_max_files: int
    ) -> List[str]:
        """Postprocess selected files: deduplicate, validate, and limit.

        Args:
            selected_files: List of files selected by LLM
            all_files: Original list of all discovered files (for validation)
            effective_max_files: Maximum number of files to return (calculated from ratios)

        Returns:
            List[str]: Postprocessed file list
        """
        all_files_set = set(all_files)

        # Validate files exist in original list
        validated_files = [f for f in selected_files if f in all_files_set]

        # Deduplicate while preserving order
        seen: set = set()
        deduped_files: List[str] = []
        for f in validated_files:
            if f not in seen:
                seen.add(f)
                deduped_files.append(f)

        # Limit to effective_max_files
        if len(deduped_files) > effective_max_files:
            deduped_files = deduped_files[:effective_max_files]

        # Normalize path separators
        normalized_files = [f.replace("\\", "/") for f in deduped_files]

        return normalized_files

    def _file_selection(self, mode: str = "fast"):
        """Stage 2: Intelligent File Selection.

        Args:
            mode: Selection mode - "fast", "smart", or "full"
                The file limit is calculated from ratios configuration.
        """
        stage_start = time.time()

        print(f"\n{'='*60}")
        print(f"📂 STAGE 2: Intelligent File Selection")
        print(f"{'='*60}")
        print(f"Mode: {mode} (ratio: {self.ratios.get(mode, 0.25)*100:.0f}%)")
        print(f"Ratios: {self.ratios}")

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
            "test",
            "tests",
        }

        all_files = []

        print("Scanning repository...")
        for root, dirs, files in os.walk(self.repo_path):
            # Filter out excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]

            for file in files:
                if any(file.endswith(ext) for ext in code_extensions):
                    file_path = Path(root) / file
                    filename_lower = file.lower()

                    # Skip files with 'test' or 'mock' in filename
                    if "test" in filename_lower or "mock" in filename_lower:
                        continue

                    try:
                        rel_path = file_path.relative_to(self.repo_path)
                        # Normalize path separators to forward slashes
                        rel_path_str = str(rel_path).replace("\\", "/")
                        all_files.append(rel_path_str)
                    except Exception:
                        continue

        print(f"Found {len(all_files)} code files after pre-filtering")

        # Calculate effective_max_files based on mode and ratios
        effective_max_files = self._get_dynamic_max_files(len(all_files), mode)
        print(
            f"   Effective max files: {effective_max_files} (mode={mode}, ratio={self.ratios.get(mode, 0.25)*100:.0f}%)"
        )

        # Preprocess: remove already processed files (for incremental updates)
        preprocessed_files = self._preprocess_files(all_files)
        if len(preprocessed_files) < len(all_files):
            print(
                f"   After preprocessing: {len(preprocessed_files)} files (removed {len(all_files) - len(preprocessed_files)} existing)"
            )

        # Decide whether to use LLM filtering
        if len(preprocessed_files) <= effective_max_files:
            # File count is within limit, use all files directly
            print(
                f"File count ({len(preprocessed_files)}) <= effective_max ({effective_max_files}), skipping LLM filtering"
            )
            selected_files = preprocessed_files
            selection_method = "direct"
        else:
            # Use LLM to filter files
            print(
                f"File count ({len(preprocessed_files)}) > effective_max ({effective_max_files}), using LLM filtering..."
            )
            selected_files = self._llm_filter_files(preprocessed_files, effective_max_files)
            selection_method = "llm_filtered"

        # Postprocess: deduplicate, validate, and limit
        self.selected_files = self._postprocess_files(
            selected_files, all_files, effective_max_files
        )

        # Save selection
        save_json_to_context(
            self.repo_identifier,
            "",
            "selected_files.json",
            {
                "files": self.selected_files,
                "count": len(self.selected_files),
                "total_scanned": len(all_files),
                "preprocessed_count": len(preprocessed_files),
                "selection_method": selection_method,
            },
        )

        print(f"\n✅ Selected {len(self.selected_files)} files for analysis")

        stage_time = time.time() - stage_start
        self.stage_timings["Stage 2: File Selection"] = stage_time
        print(f"⏱️  Stage 2 completed in {stage_time:.2f}s")

    def _filter_single_chunk(
        self, chunk_files: List[str], chunk_max_files: int
    ) -> List[str]:
        """Filter a single chunk of files using LLM.

        Args:
            chunk_files: List of file paths in this chunk
            chunk_max_files: Maximum number of files to return for this chunk

        Returns:
            List[str]: Filtered file paths from this chunk
        """
        response = self.llm.invoke(
            [
                FileFilterPrompt.get_system_prompt(),
                FileFilterPrompt.get_human_prompt(self.repo_name, chunk_files, chunk_max_files),
            ]
        )
        # Parse response - each line should be a file path
        selected = []
        for line in response.content.strip().split("\n"):
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            if len(line) > 2 and line[0].isdigit() and line[1] in ".)":
                line = line[2:].strip()
            if line in chunk_files:
                selected.append(line)
        return selected

    def _llm_filter_files(
        self, all_files: List[str], max_files: int, parallel_chunks: int = 2
    ) -> List[str]:
        """Use LLM to filter and select the most important files.

        Supports parallel processing for large file lists when token count
        exceeds LLM capacity.

        Args:
            all_files: List of all pre-filtered file paths
            max_files: Maximum number of files to select
            parallel_chunks: Number of parallel chunks when splitting (default: 2)

        Returns:
            List[str]: Selected file paths
        """
        try:
            # Check if file list tokens exceed LLM capacity
            file_list_tokens = count_tokens(str(all_files))
            max_tokens = get_llm_max_tokens(compress_ratio=0.90)

            if file_list_tokens > max_tokens:
                # Split into chunks and process in parallel
                print(
                    f"   File list tokens ({file_list_tokens}) > LLM capacity ({max_tokens}), using parallel filtering..."
                )
                files_num = len(all_files)
                chunk_max_files = max(1, max_files // parallel_chunks)
                selected_files: List[str] = []

                with ThreadPoolExecutor(max_workers=parallel_chunks) as executor:
                    futures = []
                    for i in range(parallel_chunks):
                        start_idx = i * files_num // parallel_chunks
                        end_idx = (i + 1) * files_num // parallel_chunks
                        chunk = all_files[start_idx:end_idx]
                        if not chunk:
                            continue
                        futures.append(
                            executor.submit(
                                self._filter_single_chunk, chunk, chunk_max_files
                            )
                        )

                    for future in as_completed(futures):
                        try:
                            chunk_files = future.result()
                            selected_files.extend(chunk_files)
                        except Exception as e:
                            print(f"   ⚠️ Error filtering chunk: {e}")

                print(f"   Parallel filtering selected {len(selected_files)} files")
            else:
                # Single LLM call for smaller file lists
                response = self.llm.invoke(
                    [
                        FileFilterPrompt.get_system_prompt(),
                        FileFilterPrompt.get_human_prompt(self.repo_name, all_files, max_files),
                    ]
                )

                # Parse response - each line should be a file path
                selected_files = []
                for line in response.content.strip().split("\n"):
                    line = line.strip()
                    # Skip empty lines and potential headers/descriptions
                    if not line or line.startswith("#") or line.startswith("-"):
                        continue
                    # Remove potential numbering (e.g., "1. ", "1) ")
                    if len(line) > 2 and line[0].isdigit() and line[1] in ".)":
                        line = line[2:].strip()
                    # Validate file exists in original list
                    if line in all_files:
                        selected_files.append(line)

            # Deduplicate while preserving order
            seen: set = set()
            deduped_files: List[str] = []
            for f in selected_files:
                if f not in seen:
                    seen.add(f)
                    deduped_files.append(f)
            selected_files = deduped_files

            # Ensure we don't exceed max_files
            if len(selected_files) > max_files:
                print(
                    f"   LLM returned {len(selected_files)} files, limiting to {max_files}"
                )
                selected_files = selected_files[:max_files]

            # If LLM returned too few files, fill with remaining files
            if len(selected_files) < max_files:
                remaining = [f for f in all_files if f not in selected_files]
                needed = max_files - len(selected_files)
                selected_files.extend(remaining[:needed])
                print(
                    f"   LLM returned {len(selected_files) - needed} files, filled to {len(selected_files)}"
                )

            print(f"   LLM selected {len(selected_files)} files")
            return selected_files

        except Exception as e:
            print(f"   ⚠️ LLM filtering failed: {e}, using first {max_files} files")
            return all_files[:max_files]

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

        # Determine if this is a zero-code repository
        is_zero_code_repo = len(self.selected_files) == 0

        if is_zero_code_repo:
            print("ℹ️  Detected zero-code repository (documentation-only)")
            print(
                "   Generating README.md only (skipping API, ARCHITECTURE, DEVELOPMENT)"
            )

            # Generate README only
            self.macro_results = [
                self.macro_doc_agent.generate_single_doc(
                    doc_type="README.md", repo_info=self.repo_info
                )
            ]
        else:
            # Default behavior: generate all macro documents
            self.macro_results = self.macro_doc_agent.generate_all_docs(
                repo_info=self.repo_info,
            )

        # Save macro results
        save_json_to_context(
            self.repo_identifier,
            "",
            "macro_results.json",
            {
                "results": self.macro_results,
                "total_docs": len(self.macro_results),
                "is_zero_code_repo": is_zero_code_repo,
            },
        )

        print(f"\n✅ Macro documentation complete")

        stage_time = time.time() - stage_start
        self.stage_timings["Stage 5: Macro Documentation"] = stage_time
        print(f"⏱️  Stage 5 completed in {stage_time:.2f}s")

    def _full_generate(
        self, max_workers: int, progress_callback=None, mode: str = "fast"
    ) -> Dict[str, Any]:
        """Run the full multi-stage documentation generation pipeline.

        Args:
            max_workers: Maximum number of parallel workers for module docs.
            progress_callback: Optional callback function for progress updates.
            mode: File selection mode - "fast", "smart", or "full".
                The file limit is calculated from ratios configuration.

        Returns:
            Dict[str, Any]: Aggregated results from all pipeline stages.
        """
        start_time = time.time()
        print(f"Starting MoeAgent Documentation Generation Pipeline")

        # Progress stages mapping
        progress_stages = {
            "repo_info": {
                "progress": 10,
                "message": "Collecting repository information",
            },
            "file_selection": {"progress": 30, "message": "Selecting important files"},
            "module_clustering": {
                "progress": 45,
                "message": "Clustering files into modules",
            },
            "module_docs": {
                "progress": 60,
                "message": "Generating module documentation",
            },
            "macro_docs": {"progress": 85, "message": "Generating macro documentation"},
        }

        def _notify(stage: str):
            if progress_callback and stage in progress_stages:
                progress_callback(
                    {
                        "stage": stage,
                        "message": progress_stages[stage]["message"],
                        "progress": progress_stages[stage]["progress"],
                    }
                )

        try:
            # Stage 1: Repository Information Collection (Async - Non-blocking)
            _notify("repo_info")
            executor = ThreadPoolExecutor(max_workers=1)
            stage1_future = executor.submit(self._repo_info)

            # Stage 2: Intelligent File Selection
            _notify("file_selection")
            self._file_selection(mode=mode)

            # Stage 3: Module Clustering
            _notify("module_clustering")
            self._module_clustering()

            # Stage 4: Incremental Module Documentation Generation
            _notify("module_docs")
            self._module_documentation(max_workers)

            # Wait for Stage 1 to complete before Stage 5
            print(f"\n{'='*60}")
            print(f"⏳ Waiting for Stage 1 (repo info collection) to complete...")
            print(f"{'='*60}")
            stage1_future.result()  # Block here until Stage 1 completes
            executor.shutdown(wait=True)
            print(f"✅ Stage 1 completed, continuing with Stage 5...\n")

            # Stage 5: Macro Documentation Generation
            _notify("macro_docs")
            self._macro_documentation()

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

            # Send completion progress
            if progress_callback:
                progress_callback(
                    {
                        "stage": "completed",
                        "message": "Documentation generation completed successfully",
                        "progress": 100,
                        "elapsed_time": total_time,
                    }
                )

            return results

        except Exception as e:
            print(f"\n{'='*60}")
            print(f"❌ Error in documentation generation pipeline:")
            print(f"   {str(e)}")
            print(f"{'='*60}\n")

            # Send error progress
            if progress_callback:
                progress_callback(
                    {
                        "stage": "error",
                        "message": f"Error: {str(e)}",
                        "progress": -1,
                    }
                )
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
        # Determine if this is a zero-code repository
        is_zero_code_repo = len(self.selected_files) == 0

        # Collect statistics on generated macro document types
        generated_docs = [
            r.get("doc_type")
            for r in self.macro_results
            if r.get("status") in ["success", "fallback"]
        ]
        all_doc_types = ["README.md", "API.md", "ARCHITECTURE.md", "DEVELOPMENT.md"]
        skipped_docs = [d for d in all_doc_types if d not in generated_docs]

        return {
            "repo_name": self.repo_name,
            "repo_path": str(self.repo_path),
            "wiki_path": str(self.wiki_path),
            "cache_path": str(self.cache_path),
            "is_zero_code_repo": is_zero_code_repo,  # Flag for zero-code repositories
            "ratios": self.ratios,  # File selection ratios configuration
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
                    "status": (
                        "skipped"
                        if is_zero_code_repo
                        else ("success" if self.module_structure else "error")
                    ),
                    "time": self.stage_timings.get("Stage 3: Module Clustering", 0),
                    "total_modules": (
                        self.module_structure.get("total_modules", 0)
                        if self.module_structure
                        else 0
                    ),
                    "reason": "zero_code_repository" if is_zero_code_repo else None,
                },
                "4_module_docs": {
                    "status": "skipped" if is_zero_code_repo else "success",
                    "time": self.stage_timings.get("Stage 4: Module Documentation", 0),
                    "total": len(self.module_results),
                    "successful": sum(
                        1 for r in self.module_results if r["status"] == "success"
                    ),
                    "failed": sum(
                        1 for r in self.module_results if r["status"] == "error"
                    ),
                    "reason": "zero_code_repository" if is_zero_code_repo else None,
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
                    "docs_generated": generated_docs,  # List of generated documents
                    "docs_skipped": (
                        skipped_docs if is_zero_code_repo else []
                    ),  # List of skipped documents
                },
            },
        }

    def _save_generation_summary(self, results: Dict[str, Any]):
        """Save generation summary to cache.

        Args:
            results (Dict[str, Any]): Complete results
        """
        # Add baseline information (from repo_info commits[0])
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

    custom_ratios = {
        "fast": 0.25,    # 25% for fast mode
        "smart": 0.50,   # 50% for smart mode
        "full": 1.0,     # 100% for full mode
        "minimal": 0.10, # Custom 10% mode
    }

    # Example 1: Create MoeAgent with default ratios
    agent = MoeAgent(
        owner="cloudwego",
        repo_name="eino",
        wiki_path=".wikis/moe/cloudwego_eino",
        ratios=custom_ratios,
    )

    # Generate documentation using "smart" mode (75% file selection)
    results = agent.generate(max_workers=4, mode="fast")

    # Print summary
    print("\n" + "=" * 60)
    print("Generation Summary:")
    print("=" * 60)
    print(f"Repository: {results['repo_name']}")
    print(f"Total Time: {results['total_time']:.2f}s")
    print(f"Ratios: {results.get('ratios', {})}")
    print(f"\nStages:")
    for stage_name, stage_result in results["stages"].items():
        status_icon = "✅" if stage_result.get("status") == "success" else "❌"
        print(f"{status_icon} {stage_name}: {stage_result.get('status', 'unknown')}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    test_moe_agent()
