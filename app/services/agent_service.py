from typing import List, Optional
from datetime import datetime
import os

from app.models.agents import (
    GenerateRequest,
    GenerateResponseData,
    FileInfo,
    WikiItem,
    ListWikisResponseData,
    AgentMode,
)
from agents.sub_graph.parent import ParentGraphBuilder
from agents.moe_agent.moe_agent import MoeAgent
from config import CONFIG
from utils.repo import clone_repo, pull_repo


class AgentService:

    def __init__(self):
        self.wiki_root = "./.wikis"
        # Create subdirectories for different modes
        os.makedirs(os.path.join(self.wiki_root, "sub"), exist_ok=True)
        os.makedirs(os.path.join(self.wiki_root, "moe"), exist_ok=True)

    def check_existing_wiki(
        self, request: GenerateRequest, mode: AgentMode
    ) -> Optional[GenerateResponseData]:
        # Check for existing wiki documentation based on request and mode
        if request.need_update:
            return None
        return self.get_wiki_info(request.owner, request.repo, mode)

    def get_wiki_info(
        self, owner: str, repo: str, mode: AgentMode
    ) -> Optional[GenerateResponseData]:
        # Get existing wiki information if available, based on mode
        mode_str = mode.value  # "sub" or "moe"
        wiki_dir = f"{owner}_{repo}"
        wiki_path = os.path.join(self.wiki_root, mode_str, wiki_dir)

        if not os.path.exists(wiki_path):
            return None

        wiki_url = f"/wikis/{mode_str}/{wiki_dir}"
        files = self.get_wiki_files(wiki_path, owner, repo, mode_str)

        return GenerateResponseData(
            owner=owner,
            repo=repo,
            wiki_path=wiki_path,
            wiki_url=wiki_url,
            files=files,
            total_files=len(files),
        )

    def get_wiki_files(
        self, wiki_path: str, owner: str, repo: str, mode_str: str = ""
    ) -> List[FileInfo]:
        # Retrieve list of files in the wiki directory
        files = []
        if not os.path.exists(wiki_path):
            return files

        for root, dirs, filenames in os.walk(wiki_path):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, wiki_path)
                file_size = os.path.getsize(file_path)

                # Generate URL path with mode prefix
                if mode_str:
                    url_path = f"/wikis/{mode_str}/{owner}_{repo}/{rel_path}"
                else:
                    url_path = f"/wikis/{owner}_{repo}/{rel_path}"

                files.append(
                    FileInfo(name=filename, path=rel_path, url=url_path, size=file_size)
                )

        return files

    def preprocess_repo(self, request: GenerateRequest) -> bool:
        # Preprocess the repository for documentation generation
        repo_root = "./.repos"
        repo_dir = f"{request.owner}_{request.repo}"
        repo_path = os.path.join(repo_root, repo_dir)

        try:
            cloned = clone_repo(
                platform=request.platform,
                owner=request.owner,
                repo=request.repo,
                dest=repo_path,
            )
            if cloned:
                print(
                    f"[AgentService] Cloned repository {request.platform}:{request.owner}/{request.repo} to {repo_path}."
                )
            else:
                print(
                    f"[AgentService] Repository already exists at {repo_path}. Pulling latest changes..."
                )
                pulled = pull_repo(
                    platform=request.platform,
                    owner=request.owner,
                    repo=request.repo,
                    dest=repo_path,
                )
                if pulled:
                    print(
                        f"[AgentService] Pulled latest changes for {request.platform}:{request.owner}/{request.repo}."
                    )
        except Exception as e:
            # throw exception to upper layer
            raise RuntimeError(
                f"Failed to preprocess repository for {request.platform}:{request.owner}/{request.repo}: {e}"
            ) from e

        return True

    def generate_documentation(
        self, mode: AgentMode, request: GenerateRequest, progress_callback=None
    ) -> GenerateResponseData:
        """Generate documentation using the specified agent mode."""
        CONFIG.display()

        # Dispatch to specific agent method based on mode
        if mode == AgentMode.SUB:
            self._generate_with_sub(request, progress_callback)
        elif mode == AgentMode.MOE:
            self._generate_with_moe(request, progress_callback)
        else:
            raise ValueError(f"Unknown agent mode: {mode}")

        # Unified response generation
        mode_str = mode.value  # "sub" or "moe"
        wiki_dir = f"{request.owner}_{request.repo}"
        wiki_path = os.path.join(self.wiki_root, mode_str, wiki_dir)
        wiki_url = f"/wikis/{mode_str}/{wiki_dir}"
        files = self.get_wiki_files(wiki_path, request.owner, request.repo, mode_str)

        return GenerateResponseData(
            owner=request.owner,
            repo=request.repo,
            wiki_path=wiki_path,
            wiki_url=wiki_url,
            files=files,
            total_files=len(files),
        )

    def _generate_with_sub(
        self, request: GenerateRequest, progress_callback=None
    ) -> None:
        """Generate documentation using ParentGraphBuilder (sub_graph)."""
        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Set wiki_root_path to include mode subdirectory
        wiki_root_path = os.path.join(self.wiki_root, "sub")

        inputs = {
            "owner": request.owner,
            "repo": request.repo,
            "platform": request.platform,
            "mode": request.mode,
            "max_workers": request.max_workers,
            "date": date,
            "log": request.log,
            "wiki_root_path": wiki_root_path,
        }
        config = {"configurable": {"thread_id": f"wiki-generation-{date}"}}

        parent_graph_builder = ParentGraphBuilder(branch_mode=request.branch_mode)

        if progress_callback:
            parent_graph_builder.stream(
                inputs=inputs,
                progress_callback=progress_callback,
                config=config,
                count_time=True,
            )
        else:
            parent_graph_builder.run(inputs=inputs, config=config, count_time=True)

    def _generate_with_moe(
        self, request: GenerateRequest, progress_callback=None
    ) -> None:
        """Generate documentation using MoeAgent."""
        # Set wiki_path to include mode subdirectory
        wiki_path = os.path.join(
            self.wiki_root, "moe", f"{request.owner}_{request.repo}"
        )
        moe_agent = MoeAgent(
            owner=request.owner, repo_name=request.repo, wiki_path=wiki_path
        )

        max_files = 100 if request.mode == "smart" else 30

        if progress_callback:
            moe_agent.stream(
                max_files=max_files,
                max_workers=request.max_workers,
                allow_incremental=not request.need_update,
                progress_callback=progress_callback,
            )
        else:
            moe_agent.generate(
                max_files=max_files,
                max_workers=request.max_workers,
                allow_incremental=not request.need_update,
            )

    def list_wikis(self) -> ListWikisResponseData:
        # List all generated wikis from both sub and moe directories
        wikis = []

        if not os.path.exists(self.wiki_root):
            return ListWikisResponseData(wikis=[], total_wikis=0)

        # Scan both mode subdirectories
        for mode_str in ["sub", "moe"]:
            mode_path = os.path.join(self.wiki_root, mode_str)
            if not os.path.exists(mode_path):
                continue

            # Scan wiki mode directory for owner_repo folders
            for item in os.listdir(mode_path):
                item_path = os.path.join(mode_path, item)

                # Skip files, only process directories
                if not os.path.isdir(item_path):
                    continue

                # Parse owner and repo from directory name (format: owner_repo)
                parts = item.split("_", 1)
                if len(parts) != 2:
                    continue

                owner, repo = parts

                # Get last modification time
                mtime = os.path.getmtime(item_path)
                generated_at = datetime.fromtimestamp(mtime).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                # Count files in the directory
                total_files = sum(len(files) for _, _, files in os.walk(item_path))

                wiki_url = f"/wikis/{mode_str}/{item}"

                wikis.append(
                    WikiItem(
                        owner=owner,
                        repo=repo,
                        wiki_path=item_path,
                        wiki_url=wiki_url,
                        total_files=total_files,
                        generated_at=generated_at,
                        mode=mode_str,
                    )
                )

        # Sort by generated_at (most recent first)
        wikis.sort(key=lambda x: x.generated_at, reverse=True)

        return ListWikisResponseData(wikis=wikis, total_wikis=len(wikis))
