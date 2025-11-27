from typing import List, Optional
from datetime import datetime
import os

from app.models.agents import (
    GenerateRequest,
    GenerateResponseData,
    FileInfo,
    WikiItem,
    ListWikisResponseData,
)
from agents.sub_graph.parent import ParentGraphBuilder
from config import CONFIG


class AgentService:

    def __init__(self):
        self.wiki_root = "./.wikis"

    def check_existing_wiki(
        self, request: GenerateRequest
    ) -> Optional[GenerateResponseData]:
        # Check for existing wiki documentation based on request
        if request.need_update:
            return None
        return self.get_wiki_info(request.owner, request.repo)

    def get_wiki_info(self, owner: str, repo: str) -> Optional[GenerateResponseData]:
        # Get existing wiki information if available
        wiki_dir = f"{owner}_{repo}"
        wiki_path = os.path.join(self.wiki_root, wiki_dir)

        if not os.path.exists(wiki_path):
            return None

        wiki_url = f"/wikis/{wiki_dir}"
        files = self.get_wiki_files(wiki_path, owner, repo)

        return GenerateResponseData(
            owner=owner,
            repo=repo,
            wiki_path=wiki_path,
            wiki_url=wiki_url,
            files=files,
            total_files=len(files),
        )

    def get_wiki_files(self, wiki_path: str, owner: str, repo: str) -> List[FileInfo]:
        # Retrieve list of files in the wiki directory
        files = []
        if not os.path.exists(wiki_path):
            return files

        for root, dirs, filenames in os.walk(wiki_path):
            for filename in filenames:
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, wiki_path)
                file_size = os.path.getsize(file_path)

                # Generate URL path
                url_path = f"/wikis/{owner}_{repo}/{rel_path}"

                files.append(
                    FileInfo(name=filename, path=rel_path, url=url_path, size=file_size)
                )

        return files

    def generate_documentation(
        self, request: GenerateRequest, progress_callback=None
    ) -> GenerateResponseData:
        # Generate or update documentation for a repository.
        CONFIG.display()
        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        inputs = {
            "owner": request.owner,
            "repo": request.repo,
            "platform": request.platform,
            "mode": request.mode,
            "max_workers": request.max_workers,
            "date": date,
            "log": request.log,
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

        wiki_dir = f"{request.owner}_{request.repo}"
        wiki_path = os.path.join(self.wiki_root, wiki_dir)
        wiki_url = f"/wikis/{wiki_dir}"

        files = self.get_wiki_files(wiki_path, request.owner, request.repo)

        return GenerateResponseData(
            owner=request.owner,
            repo=request.repo,
            wiki_path=wiki_path,
            wiki_url=wiki_url,
            files=files,
            total_files=len(files),
        )

    def list_wikis(self) -> ListWikisResponseData:
        # List all generated wikis
        wikis = []

        if not os.path.exists(self.wiki_root):
            return ListWikisResponseData(wikis=[], total_wikis=0)

        # Scan wiki root directory for owner_repo folders
        for item in os.listdir(self.wiki_root):
            item_path = os.path.join(self.wiki_root, item)

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
            generated_at = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

            # Count files in the directory
            total_files = sum(len(files) for _, _, files in os.walk(item_path))

            wiki_url = f"/wikis/{item}"

            wikis.append(
                WikiItem(
                    owner=owner,
                    repo=repo,
                    wiki_path=item_path,
                    wiki_url=wiki_url,
                    total_files=total_files,
                    generated_at=generated_at,
                )
            )

        # Sort by generated_at (most recent first)
        wikis.sort(key=lambda x: x.generated_at, reverse=True)

        return ListWikisResponseData(wikis=wikis, total_wikis=len(wikis))
