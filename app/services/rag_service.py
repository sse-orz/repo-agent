from typing import Iterable

from agents.rag_agent import RAGAgent
from app.models.rag import RAGQueryRequest, RAGAnswerData, RAGStreamStepData


class RAGService:
    # Service layer for repository RAG question answering

    def __init__(self):
        # For now create a new agent per service instance.
        # Can be optimized later with caching / singletons if needed.
        self.rag_agent = RAGAgent()

    def _get_repo_dir(self, owner: str, repo: str) -> str:
        """Get repository directory name from owner and repo."""
        return f"{owner}_{repo}"

    def _get_repo_name(self, request: RAGQueryRequest) -> str:
        """Get display repo name from request."""
        return f"{request.platform}:{request.owner}/{request.repo}"

    def ask(self, request: RAGQueryRequest) -> RAGAnswerData:
        """Run a single-turn RAG query and return the final answer plus sources.
        
        Args:
            request: RAG query request containing question and repo info
            
        Returns:
            RAGAnswerData with answer and sources
        """
        # Recreate agent per request to respect mode configuration (fast / smart)
        self.rag_agent = RAGAgent(mode=request.mode)
        
        # Initialize vectorstore for the repository
        repo_dir = self._get_repo_dir(request.owner, request.repo)
        self.rag_agent.init_repo(repo_dir)
        
        repo_name = self._get_repo_name(request)
        return self.rag_agent.ask(request.question, repo_name, repo_dir)

    def stream(self, request: RAGQueryRequest) -> Iterable[RAGStreamStepData]:
        """Stream RAG answering process, yielding answer and node name for each step.
        
        Args:
            request: RAG query request containing question and repo info
            
        Yields:
            RAGStreamStepData for each step in the process
        """
        # Recreate agent per request to respect mode configuration (fast / smart)
        self.rag_agent = RAGAgent(mode=request.mode)
        
        # Initialize vectorstore for the repository
        repo_dir = self._get_repo_dir(request.owner, request.repo)
        self.rag_agent.init_repo(repo_dir)
        
        repo_name = self._get_repo_name(request)
        yield from self.rag_agent.stream(request.question, repo_name, repo_dir)
