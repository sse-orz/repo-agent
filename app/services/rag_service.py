from typing import Dict, Any, Iterable, Optional

from agents.rag_agent import RAGAgent
from app.models.rag import RAGQueryRequest


class RAGService:
    # Service layer for repository RAG question answering

    def __init__(self):
        # For now create a new agent per service instance.
        # Can be optimized later with caching / singletons if needed.
        self.rag_agent = RAGAgent()

    def _build_initial_state(self, request: RAGQueryRequest) -> Dict[str, Any]:
        # Build initial RAGState for a single-turn QA
        repo_name = f"{request.platform}:{request.owner}/{request.repo}"
        return {
            "messages": [],
            "query": "",
            "question": request.question,
            "documents": [],
            "answer": "",
            "is_sufficient": False,
            "retrieval_count": 0,
            "is_repo_related": False,
            "repo_name": repo_name,
        }

    def _init_vectorstores_for_repo(self, owner: str, repo: str) -> None:
        # Ensure vectorstore is initialized for the target repo
        repo_dir = f"{owner}_{repo}"
        self.rag_agent.vectorstores = self.rag_agent._init_vectorstores(repo_dir)

    @staticmethod
    def _extract_node_name_from_state(state: Dict[str, Any]) -> Optional[str]:
        # Extract current node name from the latest system message like '[Judge]: ...'
        messages = state.get("messages") or []
        # Find the latest system node marker from the last message backwards
        for msg in reversed(messages):
            content = getattr(msg, "content", None)
            if not isinstance(content, str):
                continue
            if content.startswith("[") and "]" in content:
                return content[1 : content.index("]")]
        return None

    def ask(self, request: RAGQueryRequest) -> str:
        # Run a single-turn RAG query and return the final answer
        self._init_vectorstores_for_repo(request.owner, request.repo)
        state = self._build_initial_state(request)
        thread_id = f"{request.owner}_{request.repo}-api-chat"

        final_state = self.rag_agent.app.invoke(
            state,
            config={
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 100,
            },
        )

        return final_state.get("answer", "")

    def stream(self, request: RAGQueryRequest) -> Iterable[Dict[str, Any]]:
        # Stream RAG answering process, yielding answer and node name for each step
        self._init_vectorstores_for_repo(request.owner, request.repo)
        state = self._build_initial_state(request)
        thread_id = f"{request.owner}_{request.repo}-api-chat"

        for s in self.rag_agent.app.stream(
            state,
            stream_mode="values",
            config={
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 100,
            },
        ):
            answer = s.get("answer", "")
            node = self._extract_node_name_from_state(s)
            yield {
                "answer": answer,
                "node": node,
            }
