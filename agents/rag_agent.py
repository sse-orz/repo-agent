import os
from datetime import datetime
from typing import TypedDict, List, Annotated, Sequence
from textwrap import dedent

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END

from config import CONFIG


class RAGState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query: str
    question: str
    documents: List[Document]
    answer: str
    is_sufficient: bool
    retrieval_count: int
    is_repo_related: bool
    repo_name: str


class RAGAgent:
    """RAG Agent for repository-related question answering with intelligent retrieval.

    Two modes:
    - fast：default mode, less retrieval attempts, faster response
    - smart：more retrieval attempts, larger history window, better effect
    """

    def __init__(
        self,
        wikis_dir: str = ".wikis",
        embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_retrieval_attempts: int = 2,
        retrieval_k: int = 5,
        history_window_size: int = 10,
        mode: str = "fast",
    ):
        # normalize mode
        mode = (mode or "fast").lower()
        if mode not in {"fast", "smart"}:
            mode = "fast"

        # If user doesn't explicitly set parameters, override defaults based on mode
        if mode == "smart":
            max_retrieval_attempts = 5
            retrieval_k = 10
            history_window_size = 20

        self.wikis_dir = wikis_dir
        self.embeddings_model_name = embeddings_model_name
        self.max_retrieval_attempts = max_retrieval_attempts
        self.retrieval_k = retrieval_k
        self.history_window_size = history_window_size
        self.mode = mode
        self.memory = InMemorySaver()
        self.vectorstores = {}
        self.app = self._build_graph()

    @staticmethod
    def _extract_sources_from_docs(docs: List[Document]) -> List[str]:
        """Extract unique, human-readable source paths from retrieved documents."""
        sources: List[str] = []
        seen = set()
        for doc in docs or []:
            metadata = getattr(doc, "metadata", {}) or {}
            src = (
                metadata.get("source")
                or metadata.get("file_path")
                or metadata.get("path")
                or metadata.get("absolute_source")
            )
            if not src:
                continue
            if src in seen:
                continue
            seen.add(src)
            sources.append(src)
        return sources

    @staticmethod
    def _create_judge_prompt(question: str, context: str) -> HumanMessage:
        """Create prompt to judge document sufficiency."""
        return HumanMessage(
            content=dedent(
                f"""
                Evaluate whether the retrieved documents are sufficient to answer the user's question.

                User question:
                {question}

                Retrieved documents (combined):
                {context}

                Evaluation criteria (for "sufficient"):
                - Contain direct or highly relevant information for the core of the question
                - Include concrete technical details (code, APIs, configuration, etc.)

                Reply with ONLY:
                - "yes" if the documents are sufficient
                - "no" if more information is needed
                """
            ).strip(),
        )

    @staticmethod
    def _create_intent_check_prompt(question: str, repo_name: str) -> HumanMessage:
        """Create prompt to check if question is repository-related."""
        return HumanMessage(
            content=dedent(
                f"""
                Decide if the question is specifically about the repository "{repo_name}".

                User question:
                {question}

                Answer "yes" if the question is about:
                - This repository's codebase, architecture, or design
                - Specific files, modules, classes, or functions in this repo
                - How to use, configure, or extend this repo

                Otherwise (general programming, other projects, chit-chat, etc.), answer "no".

                Reply with ONLY "yes" or "no".
                """
            ).strip(),
        )

    @staticmethod
    def _create_rewrite_prompt(original_question: str, repo_name: str) -> HumanMessage:
        """Create prompt to rewrite question for better retrieval."""
        return HumanMessage(
            content=dedent(
                f"""
                Refine the user's question about the repository "{repo_name}" so it is clearer
                and better suited for document retrieval.

                Original question:
                {original_question}

                Guidelines:
                - Make vague questions more specific and technical
                - Use repository-specific terms (files, modules, APIs) when appropriate
                - Keep it concise but precise

                Output ONLY the refined question text (no explanations or extra text).
                If the original question is already good, you may return it unchanged.
                """
            ).strip(),
        )

    @staticmethod
    def _get_answer_system_prompt() -> SystemMessage:
        """Global system prompt for repository-based answering."""
        return SystemMessage(
            content=dedent(
                """
                You are a Repository Analysis Assistant.
                Your job is to answer questions strictly based on the provided repository context.

                Global rules:
                - Use ONLY the given context as your source of truth
                - Be clear, technical, and concise
                - Reference relevant files, functions, or code snippets when helpful
                - If the context lacks required information, say so explicitly
                """
            ).strip(),
        )

    @staticmethod
    def _create_rag_generation_prompt(
        history: str, context: str, question: str
    ) -> HumanMessage:
        """Create user prompt for RAG-based answer generation."""
        return HumanMessage(
            content=dedent(
                f"""
                Conversation history:
                {history or "(no prior conversation)"}

                Repository context (code, docs, related files):
                ---BEGIN CONTEXT---
                {context}
                ---END CONTEXT---

                User question:
                {question}

                Instructions for this answer:
                - Use ONLY the context above to answer
                - If the context does not contain enough information, say so directly
                - Be specific and technical; quote or reference code and files when useful
                """
            ).strip(),
        )

    @staticmethod
    def _create_direct_generation_prompt(history: str, question: str) -> HumanMessage:
        """Create user prompt for direct answer generation to decline unrelated questions."""
        return HumanMessage(
            content=dedent(
                f"""
                Conversation history:
                {history or "(no prior conversation)"}

                User question:
                {question}

                You have determined that this question is NOT related to the current repository.

                Instructions for this answer:
                - State that you are a Repository Analysis Assistant limited to this repository
                - Politely decline to answer because it is out of scope
                - Do NOT use general world knowledge to answer
                - Invite the user to ask repository-related questions instead
                """
            ).strip(),
        )

    def _build_graph(self):
        """Build and compile the LangGraph state graph."""
        graph = StateGraph(RAGState)

        # Add nodes
        graph.add_node("check_intent", self.check_intent_node)
        graph.add_node("rewrite", self.rewrite_node)
        graph.add_node("retrieve", self.retrieve_node)
        graph.add_node("judge", self.judge_node)
        graph.add_node("generate_rag", self.generate_rag_node)
        graph.add_node("generate_direct", self.generate_direct_node)

        # Add edges
        graph.add_edge(START, "check_intent")
        graph.add_conditional_edges(
            "check_intent",
            lambda s: "rag" if s.get("is_repo_related", False) else "direct",
            {"rag": "rewrite", "direct": "generate_direct"},
        )
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("retrieve", "judge")
        graph.add_conditional_edges(
            "judge",
            lambda s: "generate" if s.get("is_sufficient", False) else "retrieve",
            {"retrieve": "retrieve", "generate": "generate_rag"},
        )
        graph.add_edge("generate_rag", END)
        graph.add_edge("generate_direct", END)

        return graph.compile(checkpointer=self.memory)

    def _invoke_llm(self, prompt: HumanMessage) -> str:
        """Invoke LLM with prompt and return normalized response."""
        llm = CONFIG.get_llm()
        return llm.invoke([prompt]).content.lower().strip()

    def check_intent_node(self, state: RAGState) -> RAGState:
        """Determine if question is repository-related."""
        question = state["question"]
        repo_name = state["repo_name"]

        prompt = self._create_intent_check_prompt(question, repo_name)
        judgment = self._invoke_llm(prompt)
        is_repo_related = "yes" in judgment

        return {
            "is_repo_related": is_repo_related,
            "messages": state["messages"]
            + [
                SystemMessage(
                    content=f"[Intent]: {'Repository' if is_repo_related else 'General'} question"
                )
            ],
        }

    def rewrite_node(self, state: RAGState) -> RAGState:
        """Refine question for better retrieval."""
        original_question = state["question"]
        repo_name = state["repo_name"]

        prompt = self._create_rewrite_prompt(original_question, repo_name)
        refined_question = self._invoke_llm(prompt)

        return {
            "question": refined_question,
            "messages": state["messages"]
            + [
                SystemMessage(
                    content=f"[Rewrite]: '{original_question}' -> '{refined_question}'"
                )
            ],
        }

    def judge_node(self, state: RAGState) -> RAGState:
        """Judge if retrieved documents are sufficient."""
        question = state["question"]
        docs = state["documents"]
        retrieval_count = state.get("retrieval_count", 0)

        if retrieval_count >= self.max_retrieval_attempts:
            return {
                "is_sufficient": True,
                "retrieval_count": retrieval_count + 1,
                "messages": state["messages"]
                + [
                    SystemMessage(
                        content=f"[Judge]: Max attempts ({retrieval_count + 1}) reached"
                    )
                ],
            }

        if not docs:
            return {
                "is_sufficient": False,
                "retrieval_count": retrieval_count + 1,
                "messages": state["messages"]
                + [SystemMessage(content="[Judge]: No documents, retrieving more")],
            }

        context = "\n".join([doc.page_content for doc in docs])
        prompt = self._create_judge_prompt(question, context)
        judgment = self._invoke_llm(prompt)
        is_sufficient = any(
            word in judgment for word in ["yes", "sufficient", "enough"]
        )

        return {
            "is_sufficient": is_sufficient,
            "retrieval_count": retrieval_count + 1,
            "messages": state["messages"]
            + [
                SystemMessage(
                    content=f"[Judge]: {'Sufficient' if is_sufficient else 'Insufficient'} (attempt {retrieval_count + 1})"
                )
            ],
        }

    def retrieve_node(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents from vectorstore."""
        question = state["question"]
        repo_name = state["repo_name"]

        if repo_name and repo_name in self.vectorstores:
            docs = self.vectorstores[repo_name].similarity_search(
                question, k=self.retrieval_k
            )
        else:
            docs = self._fallback_search(question)

        return {
            "documents": docs,
            "messages": state["messages"]
            + [SystemMessage(content=f"[Retrieve]: Found {len(docs)} documents")],
        }

    def _fallback_search(self, question: str) -> List[Document]:
        """Search across all vectorstores when specific repo not found."""
        docs = []
        for vs in self.vectorstores.values():
            docs.extend(vs.similarity_search(question, k=3))
        return docs

    def generate_rag_node(self, state: RAGState) -> RAGState:
        """Generate answer using RAG context."""
        question = state["question"]
        docs = state["documents"]
        context = "\n".join([doc.page_content for doc in docs])
        history = self._get_history(state)

        system_prompt = self._get_answer_system_prompt()
        user_prompt = self._create_rag_generation_prompt(history, context, question)
        llm = CONFIG.get_llm()
        answer = llm.invoke([system_prompt, user_prompt]).content

        return {
            "answer": answer,
            "messages": state["messages"]
            + [
                AIMessage(content=answer),
                SystemMessage(content="[Generate]: Generated RAG answer"),
            ],
        }

    def generate_direct_node(self, state: RAGState) -> RAGState:
        """Generate answer directly without RAG."""
        question = state["question"]
        history = self._get_history(state)

        system_prompt = self._get_answer_system_prompt()
        user_prompt = self._create_direct_generation_prompt(history, question)
        llm = CONFIG.get_llm()
        answer = llm.invoke([system_prompt, user_prompt]).content

        return {
            "answer": answer,
            "messages": state["messages"]
            + [
                AIMessage(content=answer),
                SystemMessage(content="[Generate]: Generated direct answer"),
            ],
        }

    def _get_history(self, state: RAGState) -> str:
        """Extract and format conversation history."""
        history = [
            msg
            for msg in state["messages"]
            if isinstance(msg, (HumanMessage, AIMessage))
        ]

        if len(history) > self.history_window_size:
            history = history[-self.history_window_size :]

        return "\n".join(
            [
                f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                for msg in history
            ]
        )

    def _init_vectorstores(self, repo_dir: str):
        """Initialize vectorstores for the given repository.

        support the following directory structures:
        - `.wikis/{owner_repo}/...`
        - `.wikis/sub/{owner_repo}/...`
        - `.wikis/moe/{owner_repo}/...`
        - and other subdirectories in `.wikis` that contain `{repo_dir}`.
        """
        vectorstores = {}

        # collect all possible directories that contain the repository documents
        candidate_paths = []

        # 1. directly in `.wikis/{repo_dir}`
        direct_path = os.path.join(self.wikis_dir, repo_dir)
        if os.path.isdir(direct_path):
            candidate_paths.append(direct_path)

        # 2. `.wikis/*/{repo_dir}` structure (e.g., `.wikis/sub/{repo_dir}`, `.wikis/moe/{repo_dir}`)
        if os.path.isdir(self.wikis_dir):
            for name in os.listdir(self.wikis_dir):
                sub_root = os.path.join(self.wikis_dir, name)
                if not os.path.isdir(sub_root):
                    continue
                nested_repo_path = os.path.join(sub_root, repo_dir)
                if (
                    os.path.isdir(nested_repo_path)
                    and nested_repo_path not in candidate_paths
                ):
                    candidate_paths.append(nested_repo_path)

        # if no directory is found, return empty
        if not candidate_paths:
            return vectorstores

        # load all documents from all directories
        repo_docs = []
        for path in candidate_paths:
            loader = DirectoryLoader(
                path, glob="**/*", loader_cls=TextLoader, show_progress=False
            )
            docs = loader.load()
            for doc in docs:
                # Preserve original loader source (usually the absolute file path)
                original_source = doc.metadata.get("source")
                if original_source:
                    # Store original absolute path for debugging / advanced use-cases
                    doc.metadata.setdefault("absolute_source", original_source)

                    # Normalize source path to be relative to the wikis root so callers
                    # can directly use it as a display path or build URLs.
                    try:
                        rel_source = os.path.relpath(original_source, self.wikis_dir)
                    except ValueError:
                        # Fallback to the original value if relpath fails on some OS edge cases
                        rel_source = original_source
                    doc.metadata["source"] = rel_source
                else:
                    # If loader didn't set a source, at least record the directory
                    doc.metadata["source"] = os.path.relpath(path, self.wikis_dir)

                # Also mark which wiki sub-directory this document came from
                doc.metadata["wiki_dir"] = os.path.relpath(path, self.wikis_dir)
            repo_docs.extend(docs)

        # if no documents are available, return empty
        if not repo_docs:
            return vectorstores

        # merge all documents into the same collection (by repo_dir)
        vectorstore = Chroma(
            collection_name=repo_dir,
            embedding_function=HuggingFaceEmbeddings(
                model_name=self.embeddings_model_name
            ),
            persist_directory=f"./.chroma_dbs/{repo_dir}",
        )

        if not vectorstore.get()["ids"]:
            print(
                f"Indexing documents for: {repo_dir} "
                f"from {len(candidate_paths)} wiki directories"
            )
            vectorstore.add_documents(repo_docs)
        else:
            print(f"Using existing vectorstore for: {repo_dir}")

        vectorstores[repo_dir] = vectorstore
        return vectorstores

    def _stream_and_print(self, stream):
        """Process stream and print messages."""
        final_state = None
        for s in stream:
            message = s["messages"][-1]
            if hasattr(message, "pretty_print"):
                message.pretty_print()
            else:
                print(message)
            final_state = s
        # print the final answer
        if final_state is not None:
            answer = final_state.get("answer")
            if answer:
                print("\n=== Answer ===")
                print(answer)
            # print referenced document sources, if any
            docs = final_state.get("documents") or []
            sources = self._extract_sources_from_docs(docs)
            if sources:
                print("\n=== Sources ===")
                for idx, src in enumerate(sources, start=1):
                    print(f"[{idx}] {src}")
        return final_state

    def _log_state(self, state: RAGState, file_name: str):
        """Log the current state to a file."""
        log_dir = "./.logs/rag_agent"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{file_name}.log")
        with open(log_file, "a") as f:
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"State: {state}\n")
            f.write("-" * 40 + "\n")

    def run(self) -> str:
        """Interactive CLI for querying repositories."""
        repo_input = input('Repository name (e.g., "owner/repo"): ')
        repo_dir = repo_input.replace("/", "_")
        self.vectorstores = self._init_vectorstores(repo_dir)

        thread_id = f"{repo_dir}-chat"
        app_state = {
            "messages": [],
            "query": "",
            "question": "",
            "documents": [],
            "answer": "",
            "is_sufficient": False,
            "retrieval_count": 0,
            "is_repo_related": False,
            "repo_name": repo_input,
        }
        file_name = f"{thread_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._log_state(app_state, file_name)

        while True:
            user_input = input("\nQuery (or 'exit'): ")
            if user_input.lower() == "exit":
                break

            # Reset query state
            app_state["messages"] = app_state.get("messages", []) + [
                HumanMessage(content=user_input)
            ]
            app_state["question"] = user_input
            app_state["is_sufficient"] = False
            app_state["retrieval_count"] = 0
            app_state = self._stream_and_print(
                self.app.stream(
                    app_state,
                    stream_mode="values",
                    config={
                        "configurable": {"thread_id": thread_id},
                        "recursion_limit": 100,
                    },
                )
            )
            self._log_state(app_state, file_name)
        return "Session ended."


if __name__ == "__main__":
    mode = "fast"

    rag_agent = RAGAgent(mode=mode)
    rag_agent.run()
