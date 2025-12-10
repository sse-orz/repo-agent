import os
from collections import defaultdict
from datetime import datetime
from typing import TypedDict, List, Annotated, Sequence, Tuple, Dict
from textwrap import dedent

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END
from sentence_transformers import CrossEncoder

from config import CONFIG
from app.models.rag import RAGAnswerData, RAGStreamStepData


# Type alias for accumulated document: (Document, similarity_score, retrieval_round)
AccumulatedDoc = Tuple[Document, float, int]


class RAGState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    question: str
    original_question: str 
    documents: List[Document]
    accumulated_docs: List[AccumulatedDoc]  # Accumulated docs across retrieval rounds
    answer: str
    is_sufficient: bool
    retrieval_count: int
    is_repo_related: bool
    repo_name: str
    repo_dir: str  # Key for vectorstore lookup


class RAGAgent:
    """RAG Agent for repository-related question answering with intelligent retrieval.

    Two modes:
    - fast: default mode, less retrieval attempts, faster response
    - smart: more retrieval attempts, larger history window, better effect
    
    Features:
    - Accumulative retrieval: Documents from all retrieval rounds are accumulated
    - RRF fusion: Reciprocal Rank Fusion to merge multi-round results
    - Cross-Encoder reranking: Fine-grained reranking using cross-encoder model
    """

    def __init__(
        self,
        wikis_dir: str = ".wikis",
        embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_retrieval_attempts: int = 2,
        retrieval_k: int = 5,
        history_window_size: int = 10,
        mode: str = "fast",
        # RRF and reranking parameters
        rrf_k: int = 60,
        rerank_top_n: int = 20,
        final_top_k: int = 10,
        use_cross_encoder: bool = True,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
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
            rerank_top_n = 30
        else:  # fast mode
            max_retrieval_attempts = 3
            retrieval_k = 5
            history_window_size = 10
            rerank_top_n = 20
        
        # final_top_k always equals retrieval_k
        final_top_k = retrieval_k

        self.wikis_dir = wikis_dir
        self.embeddings_model_name = embeddings_model_name
        self.max_retrieval_attempts = max_retrieval_attempts
        self.retrieval_k = retrieval_k
        self.history_window_size = history_window_size
        self.mode = mode
        self.memory = InMemorySaver()
        self.vectorstores = {}
        
        # RRF and reranking settings
        self.rrf_k = rrf_k
        self.rerank_top_n = rerank_top_n
        self.final_top_k = final_top_k
        self.use_cross_encoder = use_cross_encoder
        self.cross_encoder_model = cross_encoder_model
        self._cross_encoder = None  # Lazy-loaded
        
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
                - This repository in general (introduction, overview, what it does, etc.)
                - This repository's codebase, architecture, or design
                - Specific files, modules, classes, or functions in this repo
                - How to use, configure, or extend this repo
                - Understanding or learning about this repository

                Answer "no" only if the question is:
                - About general programming concepts unrelated to this repo
                - About other projects or repositories
                - Pure chit-chat with no connection to this repository

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
        graph.add_node("check_intent", self._check_intent_node)
        graph.add_node("rewrite", self._rewrite_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("judge", self._judge_node)
        graph.add_node("generate_rag", self._generate_rag_node)
        graph.add_node("generate_direct", self._generate_direct_node)

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
        """Invoke LLM with prompt and return response."""
        llm = CONFIG.get_llm()
        return llm.invoke([prompt]).content.strip()

    def _check_intent_node(self, state: RAGState) -> RAGState:
        """Determine if question is repository-related."""
        question = state["question"]
        repo_name = state["repo_name"]

        prompt = self._create_intent_check_prompt(question, repo_name)
        judgment = self._invoke_llm(prompt)
        is_repo_related = "yes" in judgment.lower()

        return {
            "is_repo_related": is_repo_related,
            "original_question": question,  # Preserve original question
            "messages": state["messages"]
            + [
                SystemMessage(
                    content=f"[Intent]: {'Repository' if is_repo_related else 'General'} question"
                )
            ],
        }

    def _rewrite_node(self, state: RAGState) -> RAGState:
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

    def _judge_node(self, state: RAGState) -> RAGState:
        """Judge if accumulated documents are sufficient.
        
        Uses accumulated_docs (all documents from all retrieval rounds) for judgment.
        If documents are insufficient, retry with larger k .
        """
        original_question = state.get("original_question", state["question"])
        accumulated_docs: List[AccumulatedDoc] = state.get("accumulated_docs") or []
        retrieval_count = state.get("retrieval_count", 0)

        if retrieval_count >= self.max_retrieval_attempts:
            return {
                "is_sufficient": True,
                "retrieval_count": retrieval_count + 1,
                "messages": state["messages"]
                + [
                    SystemMessage(
                        content=f"[Judge]: Max attempts ({retrieval_count + 1}) reached, "
                        f"using {len(accumulated_docs)} accumulated docs"
                    )
                ],
            }

        if not accumulated_docs:
            # No documents found, retry with larger k
            return {
                "is_sufficient": False,
                "retrieval_count": retrieval_count + 1,
                "messages": state["messages"]
                + [
                    SystemMessage(
                        content=f"[Judge]: No documents found, retrying with larger k"
                    )
                ],
            }

        # Build context from ALL accumulated documents for judgment
        # Sort by score to prioritize high-quality docs in context
        sorted_docs = sorted(accumulated_docs, key=lambda x: x[1], reverse=True)
        context = "\n".join([doc.page_content for doc, _, _ in sorted_docs])
        
        prompt = self._create_judge_prompt(original_question, context)
        judgment = self._invoke_llm(prompt)
        is_sufficient = any(
            word in judgment.lower() for word in ["yes", "sufficient", "enough"]
        )

        if not is_sufficient:
            # Documents insufficient, retry with larger k
            return {
                "is_sufficient": False,
                "retrieval_count": retrieval_count + 1,
                "messages": state["messages"]
                + [
                    SystemMessage(
                        content=f"[Judge]: Insufficient (attempt {retrieval_count + 1}), "
                        f"accumulated {len(accumulated_docs)} docs, retrying with larger k"
                    )
                ],
            }

        return {
            "is_sufficient": True,
            "retrieval_count": retrieval_count + 1,
            "messages": state["messages"]
            + [
                SystemMessage(
                    content=f"[Judge]: Sufficient (attempt {retrieval_count + 1}), "
                    f"accumulated {len(accumulated_docs)} docs"
                )
            ],
        }

    def _retrieve_node(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents from vectorstore and accumulate results.
        
        Uses similarity_search_with_score to get similarity scores.
        Accumulates documents across retrieval rounds instead of overwriting.
        Deduplicates by source, keeping the highest-scored version.
        
        On retry, k is scaled up: k * (retrieval_count + 1)
        """
        question = state["question"]
        repo_dir = state["repo_dir"]
        retrieval_count = state.get("retrieval_count", 0)
        
        # Scale k based on retrieval round: k, 2k, 3k, ...
        current_k = self.retrieval_k * (retrieval_count + 1)
        
        # Get existing accumulated docs
        accumulated_docs: List[AccumulatedDoc] = list(
            state.get("accumulated_docs") or []
        )
        
        # Retrieve with scores
        if repo_dir and repo_dir in self.vectorstores:
            docs_with_scores = self.vectorstores[repo_dir].similarity_search_with_score(
                question, k=current_k
            )
        else:
            docs_with_scores = self._fallback_search_with_score(question, k=current_k)
        
        # Build a map of source -> best (doc, score, round) for deduplication
        source_to_best: Dict[str, AccumulatedDoc] = {}
        
        # First, add existing accumulated docs to the map
        for doc, score, round_num in accumulated_docs:
            source = self._get_doc_source(doc)
            if source not in source_to_best or score > source_to_best[source][1]:
                source_to_best[source] = (doc, score, round_num)
        
        # Then, add new docs (current round)
        # Note: Chroma returns (doc, distance), lower distance = more similar
        # We convert distance to similarity score: score = 1 / (1 + distance)
        for doc, distance in docs_with_scores:
            similarity_score = 1.0 / (1.0 + distance)
            source = self._get_doc_source(doc)
            if source not in source_to_best or similarity_score > source_to_best[source][1]:
                source_to_best[source] = (doc, similarity_score, retrieval_count)
        
        # Convert back to list
        new_accumulated_docs = list(source_to_best.values())
        
        # Also update documents for backward compatibility (will be replaced in generate)
        current_docs = [doc for doc, _, _ in new_accumulated_docs]
        
        new_docs_count = len(docs_with_scores)
        total_unique = len(new_accumulated_docs)
        
        return {
            "documents": current_docs,
            "accumulated_docs": new_accumulated_docs,
            "messages": state["messages"]
            + [
                SystemMessage(
                    content=f"[Retrieve]: Found {new_docs_count} docs (round {retrieval_count}, k={current_k}), "
                    f"total unique: {total_unique}"
                )
            ],
        }

    @staticmethod
    def _get_doc_source(doc: Document) -> str:
        """Extract source identifier from document for deduplication."""
        metadata = getattr(doc, "metadata", {}) or {}
        return (
            metadata.get("source")
            or metadata.get("file_path")
            or metadata.get("path")
            or metadata.get("absolute_source")
            or str(id(doc))  # Fallback to object id
        )

    def _fallback_search_with_score(
        self, question: str, k: int = 3
    ) -> List[Tuple[Document, float]]:
        """Search across all vectorstores when specific repo not found."""
        docs_with_scores = []
        # Distribute k across vectorstores
        per_store_k = max(1, k // max(1, len(self.vectorstores)))
        for vs in self.vectorstores.values():
            docs_with_scores.extend(vs.similarity_search_with_score(question, k=per_store_k))
        return docs_with_scores

    def _rrf_fusion(
        self, accumulated_docs: List[AccumulatedDoc]
    ) -> List[Tuple[Document, float]]:
        """Apply Reciprocal Rank Fusion to merge multi-round retrieval results.
        
        RRF formula: score(d) = Σ 1/(k + rank_i)
        where k is a constant (default 60) and rank_i is the rank in each round.
        
        Args:
            accumulated_docs: List of (Document, similarity_score, retrieval_round)
            
        Returns:
            List of (Document, rrf_score) sorted by RRF score descending
        """
        if not accumulated_docs:
            return []
        
        # Group documents by retrieval round
        rounds: Dict[int, List[Tuple[Document, float]]] = defaultdict(list)
        for doc, score, round_num in accumulated_docs:
            rounds[round_num].append((doc, score))
        
        # Sort each round by similarity score (descending) to get ranks
        for round_num in rounds:
            rounds[round_num].sort(key=lambda x: x[1], reverse=True)
        
        # Calculate RRF score for each document
        # Use source as key to handle same doc appearing in multiple rounds
        source_to_doc: Dict[str, Document] = {}
        rrf_scores: Dict[str, float] = defaultdict(float)
        
        for round_num, docs_in_round in rounds.items():
            for rank, (doc, _) in enumerate(docs_in_round, start=1):
                source = self._get_doc_source(doc)
                source_to_doc[source] = doc
                # RRF formula: 1 / (k + rank)
                rrf_scores[source] += 1.0 / (self.rrf_k + rank)
        
        # Build result list sorted by RRF score
        result = [
            (source_to_doc[source], rrf_score)
            for source, rrf_score in rrf_scores.items()
        ]
        result.sort(key=lambda x: x[1], reverse=True)
        
        return result

    def _get_cross_encoder(self) -> CrossEncoder:
        """Lazy-load the CrossEncoder model."""
        if self._cross_encoder is None:
            print(f"Loading CrossEncoder model: {self.cross_encoder_model}")
            self._cross_encoder = CrossEncoder(self.cross_encoder_model)
        return self._cross_encoder

    def _rerank_with_cross_encoder(
        self, question: str, docs_with_scores: List[Tuple[Document, float]]
    ) -> List[Tuple[Document, float]]:
        """Rerank documents using Cross-Encoder for fine-grained relevance scoring.
        
        Args:
            question: The user's question
            docs_with_scores: List of (Document, rrf_score) from RRF fusion
            
        Returns:
            List of (Document, cross_encoder_score) sorted by score descending
        """
        if not docs_with_scores:
            return []
        
        if not self.use_cross_encoder:
            # Skip cross-encoder, just return top-N by RRF score
            return docs_with_scores[: self.rerank_top_n]
        
        # Take top-N candidates for reranking (cross-encoder is expensive)
        candidates = docs_with_scores[: self.rerank_top_n]
        
        # Prepare query-document pairs for cross-encoder
        pairs = [(question, doc.page_content) for doc, _ in candidates]
        
        # Get cross-encoder scores
        cross_encoder = self._get_cross_encoder()
        scores = cross_encoder.predict(pairs)
        
        # Build result with cross-encoder scores
        result = [
            (doc, float(score))
            for (doc, _), score in zip(candidates, scores)
        ]
        
        # Sort by cross-encoder score (descending)
        result.sort(key=lambda x: x[1], reverse=True)
        
        return result

    def _select_final_documents(
        self, question: str, accumulated_docs: List[AccumulatedDoc]
    ) -> List[Document]:
        """Select final documents for answer generation.
        
        Pipeline:
        1. RRF fusion to merge multi-round results
        2. Cross-Encoder reranking for fine-grained relevance
        3. Truncate to final_top_k
        
        Args:
            question: The user's original question
            accumulated_docs: All accumulated documents from retrieval rounds
            
        Returns:
            List of top-K documents for answer generation
        """
        if not accumulated_docs:
            return []
        
        # Step 1: RRF fusion
        rrf_results = self._rrf_fusion(accumulated_docs)
        
        # Step 2: Cross-Encoder reranking
        reranked_results = self._rerank_with_cross_encoder(question, rrf_results)
        
        # Step 3: Truncate to final_top_k
        final_docs = [doc for doc, _ in reranked_results[: self.final_top_k]]
        
        return final_docs

    def _generate_rag_node(self, state: RAGState) -> RAGState:
        """Generate answer using RAG context with RRF fusion and Cross-Encoder reranking.
        
        Pipeline:
        1. Get accumulated documents from all retrieval rounds
        2. Apply RRF fusion to merge results
        3. Rerank with Cross-Encoder for fine-grained relevance
        4. Select top-K documents for answer generation
        """
        original_question = state.get("original_question", state["question"])
        accumulated_docs: List[AccumulatedDoc] = state.get("accumulated_docs") or []
        history = self._get_history(state)

        # Select final documents using RRF + Cross-Encoder pipeline
        final_docs = self._select_final_documents(original_question, accumulated_docs)
        
        # Build context from final selected documents
        context = "\n".join([doc.page_content for doc in final_docs])

        system_prompt = self._get_answer_system_prompt()
        user_prompt = self._create_rag_generation_prompt(
            history, context, original_question
        )
        llm = CONFIG.get_llm()
        answer = llm.invoke([system_prompt, user_prompt]).content

        return {
            "answer": answer,
            "documents": final_docs,  # Update documents with final selection
            "messages": state["messages"]
            + [
                AIMessage(content=answer),
                SystemMessage(
                    content=f"[Generate]: Generated RAG answer using {len(final_docs)} "
                    f"docs (from {len(accumulated_docs)} accumulated)"
                ),
            ],
        }

    def _generate_direct_node(self, state: RAGState) -> RAGState:
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

    def _load_vectorstores(self, repo_dir: str):
        """Load vectorstores for the given repository (internal implementation).

        Support the following directory structures:
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

    def _build_initial_state(self, question: str, repo_name: str, repo_dir: str) -> RAGState:
        """Build initial RAGState for a query."""
        return {
            "messages": [],
            "question": question,
            "original_question": question,
            "documents": [],
            "accumulated_docs": [],
            "answer": "",
            "is_sufficient": False,
            "retrieval_count": 0,
            "is_repo_related": False,
            "repo_name": repo_name,
            "repo_dir": repo_dir,
        }

    # =========================================================================
    # Public API methods
    # =========================================================================

    def init_repo(self, repo_dir: str) -> None:
        """Initialize vectorstore for a repository.
        
        Args:
            repo_dir: Repository directory name (e.g., "owner_repo")
        """
        if repo_dir in self.vectorstores:
            # Already initialized, skip
            return
        self.vectorstores = self._load_vectorstores(repo_dir)

    def ask(self, question: str, repo_name: str, repo_dir: str) -> RAGAnswerData:
        """Synchronously answer a question about a repository.
        
        Args:
            question: User's question
            repo_name: Repository name for display (e.g., "github:owner/repo")
            repo_dir: Repository directory key for vectorstore lookup (e.g., "owner_repo")
            
        Returns:
            RAGAnswerData with answer and sources
        """
        state = self._build_initial_state(question, repo_name, repo_dir)
        thread_id = f"{repo_dir}-api"

        final_state = self.app.invoke(
            state,
            config={
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 100,
            },
        )

        answer = final_state.get("answer", "") or ""
        sources = self._extract_sources_from_docs(final_state.get("documents") or [])
        return RAGAnswerData(answer=answer, sources=sources)

    def stream(self, question: str, repo_name: str, repo_dir: str):
        """Stream the RAG answering process step by step with true token streaming.
        
        For the Generate phase, uses LLM streaming to yield tokens incrementally,
        providing a typewriter effect for the answer.
        
        Args:
            question: User's question
            repo_name: Repository name for display (e.g., "github:owner/repo")
            repo_dir: Repository directory key for vectorstore lookup (e.g., "owner_repo")
            
        Yields:
            RAGStreamStepData for each step in the process.
            During Generate phase, yields incremental tokens via the `delta` field.
        """
        state = self._build_initial_state(question, repo_name, repo_dir)
        thread_id = f"{repo_dir}-api"

        # Track state for streaming generation
        last_state = None
        
        for s in self.app.stream(
            state,
            stream_mode="values",
            config={
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 100,
            },
        ):
            node = self._extract_node_name_from_state(s)
            
            # For Generate node, we'll handle streaming separately
            if node == "Generate":
                last_state = s
                # Don't yield the pre-computed answer, we'll stream it below
                continue
            
            # For non-Generate nodes, yield state without sources
            # (sources are only finalized after RRF+reranking in Generate phase)
            answer = s.get("answer", "") or ""
            yield RAGStreamStepData(answer=answer, delta=None, node=node, sources=[])
            last_state = s
        
        # Now handle the Generate phase with true token streaming
        if last_state is not None:
            yield from self._stream_generation(last_state)

    def _stream_generation(self, state: RAGState):
        """Stream the answer generation token by token.
        
        Args:
            state: The final state from the graph containing all needed context
            
        Yields:
            RAGStreamStepData with delta field for each token
        """
        is_repo_related = state.get("is_repo_related", False)
        original_question = state.get("original_question", state.get("question", ""))
        accumulated_docs: List[AccumulatedDoc] = state.get("accumulated_docs") or []
        history = self._get_history(state)
        
        llm = CONFIG.get_llm()
        system_prompt = self._get_answer_system_prompt()
        
        if is_repo_related:
            # RAG-based generation with context
            final_docs = self._select_final_documents(original_question, accumulated_docs)
            context = "\n".join([doc.page_content for doc in final_docs])
            user_prompt = self._create_rag_generation_prompt(
                history, context, original_question
            )
            sources = self._extract_sources_from_docs(final_docs)
        else:
            # Direct generation without RAG context
            user_prompt = self._create_direct_generation_prompt(history, original_question)
            sources = []
            final_docs = []
        
        # Stream the LLM response token by token
        accumulated_answer = ""
        for chunk in llm.stream([system_prompt, user_prompt]):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                accumulated_answer += token
                yield RAGStreamStepData(
                    answer=accumulated_answer,
                    delta=token,
                    node="Generate",
                    sources=sources
                )
        
        # Final yield with complete answer (no delta, indicates completion)
        if accumulated_answer:
            yield RAGStreamStepData(
                answer=accumulated_answer,
                delta=None,
                node="Generate",
                sources=sources
            )

    @staticmethod
    def _extract_node_name_from_state(state: dict) -> str | None:
        """Extract current node name from the latest system message like '[Judge]: ...'."""
        messages = state.get("messages") or []
        for msg in reversed(messages):
            content = getattr(msg, "content", None)
            if not isinstance(content, str):
                continue
            if content.startswith("[") and "]" in content:
                return content[1 : content.index("]")]
        return None

    def run(self) -> str:
        """Interactive CLI for querying repositories."""
        repo_input = input('Repository name (e.g., "owner/repo"): ')
        repo_dir = repo_input.replace("/", "_")
        self.init_repo(repo_dir)

        thread_id = f"{repo_dir}-chat"
        app_state = {
            "messages": [],
            "question": "",
            "original_question": "",
            "documents": [],
            "accumulated_docs": [],
            "answer": "",
            "is_sufficient": False,
            "retrieval_count": 0,
            "is_repo_related": False,
            "repo_name": repo_input,
            "repo_dir": repo_dir,
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
            app_state["original_question"] = user_input
            app_state["is_sufficient"] = False
            app_state["retrieval_count"] = 0
            app_state["accumulated_docs"] = []  # Reset accumulated docs for new query
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
