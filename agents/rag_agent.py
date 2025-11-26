import os
from datetime import datetime
from typing import TypedDict, List, Annotated, Sequence

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
    """RAG Agent for repository-related question answering with intelligent retrieval."""

    def __init__(
        self,
        wikis_dir: str = ".wikis",
        embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_retrieval_attempts: int = 2,
        retrieval_k: int = 5,
        history_window_size: int = 10,
    ):
        self.wikis_dir = wikis_dir
        self.embeddings_model_name = embeddings_model_name
        self.max_retrieval_attempts = max_retrieval_attempts
        self.retrieval_k = retrieval_k
        self.history_window_size = history_window_size
        self.memory = InMemorySaver()
        self.vectorstores = {}
        self.app = self._build_graph()

    @staticmethod
    def _create_judge_prompt(question: str, context: str) -> SystemMessage:
        """Create prompt to judge document sufficiency."""
        return SystemMessage(
            content=f"""Evaluate if the provided documents contain sufficient information to comprehensively answer the user's question.

                        User Question: {question}

                        Retrieved Documents:
                        {context}

                        Evaluation Criteria:
                        - The documents should contain direct or highly relevant information that addresses the core of the question
                        - Code snippets, API documentation, or implementation details are considered sufficient
                        - General descriptions without specific technical details may be insufficient
                        - If documents are mostly empty or generic, they are not sufficient

                        Your Response: Answer with ONLY "yes" if documents are sufficient to answer the question, or "no" if more information is needed."""
            )

    @staticmethod
    def _create_intent_check_prompt(question: str, repo_name: str) -> SystemMessage:
        """Create prompt to check if question is repository-related."""
        return SystemMessage(
            content=f"""Analyze whether the following question is specifically related to the repository "{repo_name}".

                        User Question: {question}

                        Repository Name: {repo_name}

                        Classification Guidelines:

                        REPOSITORY-RELATED Questions (Answer "yes"):
                        - Asks about the repository's codebase, architecture, or design patterns
                        - Requests information about specific files, modules, classes, or functions
                        - Seeks to understand algorithms, implementations, or technical workflows
                        - Asks about the repository's purpose, features, capabilities, or limitations
                        - Requests examples of how to use the code or API
                        - Questions the repository's dependencies or configuration

                        NOT Repository-Related Questions (Answer "no"):
                        - General programming questions unrelated to this specific repository
                        - Asks about the assistant itself or conversation meta-topics
                        - Requests information about other projects or general knowledge
                        - Suggests improvements or asks for critique of previous answers

                        Your Response: Answer with ONLY "yes" if the question is repository-related, or "no" otherwise."""
                    )

    @staticmethod
    def _create_rewrite_prompt(original_question: str, repo_name: str) -> SystemMessage:
        """Create prompt to rewrite question for better retrieval."""
        return SystemMessage(
            content=f"""Your task is to refine and improve the user's question about the repository "{repo_name}" to make it more specific, clear, and optimized for document retrieval.

                        Original Question: {original_question}

                        Refinement Guidelines:
                        1. Make the question more specific and detailed if it's too vague
                        2. Add technical context if the question lacks clarity
                        3. Break down complex questions into focused sub-questions if needed
                        4. Rephrase to use repository-specific terminology when applicable
                        5. Ensure the refined question targets concrete code artifacts (functions, classes, files, etc.)
                        6. Keep the refined question concise but comprehensive

                        Output ONLY the refined question without any explanation or additional text. If the original question is already well-formed, you can return it as-is."""
                    )

    @staticmethod
    def _create_rag_generation_prompt(
        history: str, context: str, question: str
    ) -> SystemMessage:
        """Create prompt for RAG-based answer generation."""
        return SystemMessage(
            content=f"""You are a knowledgeable Repository Analysis Assistant specializing in helping users understand and work with codebases.

                        Your Task: Answer the user's question based EXCLUSIVELY on the provided repository documentation and code context.

                        Conversation History:
                        {'---' if history else '(No prior conversation)'}
                        {history if history else ''}
                        {'---' if history else ''}

                        Repository Context (Code, Documentation, and Related Files):
                        ---BEGIN CONTEXT---
                        {context}
                        ---END CONTEXT---

                        User Question: {question}

                        Instructions:
                        1. Ground your answer entirely in the provided context - do not use external knowledge
                        2. If the context doesn't contain relevant information, explicitly state: "The provided documentation does not contain information about this topic"
                        3. Be specific and technical - reference actual code, file names, and line numbers when applicable
                        4. If there are multiple relevant parts in the context, cite them appropriately
                        5. Provide practical examples or code snippets from the documentation when helpful
                        6. If the question requires clarification, ask for it before attempting to answer

                        Provide a clear, comprehensive, and well-structured answer:"""
                    )

    @staticmethod
    def _create_direct_generation_prompt(history: str, question: str) -> SystemMessage:
        """Create prompt for direct answer generation to decline unrelated questions."""
        return SystemMessage(
            content=f"""You are a specialized Repository Analysis Assistant. Your sole purpose is to assist users in understanding, analyzing, and working with the specific code repository loaded into the system.

                        Conversation History:
                        {'---' if history else '(No prior conversation)'}
                        {history if history else ''}
                        {'---' if history else ''}

                        User Question: {question}

                        Instructions:
                        1. You have identified that the user's question is NOT related to the repository, codebase, or technical context.
                        2. Explicitly state that you are a Repository Analysis Agent.
                        3. Politely decline to answer the question because it falls outside the scope of the repository.
                        4. Do NOT use your general knowledge to answer the question (e.g., do not answer general world knowledge, math, or history questions).
                        5. Invite the user to ask questions specifically regarding the codebase, architecture, or implementation details of the current repository.

                        Your Response:"""
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

    def _invoke_llm(self, prompt: SystemMessage) -> str:
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

        prompt = self._create_rag_generation_prompt(history, context, question)
        llm = CONFIG.get_llm()
        answer = llm.invoke([prompt]).content

        return {
            "answer": answer,
            "messages": state["messages"] + [AIMessage(content=answer)],
        }

    def generate_direct_node(self, state: RAGState) -> RAGState:
        """Generate answer directly without RAG."""
        question = state["question"]
        history = self._get_history(state)

        prompt = self._create_direct_generation_prompt(history, question)
        llm = CONFIG.get_llm()
        answer = llm.invoke([prompt]).content

        return {
            "answer": answer,
            "messages": state["messages"] + [AIMessage(content=answer)],
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
        """Initialize vectorstores for the given repository."""
        vectorstores = {}
        repo_path = os.path.join(self.wikis_dir, repo_dir)

        if not os.path.exists(repo_path) or not os.path.isdir(repo_path):
            return vectorstores

        loader = DirectoryLoader(
            repo_path, glob="**/*", loader_cls=TextLoader, show_progress=False
        )
        repo_docs = loader.load()

        for doc in repo_docs:
            doc.metadata["source"] = repo_dir

        vectorstore = Chroma(
            collection_name=repo_dir,
            embedding_function=HuggingFaceEmbeddings(
                model_name=self.embeddings_model_name
            ),
            persist_directory=f"./.chroma_dbs/{repo_dir}",
        )

        if not vectorstore.get()["ids"]:
            print(f"Indexing documents for: {repo_dir}")
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
    rag_agent = RAGAgent()
    rag_agent.run()
