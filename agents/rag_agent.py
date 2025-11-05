from config import CONFIG

import os
import io
from datetime import datetime
from contextlib import redirect_stdout
from typing import TypedDict, List, Annotated, Sequence
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.runnables.graph import MermaidDrawMethod
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END


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
    def __init__(
        self,
        wikis_dir: str = ".wikis",
        embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.wikis_dir = wikis_dir
        self.embeddings_model_name = embeddings_model_name
        self.memory = InMemorySaver()
        self.vectorstores = {}
        self.app = self._build_app()

    def _build_app(self):
        # Build the graph
        graph = StateGraph(RAGState)
        graph.add_node("check_intent", self._check_intent)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("llm_judge", self._llm_judge)
        graph.add_node("generate_with_rag", self._generate)
        graph.add_node("generate_direct", self._generate_direct)

        graph.add_edge(START, "check_intent")
        graph.add_conditional_edges(
            "check_intent",
            self._decide_rag_or_direct,
            {
                "rag": "retrieve",
                "direct": "generate_direct",
            },
        )
        graph.add_edge("retrieve", "llm_judge")
        graph.add_conditional_edges(
            "llm_judge",
            self._decide_retrieve_or_generate,
            {
                "retrieve": "retrieve",
                "generate": "generate_with_rag",
            },
        )
        graph.add_edge("generate_with_rag", END)
        graph.add_edge("generate_direct", END)

        # Compile the graph
        return graph.compile(checkpointer=self.memory)

    def _llm_judge(self, state: RAGState) -> RAGState:
        """Use LLM to judge if retrieved documents are sufficient for answering the question."""
        question = state["question"]
        docs = state["documents"]
        retrieval_count = state.get("retrieval_count", 0)

        if not docs:
            # No documents retrieved, not sufficient
            return {
                "is_sufficient": False,
                "retrieval_count": retrieval_count + 1,
                "messages": state["messages"]
                + [
                    SystemMessage(
                        content="[llm_judge]: No documents retrieved. Need to retrieve more."
                    )
                ],
            }

        context = "\n".join([doc.page_content for doc in docs])

        # Create a prompt to judge sufficiency
        judge_prompt = SystemMessage(
            content=f"""Based on the following documents and question, determine if the documents are sufficient to answer the question.

Question: {question}

Documents:
{context}

Please answer with just "yes" or "no":"""
        )

        llm = CONFIG.get_llm()
        judgment = llm.invoke([judge_prompt]).content.lower().strip()

        # If judge says "yes" or contains "sufficient/enough", consider it sufficient
        is_sufficient = (
            "yes" in judgment or "sufficient" in judgment or "enough" in judgment
        )

        # Limit retrieval attempts to avoid infinite loops (e.g., max 3 attempts)
        if retrieval_count >= 3:
            is_sufficient = True

        judge_message = f"[llm_judge]: Document sufficiency check: {judgment}. Retrieval count: {retrieval_count + 1}"

        return {
            "is_sufficient": is_sufficient,
            "retrieval_count": retrieval_count + 1,
            "messages": state["messages"] + [SystemMessage(content=judge_message)],
        }

    def _decide_retrieve_or_generate(self, state: RAGState) -> str:
        """Decide whether to retrieve more documents or generate answer."""
        if state.get("is_sufficient", False):
            return "generate"
        else:
            return "retrieve"

    def _check_intent(self, state: RAGState) -> RAGState:
        """Check if the question is related to the repository."""
        question = state["question"]
        repo_name = state["repo_name"]

        # Create a prompt to judge if question is repo-related
        intent_prompt = SystemMessage(
            content=f"""Determine if the following question is related to the repository "{repo_name}".

Question: {question}

Consider the question is repo-related if:
1. It asks about code, structure, or content of the repository
2. It asks about algorithms, implementations, or functions in the repository
3. It asks about the repository's purpose, features, or documentation
4. It asks "what is", "how to", "explain" about the repository content

Consider the question is NOT repo-related if:
1. It asks about our previous conversation or interaction
2. It asks meta questions about the assistant itself
3. It asks for general knowledge unrelated to the repo
4. It asks to critique or self-reflect on previous responses

Please answer with just "yes" or "no":"""
        )

        llm = CONFIG.get_llm()
        judgment = llm.invoke([intent_prompt]).content.lower().strip()

        is_repo_related = "yes" in judgment
        intent_message = f"[check_intent]: {'Repository-related' if is_repo_related else 'General question'}. Question: {question}"

        return {
            "is_repo_related": is_repo_related,
            "messages": state["messages"] + [SystemMessage(content=intent_message)],
        }

    def _decide_rag_or_direct(self, state: RAGState) -> str:
        """Decide whether to use RAG or direct LLM response."""
        if state.get("is_repo_related", False):
            return "rag"
        else:
            return "direct"

    def _init_vectorstores(self, repo_dir: str):
        vectorstores = {}
        repo_path = os.path.join(self.wikis_dir, repo_dir)
        if os.path.exists(repo_path) and os.path.isdir(repo_path):
            loader = DirectoryLoader(repo_path, glob="**/*", show_progress=False)
            repo_docs = loader.load()
            for doc in repo_docs:
                doc.metadata["source"] = repo_dir

            # Create separate collection for each repo
            collection_name = repo_dir
            vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=HuggingFaceEmbeddings(
                    model_name=self.embeddings_model_name
                ),
                persist_directory=f"./.chroma_dbs/{collection_name}",  # Separate directory for each collection
            )
            # only add documents if the vectorstore is empty
            if not vectorstore.get()["ids"]:  # If no documents present
                print(f"Adding documents to vectorstore for repo: {repo_dir}")
                vectorstore.add_documents(repo_docs)
            else:
                print(f"Vectorstore for repo {repo_dir} already has documents.")
            vectorstores[repo_dir] = vectorstore
        return vectorstores

    # Define retrieval function
    def _retrieve(self, state: RAGState) -> RAGState:
        query = state["query"]

        def fallback_search(query: str) -> List[Document]:
            docs = []
            for vs in self.vectorstores.values():
                docs.extend(vs.similarity_search(query, k=1))
            return docs

        def process_query(query: str) -> tuple[str, List[Document]]:

            if ":" in query:
                repo, question = query.split(":", 1)
                repo = repo.strip()
                question = question.strip()
                if repo in self.vectorstores:
                    docs = self.vectorstores[repo].similarity_search(question, k=3)
                else:
                    docs = fallback_search(question)
            else:
                question = query
                docs = fallback_search(question)
            return question, docs

        # Parse repo from query, assume format "repo: question"
        question, docs = process_query(query)

        return {
            "query": query,
            "question": question,
            "documents": docs,
            "answer": "",
            "messages": state["messages"]
            + [SystemMessage(content="[retrieve]: Retrieved documents.")],
        }

    # Define generation function with RAG context
    def _generate_with_rag(self, state: RAGState) -> RAGState:
        query = state["query"]
        question = state["question"]
        docs = state["documents"]
        context = "\n".join([doc.page_content for doc in docs])
        history = self._get_history_messages(state)

        prompt = SystemMessage(
            content=f"You are a Repo Analysis Assistant.\nHistory of the conversation:\n===HISTORY===\n{history}\n===END HISTORY===\n\nBased on the following documents:\n===DOCUMENTS===\n{context}\n===END DOCUMENTS===\n\nQuestion: {question}\n\nPlease provide a detailed answer."
        )
        llm = CONFIG.get_llm()
        answer = llm.invoke([prompt]).content

        return {
            "query": query,
            "question": question,
            "documents": docs,
            "answer": answer,
            "messages": state["messages"] + [AIMessage(content=answer)],
        }

    def _generate_direct(self, state: RAGState) -> RAGState:
        """Generate answer directly without RAG context."""
        question = state["question"]
        history = self._get_history_messages(state)

        prompt = SystemMessage(
            content=f"You are a helpful assistant. History of the conversation:\n===HISTORY===\n{history}\n===END HISTORY===\n\nQuestion: {question}\n\nPlease provide a detailed answer."
        )

        llm = CONFIG.get_llm()
        answer = llm.invoke([prompt]).content

        return {
            "query": state["query"],
            "question": question,
            "documents": [],
            "answer": answer,
            "messages": state["messages"] + [AIMessage(content=answer)],
        }

    # For backward compatibility
    def _generate(self, state: RAGState) -> RAGState:
        return self._generate_with_rag(state)

    def _print_stream(self, file_name: str, stream):
        final_state = None
        for s in stream:
            message = s["messages"][-1]
            self._write_log(file_name, message)
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()
            final_state = s
        return final_state

    def _write_log(self, file_name: str, message):
        """Write a log message to a file.

        Args:
            message: The log message to write (can be a message object or string).
        """
        f = io.StringIO()
        with redirect_stdout(f):
            if hasattr(message, "pretty_print"):
                message.pretty_print()
            else:
                print(message)
        pretty_output = f.getvalue()
        dir_name = "./.logs"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        log_file_name = os.path.join(dir_name, f"{file_name}.log")
        with open(log_file_name, "a") as log_file:
            log_file.write(pretty_output + "\n\n")

    def _draw_graph(self):
        img = self.app.get_graph().draw_mermaid_png(
            draw_method=MermaidDrawMethod.API,
        )
        dir_name = "./.graphs"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        graph_file_name = os.path.join(
            dir_name,
            f"rag_agent_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.png",
        )
        with open(graph_file_name, "wb") as f:
            f.write(img)

    def _get_initial_state(self, repo_name: str = "") -> RAGState:
        return {
            "messages": [],
            "query": "",
            "question": "",
            "documents": [],
            "answer": "",
            "is_sufficient": False,
            "retrieval_count": 0,
            "is_repo_related": False,
            "repo_name": repo_name,
        }

    def _get_history_messages(self, state: RAGState):
        # get only user and ai answer messages from state
        # messages like check_intent, retrieve, llm_judge are not included
        history = []
        for msg in state["messages"]:
            if isinstance(msg, (HumanMessage, AIMessage)):
                history.append(msg)
        formatted_history = "\n".join(
            [
                (
                    f"User: {msg.content}"
                    if isinstance(msg, HumanMessage)
                    else f"Assistant: {msg.content}"
                )
                for msg in history
            ]
        )
        return formatted_history

    # Function to run the agent
    def run(self) -> str:
        repo_input = input(
            'Enter the repo name(such as "squatting-at-home123/back-puppet"): '
        )
        repo_dir = repo_input.replace("/", "_")
        self.vectorstores = self._init_vectorstores(repo_dir=repo_dir)
        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        thread_id = f"{repo_dir}-chat-thread"
        app_state = self._get_initial_state(repo_name=repo_input)
        while True:
            user_input = input("Enter your query (or 'exit' to quit): ")
            if user_input.lower() == "exit":
                break
            query = f"{repo_dir}: {user_input}"

            app_state["messages"] = app_state.get("messages", []) + [
                HumanMessage(content=user_input)
            ]
            app_state["query"] = query
            app_state["question"] = user_input
            app_state["is_sufficient"] = False
            app_state["retrieval_count"] = 0
            app_state["is_repo_related"] = False

            app_state = self._print_stream(
                file_name=time,
                stream=self.app.stream(
                    app_state,
                    stream_mode="values",
                    config={
                        "configurable": {"thread_id": thread_id},
                        "recursion_limit": 100,
                    },
                ),
            )


if __name__ == "__main__":
    # Example usage
    # Use "uv run python -m agents.rag_agent" to run this file
    rag_agent = RAGAgent()
    # rag_agent.run()
    rag_agent._draw_graph()
