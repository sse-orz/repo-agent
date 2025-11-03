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
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import StateGraph, START, END


class RAGState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query: str
    question: str
    documents: List[Document]
    answer: str


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

    def init_vectorstores(self, repo_dir: str):
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
            "messages": state["messages"] + [AIMessage(content="Retrieved documents.")],
        }

    # Define generation function
    def _generate(self, state: RAGState) -> RAGState:
        query = state["query"]
        question = state["question"]
        docs = state["documents"]
        context = "\n".join([doc.page_content for doc in docs])

        prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant. Use the following context to answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
        )

        llm = CONFIG.get_llm()
        chain = prompt | llm
        answer = chain.invoke({"context": context, "question": question}).content

        return {
            "query": query,
            "question": question,
            "documents": docs,
            "answer": answer,
            "messages": state["messages"] + [AIMessage(content=answer)],
        }

    def _print_stream(self, file_name: str, stream):
        for s in stream:
            message = s["messages"][-1]
            self._write_log(file_name, message)
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()

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

    def _build_app(self):
        # Build the graph
        graph = StateGraph(RAGState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("generate", self._generate)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)

        # Compile the graph
        return graph.compile(checkpointer=self.memory)

    # Function to run the agent
    def run(self) -> str:
        repo_input = input(
            'Enter the repo name(such as "squatting-at-home123/back-puppet"): '
        )
        repo_dir = repo_input.replace("/", "_")
        self.vectorstores = self.init_vectorstores(repo_dir=repo_dir)
        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        while True:
            user_input = input("Enter your query (or 'exit' to quit): ")
            if user_input.lower() == "exit":
                break
            query = f"{repo_dir}: {user_input}"
            app_state = {
                "messages": [HumanMessage(content=query)],
                "query": query,
                "question": "",
                "documents": [],
                "answer": "",
            }
            self._print_stream(
                file_name=time,
                stream=self.app.stream(
                    app_state,
                    stream_mode="values",
                    config={
                        "configurable": {"thread_id": f"{repo_dir}-chat-thread"},
                        "recursion_limit": 100,
                    },
                ),
            )


if __name__ == "__main__":
    # Example usage
    # Use "uv run python -m agents.rag_agent" to run this file
    rag_agent = RAGAgent()
    rag_agent.run()
