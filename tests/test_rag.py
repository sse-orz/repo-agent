import os
from typing import TypedDict, List
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langgraph.graph import StateGraph, START, END

from config import CONFIG


class RAGState(TypedDict):
    query: str
    question: str
    documents: List[Document]
    answer: str


def init_vectorstores(
    wikis_dir: str = ".wikis",
    embeddings_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    vectorstores = {}
    if os.path.exists(wikis_dir):
        for repo_dir in os.listdir(wikis_dir):
            repo_path = os.path.join(wikis_dir, repo_dir)
            if os.path.isdir(repo_path):
                loader = DirectoryLoader(repo_path, glob="**/*", show_progress=False)
                repo_docs = loader.load()
                for doc in repo_docs:
                    doc.metadata["source"] = repo_dir

                # Create separate collection for each repo
                collection_name = repo_dir
                vectorstore = Chroma(
                    collection_name=collection_name,
                    embedding_function=HuggingFaceEmbeddings(
                        model_name=embeddings_model_name
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
def retrieve(vectorstores, state: RAGState) -> RAGState:
    query = state["query"]

    def fallback_search(query: str) -> List[Document]:
        docs = []
        for vs in vectorstores.values():
            docs.extend(vs.similarity_search(query, k=1))
        return docs

    def process_query(query: str) -> tuple[str, List[Document]]:

        if ":" in query:
            repo, question = query.split(":", 1)
            repo = repo.strip()
            question = question.strip()
            if repo in vectorstores:
                docs = vectorstores[repo].similarity_search(question, k=3)
            else:
                docs = fallback_search(question)
        else:
            question = query
            docs = fallback_search(question)
        return question, docs

    # Parse repo from query, assume format "repo: question"
    question, docs = process_query(query)

    return {"query": query, "question": question, "documents": docs, "answer": ""}


# Define generation function
def generate(state: RAGState) -> RAGState:
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

    return {"query": query, "question": question, "documents": docs, "answer": answer}


def build_rag_agent(vectorstores):
    # Build the graph
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", lambda state: retrieve(vectorstores, state))
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    # Compile the graph
    return graph.compile()


# Function to run the agent
def run_rag_agent(rag_agent, query: str) -> str:
    initial_state = {"query": query, "question": "", "documents": [], "answer": ""}
    result = rag_agent.invoke(initial_state)
    return result["answer"]


if __name__ == "__main__":
    # Example usage
    vectorstores = init_vectorstores()
    repo_dir = "squatting-at-home123_back-puppet"
    question = "What is this repo about?"
    query = f"{repo_dir}: {question}"
    rag_agent = build_rag_agent(vectorstores)
    answer = run_rag_agent(rag_agent, query)
    print(f"Query: {query}")
    print(f"Answer: {answer}")
