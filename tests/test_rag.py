from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

from config import CONFIG


class RAGState(TypedDict):
    """State for RAG agent"""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    documents: list[str]


def _init_embedding_model():
    """Initialize embedding model lazily to avoid loading at import time"""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def initialize_vectorstore(documents: list[str]) -> VectorStore:
    """Initialize FAISS vector store with documents"""
    embedding_model = _init_embedding_model()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text("\n\n".join(documents))
    vectorstore = FAISS.from_texts(chunks, embedding_model)
    return vectorstore


@tool
def retrieve_documents(query: str, vectorstore: VectorStore) -> str:
    """Retrieve relevant documents from vector store"""
    docs = vectorstore.similarity_search(query, k=3)
    context = "\n\n".join([doc.page_content for doc in docs])
    return context


@tool
def generate_answer(question: str, context: str, llm) -> str:
    """Generate answer based on question and retrieved context"""
    prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:"""
    response = llm.invoke(prompt)
    return response.content


def retrieve_node(state: RAGState, vectorstore: VectorStore) -> RAGState:
    """Retrieve relevant documents"""
    last_message = state["messages"][-1]
    query = last_message.content

    context = retrieve_documents.invoke({"query": query, "vectorstore": vectorstore})
    state["context"] = context

    return state


def generate_node(state: RAGState) -> RAGState:
    """Generate response using LLM"""
    last_message = state["messages"][-1]
    question = last_message.content
    context = state["context"]

    llm = CONFIG.get_llm()
    answer = generate_answer.invoke(
        {"question": question, "context": context, "llm": llm}
    )

    state["messages"].append(AIMessage(content=answer))
    return state


def create_rag_agent(documents: list[str]):
    """Create and return configured RAG agent graph"""

    # Initialize vector store
    vectorstore = initialize_vectorstore(documents)

    # Create graph
    graph = StateGraph(RAGState)

    # Add nodes
    graph.add_node("retrieve", lambda state: retrieve_node(state, vectorstore))
    graph.add_node("generate", generate_node)

    # Add edges
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    # Set entry point
    graph.set_entry_point("retrieve")

    return graph.compile()


def run_rag_agent(agent, question: str) -> str:
    """Run RAG agent with a question"""
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "context": "",
        "documents": [],
    }

    result = agent.invoke(initial_state)

    # Return the last AI message
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage):
            return msg.content

    return "No response generated"


# Example usage
if __name__ == "__main__":
    # Use "uv run python -m tests.test_rag" to execute this script.
    # Sample documents for RAG
    CONFIG.display()
    sample_docs = [
        """Python is a high-level programming language known for its simplicity and readability.
        It was created by Guido van Rossum and first released in 1991.""",
        """LangChain is a framework for developing applications powered by language models.
        It provides tools for building RAG systems, agents, and other LLM applications.""",
        """LangGraph is a library for building stateful, multi-actor applications with LLMs.
        It extends LangChain and provides a graph-based interface for complex workflows.""",
    ]

    # Create agent
    agent = create_rag_agent(sample_docs)

    # Test queries
    test_queries = [
        "What is Python?",
        "Tell me about LangChain",
        "How does LangGraph work?",
    ]

    for query in test_queries:
        print(f"\nQuestion: {query}")
        answer = run_rag_agent(agent, query)
        print(f"Answer: {answer}")
