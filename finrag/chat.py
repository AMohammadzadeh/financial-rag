from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from loguru import logger
from typing_extensions import List, TypedDict

from finrag import prompts
from finrag.config import settings
from finrag.embedder import Embedder

prompt = PromptTemplate.from_template(prompts.RAG_PROMPT)
embedder = Embedder(model_type="openai", persist_directory="db-openai")


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


llm = ChatOpenAI(model="gpt-4o-mini", api_key=settings.OPENAI_API_KEY)


def retrieve(state: State) -> dict:
    logger.info("Fetching context...")
    retrieved_docs = embedder.vector_store.similarity_search(state["question"])
    logger.info(f"Context: {retrieved_docs}")
    return {"context": retrieved_docs}


def generate(state: State) -> dict:
    """
    Generates an answer based on the question and retrieved context.
    """
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    logger.info("Sending to LLM")
    response = llm.invoke(messages)
    return {"answer": response.content}


def build_graph() -> CompiledStateGraph:
    graph_builder = StateGraph(State).add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile()


# if __name__ == "__main__":
#     graph = build_graph()
#     response = graph.invoke({"question": "کارگزار ناظر یعنی چی؟"})
#     print(response["answer"])
