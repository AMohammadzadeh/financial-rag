from langchain_chroma import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from loguru import logger

from finrag.config import settings


class Embedder:
    def __init__(
        self,
        model_type="huggingface",
        model_name=None,
        cache_folder="ai-models",
        persist_directory="db",
    ):
        """
        Initializes the Embedder with the desired model and vector store.

        Args:
            model_type (str): The type of model to use ('huggingface' or 'openai').
            model_name (str): The name of the embedding model to use.
            cache_folder (str): The folder to cache the model locally.
            persist_directory (str): The directory to persist the vector store.
        """
        if model_type == "huggingface":
            model_name = model_name or "heydariAI/persian-embeddings"
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name, cache_folder=cache_folder
            )
        elif model_type == "openai":
            model_name = model_name or "text-embedding-3-large"
            self.embeddings = OpenAIEmbeddings(model=model_name,api_key=settings.OPENAI_API_KEY)
        else:
            raise ValueError(
                "Invalid model_type. Choose either 'huggingface' or 'openai'."
            )

        self.vector_store = Chroma(
            embedding_function=self.embeddings, persist_directory=persist_directory
        )

    def load_documents(self, file_path, jq_schema=None):
        """
        Loads and processes documents from a JSONL file.

        Args:
            file_path (str): The path to the JSONL file containing the documents.
            jq_schema (str, optional): A jq schema to preprocess the JSONL content.

        Returns:
            None
        """
        logger.info("Start loading the documents...")
        loader = JSONLoader(
            file_path=file_path,
            jq_schema=jq_schema,
            json_lines=True,
        )

        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        self.all_splits = text_splitter.split_documents(docs)

    def embed_documents(self):
        """
        Embeds the loaded documents into the vector store.

        Returns:
            None
        """
        if not hasattr(self, "all_splits"):
            raise ValueError("Documents are not loaded. Call `load_documents` first.")

        logger.info("Start embedding documents....")
        self.vector_store.add_documents(documents=self.all_splits)
        logger.info("Embedding finished.")

# emb = Embedder(model_type='openai',persist_directory='db-openai')
# emb.load_documents("knowledge/learning.jsonl",jq_schema=".content | join(" ")")
# emb.embed_documents()