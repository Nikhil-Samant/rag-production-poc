import logging
from typing import Optional
from nemoguardrails import LLMRails, RailsConfig
import chromadb
from chromadb.config import Settings as ChromaSettings
from langfuse.llama_index import LlamaIndexCallbackHandler
from llama_index.core.callbacks import CallbackManager
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from config import *
from prompts import get_template


class RagApp:
    def __init__(self):
        self.query_engine = None
        self.rails = None
        self.setup()
        self.index = self.init_index()
        self.init_query_engine(self.index)

    def setup(self):
        llm = Ollama(model=LLM_MODEL, request_timeout=REQUEST_TIMEOUT)
        embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
        Settings.llm = llm
        Settings.embed_model = embed_model
        Settings.chunk_size = CHUNK_SIZE
        Settings.chunk_overlap = CHUNK_OVERLAP
        self.init_langfuse()
        config = RailsConfig.from_path(RAILS_CONFIG)
        self.rails = LLMRails(config=config)
        self.rails.register_action(self.user_query, "user_query")

    @staticmethod
    def init_langfuse():
        if LANGFUSE_PUBLIC_KEY is None and LANGFUSE_SECRET_KEY is None:
            logging.warning("LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY is not set in the environment variables.")
            logging.info("Langfuse will not be integrated.")
            return None
        logging.info('Integrating Langfuse...')
        langfuse_callback_handler = LlamaIndexCallbackHandler()
        Settings.callback_manager = CallbackManager([langfuse_callback_handler])
        logging.info('Langfuse integrated successfully!')
        return

    @staticmethod
    def init_index():
        reader = SimpleDirectoryReader(input_dir=DOCUMENTS_DIR, recursive=True)
        documents = reader.load_data()

        logging.info("index creating with `%d` documents", len(documents))

        chroma_client = chromadb.EphemeralClient(ChromaSettings(anonymized_telemetry=False))
        chroma_collection = chroma_client.create_collection(COLLECTION_NAME)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        doc_index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)

        return doc_index

    def init_query_engine(self, doc_index):
        qa_template = get_template()
        self.query_engine = doc_index.as_query_engine(
            streaming=True,
            text_qa_template=qa_template,
            similarity_top_k=3)

    def user_query(self, context: Optional[dict] = None):
        question = context.get("user_message")
        response = self.query_engine.query(question)
        res = response.get_response()
        return res.response

    def start_chat(self):
        while (input_question := input("Enter your question (or 'q' to quit): ")) != 'q':
            res = self.rails.generate(prompt=input_question)
            print(res)


if __name__ == '__main__':
    app = RagApp()
    app.start_chat()
