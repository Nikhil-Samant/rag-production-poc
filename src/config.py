import os
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv('LLM_MODEL', "llama3")

EMBED_MODEL = os.getenv('EMBED_MODEL', "nomic-embed-text")

TOKEN_COUNT_FOR_MODEL = os.getenv('TOKEN_COUNT_FOR_MODEL', "gpt-3.5-turbo")

CHUNK_SIZE = int(os.getenv('CHUNK_SIZE', 1000))

CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', 150))

REQUEST_TIMEOUT = int(os.getenv('REQUEST_TIMEOUT', 60))

DOCUMENTS_DIR = os.getenv('DOCUMENTS_DIR', "../docs")

COLLECTION_NAME = os.getenv('COLLECTION_NAME', "demo_collection")

PROMPT_TEMPLATE = os.getenv('PROMPT_TEMPLATE', "demo_prompt")

RAILS_CONFIG = os.getenv('RAILS_CONFIG', "./rails_config")

LANGFUSE_PUBLIC_KEY = os.getenv('LANGFUSE_PUBLIC_KEY')

LANGFUSE_SECRET_KEY = os.getenv('LANGFUSE_SECRET_KEY')
