import os
from dotenv import load_dotenv

# load_dotenv()
load_dotenv(".env.development")

DATABASE_URL = os.environ.get("DATABASE_URL")
PERSIST_DIRECTORY_DB = os.environ.get("PERSIST_DIRECTORY_DB")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME")
COLLECTION_NAME_SQL = os.environ.get("COLLECTION_NAME_SQL")
DATABASE_URL_LANGCHAIN = os.environ.get("DATABASE_URL_LANGCHAIN")

# print(PERSIST_DIRECTORY_DB)