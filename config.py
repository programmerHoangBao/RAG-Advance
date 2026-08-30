import os
from dotenv import load_dotenv

load_dotenv("./.env")


BBLA_MODEL_PATH = "./model/bbla_model.pt"
BERT_MODEL_PATH = "/codebert-base"
EMBEDDING_MODEL_PATH = "./bge-base-en-v1-5"
LLM_MODEL_PATH = "./Qwen2.5-Coder-7B-Instruct"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("API_KEY")

MONGO_URI = os.getenv("MONGO_URI")

MONGO_DB_NAME = "chatbot_db"

SESSIONS_COLLECTION = "sessions"
MESSAGES_COLLECTION = "messages"