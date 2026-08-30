from pymongo import MongoClient

from config import (
    MONGO_URI,
    MONGO_DB_NAME,
    SESSIONS_COLLECTION,
    MESSAGES_COLLECTION
)


def create_mongodb_connection():
    client = MongoClient(MONGO_URI)

    db = client[MONGO_DB_NAME]

    sessions_collection = db[SESSIONS_COLLECTION]
    messages_collection = db[MESSAGES_COLLECTION]

    sessions_collection.create_index("id", unique=True)
    messages_collection.create_index("session_id")

    return client, db, sessions_collection, messages_collection