import uuid

from datetime import datetime


class SessionService:

    def __init__(self, sessions_collection, messages_collection):
        self.sessions_collection = sessions_collection
        self.messages_collection = messages_collection

    def get_all_sessions(self):

        return list(
            self.sessions_collection
            .find()
            .sort("created_at", -1)
        )

    def create_new_session(self):

        session_id = str(uuid.uuid4())

        session = {
            "id": session_id,
            "title": "New Chat",
            "created_at": datetime.utcnow()
        }

        self.sessions_collection.insert_one(session)

        return session_id

    def update_session_title(self, session_id, title):

        self.sessions_collection.update_one(
            {"id": session_id},
            {"$set": {"title": title}}
        )

    def save_message(
        self,
        session_id,
        user_question,
        tags,
        retrieved_docs_count,
        answer,
        time_process
    ):

        message = {
            "session_id": session_id,
            "user_question": user_question,
            "tags": tags,
            "len_retrieved_docs": retrieved_docs_count,
            "answer": answer,
            "time_process": time_process,
            "created_at": datetime.utcnow()
        }

        self.messages_collection.insert_one(message)

        # Update title if this is the first message
        if self.messages_collection.count_documents(
            {"session_id": session_id}
        ) == 1:

            short_title = (
                user_question[:35]
                + ("..." if len(user_question) > 35 else "")
            )

            self.update_session_title(
                session_id,
                short_title
            )

    def get_chat_history(self, session_id):

        return list(
            self.messages_collection
            .find({"session_id": session_id})
            .sort("created_at", 1)
        )

    def get_clean_history_for_generation(self, session_id):

        messages = self.get_chat_history(session_id)

        clean_history = []

        for msg in messages:
            clean_history.append(
                (
                    msg["user_question"],
                    msg["answer"]
                )
            )

        return clean_history

    def build_display_response(self, msg):

        candidate = msg["answer"]

        metadata = f"""
        
---

**Predicted Tags:** {
    ', '.join(msg.get('tags', []))
    if msg.get('tags')
    else 'None'
}

**Retrieved Documents:** {
    msg.get('len_retrieved_docs', 0)
}

**Processing Time:** {
    msg.get('time_process', 0):.2f
}s
"""

        return candidate + metadata