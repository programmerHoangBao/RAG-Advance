from database.mongodb import create_mongodb_connection

from services.model_service import load_models
from services.chat_service import ChatService
from services.session_service import SessionService

from ui.gradio_app import GradioApp


def main():

    # =========================
    # 1. Load AI models
    # =========================

    predictor, chatrag = load_models()

    # =========================
    # 2. Initialize MongoDB
    # =========================

    (
        client,
        db,
        sessions_collection,
        messages_collection
    ) = create_mongodb_connection()

    # =========================
    # 3. Create services
    # =========================

    chat_service = ChatService(
        predictor=predictor,
        chatrag=chatrag
    )

    session_service = SessionService(
        sessions_collection=sessions_collection,
        messages_collection=messages_collection
    )

    # =========================
    # 4. Create Gradio app
    # =========================

    app = GradioApp(
        chat_service=chat_service,
        session_service=session_service
    )

    # =========================
    # 5. Start application
    # =========================

    app.launch()


if __name__ == "__main__":
    main()