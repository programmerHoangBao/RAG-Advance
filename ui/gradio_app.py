import gradio as gr


class GradioApp:

    def __init__(self, chat_service, session_service):

        self.chat_service = chat_service
        self.session_service = session_service

        self.demo = self.create_ui()

    # =========================
    # SESSION
    # =========================

    def refresh_sessions(self):

        sessions = self.session_service.get_all_sessions()

        choices = [
            (
                f"{s['title']} ({s['id'][:8]}...)",
                s["id"]
            )
            for s in sessions
        ]

        return gr.update(
            choices=choices,
            value=None
        )

    def load_session(self, session_id):

        if not session_id:
            return [], "**No session selected**", session_id

        messages = self.session_service.get_chat_history(
            session_id
        )

        chat_history = []

        for msg in messages:

            display_msg = (
                self.session_service
                .build_display_response(msg)
            )

            chat_history.append(
                (
                    msg["user_question"],
                    display_msg
                )
            )

        session = self.session_service.sessions_collection.find_one(
            {"id": session_id}
        )

        info = f"""
**Session ID:** `{session_id}`

**Title:** {session['title']}

**Created:** {
    session['created_at'].strftime('%Y-%m-%d %H:%M:%S')
} UTC
"""

        return chat_history, info, session_id

    def on_new_session(self):

        new_id = self.session_service.create_new_session()

        updated_dropdown = self.refresh_sessions()

        chat_hist, info, sid = self.load_session(new_id)

        return (
            updated_dropdown,
            chat_hist,
            info,
            new_id
        )

    def on_session_change(self, session_id):

        return self.load_session(session_id)

    # =========================
    # CHAT
    # =========================

    def chatbot_response(
        self,
        message,
        history,
        session_id
    ):

        if not session_id:
            return (
                "Error: No session available.",
                history
            )

        clean_history = (
            self.session_service
            .get_clean_history_for_generation(session_id)
        )

        (
            pre_tags,
            retrieved_docs,
            candidate,
            time_process
        ) = self.chat_service.generate_answer(
            question=message,
            history=clean_history
        )

        self.session_service.save_message(
            session_id=session_id,
            user_question=message,
            tags=pre_tags,
            retrieved_docs_count=len(retrieved_docs),
            answer=candidate,
            time_process=time_process
        )

        display_response = (
            self.session_service
            .build_display_response({
                "answer": candidate,
                "tags": pre_tags,
                "len_retrieved_docs": len(retrieved_docs),
                "time_process": time_process
            })
        )

        history.append(
            (
                message,
                display_response
            )
        )

        return display_response, history

    def send_message(
        self,
        message,
        history,
        session_id
    ):

        if not message.strip():

            return (
                "",
                history,
                session_id,
                gr.update()
            )

        if not session_id:

            session_id = (
                self.session_service
                .create_new_session()
            )

        _, new_history = self.chatbot_response(
            message,
            history,
            session_id
        )

        updated_dropdown = self.refresh_sessions()

        return (
            "",
            new_history,
            session_id,
            updated_dropdown
        )

    # =========================
    # UI
    # =========================

    def create_ui(self):

        with gr.Blocks(
            title="Chatbot Demo",
            theme=gr.themes.Soft()
        ) as demo:

            gr.Markdown(
                "# 🤖 Intelligent Chatbot Demo"
            )

            gr.Markdown(
                "Each conversation runs in an independent "
                "session with MongoDB persistence."
            )

            with gr.Row():

                with gr.Column(scale=1):

                    gr.Markdown("### Sessions")

                    session_list = gr.Dropdown(
                        label="Select Session",
                        choices=[],
                        value=None,
                        interactive=True
                    )

                    new_session_btn = gr.Button(
                        "➕ New Session",
                        variant="primary"
                    )

                    session_info = gr.Markdown(
                        "**No session selected**"
                    )

                with gr.Column(scale=3):

                    chatbot = gr.Chatbot(
                        label="Chat History",
                        height=650,
                        show_label=True,
                        render_markdown=True,
                        bubble_full_width=False
                    )

                    with gr.Row():

                        msg_input = gr.Textbox(
                            label="Your message",
                            placeholder=(
                                "Feel free to ask me anything..."
                            ),
                            lines=3,
                            scale=5
                        )

                        send_btn = gr.Button(
                            "Send",
                            variant="primary",
                            scale=1
                        )

            current_session = gr.State(
                value=None
            )

            # Events

            new_session_btn.click(
                fn=self.on_new_session,
                outputs=[
                    session_list,
                    chatbot,
                    session_info,
                    current_session
                ]
            )

            session_list.change(
                fn=self.on_session_change,
                inputs=[session_list],
                outputs=[
                    chatbot,
                    session_info,
                    current_session
                ]
            )

            send_btn.click(
                fn=self.send_message,
                inputs=[
                    msg_input,
                    chatbot,
                    current_session
                ],
                outputs=[
                    msg_input,
                    chatbot,
                    current_session,
                    session_list
                ]
            )

            msg_input.submit(
                fn=self.send_message,
                inputs=[
                    msg_input,
                    chatbot,
                    current_session
                ],
                outputs=[
                    msg_input,
                    chatbot,
                    current_session,
                    session_list
                ]
            )

            demo.load(
                fn=self.refresh_sessions,
                outputs=[session_list]
            )

        return demo

    def launch(self):

        self.demo.launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860,
            debug=True
        )