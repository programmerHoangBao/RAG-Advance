import time


class ChatService:

    def __init__(self, predictor, chatrag):
        self.predictor = predictor
        self.chatrag = chatrag

    def generate_answer(self, question, history=None):

        start_time = time.time()

        pre_tags = self.predictor.predict(question)

        candidate, retrieved_docs = self.chatrag.generate_response(
            user_question=question,
            tags=pre_tags["predicted_tags"],
            history=history,
            top_k=2,
            threshold=0.80
        )

        if len(retrieved_docs) == 0:
            candidate += (
                "\n**Note**: *Your question is not covered in our "
                "knowledge base, so the answer is entirely based on "
                "the LLM model.*"
            )

        time_process = time.time() - start_time

        return (
            pre_tags["predicted_tags"],
            retrieved_docs,
            candidate,
            time_process
        )