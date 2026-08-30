import os
import pandas as pd
import time
from dotenv import load_dotenv
from classification_models.predictor import Predictor
from chatbot_RAG.RAGEvaluator import RAGEvaluator
from chatbot_RAG.StackOverflowRAG import StackOverflowRAG

load_dotenv("./.env")

bbla_model_path = "./model/bbla_model.pt"
bert_model_path = "/codebert-base"
embedding_model_path = "./bge-base-en-v1-5"
llm_model_path = "./Qwen2.5-Coder-7B-Instruct"
qdrant_url = os.getenv("QDRANT_URL")
api_key = os.getenv("API_KEY")
test_path = "./data/test.parquet"

def load_label_mappings_txt(model_bbla_path: str):
    save_dir = os.path.dirname(model_bbla_path)
    tags_file = os.path.join(save_dir, "TAGS.txt")
    tag_to_idx_file = os.path.join(save_dir, "TAG_TO_IDX.txt")
    idx_to_tag_file = os.path.join(save_dir, "IDX_TO_TAG.txt")
    TAGS = []
    with open(tags_file, "r", encoding="utf-8") as f:
        for line in f:
            tag = line.strip()
            if tag:
                TAGS.append(tag)
    TAG_TO_IDX = {}
    with open(tag_to_idx_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                tag, idx = line.split("\t")
                TAG_TO_IDX[tag] = int(idx)
    IDX_TO_TAG = {}
    with open(idx_to_tag_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                idx, tag = line.split("\t")
                IDX_TO_TAG[int(idx)] = tag
    print(f"Tags: {TAGS}")
    print(f"Tag to index: {TAG_TO_IDX}")
    print(f"Idx to tag: {IDX_TO_TAG}")  
    return TAGS, TAG_TO_IDX, IDX_TO_TAG

def initialize_models():
    """Initialize Predictor, StackOverflowRAG, and RAGEvaluator."""

    tags, tag_to_idx, idx_to_tag = load_label_mappings_txt(
        bbla_model_path
    )

    predictor = Predictor(
        model_path=bbla_model_path,
        bert_model_path=bert_model_path,
        tags=tags,
        tag_to_idx=tag_to_idx,
        idx_to_tag=idx_to_tag
    )

    chatrag = StackOverflowRAG(
        qdrant_url=qdrant_url,
        qdrant_api_key=api_key,
        llm_model_path=llm_model_path,
        embedding_model_path=embedding_model_path
    )

    evaluator = RAGEvaluator(
        bert_model_path=bert_model_path,
        embedding_model=chatrag.embed_model
    )

    return predictor, chatrag, evaluator

def load_test_data(test_path):
    """Load test data from a Parquet file."""

    df_test = pd.read_parquet(test_path)

    questions = df_test["question"].tolist()
    reference_answers = df_test["answer"].tolist()
    ref_tags = df_test["tags"].tolist()

    return questions, reference_answers, ref_tags

def evaluate_question(
    question,
    reference_answer,
    ref_tag,
    predictor,
    chatrag,
    evaluator
):
    """Evaluate a question."""

    # Tag prediction
    pre_tags = predictor.predict(question)

    # Measuring RAG latency
    loop_start_time = time.time()

    candidate, retrieved_docs = chatrag.generate_response(
        user_question=question,
        tags=pre_tags["predicted_tags"],
        top_k=3,
        threshold=0.80
    )

    loop_execution_time = time.time() - loop_start_time

    # Rate the answer
    scores = evaluator.evaluate_single(
        question=question,
        reference=reference_answer,
        candidate=candidate,
        retrieved_docs=retrieved_docs
    )

    result = {
        "question": question,
        "reference_answer": reference_answer,
        "candidate": candidate,
        "num_retrieved_docs": len(retrieved_docs),
        "ref_tags": ref_tag,
        "pre_tags": pre_tags["predicted_tags"],
        "bertscore_f1": scores["bertscore_f1"],
        "faithfulness": scores["faithfulness"],
        "answer_relevancy": scores["answer_relevancy"],
        "execution_time_seconds": loop_execution_time
    }

    return result

def evaluate_dataset(
    questions,
    reference_answers,
    ref_tags,
    predictor,
    chatrag,
    evaluator
):
    """Evaluate the entire dataset."""

    results_details = []
    total_execution_time = 0

    for question, reference_answer, ref_tag in zip(
        questions,
        reference_answers,
        ref_tags
    ):
        result = evaluate_question(
            question=question,
            reference_answer=reference_answer,
            ref_tag=ref_tag,
            predictor=predictor,
            chatrag=chatrag,
            evaluator=evaluator
        )

        results_details.append(result)

        total_execution_time += result["execution_time_seconds"]

    return results_details, total_execution_time

def calculate_summary(results_details, total_execution_time):
    """Calculate the average of the metrics."""

    total_questions = len(results_details)

    sum_faithfulness = sum(
        result["faithfulness"]
        for result in results_details
    )

    sum_bert_score = sum(
        result["bertscore_f1"]
        for result in results_details
    )

    sum_answer_relevancy = sum(
        result["answer_relevancy"]
        for result in results_details
    )

    sum_len_retrieved = sum(
        result["num_retrieved_docs"]
        for result in results_details
    )

    return {
        "average_faithfulness": (
            sum_faithfulness / total_questions
        ),
        "average_bert_score": (
            sum_bert_score / total_questions
        ),
        "average_answer_relevancy": (
            sum_answer_relevancy / total_questions
        ),
        "avg_num_retrieved_docs": (
            sum_len_retrieved / total_questions
        ),
        "total_execution_time_seconds": total_execution_time,
        "average_execution_time_seconds": (
            total_execution_time / total_questions
        )
    }

def save_results(results_details, results_summary):
    """Save evaluation results to CSV."""

    results_df = pd.DataFrame(results_details)
    results_df.to_csv(
        "./evaluation_results.csv",
        index=False
    )

    results_summary_df = pd.DataFrame([results_summary])
    results_summary_df.to_csv(
        "./evaluation_summary.csv",
        index=False
    )

def main():
    # 1. Initialize models
    predictor, chatrag, evaluator = initialize_models()

    # 2. Load test dataset
    questions, reference_answers, ref_tags = load_test_data(
        test_path
    )

    # 3. Fit IDF for evaluator
    evaluator.fit_idf(reference_answers)

    # 4. Evaluate dataset
    results_details, total_execution_time = evaluate_dataset(
        questions=questions,
        reference_answers=reference_answers,
        ref_tags=ref_tags,
        predictor=predictor,
        chatrag=chatrag,
        evaluator=evaluator
    )

    # 5. Calculate the summary
    results_summary = calculate_summary(
        results_details=results_details,
        total_execution_time=total_execution_time
    )

    # 6. Save results
    save_results(
        results_details=results_details,
        results_summary=results_summary
    )

if __name__ == "__main__":
    main()
