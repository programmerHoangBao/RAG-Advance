import pandas as pd
import time
from dotenv import load_dotenv
import os
from predictor import Predictor
from RAGEvaluator import RAGEvaluator
from StackOverflowRAG import StackOverflowRAG

load_dotenv("./.env")

bbla_model_path = os.getenv("BBLA_MODEL_PATH")
bert_model_path = os.getenv("BERT_MODEL_PATH")
embedding_model_path = os.getenv("EMBEDDING_MODEL_PATH")
llm_model_path = os.getenv("LLM_MODEL_PATH")
qdrant_url = os.getenv("QDRANT_URL")
api_key = os.getenv("API_KEY")
test_path = os.getenv("TEST_PATH")

tags, tag_to_idx, idx_to_tag = load_label_mappings_txt(bbla_model_path)
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
# Khởi tạo
evaluator = RAGEvaluator(
    bert_model_path=bert_model_path,
    embedding_model=chatrag.embed_model   # Truyền model đã load
)

df_test = pd.read_parquet(test_path)
questions = df_test["question"].tolist()
reference_answers = df_test["answer"].tolist()
evaluator.fit_idf(reference_answers)
ref_tags = df_test["tags"].tolist()
results_details = []
sum_faithfulness = 0
sum_bert_score = 0
sum_answer_relevancy = 0
sum_len_retrieved = 0
total_execution_time = 0
index = 0
for question, reference_answer in zip(questions, reference_answers):
    # print(f"Loop {index + 1}")
    pre_tags = predictor.predict(question)
    # print(f"Reference tags: {ref_tags[index]}")
    # print(f"Predict tags: {pre_tags['predicted_tags']}, {pre_tags['prediction_probabilities']}")
    loop_start_time = time.time()
    candidate, retrieved_docs = chatrag.generate_response(
        user_question=question,
        tags=pre_tags['predicted_tags'],
        top_k=3,
        threshold=0.80
    )
    loop_execution_time = time.time() - loop_start_time
    total_execution_time += loop_execution_time
    
    scores = evaluator.evaluate_single(
        question=question,
        reference=reference_answer,
        candidate=candidate,
        retrieved_docs=retrieved_docs
    )
    
    sum_faithfulness += scores["faithfulness"]
    sum_bert_score += scores["bertscore_f1"]
    sum_answer_relevancy += scores["answer_relevancy"]
    sum_len_retrieved += len(retrieved_docs)
    results_details.append({
        "question": question,
        "reference_answer": reference_answer,
        "candidate": candidate,
        "num_retrieved_docs": len(retrieved_docs),
        "ref_tags": ref_tags[index],
        "pre_tags": pre_tags['predicted_tags'],
        "bertscore_f1": scores["bertscore_f1"],
        "faithfulness": scores["faithfulness"],
        "answer_relevancy": scores["answer_relevancy"],
        "execution_time_seconds": loop_execution_time
    })
    index += 1

results_summary = {
    "average_faithfulness": sum_faithfulness / len(questions),
    "average_bert_score": sum_bert_score / len(questions),
    "average_answer_relevancy": sum_answer_relevancy / len(questions),
    "avg_num_retrieved_dos": sum_len_retrieved / len(questions),
    "total_execution_time_seconds": total_execution_time,
    "average_execution_time_seconds": total_execution_time / len(questions)
}
results_df = pd.DataFrame(results_details)
results_df.to_csv("./evaluation_results.csv", index=False)
results_summary_df = pd.DataFrame([results_summary])
results_summary_df.to_csv("./evaluation_summary.csv", index=False)