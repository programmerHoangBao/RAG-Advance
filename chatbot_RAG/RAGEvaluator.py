import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer, util
import numpy as np
from collections import defaultdict
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt', quiet=True)


class RAGEvaluator:
    def __init__(
        self,
        bert_model_path="microsoft/codebert-base",      # Dùng cho BERTScore
        embedding_model=None,                           # SentenceTransformer (bge)
        device=None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. BERTScore model
        print("Loading CodeBERT for BERTScore...")
        self.bert_tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
        self.bert_model = AutoModel.from_pretrained(bert_model_path).to(self.device)
        self.bert_model.eval()
        
        # 2. Embedding model (bge) cho Faithfulness và Answer Relevancy
        if isinstance(embedding_model, str):
            print(f"Loading embedding model {embedding_model}...")
            self.embed_model = SentenceTransformer(embedding_model)
        else:
            self.embed_model = embedding_model  # Đã load từ StackOverflowRAG
        
        self.idf_dict = None

    # ====================== BERTScore ======================
    def calculate_bertscore(
        self,
        reference=None,
        candidate=None
    ):
    
        if self.bert_model is None or self.bert_tokenizer is None or reference is None or candidate is None:
            raise ValueError("All parameters must be provided")
    
        self.bert_model.eval()
    
        device = next(self.bert_model.parameters()).device
    
        ref_inputs = self.bert_tokenizer(
            reference,
            return_tensors="pt",
            truncation=True
        ).to(device)
    
        cand_inputs = self.bert_tokenizer(
            candidate,
            return_tensors="pt",
            truncation=True
        ).to(device)
    
        with torch.no_grad():
            ref_outputs = self.bert_model(**ref_inputs)
            cand_outputs = self.bert_model(**cand_inputs)
    
            ref_embs = ref_outputs.last_hidden_state[0, 1:-1, :]
            cand_embs = cand_outputs.last_hidden_state[0, 1:-1, :]
    
        if ref_embs.shape[0] == 0 or cand_embs.shape[0] == 0:
            return {
                "precision": 0.0,
                "recall": 0.0,
                "f1": 0.0,
                "sim_matrix": None
            }
    
        ref_embs = F.normalize(ref_embs, p=2, dim=1)
        cand_embs = F.normalize(cand_embs, p=2, dim=1)
    
        sim_matrix = torch.matmul(cand_embs, ref_embs.T)
    
        precision_scores = sim_matrix.max(dim=1)[0]
        precision = precision_scores.mean().item()
    
        recall_scores = sim_matrix.max(dim=0)[0]
        recall = recall_scores.mean().item()
    
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
    
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "sim_matrix": sim_matrix.cpu()
        }

    # ====================== Faithfulness ======================
    def calculate_faithfulness(self, candidate: str, retrieved_docs: list) -> float:
        """Tính Faithfulness dựa trên embedding similarity với context"""
        if not retrieved_docs:
            return 0.0
            
        context_text = self._build_context_text(retrieved_docs)
        if not context_text.strip():
            return 0.0
            
        context_sentences = sent_tokenize(context_text)
        candidate_sentences = sent_tokenize(candidate)
        
        if not context_sentences or not candidate_sentences:
            return 0.0
        
        ctx_embs = self.embed_model.encode(context_sentences, convert_to_tensor=True)
        cand_embs = self.embed_model.encode(candidate_sentences, convert_to_tensor=True)
        
        max_sims = []
        for cand_emb in cand_embs:
            sims = util.cos_sim(cand_emb.unsqueeze(0), ctx_embs)
            max_sims.append(sims.max().item())
        
        return round(sum(max_sims) / len(max_sims), 4)

    def _build_context_text(self, retrieved_docs):
        parts = []
        for hit in retrieved_docs:
            q = hit.payload.get('question', '')
            a = hit.payload.get('answer', '')
            if q or a:
                parts.append(f"Question: {q}\nAnswer: {a}")
        return "\n\n".join(parts)

    # ====================== Answer Relevancy ======================
    def calculate_answer_relevancy(self, question: str, candidate: str) -> float:
        """
        Tính Answer Relevancy: Đo mức độ câu trả lời có liên quan và giải quyết được câu hỏi
        """
        if not question or not candidate:
            return 0.0
        
        # Encode question và answer
        q_emb = self.embed_model.encode(question, convert_to_tensor=True)
        a_emb = self.embed_model.encode(candidate, convert_to_tensor=True)
        
        # Cosine similarity
        score = util.cos_sim(q_emb, a_emb).item()
        return round(score, 4)

    # ====================== Utility ======================
    def fit_idf(self, corpus: list):
        """Fit IDF cho BERTScore (nên dùng trên tập reference answers)"""
        token_counts = defaultdict(int)
        doc_count = len(corpus)
        
        for text in corpus:
            inputs = self.bert_tokenizer(text, return_tensors="pt", truncation=True)
            tokens = inputs['input_ids'][0].tolist()
            unique_tokens = set(tokens)
            for token in unique_tokens:
                token_counts[token] += 1
        
        self.idf_dict = {}
        for token_id, count in token_counts.items():
            self.idf_dict[token_id] = np.log((doc_count + 1) / (count + 1)) + 1

    def evaluate_single(self, question: str, reference: str, candidate: str, retrieved_docs: list = None):
        """Đánh giá một sample duy nhất - tiện lợi khi test"""
        results = {
            "bertscore_f1": self.calculate_bertscore(reference, candidate)["f1"],
            "faithfulness": self.calculate_faithfulness(candidate, retrieved_docs) if retrieved_docs else 0.0,
            "answer_relevancy": self.calculate_answer_relevancy(question, candidate),
        }
        return results