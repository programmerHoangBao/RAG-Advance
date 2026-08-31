import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client import models
from collections import OrderedDict
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

class StackOverflowRAG:
    def __init__(
        self, 
        qdrant_url: str, 
        qdrant_api_key: str,
        llm_model_path: str,
        embedding_model_path: str,
        max_history_turns: int = 8,           # Số lượt hội thoại giữ lại
        max_history_for_rewrite: int = 2
    ):
        # Qdrant Client
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=1000
        )
        self.collection_name = "stackoverflow_questions"

        # Embedding
        print(f"Loading Embedding model: {embedding_model_path}...")
        self.embed_model = SentenceTransformer(embedding_model_path)
        self.query_instruction = "Represent this sentence for searching relevant passages: "
        
        # LLM
        print(f"Loading LLM: {llm_model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_path,
            torch_dtype="auto", 
            device_map="auto",
            # load_in_4bit=True,          # Bật nếu muốn tiết kiệm VRAM
            # load_in_8bit=True,
        )
        
        # History management
        self.max_history_turns = max_history_turns
        self.max_history_for_rewrite = max_history_for_rewrite
        
        # Cache embedding query
        self.embedding_cache = OrderedDict()  # LRU Cache
        self.cache_max_size = 100
        
        print("Initialization successful!")

    def _manage_history(self, history):
        """Giới hạn lịch sử để tránh context quá dài"""
        if not history:
            return []
        return history[-self.max_history_turns:]

    def _get_cached_embedding(self, text: str):
        """Cache embedding để tránh encode lặp lại"""
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        vector = self.embed_model.encode(text).tolist()
        self.embedding_cache[text] = vector
        
        # Giữ cache size
        if len(self.embedding_cache) > self.cache_max_size:
            self.embedding_cache.popitem(last=False)
            
        return vector

    def rewrite_query(self, question: str, history=None):
        """Rewrite the latest question into a standalone search query."""
        
        if not history:
            return question.strip()
    
        recent_history = history[-self.max_history_for_rewrite:]
    
        # Sử dụng chat template của Qwen2.5
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert search query rewriter. "
                    "Your ONLY task is to convert the latest user question into a concise, standalone search query.\n\n"
                    "Rules:\n"
                    "- If the current question is a standalone question, the result will be 'current question'.\n"
                    "- Output ONLY the rewritten query.\n"
                    "- Do not answer the question.\n"
                    "- Do not explain anything.\n"
                    "- Do not add quotes or extra text.\n"
                    "- Resolve pronouns (it, this, they, that...) using conversation context.\n"
                    "- Keep it short and natural for semantic search. \n"
                    "- Resolve pronouns, omitted subjects, and implicit references using the previous conversation.\n"
                    "- If the latest question is incomplete by itself, rewrite it into a fully self-contained search query.\n"
                )
            }
        ]
        index = 1
        # Thêm lịch sử
        for user_msg, assistant_msg in recent_history:
            
            messages.append({"role": "user", "content": f"quesion history {index}: {user_msg}"})
            messages.append({"role": "assistant", "content": f"answer history {index}: {assistant_msg}"})
    
        # Thêm câu hỏi mới
        messages.append({"role": "user", "content": "current quession: " + question})
    
        try:
            # Dùng chat template chính thức
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
    
            inputs = self.tokenizer(
                [text],
                return_tensors="pt",
                truncation=True,
                max_length=2048
            ).to(self.llm.device)
    
            with torch.no_grad():
                outputs = self.llm.generate(
                    **inputs,
                    max_new_tokens=64,           # đủ cho query
                    do_sample=False,
                    temperature=0.0,
                    top_p=1.0,
                    repetition_penalty=1.05,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
    
            # Decode chỉ phần generated
            generated_ids = outputs[0][inputs.input_ids.shape[1]:]
            rewritten = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
             # Fallback
            if not rewritten or len(rewritten) < 5 or len(rewritten.split()) > 500:
                return question.strip()
    
            return rewritten
    
        except Exception as e:
            print(f"Rewrite query error: {e}")
            return question.strip()

    def retrieve(
        self, 
        question: str, 
        tags: list = None, 
        top_k: int = 5, 
        threshold: float = 0.80
    ):
        full_query = self.query_instruction + question
        query_vector = self._get_cached_embedding(full_query)

        query_filter = None
        if tags and len(tags) > 0:
            query_filter = models.Filter(
                must=[models.FieldCondition(key="tags", match=models.MatchAny(any=tags))]
            )

        try:
            search_results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=threshold
            )

            if len(search_results.points) == 0 and query_filter is not None:
                search_results = self.qdrant_client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    limit=top_k,
                    score_threshold=threshold
                )
            
            return search_results.points
        except Exception as e:
            print(f"Retrieve error: {e}")
            return []

    def generate_response(
        self,
        user_question: str,
        history=None,
        tags=None,
        top_k=5,
        threshold=0.80
    ):
        
        managed_history = self._manage_history(history)
        search_query = self.rewrite_query(user_question, managed_history)
        print(f"Search query: {search_query}")

        retrieved_docs = self.retrieve(search_query, tags, top_k, threshold)
        print(f"Retrieved {len(retrieved_docs)} documents")

        # Xây dựng context
        if not retrieved_docs:
            context = "No highly relevant context found in the database."
        else:
            context_parts = []
            for hit in retrieved_docs:
                q = hit.payload.get('question', '')
                a = hit.payload.get('answer', '')
                score = hit.score
                context_parts.append(f"Q: {q}\nA: {a}\nRelevance: {score:.3f}")
            context = "\n\n---\n\n".join(context_parts)

        # ================== SYSTEM PROMPT MẠNH HƠN ==================
        SYSTEM_PROMPT = (
            "You are a concise and accurate technical assistant.\n"
            "Rules:\n"
            "- The answer is based on retrieved StackOverflow posts.\n"
            "- Answer directly and clearly.\n"
            "- Keep responses reasonably short unless user asks for details.\n"
            "- Always use proper formatting (headings, code blocks, tables).\n"
            "- If the answer is long, summarize the key points first.\n"
            "- Stop naturally when the answer is complete."
        )

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # for user_msg, assistant_msg in managed_history:
        #     messages.append({"role": "user", "content": user_msg})
        #     messages.append({"role": "assistant", "content": assistant_msg})

        messages.append({
            "role": "user",
            "content": f"""Retrieved StackOverflow Posts:\n\n{context}\n\nCurrent Question: {search_query}"""
        })

        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.llm.device)

            # ================== GENERATION TỐI ƯU ==================
            generated_ids = self.llm.generate(
                **model_inputs,
                max_new_tokens=1024,           # Tăng đáng kể
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
        
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response, retrieved_docs

        except Exception as e:
            print(f"Generation error: {e}")
            return "Sorry, I encountered an error while generating the answer.", []

    def clear_cache(self):
        """Xóa embedding cache"""
        self.embedding_cache.clear()