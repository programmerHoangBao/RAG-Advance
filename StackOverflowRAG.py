import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client import models
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

class StackOverflowRAG:
    def __init__(
        self, 
        qdrant_url: str, 
        qdrant_api_key: str,
        llm_model_path: str,
        embedding_model_path: str
    ):
        # 1. Khởi tạo Qdrant Client
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            timeout=1000
        )
        self.collection_name = "stackoverflow_questions"
        
        # 2. Khởi tạo Embedding Model
        print(f"Loading Embedding model {embedding_model_path}...")
        self.embed_model = SentenceTransformer(embedding_model_path)
        # Model bge-base-en-v1.5 cần thêm tiền tố này cho các câu query để đạt hiệu suất tốt nhất
        self.query_instruction = "Represent this sentence for searching relevant passages: "
        
        # 3. Khởi tạo LLM
        print(f"Loading LLM {llm_model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
        
        # Sử dụng device_map="auto" để tự động đưa model lên GPU nếu có
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_path,
            torch_dtype="auto", 
            device_map="auto"
        )
        print("Initialization successful!")

    def retrieve(
        self, 
        question: str, 
        tags: list = None, 
        top_k: int = 3, 
        threshold: float = 0.5
    ) -> list:
        """
        Nhúng câu hỏi, tạo filter và truy xuất vector database.
        """
    
        # Encode câu hỏi
        full_query = self.query_instruction + question
        query_vector = self.embed_model.encode(full_query).tolist()
    
        # Xây dựng filter theo tags
        query_filter = None
    
        if tags and len(tags) > 0:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="tags",
                        match=models.MatchAny(any=tags)
                    )
                ]
            )
    
        # Search trong Qdrant
        search_results = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=threshold
        )
        if len(search_results.points) == 0:
            search_results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=None,
                limit=top_k,
                score_threshold=threshold
            )
    
        # API mới trả về object có thuộc tính .points
        return search_results.points

    def generate_response(
        self, 
        user_question: str, 
        tags: list = None,
        top_k: int = 3,
        threshold: float = 0.75
    ) -> str:
        """
        Thực hiện toàn bộ luồng RAG và sinh câu trả lời bằng LLM.
        """
        # Bước 1: Retrieval
        retrieved_docs = self.retrieve(user_question, tags, top_k, threshold)
        print(f"Length retrieved docs: {len(retrieved_docs)}")
        
        # Bước 2: Chuẩn bị Context
        if not retrieved_docs:
            context = "No highly relevant context found in the database."
        else:
            context_parts = []
            for hit in retrieved_docs:
                q = hit.payload.get('question', 'No Question')
                a = hit.payload.get('answer', 'No Answer')
                context_parts.append(f"Related Question: {q}\nProvided Answer: {a}")
            
            context = "\n\n---\n\n".join(context_parts)
            
        # Bước 3: Build Prompt (Sử dụng mẫu hệ thống phù hợp với tiếng Anh)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a StackOverflow assistant.\n"
                    "Answer ONLY using the provided context.\n"
                    "Do not invent explanations.\n"
                    "If the context does not contain enough information, say:\n"
                    "'I do not have enough information from the retrieved StackOverflow posts.'\n"
                    "Prioritize technical correctness over completeness.\n"
                    "Do not provide alternative solutions unless they are supported by the context."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question:\n{user_question}"
                )
            }
        ]
        
        # Áp dụng template hội thoại chuẩn của Qwen
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize input và đưa lên thiết bị của model (GPU/CPU)
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.llm.device)
        
        # Bước 4: Generation
        generated_ids = self.llm.generate(
            **model_inputs,
            max_new_tokens=512, # Giới hạn độ dài câu trả lời
            temperature=0.3,    # Nhiệt độ thấp giúp câu trả lời chính xác, bám sát context
            top_p=0.9
        )
        
        # Lọc bỏ phần prompt để chỉ lấy câu trả lời sinh ra
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response, retrieved_docs