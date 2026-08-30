import os

from classification_models.predictor import Predictor
from chatbot_RAG.StackOverflowRAG import StackOverflowRAG

from config import (
    BBLA_MODEL_PATH,
    BERT_MODEL_PATH,
    EMBEDDING_MODEL_PATH,
    LLM_MODEL_PATH,
    QDRANT_URL,
    QDRANT_API_KEY
)


def load_label_mappings_txt(model_bbla_path: str):

    save_dir = os.path.dirname(model_bbla_path)

    tags_file = os.path.join(save_dir, "TAGS.txt")
    tag_to_idx_file = os.path.join(save_dir, "TAG_TO_IDX.txt")
    idx_to_tag_file = os.path.join(save_dir, "IDX_TO_TAG.txt")

    tags = []

    with open(tags_file, "r", encoding="utf-8") as f:
        for line in f:
            tag = line.strip()

            if tag:
                tags.append(tag)

    tag_to_idx = {}

    with open(tag_to_idx_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                tag, idx = line.split("\t")
                tag_to_idx[tag] = int(idx)

    idx_to_tag = {}

    with open(idx_to_tag_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                idx, tag = line.split("\t")
                idx_to_tag[int(idx)] = tag

    return tags, tag_to_idx, idx_to_tag


def create_predictor():

    tags, tag_to_idx, idx_to_tag = load_label_mappings_txt(
        BBLA_MODEL_PATH
    )

    predictor = Predictor(
        model_path=BBLA_MODEL_PATH,
        bert_model_path=BERT_MODEL_PATH,
        tags=tags,
        tag_to_idx=tag_to_idx,
        idx_to_tag=idx_to_tag
    )

    return predictor


def create_rag():

    chatrag = StackOverflowRAG(
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        llm_model_path=LLM_MODEL_PATH,
        embedding_model_path=EMBEDDING_MODEL_PATH
    )

    return chatrag


def load_models():

    predictor = create_predictor()
    chatrag = create_rag()

    return predictor, chatrag