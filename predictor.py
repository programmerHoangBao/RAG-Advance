import torch
from transformers import AutoTokenizer
import torch.nn as nn
import math
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from BBLAMultiLabelModel import BBLAMultiLabelModel
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

class Predictor:
    """Class for making predictions on new questions"""
    
    def __init__(
        self, 
        model_path: str,
        bert_model_path: str = "microsoft/codebert-base",
        device: str = "cuda",
        threshold: float = 0.80,
        lstm_hidden_size: int = 512,
        num_attention_heads: int = 4,
        dropout: float = 0.2,
        max_length: int = 512,
        tags: list = None,
        tag_to_idx: dict = None,
        idx_to_tag: dict = None
    ):
        self.device = device
        self.threshold = threshold
        self.max_length = max_length

        # Load label mappings
        self.TAGS, self.TAG_TO_IDX, self.IDX_TO_TAG = tags, tag_to_idx, idx_to_tag

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
        
        # Load model
        self.model = BBLAMultiLabelModel(
            model_path=bert_model_path,
            lstm_hidden=lstm_hidden_size,
            num_tags=len(self.TAGS),
            num_attention_heads=num_attention_heads,
            dropout=dropout
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Tags: {self.TAGS}")
        logger.info(f"Tag to Index Mapping: {self.TAG_TO_IDX}")
        logger.info(f"Index to Tag Mapping: {self.IDX_TO_TAG}")
    
    def predict(self, question: str):
        """
        Predict tags for a question
        
        Args:
            question: Input question text
        
        Returns:
            Dictionary with:
                - question: Original question
                - predicted_tags: List of predicted tags
                - prediction_probabilities: Dict of tag -> probability
                - prediction_array: Binary array
        """
        
        # Tokenize
        encoding = self.tokenizer(
            question,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Predict
        with torch.no_grad():
            probabilities = self.model(input_ids, attention_mask)
            probabilities = probabilities.cpu().numpy()[0]  # [num_tags]
        
        # Convert to binary predictions
        predictions = (probabilities > self.threshold).astype(int)
        
        # Get predicted tags
        predicted_tags = [
            self.TAGS[i] for i in range(len(self.TAGS))
            if predictions[i] == 1
        ]
        
        # Create result dictionary
        result = {
            'question': question,
            'predicted_tags': predicted_tags,
            'prediction_probabilities': {
                self.TAGS[i]: float(probabilities[i])
                for i in range(len(self.TAGS))
            },
            'prediction_array': predictions.tolist()
        }
        
        return result