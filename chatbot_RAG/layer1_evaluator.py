import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Dict
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

def load_test_data(parquet_path: str) -> pd.DataFrame:
    """Load test data từ file parquet"""
    df = pd.read_parquet(parquet_path)
    logger.info(f"Loaded test data: {len(df)} samples")
    logger.info(f"Columns: {df.columns.tolist()}")
    return df


def evaluate_model(predictor, df_test: pd.DataFrame, threshold: float = 0.50) -> Dict:
    """
        Evaluate the model on the test set

        Args:
            predictor: instance of the Predictor class
            df_test: DataFrame with 'text' and 'tags' columns
            threshold: prediction threshold (default is 0.5 as in Predictor)

        Returns:
            Dict containing the metrics
    """
    y_true_list = []
    y_pred_list = []
    y_prob_list = []
    
    logger.info("Starting evaluation...")
    
    for idx, row in df_test.iterrows():
        question = row['text']
        true_tags = row['tags']  # list of strings
        
        # Predict
        result = predictor.predict(question)
        
        # True labels (binary vector)
        true_vector = np.zeros(len(predictor.TAGS), dtype=int)
        for tag in true_tags:
            if tag in predictor.TAG_TO_IDX:
                true_vector[predictor.TAG_TO_IDX[tag]] = 1
        
        # Predicted binary vector
        pred_vector = np.array(result['prediction_array'])
        
        y_true_list.append(true_vector)
        y_pred_list.append(pred_vector)
        y_prob_list.append(list(result['prediction_probabilities'].values()))
    
    # Convert to numpy arrays
    y_true = np.array(y_true_list)   # [n_samples, n_tags]
    y_pred = np.array(y_pred_list)   # [n_samples, n_tags]
     
    # ==================== METRICS ====================
    
    # 1. Subset Accuracy (Exact Match)
    subset_accuracy = accuracy_score(y_true, y_pred)
    
    # 2. Hamming Accuracy (accuracy rate per label)
    hamming_accuracy = (y_true == y_pred).mean()
    
    # 3. Macro F1
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    macro_precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    macro_recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    
    # 4. Micro F1
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    micro_precision = precision_score(y_true, y_pred, average='micro', zero_division=0)
    micro_recall = recall_score(y_true, y_pred, average='micro', zero_division=0)
    
    
    
    metrics = {
        "accuracy": float(subset_accuracy),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "micro_f1": float(micro_f1),
        "micro_precision": float(micro_precision),
        "micro_recall": float(micro_recall),
    }
    
    print("="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"{'Metric':<20} {'Value':<10}")
    print("-"*60)
    print(f"{'Subset Accuracy':<20} {subset_accuracy:.4f}")
    print(f"{'Hamming Accuracy':<20} {hamming_accuracy:.4f}")
    print(f"{'Macro F1':<20} {macro_f1:.4f}")
    print(f"{'Micro F1':<20} {micro_f1:.4f}")
    print(f"{'Macro Precision':<20} {macro_precision:.4f}")
    print(f"{'Micro Precision':<20} {micro_precision:.4f}")
    print("="*60)
    
    return metrics