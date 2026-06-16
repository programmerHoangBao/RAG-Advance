import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from transformers import AutoModel
from custom_multi_head_attention import CustomMultiHeadAttention, RMSNorm

class BBLAMultiLabelModel(nn.Module):
    """
    BERT-Bi-LSTM-Attention Multi-Label Classification Model
    
    Architecture:
    CodeBERT → Bi-LSTM → Multi-Head Attention → num_tag Classification Heads
    """
    
    def __init__(self, 
                 model_path: str = "microsoft/codebert-base",
                 lstm_hidden: int = 512,
                 num_tags: int = 10,
                 num_attention_heads: int = 8,
                 dropout: float = 0.2):
        super().__init__()
        
        # 1. CodeBERT (Pretrained)
        self.codebert_model = AutoModel.from_pretrained(model_path)
        codebert_hidden = self.codebert_model.config.hidden_size  # 768
        
        # Freeze CodeBERT layers to save memory and training time
        for param in self.codebert_model.parameters():
            param.requires_grad = False
        
        # 2. Bi-LSTM (2 layers)
        self.bilstm = nn.LSTM(
            input_size=codebert_hidden,           # 768
            hidden_size=lstm_hidden,              # 512
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        
        bilstm_output_size = lstm_hidden * 2     # 1024 (512 * 2 directions)
        
        # 3. Multi-Head Self-Attention
        self.attention = CustomMultiHeadAttention(
                    embed_dim=bilstm_output_size,         # 1024
                    num_heads=num_attention_heads,        # 4
                    dropout=dropout
                )
        
        # 4. Layer Normalization and Feed-Forward (Optional but recommended)
        self.norm = nn.LayerNorm(bilstm_output_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(bilstm_output_size, bilstm_output_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(bilstm_output_size * 2, bilstm_output_size)
        )
        self.norm2 = nn.LayerNorm(bilstm_output_size)
        
        # 5. Classification Heads (10 independent heads for each tag)
        self.classification_heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(bilstm_output_size, 256),
                nn.ReLU(),
                nn.BatchNorm1d(256),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
            for _ in range(num_tags)
        ])
        
        self.num_tags = num_tags
        
    def forward(self, 
                input_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            predictions: [batch_size, num_tags] - probabilities (0-1)
        """
        
        # Step 1: CodeBERT embeddings
        with torch.no_grad():
            codebert_output = self.codebert_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            embeddings = codebert_output.last_hidden_state  # [batch, seq_len, 768]
        
        # Step 2: Bi-LSTM
        lstm_out, (h_n, c_n) = self.bilstm(embeddings)
        # lstm_out: [batch_size, seq_len, 1024]
        
        # Step 3: Multi-Head Self-Attention
        # Create padding mask for attention
        src_key_padding_mask = ~attention_mask.bool()  # Invert mask
        
        attention_out, attention_weights = self.attention(
            lstm_out, 
            lstm_out, 
            lstm_out,
            key_padding_mask=src_key_padding_mask
        )
        # attention_out: [batch_size, seq_len, 1024]
        
        # Add & Norm (Residual connection)
        lstm_out = self.norm(lstm_out + attention_out)
        
        # Feed-Forward Network
        ff_out = self.feed_forward(lstm_out)
        lstm_out = self.norm2(lstm_out + ff_out)
        # lstm_out: [batch_size, seq_len, 1024]
        
        # Step 4: Pooling - Take last token
        pooled_output = lstm_out[:, -1, :]  # [batch_size, 1024]
        
        # Step 5: Classification Heads
        outputs = []
        for head in self.classification_heads:
            output = head(pooled_output)  # [batch_size, 1]
            outputs.append(output)
        
        predictions = torch.cat(outputs, dim=1)  # [batch_size, num_tags]
        
        return predictions