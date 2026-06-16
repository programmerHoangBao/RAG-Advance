import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from transformers import AutoModel

class RMSNorm(nn.Module):
    """Gated RMS Normalization - tương tự gated_rms_norm trong code TensorFlow"""
    
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.gate = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        # RMS normalization
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        normalized = x / rms
        
        # Gated mechanism
        return normalized * self.weight * torch.sigmoid(self.gate)


class CustomMultiHeadAttention(nn.Module):
    """
    Multi-Head Attention với ReLU thay vì Softmax
    Dựa trên cấu trúc dot_attention từ TensorFlow
    
    :param embed_dim: embedding dimension
    :param num_heads: số attention heads
    :param hidden_size: attention space dimension (mặc định = embed_dim)
    :param dropout: attention dropout rate
    :param use_layer_norm: có sử dụng layer normalization hay không
    :param use_out_projection: có sử dụng output projection hay không
    :param use_rms_norm: có sử dụng RMSNorm post-normalization hay không
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        hidden_size: int = None,
        dropout: float = 0.0,
        use_layer_norm: bool = False,
        use_out_projection: bool = True,
        use_rms_norm: bool = True
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_size = hidden_size or embed_dim
        self.head_dim = self.hidden_size // num_heads
        self.use_out_projection = use_out_projection
        self.use_rms_norm = use_rms_norm
        self.dropout_rate = dropout

        assert self.head_dim * num_heads == self.hidden_size, \
            f"hidden_size ({self.hidden_size}) phải chia hết cho num_heads ({num_heads})"

        # Query, Key, Value projections
        self.q_proj = nn.Linear(embed_dim, self.hidden_size)
        self.k_proj = nn.Linear(embed_dim, self.hidden_size)
        self.v_proj = nn.Linear(embed_dim, self.hidden_size)

        # Optional layer normalization
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.ln_q = nn.LayerNorm(self.hidden_size)
            self.ln_k = nn.LayerNorm(self.hidden_size)
            self.ln_v = nn.LayerNorm(self.hidden_size)

        # Post-attention RMSNorm (thay vì layer norm thông thường)
        if use_rms_norm:
            self.rms_norm = RMSNorm(embed_dim)
        
        # Output projection (tương tự o_map trong TensorFlow)
        if use_out_projection:
            self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Chia x thành multiple heads
        Input: [batch_size, seq_len, hidden_size]
        Output: [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def combine_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Kết hợp multiple heads lại
        Input: [batch_size, num_heads, seq_len, head_dim]
        Output: [batch_size, seq_len, hidden_size]
        """
        batch_size, num_heads, seq_len, head_dim = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, num_heads * head_dim)

    def forward(
        self,
        query,
        key=None,
        value=None,
        key_padding_mask=None,
        cache=None,
        return_cache=False
    ):
        """
        Forward pass cho attention mechanism
        
        :param query: [batch_size, query_len, embed_dim]
        :param key: [batch_size, seq_len, embed_dim] hoặc None (self-attention)
        :param value: [batch_size, seq_len, embed_dim] hoặc None (self-attention)
        :param key_padding_mask: [batch_size, seq_len] - 0 cho padding, 1 cho valid tokens
        :param cache: Cache từ lần forward trước (cho decoding)
        :param return_cache: Có trả lại cache hay không
        :return: output và attention weights (hoặc kèm cache)
        """
        batch_size = query.size(0)

        # Self-attention: query = key = value
        if key is None:
            key = query
        if value is None:
            value = query

        # ========== Tính Q, K, V ==========
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # Optional layer normalization
        if self.use_layer_norm:
            q = self.ln_q(q)
            k = self.ln_k(k)
            v = self.ln_v(v)

        # Xử lý cache (tương tự cache handling trong TensorFlow code)
        if cache is not None:
            k = torch.cat([cache.get('k', torch.tensor([])), k], dim=1)
            v = torch.cat([cache.get('v', torch.tensor([])), v], dim=1)

        # Chia thành multiple heads
        q = self.split_heads(q)  # [batch_size, num_heads, query_len, head_dim]
        k = self.split_heads(k)  # [batch_size, num_heads, seq_len, head_dim]
        v = self.split_heads(v)  # [batch_size, num_heads, seq_len, head_dim]

        # ========== Scaled dot-product attention ==========
        # Scale query (tương tự q *= (hidden_size // num_heads) ** (-0.5))
        q = q * (self.head_dim ** (-0.5))

        # Q * K^T => attention logits
        logits = torch.matmul(q, k.transpose(-2, -1))

        # ========== Áp dụng mask ==========
        if key_padding_mask is not None:
            # Chuyển mask từ [batch_size, seq_len] thành [batch_size, 1, 1, seq_len]
            # Đảo ngược: 0 -> 1 (mask), 1 -> 0 (không mask)
            # mask = (1.0 - key_padding_mask.unsqueeze(1).unsqueeze(2)).bool()
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            logits = logits.masked_fill(mask, float('-inf'))

        # ========== Thay softmax bằng ReLU (như TensorFlow code) ==========
        weights = F.relu(logits)

        # Dropout
        weights = self.dropout(weights)

        # ========== Weights * V => attention output ==========
        output = torch.matmul(weights, v)

        # Kết hợp multiple heads
        output = self.combine_heads(output)  # [batch_size, query_len, hidden_size]

        # ========== Post-attention normalization ==========
        # RMSNorm để ổn định (tương tự gated_rms_norm trong TensorFlow)
        if self.use_rms_norm:
            output = self.rms_norm(output)

        # ========== Output projection ==========
        if self.use_out_projection:
            output = self.out_proj(output)

        # # Chuẩn bị results
        # results = {
        #     'output': output,
        #     'weights': weights,
        # }

        # # Trả lại cache nếu được yêu cầu
        # if return_cache:
        #     results['cache'] = {'k': k, 'v': v}

        return output, weights