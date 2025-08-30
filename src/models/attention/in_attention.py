import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class InAttention(nn.Module):
    """
    InAttention mechanism for processing millions of tokens.
    Tokens only attend to initial states for linear scaling.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        initial_context_len: int = 1024,
        bias: bool = False,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.initial_context_len = initial_context_len
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # Projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)
        
        # Initial context compression
        self.context_compressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model, device=device, dtype=dtype),
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        """
        InAttention forward pass with initial context focus.
        
        Args:
            hidden_states: Input tensor (batch, seq_len, d_model)
            attention_mask: Attention mask
            past_key_value: Cached initial context
            use_cache: Whether to cache initial context
            
        Returns:
            Tuple of (output, cached_context)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Split into initial context and remaining sequence
        if seq_len <= self.initial_context_len:
            # Short sequence - use standard attention
            return self._standard_attention(hidden_states, attention_mask)
        
        initial_context = hidden_states[:, :self.initial_context_len]
        remaining_sequence = hidden_states[:, self.initial_context_len:]
        
        # Process initial context
        if past_key_value is not None:
            compressed_context = past_key_value
        else:
            compressed_context = self._compress_initial_context(initial_context)
            
        # Apply InAttention to remaining sequence
        remaining_output = self._in_attention(remaining_sequence, compressed_context)
        
        # Process initial context normally
        initial_output = self._standard_attention(initial_context, attention_mask)
        
        # Combine outputs
        output = torch.cat([initial_output, remaining_output], dim=1)
        
        cached_context = compressed_context if use_cache else None
        
        return output, cached_context
        
    def _compress_initial_context(self, context: torch.Tensor) -> torch.Tensor:
        """Compress initial context for efficient reuse."""
        # Apply compression network
        compressed = self.context_compressor(context)
        
        # Pool to fixed size representation
        compressed = F.adaptive_avg_pool1d(
            compressed.transpose(1, 2), 
            output_size=min(256, context.shape[1])
        ).transpose(1, 2)
        
        return compressed
        
    def _in_attention(self, sequence: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Apply InAttention mechanism - sequence attends only to context."""
        batch_size, seq_len, _ = sequence.shape
        context_len = context.shape[1]
        
        # Project sequence to queries
        Q = self.q_proj(sequence)
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Project context to keys and values
        K = self.k_proj(context)
        V = self.v_proj(context)
        K = K.view(batch_size, context_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, context_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention
        attn_output = torch.matmul(attn_probs, V)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        attn_output = self.out_proj(attn_output)
        
        return attn_output
        
    def _standard_attention(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Standard multi-head attention for short sequences."""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        Q = self.q_proj(hidden_states)
        K = self.k_proj(hidden_states)
        V = self.v_proj(hidden_states)
        
        Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        
        # Apply causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=hidden_states.device))
        attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))
        
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
            
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention
        attn_output = torch.matmul(attn_probs, V)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        attn_output = self.out_proj(attn_output)
        
        return attn_output