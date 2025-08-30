import torch
import torch.nn as nn
from typing import Optional, Tuple
from .linear_attention import LinearAttention, CoreContextAwareAttention
from .in_attention import InAttention


class HybridAttentionLayer(nn.Module):
    """
    Hybrid Attention Layer combining different attention mechanisms.
    Represents 20% of the model architecture with advanced attention.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        attention_type: str = "linear",
        bias: bool = False,
        dropout: float = 0.0,
        norm_epsilon: float = 1e-5,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.attention_type = attention_type
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model, eps=norm_epsilon, device=device, dtype=dtype)
        
        # Select attention mechanism
        if attention_type == "linear":
            self.attention = LinearAttention(
                d_model=d_model,
                n_heads=n_heads,
                bias=bias,
                dropout=dropout,
                device=device,
                dtype=dtype,
            )
        elif attention_type == "cca":
            self.attention = CoreContextAwareAttention(
                d_model=d_model,
                n_heads=n_heads,
                bias=bias,
                dropout=dropout,
                device=device,
                dtype=dtype,
            )
        elif attention_type == "in_attention":
            self.attention = InAttention(
                d_model=d_model,
                n_heads=n_heads,
                bias=bias,
                dropout=dropout,
                device=device,
                dtype=dtype,
            )
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
            
        # Feed-forward network
        self.ffn = FeedForward(
            d_model=d_model,
            d_ff=d_model * 4,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        use_mixed_precision: bool = True,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass of hybrid attention layer with optimizations.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_value: Cached states
            output_attentions: Whether to output attention weights
            use_cache: Whether to cache states
            use_mixed_precision: Enable mixed precision computation
            
        Returns:
            Tuple of (output, attention_weights, past_key_value)
        """
        # Pre-normalization
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        
        # Apply attention with mixed precision if available
        autocast_context = torch.cuda.amp.autocast() if (use_mixed_precision and torch.cuda.is_available()) else torch.no_grad()
        
        with autocast_context:
            if self.attention_type in ["linear", "cca"]:
                attn_output, attn_weights, past_key_value = self.attention(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            else:  # in_attention
                attn_output, past_key_value = self.attention(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                )
                attn_weights = None
                
        # Residual connection
        hidden_states = residual + self.dropout(attn_output)
        
        # Feed-forward network with residual and mixed precision
        residual = hidden_states
        with autocast_context:
            ffn_output = self.ffn(hidden_states)
        hidden_states = residual + self.dropout(ffn_output)
        
        return hidden_states, attn_weights, past_key_value


class FeedForward(nn.Module):
    """Feed-forward network with SiLU activation."""
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False, device=device, dtype=dtype)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Optimized SwiGLU activation with fused operations
        gate = torch.nn.functional.silu(self.gate_proj(x))
        up = self.up_proj(x)
        # Use fused multiply-add when possible
        gated = gate * up
        return self.down_proj(self.dropout(gated))


class AdaptiveAttentionRouter(nn.Module):
    """
    Adaptive router to select optimal attention mechanism based on sequence length and complexity.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        bias: bool = False,
        dropout: float = 0.0,
        short_seq_threshold: int = 2048,
        long_seq_threshold: int = 32768,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.short_seq_threshold = short_seq_threshold
        self.long_seq_threshold = long_seq_threshold
        
        # Different attention mechanisms for different sequence lengths
        self.short_attention = LinearAttention(
            d_model=d_model,
            n_heads=n_heads,
            bias=bias,
            dropout=dropout,
            feature_map="softmax",
            device=device,
            dtype=dtype,
        )
        
        self.medium_attention = CoreContextAwareAttention(
            d_model=d_model,
            n_heads=n_heads,
            bias=bias,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )
        
        self.long_attention = InAttention(
            d_model=d_model,
            n_heads=n_heads,
            bias=bias,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )
        
        # Complexity predictor for dynamic routing
        self.complexity_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        
        # Performance statistics
        self.routing_stats = {
            'short_attention_calls': 0,
            'medium_attention_calls': 0,
            'long_attention_calls': 0,
        }
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        adaptive_routing: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Route to appropriate attention mechanism based on sequence length and complexity.
        """
        seq_len = hidden_states.shape[1]
        
        # Adaptive complexity-based routing
        if adaptive_routing and seq_len > self.short_seq_threshold:
            complexity_score = self.complexity_predictor(hidden_states.mean(dim=1)).mean()
            
            # Adjust thresholds based on complexity
            if complexity_score > 0.7:  # High complexity
                effective_short_threshold = self.short_seq_threshold // 2
                effective_long_threshold = self.long_seq_threshold // 2
            else:  # Low complexity
                effective_short_threshold = self.short_seq_threshold * 2
                effective_long_threshold = self.long_seq_threshold * 2
        else:
            effective_short_threshold = self.short_seq_threshold
            effective_long_threshold = self.long_seq_threshold
        
        # Route based on effective thresholds
        if seq_len <= effective_short_threshold:
            # Use optimized linear attention for short sequences
            self.routing_stats['short_attention_calls'] += 1
            with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
                output, _, _ = self.short_attention(
                    hidden_states, 
                    attention_mask=attention_mask,
                    **kwargs
                )
        elif seq_len <= effective_long_threshold:
            # Use CCA for medium sequences
            self.routing_stats['medium_attention_calls'] += 1
            with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
                output, _, _ = self.medium_attention(
                    hidden_states,
                    attention_mask=attention_mask,
                    **kwargs
                )
        else:
            # Use InAttention for very long sequences
            self.routing_stats['long_attention_calls'] += 1
            with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
                output, _ = self.long_attention(
                    hidden_states,
                    attention_mask=attention_mask,
                    **kwargs
                )
            
        return output
        
    def get_routing_stats(self) -> dict:
        """Get routing statistics."""
        total_calls = sum(self.routing_stats.values())
        if total_calls == 0:
            return self.routing_stats
        
        stats = self.routing_stats.copy()
        stats['total_calls'] = total_calls
        stats['short_attention_ratio'] = stats['short_attention_calls'] / total_calls
        stats['medium_attention_ratio'] = stats['medium_attention_calls'] / total_calls
        stats['long_attention_ratio'] = stats['long_attention_calls'] / total_calls
        
        return stats
        
    def reset_routing_stats(self):
        """Reset routing statistics."""
        for key in self.routing_stats:
            self.routing_stats[key] = 0