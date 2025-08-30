import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from einops import rearrange


class LinearAttention(nn.Module):
    """
    Linear Attention mechanism with O(L) complexity.
    Replaces quadratic attention for ultra-long sequences.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        bias: bool = False,
        dropout: float = 0.0,
        eps: float = 1e-6,
        feature_map: str = "elu",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.eps = eps
        self.feature_map = feature_map
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)
        
        self.dropout = nn.Dropout(dropout)
        
    def feature_map_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feature map to keys and queries for linear attention."""
        if self.feature_map == "elu":
            return F.elu(x) + 1
        elif self.feature_map == "relu":
            return F.relu(x)
        elif self.feature_map == "softmax":
            return F.softmax(x, dim=-1)
        else:
            return x
            
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Linear attention forward pass.
        
        Args:
            hidden_states: Input tensor (batch, seq_len, d_model)
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_value: Cached key/value states
            output_attentions: Whether to output attention weights
            use_cache: Whether to cache key/value states
            
        Returns:
            Tuple of (output, attention_weights, past_key_value)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Linear projections
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        query_states = query_states.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply feature map
        query_states = self.feature_map_fn(query_states)
        key_states = self.feature_map_fn(key_states)
        
        # Linear attention computation
        # KV = K^T V (d_k x d_v)
        kv = torch.einsum('bhnd,bhnv->bhdv', key_states, value_states)
        
        # Normalizer: K^T 1 (d_k x 1)  
        k_cumsum = key_states.sum(dim=-2, keepdim=True)
        
        # QKV = Q (K^T V) (batch x heads x seq x d_v)
        qkv = torch.einsum('bhnd,bhdv->bhnv', query_states, kv)
        
        # Normalize: Q (K^T 1) (batch x heads x seq x 1)
        normalizer = torch.einsum('bhnd,bhd->bhn', query_states, k_cumsum.squeeze(-2))
        normalizer = normalizer.unsqueeze(-1).clamp(min=self.eps)
        
        # Final output
        attn_output = qkv / normalizer
        
        # Apply causal mask if needed  
        if attention_mask is not None and attention_mask.dim() == 2:
            # Use the attention mask directly instead of creating causal mask
            # attention_mask is already [batch_size, seq_len]
            mask = attention_mask.view(batch_size, 1, seq_len, 1)  # [batch, 1, seq_len, 1]
            attn_output = attn_output * mask
            
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.d_model)
        attn_output = self.out_proj(attn_output)
        
        attn_weights = None
        if output_attentions:
            # Compute attention weights for visualization (approximate)
            attn_weights = torch.einsum('bhnd,bhnk->bhndk', query_states, key_states)
            attn_weights = attn_weights.sum(dim=-1) / normalizer
            
        past_key_value = None
        if use_cache:
            past_key_value = (key_states, value_states)
            
        return attn_output, attn_weights, past_key_value


class CoreContextAwareAttention(nn.Module):
    """
    Core Context Aware (CCA) Attention with linear scaling.
    Groups tokens by importance and fuses significant groups.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        n_groups: int = 64,
        group_size: int = 16,
        bias: bool = False,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_groups = n_groups
        self.group_size = group_size
        
        # Core projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias, device=device, dtype=dtype)
        
        # Group scoring network
        self.group_scorer = nn.Sequential(
            nn.Linear(d_model, d_model // 4, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1, device=device, dtype=dtype),
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        CCA forward pass with token grouping and selective attention.
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Step 1: Group tokens
        groups = self._group_tokens(hidden_states)
        
        # Step 2: Score groups
        group_scores = self._score_groups(groups)
        
        # Step 3: Select important groups
        selected_groups, selected_indices = self._select_groups(groups, group_scores)
        
        # Step 4: Apply attention on selected groups
        attn_output = self._apply_attention(selected_groups, selected_indices, seq_len)
        
        return attn_output
        
    def _group_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Group consecutive tokens together."""
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Pad if necessary
        pad_len = (self.group_size - seq_len % self.group_size) % self.group_size
        if pad_len > 0:
            padding = torch.zeros(batch_size, pad_len, d_model, device=hidden_states.device)
            hidden_states = torch.cat([hidden_states, padding], dim=1)
            
        # Reshape into groups
        new_seq_len = hidden_states.shape[1]
        n_groups = new_seq_len // self.group_size
        
        groups = hidden_states.view(batch_size, n_groups, self.group_size, d_model)
        # Average pool within groups
        groups = groups.mean(dim=2)  # (batch, n_groups, d_model)
        
        return groups
        
    def _score_groups(self, groups: torch.Tensor) -> torch.Tensor:
        """Score groups by importance."""
        group_scores = self.group_scorer(groups).squeeze(-1)  # (batch, n_groups)
        return F.softmax(group_scores, dim=-1)
        
    def _select_groups(self, groups: torch.Tensor, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Select top-k groups based on scores."""
        batch_size, n_groups, d_model = groups.shape
        
        # Select top groups
        k = min(self.n_groups, n_groups)
        top_scores, top_indices = torch.topk(scores, k, dim=-1)
        
        # Gather selected groups
        batch_indices = torch.arange(batch_size, device=groups.device).unsqueeze(1)
        selected_groups = groups[batch_indices, top_indices]  # (batch, k, d_model)
        
        return selected_groups, top_indices
        
    def _apply_attention(self, groups: torch.Tensor, indices: torch.Tensor, orig_seq_len: int) -> torch.Tensor:
        """Apply multi-head attention on selected groups."""
        batch_size, n_groups, d_model = groups.shape
        
        # Project to Q, K, V
        Q = self.q_proj(groups).view(batch_size, n_groups, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(groups).view(batch_size, n_groups, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(groups).view(batch_size, n_groups, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Apply attention
        attn_output = torch.matmul(attn_probs, V)
        
        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, n_groups, d_model)
        attn_output = self.out_proj(attn_output)
        
        # Expand back to original sequence length
        output = self._expand_to_sequence(attn_output, indices, orig_seq_len)
        
        return output
        
    def _expand_to_sequence(self, group_output: torch.Tensor, indices: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Expand grouped output back to sequence length."""
        batch_size, n_groups, d_model = group_output.shape
        
        # Create output tensor
        output = torch.zeros(batch_size, seq_len, d_model, device=group_output.device)
        
        # Map groups back to positions
        for i in range(n_groups):
            start_idx = indices[:, i] * self.group_size
            end_idx = torch.min(start_idx + self.group_size, torch.tensor(seq_len))
            
            for b in range(batch_size):
                s, e = start_idx[b].item(), end_idx[b].item()
                if e <= seq_len:
                    output[b, s:e] = group_output[b, i].unsqueeze(0).repeat(e-s, 1)
                    
        return output