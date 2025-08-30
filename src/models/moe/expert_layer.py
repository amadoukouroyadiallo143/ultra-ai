import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ExpertLayer(nn.Module):
    """
    Individual expert in Mixture of Experts architecture.
    Each expert is a specialized feed-forward network.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str = "silu",
        dropout: float = 0.0,
        bias: bool = False,
        expert_id: int = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.expert_id = expert_id
        self.activation = activation
        
        # Gate and up projections for GLU-style activation
        self.gate_proj = nn.Linear(d_model, d_ff, bias=bias, device=device, dtype=dtype)
        self.up_proj = nn.Linear(d_model, d_ff, bias=bias, device=device, dtype=dtype)
        self.down_proj = nn.Linear(d_ff, d_model, bias=bias, device=device, dtype=dtype)
        
        self.dropout = nn.Dropout(dropout)
        
        # Expert-specific parameters for specialization
        self.expert_bias = nn.Parameter(torch.zeros(d_model, device=device, dtype=dtype))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of expert layer.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Expert output (batch_size, seq_len, d_model)
        """
        # Add expert-specific bias for specialization
        x = x + self.expert_bias
        
        # GLU-style feed-forward with chosen activation
        gate = self._apply_activation(self.gate_proj(x))
        up = self.up_proj(x)
        hidden = gate * up
        hidden = self.dropout(hidden)
        
        output = self.down_proj(hidden)
        return output
        
    def _apply_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the specified activation function."""
        if self.activation == "silu":
            return F.silu(x)
        elif self.activation == "gelu":
            return F.gelu(x)
        elif self.activation == "relu":
            return F.relu(x)
        elif self.activation == "swish":
            return x * torch.sigmoid(x)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")


class SpecializedExpert(ExpertLayer):
    """
    Specialized expert with domain-specific parameters.
    Can be configured for specific tasks (e.g., math, coding, reasoning).
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        specialization: str = "general",
        activation: str = "silu",
        dropout: float = 0.0,
        bias: bool = False,
        expert_id: int = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            d_model=d_model,
            d_ff=d_ff,
            activation=activation,
            dropout=dropout,
            bias=bias,
            expert_id=expert_id,
            device=device,
            dtype=dtype,
        )
        self.specialization = specialization
        
        # Specialization-specific layers
        if specialization == "math":
            # Mathematical reasoning components
            self.math_gate = nn.Linear(d_model, d_model // 4, device=device, dtype=dtype)
            self.math_transform = nn.Linear(d_model // 4, d_model, device=device, dtype=dtype)
        elif specialization == "code":
            # Code understanding components  
            self.code_attention = nn.MultiheadAttention(
                d_model, num_heads=8, dropout=dropout, 
                device=device, dtype=dtype
            )
        elif specialization == "reasoning":
            # Multi-step reasoning components
            self.reasoning_memory = nn.LSTM(
                d_model, d_model // 2, batch_first=True,
                device=device, dtype=dtype
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with specialization-specific processing."""
        # Base expert processing
        base_output = super().forward(x)
        
        # Apply specialization
        if self.specialization == "math":
            math_gate = torch.sigmoid(self.math_gate(x))
            math_features = self.math_transform(math_gate)
            output = base_output + 0.1 * math_features
        elif self.specialization == "code":
            # Self-attention for code structure understanding
            x_t = x.transpose(0, 1)  # (seq, batch, dim)
            attn_out, _ = self.code_attention(x_t, x_t, x_t)
            attn_out = attn_out.transpose(0, 1)  # (batch, seq, dim)
            output = base_output + 0.1 * attn_out
        elif self.specialization == "reasoning":
            # LSTM for sequential reasoning
            reasoning_out, _ = self.reasoning_memory(x)
            output = base_output + 0.1 * reasoning_out
        else:
            output = base_output
            
        return output


class AdaptiveExpert(ExpertLayer):
    """
    Adaptive expert that can adjust its capacity based on input complexity.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        min_capacity: float = 0.25,
        max_capacity: float = 1.0,
        activation: str = "silu",
        dropout: float = 0.0,
        bias: bool = False,
        expert_id: int = 0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            d_model=d_model,
            d_ff=d_ff,
            activation=activation,
            dropout=dropout,
            bias=bias,
            expert_id=expert_id,
            device=device,
            dtype=dtype,
        )
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity
        
        # Capacity predictor
        self.capacity_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 4, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1, device=device, dtype=dtype),
            nn.Sigmoid(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with adaptive capacity."""
        batch_size, seq_len, d_model = x.shape
        
        # Predict required capacity
        capacity_scores = self.capacity_predictor(x)  # (batch, seq, 1)
        capacity = (
            self.min_capacity + 
            (self.max_capacity - self.min_capacity) * capacity_scores
        )
        
        # Adaptive processing
        gate = self._apply_activation(self.gate_proj(x))
        up = self.up_proj(x)
        
        # Scale by predicted capacity
        hidden = gate * up * capacity
        hidden = self.dropout(hidden)
        
        output = self.down_proj(hidden)
        return output