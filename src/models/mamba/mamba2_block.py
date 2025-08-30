import torch
import torch.nn as nn
from typing import Optional
from .mamba2_layer import Mamba2Layer


class Mamba2Block(nn.Module):
    """
    Mamba-2 Block with RMSNorm and residual connections.
    Forms the building block of the Mamba-2 model.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,
        layer_idx: Optional[int] = None,
        norm_epsilon: float = 1e-5,
        residual_in_fp32: bool = False,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        
        # Layer normalization
        self.norm = RMSNorm(d_model, eps=norm_epsilon, device=device, dtype=dtype)
        
        # Mamba layer
        self.mixer = Mamba2Layer(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            conv_bias=conv_bias,
            bias=bias,
            use_fast_path=use_fast_path,
            layer_idx=layer_idx,
            device=device,
            dtype=dtype,
        )
        
    def forward(
        self, 
        hidden_states: torch.Tensor, 
        residual: Optional[torch.Tensor] = None,
        inference_params=None,
        **mixer_kwargs
    ):
        """
        Forward pass with pre-normalization and residual connection.
        
        Args:
            hidden_states: Input tensor
            residual: Residual connection (optional)
            inference_params: Parameters for inference
            
        Returns:
            Tuple of (output, residual)
        """
        if residual is None:
            residual = hidden_states
        else:
            hidden_states = hidden_states + residual
            
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
            
        # Pre-normalization
        hidden_states = self.norm(hidden_states.to(dtype=self.norm.weight.dtype))
        
        # Apply Mamba layer
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)
        
        return hidden_states, residual
    
    def allocate_inference_cache(self, batch_size: int, max_seqlen: int, dtype=None, **kwargs):
        """Allocate cache for inference."""
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""
    
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, device=device, dtype=dtype))
        self.variance_epsilon = eps
        
    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)