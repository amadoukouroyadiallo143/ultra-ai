import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import math
from einops import rearrange, repeat
from .cuda_kernels import OptimizedMambaKernels, FusedMambaOps, optimized_einsum


class Mamba2Layer(nn.Module):
    """
    Mamba-2 Layer with selective state space mechanism.
    Implements State Space Duality with linear complexity O(L).
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
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias, device=device, dtype=dtype)
        
        # Convolution layer
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            device=device,
            dtype=dtype,
        )
        
        # x_proj for computing selective parameters
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, device=device, dtype=dtype
        )
        
        # dt_proj for time step computation
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, device=device, dtype=dtype)
        
        # Initialize dt_proj bias
        dt = torch.exp(
            torch.rand(self.d_inner, device=device, dtype=dtype) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp_(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
            
        # A parameter (state transition matrix)
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        
        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device, dtype=dtype))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias, device=device, dtype=dtype)
        
    def forward(
        self, 
        hidden_states: Tensor, 
        inference_params=None,
        **kwargs
    ) -> Tensor:
        """
        Forward pass of Mamba-2 layer.
        
        Args:
            hidden_states: Input tensor of shape (batch, seqlen, dim)
            inference_params: Parameters for inference optimization
            
        Returns:
            Output tensor of shape (batch, seqlen, dim)
        """
        batch, seqlen, dim = hidden_states.shape
        
        # Input projection
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)  # (batch, seqlen, d_inner)
        
        # Apply convolution
        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[..., :seqlen]  # Trim to original length
        x = rearrange(x, "b d l -> b l d")
        
        # Apply SiLU activation
        x = F.silu(x)
        
        # Compute selective parameters
        x_dbl = self.x_proj(x)  # (batch, seqlen, dt_rank + 2*d_state)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        # Compute time step
        dt = self.dt_proj(dt)  # (batch, seqlen, d_inner)
        dt = F.softplus(dt)
        
        # Get A matrix
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        
        # Selective scan
        y = self.selective_scan(x, dt, A, B, C, self.D.float())
        
        # Apply output gate
        z = F.silu(z)
        out = y * z
        
        # Output projection
        out = self.out_proj(out)
        
        return out
    
    def selective_scan(
        self,
        u: Tensor,
        delta: Tensor, 
        A: Tensor,
        B: Tensor,
        C: Tensor,
        D: Tensor,
    ) -> Tensor:
        """
        Selective scan operation with linear complexity.
        Implements the core State Space Model computation.
        """
        batch, seqlen, d_inner = u.shape
        d_state = A.shape[-1]
        
        # Utiliser les kernels CUDA optimisés
        if OptimizedMambaKernels.is_available() and u.is_cuda:
            return OptimizedMambaKernels.selective_scan(u, delta, A, B, C, D)
        
        # Discretize A and B avec einsum optimisé
        deltaA = torch.exp(optimized_einsum("bld,dn->bldn", delta, A))
        deltaB_u = optimized_einsum("bld,bln,bld->bldn", delta, B, u)
        
        # Initialize state
        x = torch.zeros((batch, d_inner, d_state), device=deltaA.device, dtype=deltaA.dtype)
        ys = []
        
        # Sequential scan
        for i in range(seqlen):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = optimized_einsum("bdn,bn->bd", x, C[:, i, :])
            ys.append(y)
            
        y = torch.stack(ys, dim=1)  # (batch, seqlen, d_inner)
        
        # Add skip connection
        y = y + u * rearrange(D, "d -> d")
        
        return y


def einsum(*args):
    """Helper function for einsum operations with proper type handling."""
    # Déterminer le format basé sur le type du dernier argument
    if isinstance(args[-1], str):
        # Format: einsum(tensor1, tensor2, ..., equation)
        tensors = args[:-1]
        equation = args[-1]
        float_tensors = [t.float() if hasattr(t, 'float') else t for t in tensors]
        result = torch.einsum(equation, *float_tensors)
        return result.type_as(tensors[0]) if hasattr(tensors[0], 'dtype') else result
    else:
        # Format standard: einsum(equation, tensor1, tensor2, ...)
        equation = args[0]
        tensors = args[1:]
        float_tensors = [t.float() if hasattr(t, 'float') else t for t in tensors]
        result = torch.einsum(equation, *float_tensors)
        return result.type_as(tensors[0]) if hasattr(tensors[0], 'dtype') else result