import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any
from .expert_layer import ExpertLayer, SpecializedExpert
from .routing import TopKRouter, ExpertChoiceRouter, SwitchRouter
from .load_balancing import LoadBalancer


class MoELayer(nn.Module):
    """
    Mixture of Experts Layer with configurable routing and load balancing.
    Represents 8% of the model architecture with massive capacity.
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int = 256,
        top_k: int = 2,
        expert_type: str = "standard",
        router_type: str = "top_k",
        d_ff: Optional[int] = None,
        activation: str = "silu",
        dropout: float = 0.0,
        bias: bool = False,
        capacity_factor: float = 1.25,
        balance_loss_weight: float = 0.01,
        z_loss_weight: float = 0.001,
        jitter_noise: float = 0.01,
        normalize_gates: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_type = expert_type
        self.router_type = router_type
        
        if d_ff is None:
            d_ff = d_model * 4
            
        # Create experts
        self.experts = nn.ModuleList()
        for i in range(num_experts):
            if expert_type == "specialized":
                # Create specialized experts
                specializations = ["general", "math", "code", "reasoning"]
                specialization = specializations[i % len(specializations)]
                expert = SpecializedExpert(
                    d_model=d_model,
                    d_ff=d_ff,
                    specialization=specialization,
                    activation=activation,
                    dropout=dropout,
                    bias=bias,
                    expert_id=i,
                    device=device,
                    dtype=dtype,
                )
            else:
                expert = ExpertLayer(
                    d_model=d_model,
                    d_ff=d_ff,
                    activation=activation,
                    dropout=dropout,
                    bias=bias,
                    expert_id=i,
                    device=device,
                    dtype=dtype,
                )
            self.experts.append(expert)
            
        # Create router
        if router_type == "top_k":
            self.router = TopKRouter(
                d_model=d_model,
                num_experts=num_experts,
                top_k=top_k,
                capacity_factor=capacity_factor,
                jitter_noise=jitter_noise,
                normalize_gates=normalize_gates,
                device=device,
                dtype=dtype,
            )
        elif router_type == "expert_choice":
            self.router = ExpertChoiceRouter(
                d_model=d_model,
                num_experts=num_experts,
                capacity_factor=capacity_factor,
                device=device,
                dtype=dtype,
            )
        elif router_type == "switch":
            self.router = SwitchRouter(
                d_model=d_model,
                num_experts=num_experts,
                capacity_factor=capacity_factor,
                jitter_noise=jitter_noise,
                device=device,
                dtype=dtype,
            )
        else:
            raise ValueError(f"Unknown router type: {router_type}")
            
        # Load balancer
        self.load_balancer = LoadBalancer(
            num_experts=num_experts,
            balance_loss_weight=balance_loss_weight,
            z_loss_weight=z_loss_weight,
            device=device,
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_router_logits: bool = False,
        use_parallel_experts: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass of MoE layer with optimizations.
        
        Args:
            hidden_states: Input tensor (batch_size, seq_len, d_model)
            output_router_logits: Whether to output router information
            use_parallel_experts: Whether to use parallelized expert computation
            
        Returns:
            Tuple of (output, router_outputs)
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Store residual for skip connection
        residual = hidden_states
        
        # Pre-normalization
        hidden_states = self.norm(hidden_states)
        
        # Enable automatic mixed precision if available
        with torch.cuda.amp.autocast() if torch.cuda.is_available() else torch.no_grad():
            # Route tokens to experts
            router_outputs = self._route_tokens(hidden_states)
            
            if use_parallel_experts:
                expert_outputs = self._compute_expert_outputs_parallel(hidden_states, router_outputs)
            else:
                expert_outputs = self._compute_expert_outputs(hidden_states, router_outputs)
        
        # Combine expert outputs
        output = self._combine_expert_outputs(expert_outputs, router_outputs, hidden_states.shape)
        
        # Residual connection
        output = residual + output
        
        # Prepare router outputs for loss computation
        if output_router_logits:
            return output, router_outputs
        else:
            return output, None
            
    def _compute_expert_outputs_parallel(
        self, 
        hidden_states: torch.Tensor,
        router_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute expert outputs with better parallelization."""
        if router_outputs["router_type"] == "expert_choice":
            return self._compute_expert_choice_outputs_parallel(hidden_states, router_outputs)
        else:
            return self._compute_top_k_outputs_parallel(hidden_states, router_outputs)
            
    def _compute_top_k_outputs_parallel(
        self,
        hidden_states: torch.Tensor,
        router_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Optimized parallel computation for top-k routing."""
        batch_size, seq_len, d_model = hidden_states.shape
        tokens = hidden_states.view(-1, d_model)
        
        expert_weights = router_outputs["expert_weights"]
        expert_indices = router_outputs["expert_indices"]
        
        # Prepare expert computations in parallel
        all_expert_outputs = []
        expert_token_counts = []
        
        # Collect tokens for each expert
        expert_inputs = [[] for _ in range(self.num_experts)]
        expert_token_maps = [[] for _ in range(self.num_experts)]
        
        for token_idx in range(tokens.shape[0]):
            for k in range(self.top_k):
                expert_idx = expert_indices[token_idx, k].item()
                weight = expert_weights[token_idx, k]
                if weight > 0:
                    expert_inputs[expert_idx].append(tokens[token_idx])
                    expert_token_maps[expert_idx].append((token_idx, k, weight))
        
        # Batch process each expert
        output = torch.zeros_like(tokens)
        
        for expert_idx in range(self.num_experts):
            if len(expert_inputs[expert_idx]) == 0:
                continue
                
            # Stack inputs for batch processing
            expert_batch = torch.stack(expert_inputs[expert_idx])
            expert_output = self.experts[expert_idx](expert_batch)
            
            # Distribute outputs back to tokens
            for i, (token_idx, k, weight) in enumerate(expert_token_maps[expert_idx]):
                output[token_idx] += weight * expert_output[i]
        
        return output.view(batch_size, seq_len, d_model)
        
    def _compute_expert_choice_outputs_parallel(
        self,
        hidden_states: torch.Tensor,
        router_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Optimized parallel computation for expert choice routing."""
        batch_size, seq_len, d_model = hidden_states.shape
        tokens = hidden_states.view(-1, d_model)
        
        expert_weights = router_outputs["expert_weights"]
        expert_assignments = router_outputs["expert_assignments"]
        
        output = torch.zeros_like(tokens)
        
        # Process experts in parallel where possible
        for expert_idx in range(self.num_experts):
            expert_mask = expert_assignments[:, expert_idx] > 0
            if expert_mask.sum() == 0:
                continue
                
            expert_tokens = tokens[expert_mask]
            expert_output = self.experts[expert_idx](expert_tokens)
            
            weights = expert_weights[expert_mask, expert_idx].unsqueeze(-1)
            weighted_output = expert_output * weights
            
            output[expert_mask] += weighted_output
            
        return output.view(batch_size, seq_len, d_model)
            
    def _route_tokens(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Route tokens to appropriate experts."""
        if self.router_type == "expert_choice":
            expert_weights, expert_assignments, capacity = self.router(hidden_states)
            return {
                "expert_weights": expert_weights,
                "expert_assignments": expert_assignments,
                "capacity_per_expert": capacity,
                "router_type": "expert_choice"
            }
        else:
            expert_weights, expert_indices, tokens_per_expert, aux_loss = self.router(
                hidden_states, use_aux_loss=self.training
            )
            return {
                "expert_weights": expert_weights,
                "expert_indices": expert_indices,
                "tokens_per_expert": tokens_per_expert,
                "aux_loss": aux_loss,
                "router_type": self.router_type
            }
            
    def _compute_expert_outputs(
        self, 
        hidden_states: torch.Tensor,
        router_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute outputs from selected experts."""
        batch_size, seq_len, d_model = hidden_states.shape
        
        if router_outputs["router_type"] == "expert_choice":
            return self._compute_expert_choice_outputs(hidden_states, router_outputs)
        else:
            return self._compute_top_k_outputs(hidden_states, router_outputs)
            
    def _compute_expert_choice_outputs(
        self,
        hidden_states: torch.Tensor,
        router_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute outputs for expert choice routing."""
        batch_size, seq_len, d_model = hidden_states.shape
        tokens = hidden_states.view(-1, d_model)
        
        expert_weights = router_outputs["expert_weights"]  # (num_tokens, num_experts)
        expert_assignments = router_outputs["expert_assignments"]  # (num_tokens, num_experts)
        
        # Initialize output
        output = torch.zeros_like(tokens)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Get tokens assigned to this expert
            expert_mask = expert_assignments[:, expert_idx] > 0
            if expert_mask.sum() == 0:
                continue
                
            expert_tokens = tokens[expert_mask]
            expert_output = self.experts[expert_idx](expert_tokens)
            
            # Weight by expert assignment
            weights = expert_weights[expert_mask, expert_idx].unsqueeze(-1)
            weighted_output = expert_output * weights
            
            # Add to final output
            output[expert_mask] += weighted_output
            
        return output.view(batch_size, seq_len, d_model)
        
    def _compute_top_k_outputs(
        self,
        hidden_states: torch.Tensor,
        router_outputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute outputs for top-k routing with parallelization."""
        batch_size, seq_len, d_model = hidden_states.shape
        tokens = hidden_states.view(-1, d_model)  # (num_tokens, d_model)
        
        expert_weights = router_outputs["expert_weights"]  # (num_tokens, top_k)
        expert_indices = router_outputs["expert_indices"]  # (num_tokens, top_k)
        
        # Initialize output
        output = torch.zeros_like(tokens)
        
        # Parallelized expert computation
        # Group tokens by expert to enable batched computation
        for expert_idx in range(self.num_experts):
            # Find all tokens assigned to this expert
            expert_mask = (expert_indices == expert_idx).any(dim=1)
            if not expert_mask.any():
                continue
                
            expert_tokens = tokens[expert_mask]
            if expert_tokens.shape[0] == 0:
                continue
                
            # Batch compute for this expert
            expert_output = self.experts[expert_idx](expert_tokens)
            
            # Apply weights and accumulate
            token_indices = torch.where(expert_mask)[0]
            for i, token_idx in enumerate(token_indices):
                # Find which k position this expert occupies for this token
                expert_positions = torch.where(expert_indices[token_idx] == expert_idx)[0]
                for pos in expert_positions:
                    weight = expert_weights[token_idx, pos]
                    output[token_idx] += weight * expert_output[i]
                    
        return output.view(batch_size, seq_len, d_model)
        
    def _combine_expert_outputs(
        self,
        expert_outputs: torch.Tensor,
        router_outputs: Dict[str, torch.Tensor],
        original_shape: torch.Size
    ) -> torch.Tensor:
        """Combine expert outputs (already done in compute methods)."""
        return expert_outputs
        
    def get_aux_loss(self, router_outputs: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Get auxiliary loss for load balancing."""
        if router_outputs is None or "aux_loss" not in router_outputs:
            return torch.tensor(0.0, device=self.norm.weight.device)
        return router_outputs["aux_loss"]
        
    def get_router_stats(self) -> Dict[str, Any]:
        """Get router statistics for monitoring."""
        return self.load_balancer.get_expert_utilization_stats()


class SparseMoELayer(MoELayer):
    """
    Sparse MoE layer with dynamic expert activation based on input complexity.
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int = 256,
        base_top_k: int = 2,
        max_top_k: int = 8,
        complexity_threshold: float = 0.5,
        **kwargs
    ):
        super().__init__(
            d_model=d_model,
            num_experts=num_experts,
            top_k=base_top_k,
            **kwargs
        )
        self.base_top_k = base_top_k
        self.max_top_k = max_top_k
        self.complexity_threshold = complexity_threshold
        
        # Complexity predictor
        self.complexity_predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )
        
    def _adaptive_top_k(self, hidden_states: torch.Tensor) -> int:
        """Determine top-k based on input complexity."""
        complexity_scores = self.complexity_predictor(hidden_states)
        avg_complexity = complexity_scores.mean()
        
        if avg_complexity > self.complexity_threshold:
            return min(self.max_top_k, self.num_experts)
        else:
            return self.base_top_k
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        output_router_logits: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """Forward pass with adaptive expert selection."""
        # Adapt top-k based on complexity
        original_top_k = self.top_k
        self.top_k = self._adaptive_top_k(hidden_states)
        self.router.top_k = self.top_k
        
        # Call parent forward
        output, router_outputs = super().forward(hidden_states, output_router_logits)
        
        # Restore original top-k
        self.top_k = original_top_k
        self.router.top_k = original_top_k
        
        return output, router_outputs