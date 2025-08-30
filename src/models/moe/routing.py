import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math


class TopKRouter(nn.Module):
    """
    Top-K routing for Mixture of Experts.
    Routes tokens to top-k experts based on learned gating function.
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        jitter_noise: float = 0.01,
        normalize_gates: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.jitter_noise = jitter_noise
        self.normalize_gates = normalize_gates
        
        # Gating network
        self.gate = nn.Linear(d_model, num_experts, bias=False, device=device, dtype=dtype)
        
        # Initialize gate weights
        nn.init.normal_(self.gate.weight, 0.0, 0.02)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        use_aux_loss: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Route tokens to experts.
        
        Args:
            hidden_states: Input tokens (batch_size, seq_len, d_model)
            use_aux_loss: Whether to compute auxiliary load balancing loss
            
        Returns:
            Tuple of (expert_weights, expert_indices, tokens_per_expert, aux_loss)
        """
        batch_size, seq_len, d_model = hidden_states.shape
        num_tokens = batch_size * seq_len
        
        # Reshape for routing
        hidden_states = hidden_states.view(-1, d_model)  # (num_tokens, d_model)
        
        # Compute gate scores
        gate_logits = self.gate(hidden_states)  # (num_tokens, num_experts)
        
        # Add jitter noise during training
        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(gate_logits) * self.jitter_noise
            gate_logits = gate_logits + noise
            
        # Convert to probabilities
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(gate_probs, self.top_k, dim=-1)
        
        # Normalize top-k probabilities if requested
        if self.normalize_gates:
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            
        # Compute expert assignment statistics
        tokens_per_expert = torch.zeros(self.num_experts, device=hidden_states.device)
        for expert_idx in range(self.num_experts):
            tokens_per_expert[expert_idx] = (top_k_indices == expert_idx).sum().float()
            
        # Auxiliary loss for load balancing
        aux_loss = None
        if use_aux_loss:
            aux_loss = self._compute_aux_loss(gate_probs, tokens_per_expert, num_tokens)
            
        return top_k_probs, top_k_indices, tokens_per_expert, aux_loss
        
    def _compute_aux_loss(
        self, 
        gate_probs: torch.Tensor, 
        tokens_per_expert: torch.Tensor,
        num_tokens: int
    ) -> torch.Tensor:
        """Compute auxiliary loss for load balancing."""
        # Fraction of tokens routed to each expert
        fraction_per_expert = tokens_per_expert / num_tokens
        
        # Average gate probability for each expert
        mean_gate_probs = gate_probs.mean(dim=0)
        
        # Load balancing loss
        aux_loss = (fraction_per_expert * mean_gate_probs).sum() * self.num_experts
        
        return aux_loss


class ExpertChoiceRouter(nn.Module):
    """
    Expert Choice routing where experts select their top tokens.
    Ensures perfect load balancing with no auxiliary loss needed.
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        expert_capacity: Optional[int] = None,
        capacity_factor: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.expert_capacity = expert_capacity
        
        # Gating network
        self.gate = nn.Linear(d_model, num_experts, bias=False, device=device, dtype=dtype)
        
        # Expert-specific token selectors
        self.expert_selectors = nn.ModuleList([
            nn.Linear(d_model, 1, bias=False, device=device, dtype=dtype)
            for _ in range(num_experts)
        ])
        
    def forward(
        self, 
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Expert choice routing.
        
        Args:
            hidden_states: Input tokens (batch_size, seq_len, d_model)
            
        Returns:
            Tuple of (expert_weights, selected_tokens, expert_assignments)
        """
        batch_size, seq_len, d_model = hidden_states.shape
        num_tokens = batch_size * seq_len
        
        # Determine expert capacity
        if self.expert_capacity is None:
            capacity_per_expert = int(num_tokens * self.capacity_factor / self.num_experts)
        else:
            capacity_per_expert = self.expert_capacity
            
        # Reshape tokens
        tokens = hidden_states.view(-1, d_model)  # (num_tokens, d_model)
        
        # Each expert selects its top tokens
        expert_assignments = torch.zeros(num_tokens, self.num_experts, device=hidden_states.device)
        expert_weights = torch.zeros(num_tokens, self.num_experts, device=hidden_states.device)
        
        for expert_idx in range(self.num_experts):
            # Expert computes affinity scores for all tokens
            affinity_scores = self.expert_selectors[expert_idx](tokens).squeeze(-1)
            
            # Select top tokens for this expert
            top_scores, top_indices = torch.topk(
                affinity_scores, 
                min(capacity_per_expert, num_tokens), 
                dim=0
            )
            
            # Assign selected tokens to expert
            expert_assignments[top_indices, expert_idx] = 1.0
            expert_weights[top_indices, expert_idx] = F.softmax(top_scores, dim=0)
            
        # Normalize weights across experts for each token
        token_expert_counts = expert_assignments.sum(dim=1, keepdim=True)
        token_expert_counts = torch.clamp(token_expert_counts, min=1.0)
        expert_weights = expert_weights / token_expert_counts
        
        return expert_weights, expert_assignments, capacity_per_expert


class SwitchRouter(nn.Module):
    """
    Switch routing that routes each token to exactly one expert.
    Simpler than Top-K routing but with potential load imbalance.
    """
    
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        capacity_factor: float = 1.25,
        jitter_noise: float = 0.01,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.jitter_noise = jitter_noise
        
        # Gating network
        self.gate = nn.Linear(d_model, num_experts, bias=False, device=device, dtype=dtype)
        
    def forward(
        self, 
        hidden_states: torch.Tensor,
        use_aux_loss: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Switch routing - route each token to single expert.
        
        Args:
            hidden_states: Input tokens (batch_size, seq_len, d_model)
            use_aux_loss: Whether to compute auxiliary load balancing loss
            
        Returns:
            Tuple of (expert_weights, expert_indices, tokens_per_expert, aux_loss)
        """
        batch_size, seq_len, d_model = hidden_states.shape
        num_tokens = batch_size * seq_len
        
        # Reshape for routing
        hidden_states = hidden_states.view(-1, d_model)
        
        # Compute gate scores
        gate_logits = self.gate(hidden_states)
        
        # Add jitter noise during training
        if self.training and self.jitter_noise > 0:
            noise = torch.randn_like(gate_logits) * self.jitter_noise
            gate_logits = gate_logits + noise
            
        # Convert to probabilities and select best expert
        gate_probs = F.softmax(gate_logits, dim=-1)
        expert_weights, expert_indices = torch.max(gate_probs, dim=-1)
        
        # Compute capacity per expert
        capacity_per_expert = int(num_tokens * self.capacity_factor / self.num_experts)
        
        # Count tokens per expert
        tokens_per_expert = torch.zeros(self.num_experts, device=hidden_states.device)
        for expert_idx in range(self.num_experts):
            tokens_per_expert[expert_idx] = (expert_indices == expert_idx).sum().float()
            
        # Drop tokens that exceed expert capacity
        for expert_idx in range(self.num_experts):
            expert_mask = (expert_indices == expert_idx)
            if expert_mask.sum() > capacity_per_expert:
                # Keep only top tokens based on weight
                expert_positions = torch.where(expert_mask)[0]
                expert_token_weights = expert_weights[expert_positions]
                
                # Select top tokens
                _, top_token_indices = torch.topk(
                    expert_token_weights, 
                    capacity_per_expert
                )
                
                # Drop tokens that didn't make the cut
                keep_mask = torch.zeros_like(expert_mask)
                keep_mask[expert_positions[top_token_indices]] = True
                expert_weights = expert_weights * keep_mask.float()
                
        # Auxiliary loss
        aux_loss = None
        if use_aux_loss:
            mean_gate_probs = gate_probs.mean(dim=0)
            fraction_per_expert = tokens_per_expert / num_tokens
            aux_loss = (fraction_per_expert * mean_gate_probs).sum() * self.num_experts
            
        return expert_weights.unsqueeze(-1), expert_indices.unsqueeze(-1), tokens_per_expert, aux_loss