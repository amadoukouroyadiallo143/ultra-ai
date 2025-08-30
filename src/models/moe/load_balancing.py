import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional


class LoadBalancer(nn.Module):
    """
    Load balancing utilities for Mixture of Experts.
    Implements various strategies to ensure balanced expert utilization.
    """
    
    def __init__(
        self,
        num_experts: int,
        balance_loss_weight: float = 0.01,
        z_loss_weight: float = 0.001,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.balance_loss_weight = balance_loss_weight
        self.z_loss_weight = z_loss_weight
        
        # Expert usage statistics
        self.register_buffer(
            'expert_usage_count', 
            torch.zeros(num_experts, device=device)
        )
        self.register_buffer(
            'total_tokens_processed', 
            torch.tensor(0.0, device=device)
        )
        
    def compute_balance_loss(
        self,
        gate_probs: torch.Tensor,
        expert_indices: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """
        Compute load balancing loss to encourage equal expert utilization.
        
        Args:
            gate_probs: Gate probabilities (num_tokens, num_experts)
            expert_indices: Selected expert indices (num_tokens, top_k)
            tokens_per_expert: Number of tokens assigned to each expert
            num_tokens: Total number of tokens
            
        Returns:
            Balance loss tensor
        """
        # Method 1: Standard load balancing loss
        fraction_per_expert = tokens_per_expert / max(num_tokens, 1)
        mean_gate_probs = gate_probs.mean(dim=0)
        balance_loss = (fraction_per_expert * mean_gate_probs).sum() * self.num_experts
        
        return balance_loss * self.balance_loss_weight
        
    def compute_z_loss(self, gate_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute Z-loss to encourage sparsity in gate activations.
        
        Args:
            gate_logits: Raw gate logits (num_tokens, num_experts)
            
        Returns:
            Z-loss tensor
        """
        # Z-loss encourages the squared log-sum-exp to be small
        log_sum_exp = torch.logsumexp(gate_logits, dim=-1)
        z_loss = (log_sum_exp ** 2).mean()
        
        return z_loss * self.z_loss_weight
        
    def compute_router_loss(
        self,
        gate_logits: torch.Tensor,
        gate_probs: torch.Tensor,
        expert_indices: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        num_tokens: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive router loss including balance and z-loss.
        
        Returns:
            Dictionary containing different loss components
        """
        losses = {}
        
        # Balance loss
        balance_loss = self.compute_balance_loss(
            gate_probs, expert_indices, tokens_per_expert, num_tokens
        )
        losses['balance_loss'] = balance_loss
        
        # Z-loss for sparsity
        z_loss = self.compute_z_loss(gate_logits)
        losses['z_loss'] = z_loss
        
        # Total router loss
        total_loss = balance_loss + z_loss
        losses['router_loss'] = total_loss
        
        return losses
        
    def update_expert_usage(
        self,
        expert_indices: torch.Tensor,
        num_tokens: int,
    ):
        """Update expert usage statistics for monitoring."""
        if self.training:
            # Count how many tokens went to each expert
            for expert_idx in range(self.num_experts):
                count = (expert_indices == expert_idx).sum().float()
                self.expert_usage_count[expert_idx] += count
                
            self.total_tokens_processed += num_tokens
            
    def get_expert_utilization_stats(self) -> Dict[str, Any]:
        """Get statistics about expert utilization."""
        if self.total_tokens_processed == 0:
            return {
                'expert_usage_fraction': torch.zeros(self.num_experts),
                'usage_variance': 0.0,
                'max_usage': 0.0,
                'min_usage': 0.0,
            }
            
        # Compute usage fractions
        usage_fractions = self.expert_usage_count / self.total_tokens_processed
        
        stats = {
            'expert_usage_fraction': usage_fractions,
            'usage_variance': usage_fractions.var().item(),
            'max_usage': usage_fractions.max().item(),
            'min_usage': usage_fractions.min().item(),
            'usage_std': usage_fractions.std().item(),
            'ideal_usage': 1.0 / self.num_experts,
        }
        
        return stats
        
    def reset_stats(self):
        """Reset usage statistics."""
        self.expert_usage_count.zero_()
        self.total_tokens_processed.zero_()


class DynamicLoadBalancer(LoadBalancer):
    """
    Dynamic load balancer that adjusts routing based on expert performance.
    """
    
    def __init__(
        self,
        num_experts: int,
        balance_loss_weight: float = 0.01,
        z_loss_weight: float = 0.001,
        adaptation_rate: float = 0.001,
        device: Optional[torch.device] = None,
    ):
        super().__init__(num_experts, balance_loss_weight, z_loss_weight, device)
        self.adaptation_rate = adaptation_rate
        
        # Expert performance tracking
        self.register_buffer(
            'expert_performance', 
            torch.ones(num_experts, device=device)
        )
        self.register_buffer(
            'expert_update_count',
            torch.zeros(num_experts, device=device)
        )
        
    def update_expert_performance(
        self,
        expert_indices: torch.Tensor,
        expert_outputs: torch.Tensor,
        targets: torch.Tensor,
        loss_fn: Optional[callable] = None,
    ):
        """Update expert performance based on their outputs."""
        if loss_fn is None:
            loss_fn = nn.MSELoss(reduction='none')
            
        # Compute per-token losses
        token_losses = loss_fn(expert_outputs, targets)
        if token_losses.dim() > 1:
            token_losses = token_losses.mean(dim=list(range(1, token_losses.dim())))
            
        # Update expert performance
        for expert_idx in range(self.num_experts):
            expert_mask = (expert_indices == expert_idx)
            if expert_mask.sum() > 0:
                expert_loss = token_losses[expert_mask].mean()
                
                # Exponential moving average update
                alpha = self.adaptation_rate
                self.expert_performance[expert_idx] = (
                    (1 - alpha) * self.expert_performance[expert_idx] +
                    alpha * (1.0 / (1.0 + expert_loss))  # Higher performance = lower loss
                )
                self.expert_update_count[expert_idx] += 1
                
    def get_expert_weights(self) -> torch.Tensor:
        """Get expert weights based on performance for routing adjustment."""
        # Normalize performance scores
        normalized_performance = self.expert_performance / self.expert_performance.sum()
        return normalized_performance
        
    def compute_performance_adjusted_loss(
        self,
        gate_probs: torch.Tensor,
        expert_indices: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        num_tokens: int,
    ) -> torch.Tensor:
        """Compute load balancing loss adjusted for expert performance."""
        # Get expert weights based on performance
        expert_weights = self.get_expert_weights()
        
        # Adjust target distribution based on performance
        target_fraction = expert_weights
        actual_fraction = tokens_per_expert / max(num_tokens, 1)
        
        # Compute weighted balance loss
        balance_loss = ((actual_fraction - target_fraction) ** 2).sum()
        
        return balance_loss * self.balance_loss_weight


class AdaptiveCapacityBalancer(LoadBalancer):
    """
    Load balancer with adaptive expert capacity based on demand.
    """
    
    def __init__(
        self,
        num_experts: int,
        initial_capacity_factor: float = 1.25,
        min_capacity_factor: float = 0.5,
        max_capacity_factor: float = 2.0,
        capacity_adaptation_rate: float = 0.01,
        balance_loss_weight: float = 0.01,
        z_loss_weight: float = 0.001,
        device: Optional[torch.device] = None,
    ):
        super().__init__(num_experts, balance_loss_weight, z_loss_weight, device)
        self.min_capacity_factor = min_capacity_factor
        self.max_capacity_factor = max_capacity_factor
        self.capacity_adaptation_rate = capacity_adaptation_rate
        
        # Per-expert capacity factors
        self.register_buffer(
            'expert_capacity_factors',
            torch.full((num_experts,), initial_capacity_factor, device=device)
        )
        
    def adapt_expert_capacities(
        self,
        tokens_per_expert: torch.Tensor,
        num_tokens: int,
    ):
        """Adapt expert capacities based on demand."""
        if num_tokens == 0:
            return
            
        # Compute demand ratio for each expert
        average_tokens = num_tokens / self.num_experts
        demand_ratios = tokens_per_expert / max(average_tokens, 1)
        
        # Adapt capacity factors
        for expert_idx in range(self.num_experts):
            current_factor = self.expert_capacity_factors[expert_idx]
            demand_ratio = demand_ratios[expert_idx]
            
            # Increase capacity for overloaded experts
            if demand_ratio > 1.2:
                new_factor = current_factor * (1 + self.capacity_adaptation_rate)
            # Decrease capacity for underutilized experts
            elif demand_ratio < 0.8:
                new_factor = current_factor * (1 - self.capacity_adaptation_rate)
            else:
                new_factor = current_factor
                
            # Clamp to valid range
            new_factor = torch.clamp(
                new_factor, 
                self.min_capacity_factor, 
                self.max_capacity_factor
            )
            
            self.expert_capacity_factors[expert_idx] = new_factor
            
    def get_expert_capacities(self, base_capacity: int) -> torch.Tensor:
        """Get per-expert capacities."""
        return (self.expert_capacity_factors * base_capacity).long()