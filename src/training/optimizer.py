import torch
from torch import nn
from torch.optim import Optimizer
from typing import Dict, Any, List, Optional, Tuple
import math
import logging

logger = logging.getLogger(__name__)


def get_optimizer(model: nn.Module, config: Dict[str, Any]) -> Optimizer:
    """
    Get optimizer based on configuration.
    Supports advanced optimizers for ultra-large model training.
    """
    optimizer_name = config.get("optimizer", "adamw").lower()
    learning_rate = config.get("learning_rate", 1e-4)
    weight_decay = config.get("weight_decay", 0.01)
    
    # Get parameter groups with different settings
    param_groups = get_parameter_groups(model, config)
    
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(config.get("beta1", 0.9), config.get("beta2", 0.95)),
            eps=config.get("eps", 1e-8),
        )
    elif optimizer_name == "lion":
        optimizer = Lion(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(config.get("beta1", 0.9), config.get("beta2", 0.99)),
        )
    elif optimizer_name == "sophia":
        optimizer = SophiaG(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(config.get("beta1", 0.965), config.get("beta2", 0.99)),
        )
    elif optimizer_name == "galore":
        optimizer = GaLoreAdamW(
            param_groups,
            lr=learning_rate,
            weight_decay=weight_decay,
            rank=config.get("galore_rank", 256),
            scale=config.get("galore_scale", 0.25),
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
        
    logger.info(f"Using optimizer: {optimizer.__class__.__name__}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Weight decay: {weight_decay}")
    
    return optimizer


def get_parameter_groups(model: nn.Module, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create parameter groups with different learning rates and weight decay.
    """
    no_decay_modules = ["LayerNorm", "RMSNorm", "bias", "embedding"]
    large_lr_modules = ["attention", "ffn", "moe"]
    small_lr_modules = ["embedding", "position"]
    
    param_groups = []
    
    # Default parameters
    default_params = []
    # No weight decay parameters
    no_decay_params = []
    # Large learning rate parameters
    large_lr_params = []
    # Small learning rate parameters  
    small_lr_params = []
    
    base_lr = config.get("learning_rate", 1e-4)
    large_lr_multiplier = config.get("large_lr_multiplier", 1.0)
    small_lr_multiplier = config.get("small_lr_multiplier", 0.1)
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
            
        # Check if parameter should have no weight decay
        no_decay = any(nd in name.lower() for nd in no_decay_modules)
        
        # Check if parameter should have different learning rate
        large_lr = any(lg in name.lower() for lg in large_lr_modules)
        small_lr = any(sm in name.lower() for sm in small_lr_modules)
        
        if no_decay:
            no_decay_params.append(param)
        elif large_lr:
            large_lr_params.append(param)
        elif small_lr:
            small_lr_params.append(param)
        else:
            default_params.append(param)
            
    # Create parameter groups
    if default_params:
        param_groups.append({
            "params": default_params,
            "lr": base_lr,
            "weight_decay": config.get("weight_decay", 0.01),
        })
        
    if no_decay_params:
        param_groups.append({
            "params": no_decay_params,
            "lr": base_lr,
            "weight_decay": 0.0,
        })
        
    if large_lr_params:
        param_groups.append({
            "params": large_lr_params,
            "lr": base_lr * large_lr_multiplier,
            "weight_decay": config.get("weight_decay", 0.01),
        })
        
    if small_lr_params:
        param_groups.append({
            "params": small_lr_params,
            "lr": base_lr * small_lr_multiplier,
            "weight_decay": config.get("weight_decay", 0.01) * 0.1,
        })
        
    logger.info(f"Created {len(param_groups)} parameter groups")
    for i, group in enumerate(param_groups):
        num_params = sum(p.numel() for p in group["params"])
        logger.info(f"Group {i}: {num_params:,} parameters, lr={group['lr']:.2e}, wd={group['weight_decay']}")
        
    return param_groups


class Lion(Optimizer):
    """
    Lion optimizer implementation.
    More memory efficient than AdamW for large models.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.01,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")
                    
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    
                exp_avg = state["exp_avg"]
                beta1, beta2 = group["betas"]
                state["step"] += 1
                
                # Weight decay
                if group["weight_decay"] > 0:
                    p.data.mul_(1 - group["lr"] * group["weight_decay"])
                    
                # Lion update
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.data.add_(torch.sign(update), alpha=-group["lr"])
                
                # Update exponential moving average
                exp_avg.mul_(beta2).add_(grad, alpha=(1 - beta2))
                
        return loss


class SophiaG(Optimizer):
    """
    Sophia-G optimizer for large language model training.
    Uses second-order information for better convergence.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.965, 0.99),
        rho: float = 0.04,
        weight_decay: float = 0.01,
        maximize: bool = False,
        capturable: bool = False,
    ):
        defaults = dict(
            lr=lr, betas=betas, rho=rho, weight_decay=weight_decay,
            maximize=maximize, capturable=capturable
        )
        super().__init__(params, defaults)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            hut_traces = []
            state_steps = []
            
            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    grads.append(p.grad if not group["maximize"] else -p.grad)
                    
                    state = self.state[p]
                    if len(state) == 0:
                        state["step"] = torch.tensor(0.)
                        state["exp_avg"] = torch.zeros_like(p)
                        state["hessian_diag"] = torch.zeros_like(p)
                        
                    exp_avgs.append(state["exp_avg"])
                    hut_traces.append(state["hessian_diag"])
                    state_steps.append(state["step"])
                    
            self._single_tensor_sophia(
                params_with_grad,
                grads,
                exp_avgs,
                hut_traces,
                state_steps,
                beta1=group["betas"][0],
                beta2=group["betas"][1],
                rho=group["rho"],
                lr=group["lr"],
                weight_decay=group["weight_decay"],
            )
            
        return loss
        
    def _single_tensor_sophia(
        self,
        params: List[torch.Tensor],
        grads: List[torch.Tensor],
        exp_avgs: List[torch.Tensor],
        hut_traces: List[torch.Tensor],
        state_steps: List[torch.Tensor],
        beta1: float,
        beta2: float,
        rho: float,
        lr: float,
        weight_decay: float,
    ):
        for param, grad, exp_avg, hut_trace, state_step in zip(
            params, grads, exp_avgs, hut_traces, state_steps
        ):
            state_step += 1
            
            # Weight decay
            if weight_decay > 0:
                param.mul_(1 - lr * weight_decay)
                
            # Update biased first moment estimate
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            
            # Update Hutchinson trace estimator
            if state_step % 2 == 1:  # Odd steps: compute diagonal Hessian
                # Use gradient norm as proxy for diagonal Hessian
                hut_trace.mul_(beta2).add_(grad.pow(2), alpha=1 - beta2)
            else:  # Even steps: use previous estimate
                pass
                
            # Bias correction
            bias_correction1 = 1 - beta1 ** state_step
            bias_correction2 = 1 - beta2 ** state_step
            
            corrected_exp_avg = exp_avg / bias_correction1
            corrected_hut_trace = hut_trace / bias_correction2
            
            # Clip update
            denominator = torch.clamp(corrected_hut_trace, min=rho)
            update = corrected_exp_avg / denominator
            
            param.add_(update, alpha=-lr)


class GaLoreAdamW(Optimizer):
    """
    GaLore (Gradient Low-Rank Projection) optimizer.
    Reduces memory usage by projecting gradients to low-rank space.
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        rank: int = 256,
        scale: float = 0.25,
        proj_type: str = "std",  # "std", "reverse_std", "right", "left"
        update_proj_gap: int = 200,
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            rank=rank, scale=scale, proj_type=proj_type,
            update_proj_gap=update_proj_gap
        )
        super().__init__(params, defaults)
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                    
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    
                    # Determine if parameter is 2D (matrix)
                    if len(p.shape) >= 2:
                        state["rank"] = min(group["rank"], min(p.shape))
                        state["use_galore"] = True
                        
                        # Initialize projection matrices
                        if group["proj_type"] in ["std", "reverse_std"]:
                            state["projector_left"] = None
                            state["projector_right"] = None
                        else:
                            state["projector"] = None
                            
                        # Low-rank gradient states
                        if group["proj_type"] == "std":
                            proj_shape = (state["rank"], p.shape[1])
                        elif group["proj_type"] == "reverse_std":
                            proj_shape = (p.shape[0], state["rank"])
                        elif group["proj_type"] == "right":
                            proj_shape = (p.shape[0], state["rank"])
                        elif group["proj_type"] == "left":
                            proj_shape = (state["rank"], p.shape[1])
                        else:
                            raise ValueError(f"Unknown proj_type: {group['proj_type']}")
                            
                        state["exp_avg_proj"] = torch.zeros(proj_shape, device=p.device, dtype=p.dtype)
                        state["exp_avg_sq_proj"] = torch.zeros(proj_shape, device=p.device, dtype=p.dtype)
                    else:
                        # 1D parameter - use regular AdamW
                        state["use_galore"] = False
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        
                state["step"] += 1
                step = state["step"]
                
                if state["use_galore"]:
                    # Update projection matrices
                    if step % group["update_proj_gap"] == 1 or step == 1:
                        self._update_projectors(state, grad, group["proj_type"])
                        
                    # Project gradient
                    grad_proj = self._project_gradient(grad, state, group["proj_type"])
                    
                    # Update projected moments
                    beta1, beta2 = group["betas"]
                    state["exp_avg_proj"].mul_(beta1).add_(grad_proj, alpha=1 - beta1)
                    state["exp_avg_sq_proj"].mul_(beta2).addcmul_(grad_proj, grad_proj, value=1 - beta2)
                    
                    # Bias correction
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step
                    
                    denom = (state["exp_avg_sq_proj"] / bias_correction2).sqrt_().add_(group["eps"])
                    step_size = group["lr"] / bias_correction1
                    
                    # Compute update in low-rank space
                    update_proj = state["exp_avg_proj"] / denom
                    
                    # Project back to full space
                    update = self._project_back(update_proj, state, group["proj_type"])
                    
                    # Apply weight decay
                    if group["weight_decay"] > 0:
                        p.data.add_(p.data, alpha=-group["weight_decay"] * group["lr"])
                        
                    # Apply update
                    p.data.add_(update, alpha=-step_size * group["scale"])
                    
                else:
                    # Regular AdamW for 1D parameters
                    beta1, beta2 = group["betas"]
                    state["exp_avg"].mul_(beta1).add_(grad, alpha=1 - beta1)
                    state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    
                    bias_correction1 = 1 - beta1 ** step
                    bias_correction2 = 1 - beta2 ** step
                    
                    denom = (state["exp_avg_sq"] / bias_correction2).sqrt_().add_(group["eps"])
                    step_size = group["lr"] / bias_correction1
                    
                    # Apply weight decay
                    if group["weight_decay"] > 0:
                        p.data.add_(p.data, alpha=-group["weight_decay"] * group["lr"])
                        
                    # Apply update
                    p.data.addcdiv_(state["exp_avg"], denom, value=-step_size)
                    
        return loss
        
    def _update_projectors(self, state, grad, proj_type):
        """Update projection matrices using SVD."""
        if proj_type == "std":
            U, _, Vt = torch.linalg.svd(grad, full_matrices=False)
            state["projector_left"] = U[:, :state["rank"]]
            state["projector_right"] = Vt[:state["rank"], :]
        elif proj_type == "reverse_std":
            U, _, Vt = torch.linalg.svd(grad.T, full_matrices=False)
            state["projector_left"] = U[:, :state["rank"]]
            state["projector_right"] = Vt[:state["rank"], :]
        elif proj_type == "right":
            _, _, Vt = torch.linalg.svd(grad, full_matrices=False)
            state["projector"] = Vt[:state["rank"], :]
        elif proj_type == "left":
            U, _, _ = torch.linalg.svd(grad, full_matrices=False)
            state["projector"] = U[:, :state["rank"]]
            
    def _project_gradient(self, grad, state, proj_type):
        """Project gradient to low-rank space."""
        if proj_type == "std":
            return state["projector_left"].T @ grad @ state["projector_right"].T
        elif proj_type == "reverse_std":
            return grad @ state["projector_left"] @ state["projector_right"]
        elif proj_type == "right":
            return grad @ state["projector"].T
        elif proj_type == "left":
            return state["projector"].T @ grad
            
    def _project_back(self, update_proj, state, proj_type):
        """Project update back to full parameter space."""
        if proj_type == "std":
            return state["projector_left"] @ update_proj @ state["projector_right"]
        elif proj_type == "reverse_std":
            return update_proj @ state["projector_right"].T @ state["projector_left"].T
        elif proj_type == "right":
            return update_proj @ state["projector"]
        elif proj_type == "left":
            return state["projector"] @ update_proj


def get_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine",
    num_training_steps: int = 1000,
    num_warmup_steps: int = 100,
    **kwargs
) -> torch.optim.lr_scheduler.LRScheduler:
    """
    Get learning rate scheduler.
    """
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_training_steps - num_warmup_steps,
            eta_min=kwargs.get("min_lr", 0.0),
        )
    elif scheduler_type == "linear":
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=kwargs.get("start_factor", 1.0),
            end_factor=kwargs.get("end_factor", 0.0),
            total_iters=num_training_steps,
        )
    elif scheduler_type == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get("gamma", 0.95),
        )
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get("mode", "min"),
            factor=kwargs.get("factor", 0.5),
            patience=kwargs.get("patience", 10),
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
    # Wrap with warmup if specified
    if num_warmup_steps > 0 and scheduler_type != "plateau":
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=kwargs.get("warmup_start_factor", 0.1),
            end_factor=1.0,
            total_iters=num_warmup_steps,
        )
        
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[num_warmup_steps],
        )
        
    return scheduler