import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List, Tuple
import math
import logging

logger = logging.getLogger(__name__)


class UltraAILoss(nn.Module):
    """
    Comprehensive loss function for Ultra-AI multimodal model.
    Combines multiple objectives: language modeling, diffusion, contrastive, and auxiliary losses.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        vocab_size: int = 50432,
        temperature: float = 0.07,
        lambda_lm: float = 1.0,
        lambda_diffusion: float = 1.0,
        lambda_contrastive: float = 0.1,
        lambda_moe: float = 0.01,
        lambda_router: float = 0.01,
        label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        self.temperature = temperature
        
        # Loss weights
        self.lambda_lm = lambda_lm
        self.lambda_diffusion = lambda_diffusion
        self.lambda_contrastive = lambda_contrastive
        self.lambda_moe = lambda_moe
        self.lambda_router = lambda_router
        
        # Language modeling loss
        self.lm_loss_fn = nn.CrossEntropyLoss(
            ignore_index=-100,
            label_smoothing=label_smoothing,
        )
        
        # Diffusion loss
        self.diffusion_loss_fn = nn.MSELoss()
        
        # Contrastive loss
        self.contrastive_loss_fn = nn.CrossEntropyLoss()
        
        # Regularization losses
        self.l2_loss = nn.MSELoss()
        
    def forward(
        self, 
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive loss across all objectives.
        
        Args:
            outputs: Model outputs dictionary
            batch: Input batch dictionary
            
        Returns:
            Dictionary of loss components
        """
        losses = {}
        total_loss = torch.tensor(0.0, device=next(iter(outputs.values())).device)
        
        # Language modeling loss
        if "text_logits" in outputs and "input_ids" in batch:
            lm_loss = self._compute_language_modeling_loss(outputs, batch)
            losses["lm_loss"] = lm_loss
            total_loss += self.lambda_lm * lm_loss
            
        # Diffusion loss for image generation
        if "noise_pred" in outputs and "target_noise" in outputs:
            diffusion_loss = self._compute_diffusion_loss(outputs)
            losses["diffusion_loss"] = diffusion_loss
            total_loss += self.lambda_diffusion * diffusion_loss
            
        # Contrastive loss for multimodal alignment
        if "projected_features" in outputs:
            contrastive_loss = self._compute_contrastive_loss(outputs)
            losses["contrastive_loss"] = contrastive_loss
            total_loss += self.lambda_contrastive * contrastive_loss
            
        # MoE auxiliary losses
        if "router_outputs" in outputs:
            moe_losses = self._compute_moe_losses(outputs["router_outputs"])
            losses.update(moe_losses)
            
            if "aux_loss" in moe_losses:
                total_loss += self.lambda_moe * moe_losses["aux_loss"]
            if "router_z_loss" in moe_losses:
                total_loss += self.lambda_router * moe_losses["router_z_loss"]
                
        # Regularization losses
        reg_losses = self._compute_regularization_losses(outputs)
        losses.update(reg_losses)
        total_loss += sum(reg_losses.values())
        
        losses["total_loss"] = total_loss
        
        return losses
        
    def _compute_language_modeling_loss(
        self, 
        outputs: Dict[str, torch.Tensor],
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute causal language modeling loss."""
        logits = outputs["text_logits"]  # (batch, seq_len, vocab_size)
        labels = batch["input_ids"]      # (batch, seq_len)
        
        # Shift logits and labels for causal modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Flatten for cross entropy
        shift_logits = shift_logits.view(-1, self.vocab_size)
        shift_labels = shift_labels.view(-1)
        
        # Apply attention mask if available
        if "attention_mask" in batch:
            attention_mask = batch["attention_mask"][..., 1:].contiguous()
            mask = attention_mask.view(-1).bool()
            
            if mask.any():
                shift_logits = shift_logits[mask]
                shift_labels = shift_labels[mask]
            else:
                return torch.tensor(0.0, device=logits.device)
                
        # Compute loss
        loss = self.lm_loss_fn(shift_logits, shift_labels)
        
        return loss
        
    def _compute_diffusion_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute diffusion loss for image generation."""
        noise_pred = outputs["noise_pred"]
        target_noise = outputs["target_noise"]
        
        # MSE loss between predicted and target noise
        loss = self.diffusion_loss_fn(noise_pred, target_noise)
        
        return loss
        
    def _compute_contrastive_loss(self, outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute contrastive loss for multimodal alignment."""
        projected_features = outputs["projected_features"]
        
        if len(projected_features) < 2:
            return torch.tensor(0.0, device=next(iter(projected_features.values())).device)
            
        # Get feature pairs
        modalities = list(projected_features.keys())
        total_loss = torch.tensor(0.0, device=projected_features[modalities[0]].device)
        num_pairs = 0
        
        for i in range(len(modalities)):
            for j in range(i + 1, len(modalities)):
                mod1, mod2 = modalities[i], modalities[j]
                feat1, feat2 = projected_features[mod1], projected_features[mod2]
                
                # Ensure features are normalized
                feat1 = F.normalize(feat1, p=2, dim=-1)
                feat2 = F.normalize(feat2, p=2, dim=-1)
                
                # Compute similarity matrix
                similarity = torch.matmul(feat1, feat2.t()) / self.temperature
                
                # Labels for contrastive learning (positive pairs on diagonal)
                batch_size = feat1.shape[0]
                labels = torch.arange(batch_size, device=feat1.device)
                
                # Symmetric contrastive loss
                loss_12 = self.contrastive_loss_fn(similarity, labels)
                loss_21 = self.contrastive_loss_fn(similarity.t(), labels)
                
                pair_loss = (loss_12 + loss_21) / 2
                total_loss += pair_loss
                num_pairs += 1
                
        return total_loss / max(num_pairs, 1)
        
    def _compute_moe_losses(self, router_outputs: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Compute MoE auxiliary losses for load balancing."""
        moe_losses = {}
        
        if not router_outputs:
            return moe_losses
            
        # Aggregate losses from all MoE layers
        total_aux_loss = torch.tensor(0.0)
        total_router_z_loss = torch.tensor(0.0)
        num_layers = 0
        
        for router_output in router_outputs:
            if router_output is None:
                continue
                
            if "aux_loss" in router_output:
                total_aux_loss += router_output["aux_loss"]
                num_layers += 1
                
            if "router_z_loss" in router_output:
                total_router_z_loss += router_output["router_z_loss"]
                
        if num_layers > 0:
            moe_losses["aux_loss"] = total_aux_loss / num_layers
            moe_losses["router_z_loss"] = total_router_z_loss / num_layers
            
        return moe_losses
        
    def _compute_regularization_losses(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute regularization losses."""
        reg_losses = {}
        
        # Activity regularization for sparse activations
        if "hidden_states" in outputs:
            hidden_states = outputs["hidden_states"]
            l1_activity = torch.mean(torch.abs(hidden_states))
            reg_losses["l1_activity"] = 0.01 * l1_activity
            
        # Gradient penalty for stable training
        if "gradient_penalty" in outputs:
            reg_losses["gradient_penalty"] = 0.1 * outputs["gradient_penalty"]
            
        return reg_losses


class MultiObjectiveLoss(nn.Module):
    """
    Multi-objective loss with adaptive weighting based on task performance.
    """
    
    def __init__(
        self,
        objectives: List[str],
        initial_weights: Optional[Dict[str, float]] = None,
        adaptation_rate: float = 0.01,
        min_weight: float = 0.01,
        max_weight: float = 10.0,
    ):
        super().__init__()
        self.objectives = objectives
        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Initialize weights
        if initial_weights is None:
            initial_weights = {obj: 1.0 for obj in objectives}
            
        self.weights = nn.ParameterDict({
            obj: nn.Parameter(torch.tensor(initial_weights.get(obj, 1.0)))
            for obj in objectives
        })
        
        # Track loss history for adaptive weighting
        self.register_buffer("loss_history", torch.zeros(len(objectives), 100))
        self.register_buffer("history_index", torch.tensor(0))
        
    def forward(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute weighted multi-objective loss.
        
        Args:
            losses: Dictionary of individual loss components
            
        Returns:
            Weighted total loss
        """
        total_loss = torch.tensor(0.0, device=next(iter(losses.values())).device)
        current_losses = []
        
        for i, objective in enumerate(self.objectives):
            if objective in losses:
                loss_value = losses[objective]
                weight = torch.clamp(self.weights[objective], self.min_weight, self.max_weight)
                
                weighted_loss = weight * loss_value
                total_loss += weighted_loss
                
                current_losses.append(loss_value.detach())
            else:
                current_losses.append(torch.tensor(0.0))
                
        # Update loss history
        if self.training:
            self._update_loss_history(current_losses)
            self._adapt_weights()
            
        return total_loss
        
    def _update_loss_history(self, losses: List[torch.Tensor]):
        """Update loss history for adaptive weighting."""
        history_idx = self.history_index % self.loss_history.shape[1]
        
        for i, loss in enumerate(losses):
            self.loss_history[i, history_idx] = loss.item()
            
        self.history_index += 1
        
    def _adapt_weights(self):
        """Adapt weights based on loss trends."""
        if self.history_index < 10:  # Need some history
            return
            
        # Compute recent loss trends
        recent_window = min(20, self.history_index.item())
        start_idx = max(0, (self.history_index - recent_window) % self.loss_history.shape[1])
        end_idx = self.history_index % self.loss_history.shape[1]
        
        for i, objective in enumerate(self.objectives):
            if start_idx < end_idx:
                recent_losses = self.loss_history[i, start_idx:end_idx]
            else:
                recent_losses = torch.cat([
                    self.loss_history[i, start_idx:],
                    self.loss_history[i, :end_idx]
                ])
                
            # Compute trend (positive = increasing loss)
            if len(recent_losses) > 1:
                trend = torch.mean(recent_losses[-10:]) - torch.mean(recent_losses[:10])
                
                # Increase weight if loss is increasing (task is struggling)
                if trend > 0:
                    self.weights[objective].data *= (1 + self.adaptation_rate)
                else:
                    self.weights[objective].data *= (1 - self.adaptation_rate)
                    
                # Clamp weights
                self.weights[objective].data.clamp_(self.min_weight, self.max_weight)


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    """
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Predictions (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
            
        Returns:
            Focal loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    """
    
    def __init__(self, smooth: float = 1e-7):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute Dice loss.
        
        Args:
            inputs: Predictions (batch_size, num_classes, ...)
            targets: Ground truth (batch_size, num_classes, ...)
            
        Returns:
            Dice loss
        """
        inputs = F.softmax(inputs, dim=1)
        
        # Flatten spatial dimensions
        inputs = inputs.view(inputs.size(0), inputs.size(1), -1)
        targets = targets.view(targets.size(0), targets.size(1), -1)
        
        # Compute Dice coefficient
        intersection = (inputs * targets).sum(dim=2)
        dice = (2.0 * intersection + self.smooth) / (inputs.sum(dim=2) + targets.sum(dim=2) + self.smooth)
        
        # Return 1 - Dice as loss
        return 1.0 - dice.mean()


class PerceptualLoss(nn.Module):
    """
    Perceptual Loss using pre-trained VGG features.
    """
    
    def __init__(self, feature_layers: List[str] = ["conv1_2", "conv2_2", "conv3_3", "conv4_3"]):
        super().__init__()
        try:
            import torchvision.models as models
            vgg = models.vgg19(pretrained=True).features
            self.feature_layers = feature_layers
            
            # Extract specific layers
            self.layers = nn.ModuleDict()
            layer_names = {
                '1': 'conv1_1', '3': 'conv1_2', '6': 'conv2_1', '8': 'conv2_2',
                '11': 'conv3_1', '13': 'conv3_2', '15': 'conv3_3', '17': 'conv3_4',
                '20': 'conv4_1', '22': 'conv4_2', '24': 'conv4_3', '26': 'conv4_4',
                '29': 'conv5_1', '31': 'conv5_2', '33': 'conv5_3', '35': 'conv5_4'
            }
            
            for idx, layer in enumerate(vgg):
                name = layer_names.get(str(idx))
                if name in feature_layers:
                    self.layers[name] = layer
                    
            # Freeze VGG parameters
            for param in self.parameters():
                param.requires_grad = False
                
            self.mse_loss = nn.MSELoss()
            
        except ImportError:
            logger.warning("torchvision not available, perceptual loss disabled")
            self.layers = None
            
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute perceptual loss.
        
        Args:
            inputs: Generated images (batch_size, 3, height, width)
            targets: Target images (batch_size, 3, height, width)
            
        Returns:
            Perceptual loss
        """
        if self.layers is None:
            return torch.tensor(0.0, device=inputs.device)
            
        total_loss = torch.tensor(0.0, device=inputs.device)
        
        input_features = self._extract_features(inputs)
        target_features = self._extract_features(targets)
        
        for layer_name in self.feature_layers:
            if layer_name in input_features and layer_name in target_features:
                layer_loss = self.mse_loss(input_features[layer_name], target_features[layer_name])
                total_loss += layer_loss
                
        return total_loss / len(self.feature_layers)
        
    def _extract_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract VGG features."""
        features = {}
        
        for name, layer in self.layers.items():
            x = layer(x)
            features[name] = x
            
        return features


class GradientPenalty(nn.Module):
    """
    Gradient Penalty for Wasserstein GANs and stable training.
    """
    
    def __init__(self, lambda_gp: float = 10.0):
        super().__init__()
        self.lambda_gp = lambda_gp
        
    def forward(
        self, 
        discriminator: nn.Module,
        real_samples: torch.Tensor,
        fake_samples: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gradient penalty.
        
        Args:
            discriminator: Discriminator network
            real_samples: Real data samples
            fake_samples: Generated samples
            
        Returns:
            Gradient penalty loss
        """
        batch_size = real_samples.size(0)
        device = real_samples.device
        
        # Random interpolation
        alpha = torch.rand(batch_size, 1, device=device)
        while alpha.dim() < real_samples.dim():
            alpha = alpha.unsqueeze(-1)
            
        interpolated = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated.requires_grad_(True)
        
        # Compute discriminator output
        d_interpolated = discriminator(interpolated)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        
        # Compute gradient penalty
        gradient_norm = gradients.view(batch_size, -1).norm(2, dim=1)
        penalty = ((gradient_norm - 1) ** 2).mean()
        
        return self.lambda_gp * penalty