import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
import math
from einops import rearrange


class MultimodalFusion(nn.Module):
    """
    Multimodal fusion layer that combines text, image, audio, and video representations.
    Implements cross-modal attention and adaptive fusion strategies.
    """
    
    def __init__(
        self,
        d_model: int = 2560,
        num_heads: int = 16,
        modalities: List[str] = ["text", "image", "audio", "video"],
        fusion_strategy: str = "cross_attention",
        max_seq_len: int = 100000,  # Support for ultra-long context
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.modalities = modalities
        self.fusion_strategy = fusion_strategy
        self.num_modalities = len(modalities)
        
        # Modality-specific projections
        self.modality_projections = nn.ModuleDict({
            modality: nn.Linear(d_model, d_model, bias=False, device=device, dtype=dtype)
            for modality in modalities
        })
        
        # Cross-modal attention layers
        self.cross_modal_attention = CrossModalAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_modalities=self.num_modalities,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )
        
        # Adaptive fusion network
        self.adaptive_fusion = AdaptiveFusion(
            d_model=d_model,
            num_modalities=self.num_modalities,
            device=device,
            dtype=dtype,
        )
        
        # Modality alignment
        self.modality_alignment = ModalityAlignment(
            d_model=d_model,
            num_modalities=self.num_modalities,
            device=device,
            dtype=dtype,
        )
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model, device=device, dtype=dtype)
        
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        attention_mask: Optional[Dict[str, torch.Tensor]] = None,
        fusion_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fuse multiple modalities into unified representation.
        
        Args:
            inputs: Dictionary with modality names as keys and tensors as values
                   Each tensor shape: (batch_size, seq_len, d_model)
            attention_mask: Optional attention masks per modality
            fusion_weights: Manual fusion weights (batch_size, num_modalities)
            
        Returns:
            Fused multimodal representation (batch_size, max_seq_len, d_model)
        """
        batch_size = next(iter(inputs.values())).shape[0]
        
        # Project each modality
        projected_inputs = {}
        for modality, tensor in inputs.items():
            if modality in self.modality_projections:
                projected_inputs[modality] = self.modality_projections[modality](tensor)
            else:
                projected_inputs[modality] = tensor
                
        # Align modalities to common sequence length
        aligned_inputs = self.modality_alignment(projected_inputs)
        
        # Cross-modal attention
        attended_features = self.cross_modal_attention(
            aligned_inputs, attention_mask=attention_mask
        )
        
        # Adaptive fusion
        if fusion_weights is None:
            fused_output = self.adaptive_fusion(attended_features, aligned_inputs)
        else:
            fused_output = self._manual_fusion(attended_features, fusion_weights)
            
        # Output projection and normalization
        output = self.output_proj(fused_output)
        output = self.norm(output)
        
        return output
        
    def _manual_fusion(
        self, 
        features: Dict[str, torch.Tensor], 
        weights: torch.Tensor
    ) -> torch.Tensor:
        """Manual weighted fusion of modalities."""
        modality_list = list(features.keys())
        stacked_features = torch.stack([features[mod] for mod in modality_list], dim=1)
        
        # Apply fusion weights
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # (batch, modalities, 1, 1)
        weighted_features = stacked_features * weights
        
        # Sum across modalities
        fused = weighted_features.sum(dim=1)
        return fused


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention mechanism for multimodal fusion.
    Allows each modality to attend to all other modalities.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_modalities: int,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.num_modalities = num_modalities
        
        # Multi-head attention for each modality pair
        self.attention_layers = nn.ModuleDict()
        for i in range(num_modalities):
            for j in range(num_modalities):
                if i != j:  # Cross-modal attention (not self-attention)
                    self.attention_layers[f"{i}_{j}"] = nn.MultiheadAttention(
                        d_model, num_heads, dropout=dropout,
                        device=device, dtype=dtype, batch_first=True
                    )
                    
        # Gating mechanism for attention fusion
        self.attention_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model, device=device, dtype=dtype),
            nn.Sigmoid(),
        )
        
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        attention_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Apply cross-modal attention between all modality pairs.
        
        Args:
            inputs: Dictionary of aligned modality features
            attention_mask: Optional attention masks
            
        Returns:
            Dictionary of cross-attended features
        """
        modality_names = list(inputs.keys())
        attended_features = {}
        
        for i, query_modality in enumerate(modality_names):
            query_features = inputs[query_modality]
            cross_attended = []
            
            for j, key_modality in enumerate(modality_names):
                if i != j:  # Cross-modal attention
                    key_features = inputs[key_modality]
                    value_features = key_features
                    
                    # Get attention mask if available
                    mask = None
                    if attention_mask and key_modality in attention_mask:
                        mask = attention_mask[key_modality]
                        
                    # Apply cross-attention
                    attention_layer = self.attention_layers[f"{i}_{j}"]
                    attended, _ = attention_layer(
                        query_features, key_features, value_features,
                        key_padding_mask=mask
                    )
                    cross_attended.append(attended)
                    
            if cross_attended:
                # Combine cross-attended features
                combined_attended = torch.stack(cross_attended, dim=0).mean(dim=0)
                
                # Gate the attended features with original features
                gate_input = torch.cat([query_features, combined_attended], dim=-1)
                gate = self.attention_gate(gate_input)
                
                attended_features[query_modality] = (
                    gate * combined_attended + (1 - gate) * query_features
                )
            else:
                attended_features[query_modality] = query_features
                
        return attended_features


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion mechanism that learns optimal combination weights.
    """
    
    def __init__(
        self,
        d_model: int,
        num_modalities: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_modalities = num_modalities
        
        # Fusion weight predictor
        self.fusion_predictor = nn.Sequential(
            nn.Linear(d_model * num_modalities, d_model, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(d_model, num_modalities, device=device, dtype=dtype),
            nn.Softmax(dim=-1),
        )
        
        # Content-aware fusion
        self.content_fusion = nn.Sequential(
            nn.Linear(d_model * num_modalities, d_model * 2, device=device, dtype=dtype),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model, device=device, dtype=dtype),
        )
        
    def forward(
        self,
        attended_features: Dict[str, torch.Tensor],
        original_features: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Adaptively fuse attended features.
        
        Args:
            attended_features: Cross-attended modality features
            original_features: Original modality features
            
        Returns:
            Fused multimodal representation
        """
        modality_names = list(attended_features.keys())
        batch_size, seq_len, d_model = attended_features[modality_names[0]].shape
        
        # Stack all modality features
        stacked_attended = torch.stack([
            attended_features[mod] for mod in modality_names
        ], dim=1)  # (batch, num_modalities, seq_len, d_model)
        
        stacked_original = torch.stack([
            original_features[mod] for mod in modality_names
        ], dim=1)
        
        # Flatten for fusion weight prediction
        flattened_features = torch.cat([
            stacked_attended.flatten(start_dim=2),  # (batch, num_modalities, seq_len*d_model)
        ], dim=-1).mean(dim=1)  # (batch, num_modalities*d_model)
        
        # Predict fusion weights
        fusion_weights = self.fusion_predictor(flattened_features)  # (batch, num_modalities)
        fusion_weights = fusion_weights.unsqueeze(-1).unsqueeze(-1)  # (batch, num_modalities, 1, 1)
        
        # Weighted fusion
        weighted_attended = (stacked_attended * fusion_weights).sum(dim=1)  # (batch, seq_len, d_model)
        
        # Content-aware fusion
        all_features = torch.cat([
            feat.flatten(start_dim=1) for feat in stacked_attended.transpose(0, 1)
        ], dim=-1)  # (batch, seq_len, num_modalities*d_model)
        
        content_fused = self.content_fusion(all_features)  # (batch, seq_len, d_model)
        
        # Combine weighted and content-aware fusion
        final_output = weighted_attended + content_fused
        
        return final_output


class ModalityAlignment(nn.Module):
    """
    Aligns different modalities to a common sequence length and representation space.
    """
    
    def __init__(
        self,
        d_model: int,
        num_modalities: int,
        max_seq_len: int = 100000,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Adaptive pooling for sequence length alignment
        self.adaptive_pools = nn.ModuleDict()
        
        # Learned interpolation for sequence alignment
        self.interpolation_weights = nn.ParameterDict()
        
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Align all modalities to consistent sequence length.
        
        Args:
            inputs: Dictionary of modality tensors with varying sequence lengths
            
        Returns:
            Dictionary of aligned tensors
        """
        # Find target sequence length (longest sequence or max_seq_len)
        seq_lengths = [tensor.shape[1] for tensor in inputs.values()]
        target_seq_len = min(max(seq_lengths), self.max_seq_len)
        
        aligned_inputs = {}
        
        for modality, tensor in inputs.items():
            batch_size, current_seq_len, d_model = tensor.shape
            
            if current_seq_len == target_seq_len:
                aligned_inputs[modality] = tensor
            elif current_seq_len < target_seq_len:
                # Upsample using learned interpolation
                aligned_inputs[modality] = self._upsample(tensor, target_seq_len)
            else:
                # Downsample using adaptive pooling
                aligned_inputs[modality] = self._downsample(tensor, target_seq_len)
                
        return aligned_inputs
        
    def _upsample(self, tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        """Upsample sequence to target length."""
        # Use linear interpolation
        tensor_t = tensor.transpose(1, 2)  # (batch, d_model, seq_len)
        upsampled = F.interpolate(tensor_t, size=target_len, mode='linear', align_corners=False)
        return upsampled.transpose(1, 2)  # (batch, target_len, d_model)
        
    def _downsample(self, tensor: torch.Tensor, target_len: int) -> torch.Tensor:
        """Downsample sequence to target length."""
        tensor_t = tensor.transpose(1, 2)  # (batch, d_model, seq_len)
        
        # Adaptive average pooling
        downsampled = F.adaptive_avg_pool1d(tensor_t, target_len)
        return downsampled.transpose(1, 2)  # (batch, target_len, d_model)


class HierarchicalFusion(MultimodalFusion):
    """
    Hierarchical multimodal fusion with multiple levels of abstraction.
    """
    
    def __init__(
        self,
        d_model: int = 2560,
        num_heads: int = 16,
        modalities: List[str] = ["text", "image", "audio", "video"],
        num_levels: int = 3,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            modalities=modalities,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )
        self.num_levels = num_levels
        
        # Multi-level fusion layers
        self.fusion_levels = nn.ModuleList([
            MultimodalFusion(
                d_model=d_model,
                num_heads=max(1, num_heads // (2 ** level)),
                modalities=modalities,
                dropout=dropout,
                device=device,
                dtype=dtype,
            )
            for level in range(num_levels)
        ])
        
        # Level aggregation
        self.level_weights = nn.Parameter(
            torch.ones(num_levels, device=device, dtype=dtype) / num_levels
        )
        
    def forward(
        self,
        inputs: Dict[str, torch.Tensor],
        attention_mask: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Hierarchical fusion across multiple abstraction levels.
        """
        level_outputs = []
        
        # Process at each hierarchical level
        for level, fusion_layer in enumerate(self.fusion_levels):
            # Apply different pooling strategies at different levels
            level_inputs = self._prepare_level_inputs(inputs, level)
            level_output = fusion_layer(level_inputs, attention_mask)
            level_outputs.append(level_output)
            
        # Combine levels with learned weights
        stacked_levels = torch.stack(level_outputs, dim=0)  # (num_levels, batch, seq, d_model)
        weights = F.softmax(self.level_weights, dim=0).view(-1, 1, 1, 1)
        
        hierarchical_output = (stacked_levels * weights).sum(dim=0)
        
        return hierarchical_output
        
    def _prepare_level_inputs(
        self, 
        inputs: Dict[str, torch.Tensor], 
        level: int
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs for specific hierarchical level."""
        level_inputs = {}
        
        for modality, tensor in inputs.items():
            if level == 0:
                # Fine-grained level - full resolution
                level_inputs[modality] = tensor
            else:
                # Coarser levels - progressively pool
                pooling_factor = 2 ** level
                pooled = F.avg_pool1d(
                    tensor.transpose(1, 2), 
                    kernel_size=min(pooling_factor, tensor.shape[1]),
                    stride=min(pooling_factor, tensor.shape[1])
                ).transpose(1, 2)
                level_inputs[modality] = pooled
                
        return level_inputs


class ContrastiveFusion(nn.Module):
    """
    Contrastive multimodal fusion using contrastive learning objectives.
    """
    
    def __init__(
        self,
        d_model: int = 2560,
        projection_dim: int = 512,
        temperature: float = 0.07,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.projection_dim = projection_dim
        self.temperature = temperature
        
        # Projection heads for each modality
        self.projections = nn.ModuleDict()
        
        # Contrastive loss
        self.contrastive_loss = nn.CrossEntropyLoss()
        
    def add_modality_projection(self, modality: str, device=None, dtype=None):
        """Add projection head for a new modality."""
        self.projections[modality] = nn.Sequential(
            nn.Linear(self.d_model, self.d_model, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(self.d_model, self.projection_dim, device=device, dtype=dtype),
        )
        
    def forward(
        self,
        features: Dict[str, torch.Tensor],
        return_loss: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Contrastive multimodal fusion.
        
        Args:
            features: Dictionary of modality features
            return_loss: Whether to compute contrastive loss
            
        Returns:
            Dictionary with projected features and optional loss
        """
        projected_features = {}
        
        # Project each modality
        for modality, tensor in features.items():
            if modality in self.projections:
                # Global average pooling for sequence-level representation
                pooled = tensor.mean(dim=1)  # (batch, d_model)
                projected = self.projections[modality](pooled)
                projected = F.normalize(projected, p=2, dim=-1)
                projected_features[modality] = projected
                
        outputs = {"projected_features": projected_features}
        
        if return_loss and len(projected_features) >= 2:
            # Compute contrastive loss between modality pairs
            modality_names = list(projected_features.keys())
            total_loss = 0.0
            num_pairs = 0
            
            for i in range(len(modality_names)):
                for j in range(i + 1, len(modality_names)):
                    mod1, mod2 = modality_names[i], modality_names[j]
                    feat1, feat2 = projected_features[mod1], projected_features[mod2]
                    
                    # Compute similarity matrix
                    similarity = torch.matmul(feat1, feat2.t()) / self.temperature
                    
                    # Labels for contrastive learning (diagonal should be positive pairs)
                    labels = torch.arange(feat1.shape[0], device=feat1.device)
                    
                    # Contrastive loss (both directions)
                    loss = (
                        self.contrastive_loss(similarity, labels) +
                        self.contrastive_loss(similarity.t(), labels)
                    ) / 2
                    
                    total_loss += loss
                    num_pairs += 1
                    
            outputs["contrastive_loss"] = total_loss / num_pairs
            
        return outputs