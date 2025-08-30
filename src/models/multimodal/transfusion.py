import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math
from einops import rearrange


class TransfusionLayer(nn.Module):
    """
    Transfusion Layer combining language modeling and diffusion objectives.
    Unifies text and image generation with dual objectives.
    """
    
    def __init__(
        self,
        d_model: int = 2560,
        num_heads: int = 16,
        vocab_size: int = 50432,
        image_vocab_size: int = 8192,  # For discrete image tokens
        diffusion_steps: int = 1000,
        noise_schedule: str = "cosine",
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.vocab_size = vocab_size
        self.image_vocab_size = image_vocab_size
        self.diffusion_steps = diffusion_steps
        
        # Dual objective heads
        self.text_head = nn.Linear(d_model, vocab_size, bias=False, device=device, dtype=dtype)
        self.image_diffusion_head = DiffusionHead(
            d_model=d_model,
            diffusion_steps=diffusion_steps,
            noise_schedule=noise_schedule,
            device=device,
            dtype=dtype,
        )
        
        # Modality detection
        self.modality_detector = nn.Sequential(
            nn.Linear(d_model, d_model // 4, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(d_model // 4, 2, device=device, dtype=dtype),  # text vs image
        )
        
        # Cross-modal attention
        self.cross_modal_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout,
            device=device, dtype=dtype, batch_first=True
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model, device=device, dtype=dtype)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        modality_labels: Optional[torch.Tensor] = None,
        noise_level: Optional[torch.Tensor] = None,
        return_both_objectives: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with dual objectives.
        
        Args:
            hidden_states: Input features (batch, seq_len, d_model)
            modality_labels: 0 for text, 1 for image tokens
            noise_level: Diffusion noise level for image tokens
            return_both_objectives: Whether to compute both text and image losses
            
        Returns:
            Dictionary with text logits and/or diffusion outputs
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Normalize features
        hidden_states = self.norm(hidden_states)
        
        # Detect modalities if not provided
        if modality_labels is None:
            modality_logits = self.modality_detector(hidden_states)
            modality_probs = F.softmax(modality_logits, dim=-1)
            modality_labels = torch.argmax(modality_probs, dim=-1)
            
        outputs = {}
        
        # Separate text and image tokens
        text_mask = (modality_labels == 0)
        image_mask = (modality_labels == 1)
        
        # Text modeling objective
        if text_mask.any().item() or return_both_objectives:
            text_logits = self.text_head(hidden_states)
            outputs['text_logits'] = text_logits
            
        # Image diffusion objective  
        if image_mask.any().item() or return_both_objectives:
            diffusion_outputs = self.image_diffusion_head(
                hidden_states, noise_level=noise_level
            )
            outputs.update(diffusion_outputs)
            
        # Cross-modal attention between text and image features
        if text_mask.any().item() and image_mask.any().item():
            text_features = hidden_states[text_mask.unsqueeze(-1).expand_as(hidden_states)]
            image_features = hidden_states[image_mask.unsqueeze(-1).expand_as(hidden_states)]
            
            if text_features.numel() > 0 and image_features.numel() > 0:
                text_features = text_features.view(-1, text_features.shape[-1]).unsqueeze(0)
                image_features = image_features.view(-1, image_features.shape[-1]).unsqueeze(0)
                
                cross_attn_out, _ = self.cross_modal_attn(
                    text_features, image_features, image_features
                )
                outputs['cross_modal_features'] = cross_attn_out
                
        return outputs
        
    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        text_targets: Optional[torch.Tensor] = None,
        image_targets: Optional[torch.Tensor] = None,
        modality_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute dual objective losses."""
        losses = {}
        
        # Text modeling loss
        if 'text_logits' in outputs and text_targets is not None:
            text_logits = outputs['text_logits']
            shift_logits = text_logits[..., :-1, :].contiguous()
            shift_labels = text_targets[..., 1:].contiguous()
            
            text_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
            losses['text_loss'] = text_loss
            
        # Image diffusion loss
        if 'noise_pred' in outputs and image_targets is not None:
            noise_pred = outputs['noise_pred']
            target_noise = outputs.get('target_noise', image_targets)
            
            diffusion_loss = F.mse_loss(noise_pred, target_noise)
            losses['diffusion_loss'] = diffusion_loss
            
        # Total loss
        total_loss = sum(losses.values())
        losses['total_loss'] = total_loss
        
        return losses


class DiffusionHead(nn.Module):
    """
    Diffusion head for image generation within the unified architecture.
    """
    
    def __init__(
        self,
        d_model: int,
        diffusion_steps: int = 1000,
        noise_schedule: str = "cosine",
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.diffusion_steps = diffusion_steps
        
        # Noise prediction network
        self.noise_predictor = nn.Sequential(
            nn.Linear(d_model, d_model * 2, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(d_model, d_model, device=device, dtype=dtype),
        )
        
        # Time embedding for diffusion steps
        self.time_mlp = TimestepEmbedding(
            dim=d_model,
            max_period=10000,
            device=device,
            dtype=dtype,
        )
        
        # Noise schedule
        self.register_buffer(
            'betas', 
            self._get_noise_schedule(noise_schedule, diffusion_steps, device)
        )
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
    def _get_noise_schedule(self, schedule: str, steps: int, device) -> torch.Tensor:
        """Get noise schedule for diffusion."""
        if schedule == "linear":
            return torch.linspace(0.0001, 0.02, steps, device=device)
        elif schedule == "cosine":
            # Cosine schedule
            s = 0.008
            steps_tensor = torch.arange(steps + 1, device=device) / steps
            alphas_cumprod = torch.cos((steps_tensor + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown noise schedule: {schedule}")
            
    def forward(
        self,
        x: torch.Tensor,
        noise_level: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for diffusion prediction.
        
        Args:
            x: Input features (batch, seq_len, d_model)
            noise_level: Diffusion timestep (batch,)
            
        Returns:
            Dictionary with diffusion outputs
        """
        batch_size, seq_len, d_model = x.shape
        
        # Sample random timesteps if not provided
        if noise_level is None:
            noise_level = torch.randint(
                0, self.diffusion_steps, 
                (batch_size,), device=x.device
            )
            
        # Add time embedding
        time_emb = self.time_mlp(noise_level)  # (batch, d_model)
        time_emb = time_emb.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine with input features
        x_with_time = x + time_emb
        
        # Predict noise
        noise_pred = self.noise_predictor(x_with_time)
        
        # Add noise for training
        noise = torch.randn_like(x)
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[noise_level])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[noise_level])
        
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.view(-1, 1, 1)
        
        noisy_x = sqrt_alphas_cumprod * x + sqrt_one_minus_alphas_cumprod * noise
        
        return {
            'noise_pred': noise_pred,
            'target_noise': noise,
            'noisy_input': noisy_x,
            'timestep': noise_level,
        }
        
    def sample(
        self,
        shape: Tuple[int, ...],
        device: torch.device,
        num_inference_steps: int = 50,
    ) -> torch.Tensor:
        """
        Sample from the diffusion model.
        
        Args:
            shape: Shape of output tensor
            device: Device to generate on
            num_inference_steps: Number of denoising steps
            
        Returns:
            Generated samples
        """
        # Start with random noise
        x = torch.randn(shape, device=device)
        
        # Denoising loop
        step_size = self.diffusion_steps // num_inference_steps
        
        for i in reversed(range(0, self.diffusion_steps, step_size)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            
            # Predict noise
            time_emb = self.time_mlp(t).unsqueeze(1).expand(-1, shape[1], -1)
            x_with_time = x + time_emb
            predicted_noise = self.noise_predictor(x_with_time)
            
            # Denoise
            beta_t = self.betas[i]
            sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[i])
            sqrt_alpha = torch.sqrt(self.alphas[i])
            
            x = (x - beta_t / sqrt_one_minus_alpha_cumprod * predicted_noise) / sqrt_alpha
            
            # Add noise (except for final step)
            if i > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta_t) * noise
                
        return x


class TimestepEmbedding(nn.Module):
    """
    Timestep embedding for diffusion models.
    """
    
    def __init__(
        self,
        dim: int,
        max_period: int = 10000,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        
        # Sinusoidal position embedding
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float, device=device) / half
        )
        self.register_buffer('freqs', freqs)
        
        # MLP for timestep processing
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4, device=device, dtype=dtype),
            nn.SiLU(),
            nn.Linear(dim * 4, dim, device=device, dtype=dtype),
        )
        
    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timesteps: Timestep indices (batch,)
            
        Returns:
            Timestep embeddings (batch, dim)
        """
        args = timesteps[:, None].float() * self.freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
            
        return self.mlp(embedding)


class CausalFusion(nn.Module):
    """
    CausalFusion: Unified autoregressive and diffusion model.
    Combines causal language modeling with diffusion-based image generation.
    """
    
    def __init__(
        self,
        d_model: int = 2560,
        num_layers: int = 24,
        num_heads: int = 16,
        vocab_size: int = 50432,
        max_image_tokens: int = 1024,
        diffusion_steps: int = 1000,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_image_tokens = max_image_tokens
        
        # Unified transformer layers
        self.layers = nn.ModuleList([
            CausalFusionLayer(
                d_model=d_model,
                num_heads=num_heads,
                diffusion_steps=diffusion_steps,
                dropout=dropout,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(d_model, device=device, dtype=dtype)
        
        # Dual output heads
        self.transfusion = TransfusionLayer(
            d_model=d_model,
            num_heads=num_heads,
            vocab_size=vocab_size,
            diffusion_steps=diffusion_steps,
            device=device,
            dtype=dtype,
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        modality_labels: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        noise_level: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Unified forward pass for text and image generation.
        
        Args:
            input_ids: Text token IDs
            image_features: Image feature representations
            modality_labels: Modality indicators per token
            attention_mask: Causal attention mask
            noise_level: Diffusion noise level
            
        Returns:
            Dictionary with generation outputs
        """
        # Combine text and image features
        if image_features is not None:
            hidden_states = torch.cat([input_ids, image_features], dim=1)
        else:
            hidden_states = input_ids
            
        batch_size, seq_len = hidden_states.shape[:2]
        
        # Apply causal mask
        if attention_mask is None:
            attention_mask = torch.tril(torch.ones(seq_len, seq_len, device=hidden_states.device))
            
        # Pass through unified layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                noise_level=noise_level,
            )
            
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        # Dual objective outputs
        outputs = self.transfusion(
            hidden_states,
            modality_labels=modality_labels,
            noise_level=noise_level,
        )
        
        return outputs


class CausalFusionLayer(nn.Module):
    """
    Individual layer in the CausalFusion architecture.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        diffusion_steps: int = 1000,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        # Causal self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout,
            device=device, dtype=dtype, batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4, device=device, dtype=dtype),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model, device=device, dtype=dtype),
            nn.Dropout(dropout),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, device=device, dtype=dtype)
        self.norm2 = nn.LayerNorm(d_model, device=device, dtype=dtype)
        
        # Time conditioning for diffusion
        self.time_mlp = TimestepEmbedding(d_model, device=device, dtype=dtype)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_level: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional time conditioning.
        """
        # Add time conditioning if provided
        if noise_level is not None:
            time_emb = self.time_mlp(noise_level)
            time_emb = time_emb.unsqueeze(1).expand(-1, hidden_states.shape[1], -1)
            hidden_states = hidden_states + time_emb
            
        # Self-attention with causal mask
        normed = self.norm1(hidden_states)
        attn_out, _ = self.self_attn(normed, normed, normed, attn_mask=attention_mask)
        hidden_states = hidden_states + attn_out
        
        # Feed-forward
        hidden_states = hidden_states + self.ffn(self.norm2(hidden_states))
        
        return hidden_states