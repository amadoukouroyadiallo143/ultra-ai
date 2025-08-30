import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
from einops import rearrange


class ImageEncoder(nn.Module):
    """
    Image encoder for multimodal processing.
    Converts images to token representations compatible with text model.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        d_model: int = 2560,
        num_layers: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_image_tokens: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.d_model = d_model
        self.max_image_tokens = max_image_tokens
        
        # Calculate number of patches
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            embed_dim=d_model,
            device=device,
            dtype=dtype,
        )
        
        # Vision transformer
        self.transformer = VisionTransformer(
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )
        
        # Adaptive pooling for consistent output size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(max_image_tokens)
        
        # Output projection to align with text model
        self.output_proj = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to token representations.
        
        Args:
            images: Image tensor (batch_size, channels, height, width)
            
        Returns:
            Image tokens (batch_size, max_image_tokens, d_model)
        """
        batch_size = images.shape[0]
        
        # Patch embedding
        x = self.patch_embed(images)  # (batch_size, num_patches, d_model)
        
        # Vision transformer
        x = self.transformer(x)  # (batch_size, num_patches, d_model)
        
        # Adaptive pooling to consistent size
        x = x.transpose(1, 2)  # (batch_size, d_model, num_patches)
        x = self.adaptive_pool(x)  # (batch_size, d_model, max_image_tokens)
        x = x.transpose(1, 2)  # (batch_size, max_image_tokens, d_model)
        
        # Output projection
        x = self.output_proj(x)
        
        return x


class PatchEmbedding(nn.Module):
    """
    Convert image patches to embeddings.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        embed_dim: int = 768,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch projection
        self.proj = nn.Conv2d(
            num_channels, embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size,
            device=device, 
            dtype=dtype
        )
        
        # Position embeddings
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim, device=device, dtype=dtype) * 0.02
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Images (batch_size, channels, height, width)
            
        Returns:
            Patch embeddings (batch_size, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        
        # Ensure input size matches expected
        assert H == self.image_size and W == self.image_size, \
            f"Input size {H}x{W} doesn't match expected {self.image_size}x{self.image_size}"
            
        # Extract patches and flatten
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = rearrange(x, 'b e h w -> b (h w) e')  # (B, num_patches, embed_dim)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        return x


class VisionTransformer(nn.Module):
    """
    Vision Transformer for processing image patches.
    """
    
    def __init__(
        self,
        d_model: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            VisionTransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                norm_layer=norm_layer,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])
        
        # Final normalization
        self.norm = norm_layer(d_model, device=device, dtype=dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Patch embeddings (batch_size, num_patches, d_model)
            
        Returns:
            Processed features (batch_size, num_patches, d_model)
        """
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
            
        # Final normalization
        x = self.norm(x)
        
        return x


class VisionTransformerBlock(nn.Module):
    """
    Vision Transformer block with self-attention and MLP.
    """
    
    def __init__(
        self,
        d_model: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        # Pre-normalization
        self.norm1 = norm_layer(d_model, device=device, dtype=dtype)
        
        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout,
            device=device, dtype=dtype, batch_first=True
        )
        
        # Pre-normalization for MLP
        self.norm2 = norm_layer(d_model, device=device, dtype=dtype)
        
        # MLP
        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_dim, device=device, dtype=dtype),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, d_model, device=device, dtype=dtype),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features (batch_size, seq_len, d_model)
            
        Returns:
            Output features (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


class CLIPImageEncoder(ImageEncoder):
    """
    CLIP-style image encoder with contrastive learning capabilities.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        d_model: int = 2560,
        num_layers: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        max_image_tokens: int = 256,
        projection_dim: int = 512,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            max_image_tokens=max_image_tokens,
            device=device,
            dtype=dtype,
        )
        
        # Global average pooling for image representation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(d_model, projection_dim, device=device, dtype=dtype),
        )
        
    def forward(self, images: torch.Tensor, return_global: bool = False) -> torch.Tensor:
        """
        Forward pass with optional global representation.
        
        Args:
            images: Image tensor
            return_global: Whether to return global image representation
            
        Returns:
            Image tokens or global representation
        """
        # Get patch-level tokens
        image_tokens = super().forward(images)
        
        if return_global:
            # Global image representation
            global_features = self.global_pool(image_tokens.transpose(1, 2)).squeeze(-1)
            global_features = self.projection_head(global_features)
            return global_features
        else:
            return image_tokens