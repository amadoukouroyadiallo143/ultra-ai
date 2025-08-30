import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math
from einops import rearrange
from .image_encoder import VisionTransformer, PatchEmbedding


class VideoEncoder(nn.Module):
    """
    Video encoder for processing video sequences.
    Handles both spatial and temporal modeling for video understanding.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        num_frames: int = 8,
        d_model: int = 2560,
        spatial_layers: int = 12,
        temporal_layers: int = 4,
        num_heads: int = 16,
        max_video_tokens: int = 512,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_frames = num_frames
        self.d_model = d_model
        self.max_video_tokens = max_video_tokens
        
        # Calculate spatial dimensions
        self.num_patches_per_frame = (image_size // patch_size) ** 2
        self.total_patches = self.num_patches_per_frame * num_frames
        
        # Patch embedding (same as image encoder)
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            embed_dim=d_model,
            device=device,
            dtype=dtype,
        )
        
        # Temporal position embeddings
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, num_frames, d_model, device=device, dtype=dtype) * 0.02
        )
        
        # Spatial encoder (per-frame processing)
        self.spatial_encoder = VisionTransformer(
            d_model=d_model,
            num_layers=spatial_layers,
            num_heads=num_heads,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )
        
        # Temporal encoder (cross-frame modeling)
        self.temporal_encoder = nn.ModuleList([
            TemporalTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                dropout=dropout,
                device=device,
                dtype=dtype,
            )
            for _ in range(temporal_layers)
        ])
        
        # Frame aggregation
        self.frame_aggregator = FrameAggregator(
            d_model=d_model,
            num_frames=num_frames,
            num_patches_per_frame=self.num_patches_per_frame,
            device=device,
            dtype=dtype,
        )
        
        # Adaptive pooling for consistent output size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(max_video_tokens)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode video to token representations.
        
        Args:
            video: Video tensor (batch_size, num_frames, channels, height, width)
            
        Returns:
            Video tokens (batch_size, max_video_tokens, d_model)
        """
        batch_size, num_frames, channels, height, width = video.shape
        
        # Reshape for batch processing
        video_flat = video.view(batch_size * num_frames, channels, height, width)
        
        # Extract patches for all frames
        patches = self.patch_embed(video_flat)  # (batch*frames, patches_per_frame, d_model)
        
        # Reshape back to separate frames
        patches = patches.view(batch_size, num_frames, self.num_patches_per_frame, self.d_model)
        
        # Process each frame spatially
        frame_features = []
        for frame_idx in range(num_frames):
            frame_patches = patches[:, frame_idx]  # (batch, patches_per_frame, d_model)
            frame_feat = self.spatial_encoder(frame_patches)
            frame_features.append(frame_feat)
            
        # Stack frame features
        frame_features = torch.stack(frame_features, dim=1)  # (batch, frames, patches, d_model)
        
        # Add temporal position embeddings
        # Average pool each frame to single token for temporal modeling
        frame_tokens = frame_features.mean(dim=2)  # (batch, frames, d_model)
        frame_tokens = frame_tokens + self.temporal_pos_embed
        
        # Temporal modeling
        for temporal_layer in self.temporal_encoder:
            frame_tokens = temporal_layer(frame_tokens)
            
        # Aggregate frame and patch information
        video_tokens = self.frame_aggregator(frame_features, frame_tokens)
        
        # Adaptive pooling
        video_tokens = video_tokens.transpose(1, 2)  # (batch, d_model, tokens)
        video_tokens = self.adaptive_pool(video_tokens)  # (batch, d_model, max_tokens)
        video_tokens = video_tokens.transpose(1, 2)  # (batch, max_tokens, d_model)
        
        # Output projection
        video_tokens = self.output_proj(video_tokens)
        
        return video_tokens


class TemporalTransformerLayer(nn.Module):
    """
    Transformer layer for temporal modeling across video frames.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        # Temporal self-attention
        self.temporal_attn = nn.MultiheadAttention(
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Frame tokens (batch_size, num_frames, d_model)
            
        Returns:
            Temporally processed tokens (batch_size, num_frames, d_model)
        """
        # Temporal self-attention
        attn_out, _ = self.temporal_attn(
            self.norm1(x), self.norm1(x), self.norm1(x)
        )
        x = x + attn_out
        
        # Feed-forward
        x = x + self.ffn(self.norm2(x))
        
        return x


class FrameAggregator(nn.Module):
    """
    Aggregates spatial and temporal information from video frames.
    """
    
    def __init__(
        self,
        d_model: int,
        num_frames: int,
        num_patches_per_frame: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_frames = num_frames
        self.num_patches_per_frame = num_patches_per_frame
        
        # Cross-attention between frame-level and patch-level features
        self.cross_attn = nn.MultiheadAttention(
            d_model, num_heads=8, device=device, dtype=dtype, batch_first=True
        )
        
        # Spatial-temporal fusion
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model, device=device, dtype=dtype),
            nn.GELU(),
            nn.Linear(d_model, d_model, device=device, dtype=dtype),
        )
        
    def forward(
        self, 
        frame_features: torch.Tensor,  # (batch, frames, patches, d_model)
        frame_tokens: torch.Tensor     # (batch, frames, d_model)
    ) -> torch.Tensor:
        """
        Aggregate spatial and temporal information.
        
        Returns:
            Aggregated video tokens
        """
        batch_size, num_frames, num_patches, d_model = frame_features.shape
        
        # Flatten spatial-temporal dimensions
        all_patches = frame_features.view(batch_size, num_frames * num_patches, d_model)
        
        # Expand frame tokens to match patch dimensions
        expanded_frame_tokens = frame_tokens.unsqueeze(2).expand(-1, -1, num_patches, -1)
        expanded_frame_tokens = expanded_frame_tokens.view(batch_size, num_frames * num_patches, d_model)
        
        # Cross-attention between patches and frame tokens
        attended_patches, _ = self.cross_attn(
            all_patches, expanded_frame_tokens, expanded_frame_tokens
        )
        
        # Fusion
        combined = torch.cat([all_patches, attended_patches], dim=-1)
        output = self.fusion(combined)
        
        return output


class Video3DEncoder(VideoEncoder):
    """
    3D CNN-based video encoder for efficient spatial-temporal processing.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        num_channels: int = 3,
        num_frames: int = 16,
        d_model: int = 2560,
        conv_layers: List[int] = [64, 128, 256, 512],
        max_video_tokens: int = 512,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        # Initialize parent with minimal spatial processing
        super().__init__(
            image_size=image_size,
            num_channels=num_channels,
            num_frames=num_frames,
            d_model=d_model,
            spatial_layers=0,  # No separate spatial layers
            temporal_layers=0,  # No separate temporal layers
            max_video_tokens=max_video_tokens,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )
        
        # 3D convolutional layers
        self.conv3d_layers = nn.ModuleList()
        in_channels = num_channels
        
        for out_channels in conv_layers:
            self.conv3d_layers.append(
                nn.Sequential(
                    nn.Conv3d(
                        in_channels, out_channels, 
                        kernel_size=(3, 3, 3), 
                        stride=(1, 2, 2), 
                        padding=(1, 1, 1),
                        device=device, dtype=dtype
                    ),
                    nn.BatchNorm3d(out_channels, device=device, dtype=dtype),
                    nn.ReLU(),
                    nn.Dropout3d(dropout),
                )
            )
            in_channels = out_channels
            
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Projection to model dimension
        self.feature_proj = nn.Linear(conv_layers[-1], d_model, device=device, dtype=dtype)
        
        # Position embeddings for 3D features
        self.pos_embed_3d = nn.Parameter(
            torch.randn(1, max_video_tokens, d_model, device=device, dtype=dtype) * 0.02
        )
        
    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Process video with 3D convolutions.
        
        Args:
            video: Video tensor (batch_size, num_frames, channels, height, width)
            
        Returns:
            Video tokens (batch_size, max_video_tokens, d_model)
        """
        # Rearrange for 3D conv: (batch, channels, frames, height, width)
        video = video.transpose(1, 2)
        
        # Apply 3D convolutions
        x = video
        feature_maps = []
        
        for conv_layer in self.conv3d_layers:
            x = conv_layer(x)
            # Store intermediate features
            pooled_features = self.global_pool(x).squeeze(-1).squeeze(-1).squeeze(-1)
            feature_maps.append(pooled_features)
            
        # Combine multi-scale features
        combined_features = torch.cat(feature_maps, dim=-1)  # (batch, sum_of_channels)
        
        # Project to model dimension
        video_features = self.feature_proj(combined_features)  # (batch, d_model)
        
        # Expand to sequence length
        video_tokens = video_features.unsqueeze(1).expand(-1, self.max_video_tokens, -1)
        
        # Add positional embeddings
        video_tokens = video_tokens + self.pos_embed_3d
        
        return video_tokens


class ActionRecognitionEncoder(VideoEncoder):
    """
    Specialized video encoder for action recognition tasks.
    """
    
    def __init__(
        self,
        num_action_classes: int = 400,
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        num_frames: int = 16,
        d_model: int = 2560,
        max_video_tokens: int = 512,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            image_size=image_size,
            patch_size=patch_size,
            num_channels=num_channels,
            num_frames=num_frames,
            d_model=d_model,
            max_video_tokens=max_video_tokens,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )
        
        # Action classification head
        self.action_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_action_classes, device=device, dtype=dtype),
        )
        
        # Temporal attention for action focus
        self.action_attention = nn.MultiheadAttention(
            d_model, num_heads=8, device=device, dtype=dtype, batch_first=True
        )
        
    def forward(
        self, 
        video: torch.Tensor, 
        classify_action: bool = False
    ) -> torch.Tensor:
        """
        Process video with optional action classification.
        
        Args:
            video: Video tensor
            classify_action: Whether to output action predictions
            
        Returns:
            Video tokens or action predictions
        """
        # Get base video tokens
        video_tokens = super().forward(video)
        
        if classify_action:
            # Apply temporal attention for action-relevant features
            action_features, _ = self.action_attention(
                video_tokens, video_tokens, video_tokens
            )
            
            # Global average pooling
            pooled_features = action_features.mean(dim=1)  # (batch, d_model)
            
            # Action classification
            action_logits = self.action_classifier(pooled_features)
            return action_logits
        else:
            return video_tokens