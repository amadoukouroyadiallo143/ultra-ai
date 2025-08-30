import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class AudioEncoder(nn.Module):
    """
    Audio encoder for processing speech and audio signals.
    Converts raw audio or spectrograms to token representations.
    """
    
    def __init__(
        self,
        n_mels: int = 80,
        n_fft: int = 1024,
        hop_length: int = 320,
        sample_rate: int = 16000,
        d_model: int = 2560,
        num_layers: int = 6,
        num_heads: int = 8,
        max_audio_tokens: int = 1500,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.d_model = d_model
        self.max_audio_tokens = max_audio_tokens
        
        # Spectrogram extraction (if processing raw audio)
        self.mel_transform = nn.Sequential(
            # Note: In practice, use torchaudio.transforms.MelSpectrogram
            # This is a placeholder for the concept
        )
        
        # Convolutional feature extraction
        self.conv_layers = nn.Sequential(
            nn.Conv1d(n_mels, d_model // 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model // 4, d_model // 2, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        # Positional encoding for audio sequences
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_audio_tokens)
        
        # Transformer layers for temporal modeling
        self.transformer_layers = nn.ModuleList([
            AudioTransformerLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                device=device,
                dtype=dtype,
            )
            for _ in range(num_layers)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_model, device=device, dtype=dtype)
        
        # Adaptive pooling for consistent output size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(max_audio_tokens)
        
    def forward(
        self, 
        audio_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode audio features to tokens.
        
        Args:
            audio_features: Mel spectrogram or raw audio features
                           (batch_size, n_mels, time_steps)
            attention_mask: Optional attention mask
            
        Returns:
            Audio tokens (batch_size, max_audio_tokens, d_model)
        """
        batch_size, n_mels, time_steps = audio_features.shape
        
        # Convolutional feature extraction
        x = self.conv_layers(audio_features)  # (batch_size, d_model, reduced_time)
        
        # Transpose for transformer: (batch_size, time, d_model)
        x = x.transpose(1, 2)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x, attention_mask=attention_mask)
            
        # Layer normalization
        x = self.norm(x)
        
        # Adaptive pooling to consistent size
        x = x.transpose(1, 2)  # (batch_size, d_model, time)
        x = self.adaptive_pool(x)  # (batch_size, d_model, max_audio_tokens)
        x = x.transpose(1, 2)  # (batch_size, max_audio_tokens, d_model)
        
        return x


class AudioTransformerLayer(nn.Module):
    """
    Transformer layer optimized for audio processing.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout,
            device=device, dtype=dtype, batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model, device=device, dtype=dtype),
            nn.Dropout(dropout),
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model, device=device, dtype=dtype)
        self.norm2 = nn.LayerNorm(d_model, device=device, dtype=dtype)
        
    def forward(
        self, 
        x: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input features (batch_size, seq_len, d_model)
            attention_mask: Optional attention mask
            
        Returns:
            Output features (batch_size, seq_len, d_model)
        """
        # Self-attention with residual connection
        attn_out, _ = self.self_attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            key_padding_mask=attention_mask
        )
        x = x + attn_out
        
        # Feed-forward with residual connection
        x = x + self.ffn(self.norm2(x))
        
        return x


class WhisperEncoder(AudioEncoder):
    """
    Whisper-style encoder for speech recognition and understanding.
    """
    
    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 2560,
        num_layers: int = 24,
        num_heads: int = 16,
        max_audio_tokens: int = 1500,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            n_mels=n_mels,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            max_audio_tokens=max_audio_tokens,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )
        
        # Whisper-style convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        
    def forward(
        self,
        mel_spectrogram: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode mel spectrogram to audio tokens.
        
        Args:
            mel_spectrogram: Mel spectrogram (batch_size, n_mels, time_steps)
            attention_mask: Optional attention mask
            
        Returns:
            Audio tokens (batch_size, max_audio_tokens, d_model)
        """
        return super().forward(mel_spectrogram, attention_mask)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for sequence data.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(1), :].transpose(0, 1)


class SpeechSeparationEncoder(AudioEncoder):
    """
    Encoder for speech separation and source separation tasks.
    """
    
    def __init__(
        self,
        num_sources: int = 2,
        n_mels: int = 80,
        d_model: int = 2560,
        num_layers: int = 8,
        num_heads: int = 8,
        max_audio_tokens: int = 1500,
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            n_mels=n_mels,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            max_audio_tokens=max_audio_tokens,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )
        self.num_sources = num_sources
        
        # Source separation heads
        self.separation_heads = nn.ModuleList([
            nn.Linear(d_model, d_model, device=device, dtype=dtype)
            for _ in range(num_sources)
        ])
        
    def forward(
        self,
        mixed_audio_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        separate_sources: bool = False
    ) -> torch.Tensor:
        """
        Process mixed audio and optionally separate sources.
        
        Args:
            mixed_audio_features: Mixed audio features
            attention_mask: Optional attention mask
            separate_sources: Whether to output separated sources
            
        Returns:
            Audio tokens or separated source tokens
        """
        # Get base audio tokens
        audio_tokens = super().forward(mixed_audio_features, attention_mask)
        
        if separate_sources:
            # Apply separation heads
            separated_tokens = []
            for head in self.separation_heads:
                source_tokens = head(audio_tokens)
                separated_tokens.append(source_tokens)
            
            # Stack separated sources
            return torch.stack(separated_tokens, dim=1)  # (batch, num_sources, tokens, d_model)
        else:
            return audio_tokens


class MusicEncoder(AudioEncoder):
    """
    Specialized encoder for music understanding and generation.
    """
    
    def __init__(
        self,
        n_mels: int = 128,  # Higher resolution for music
        d_model: int = 2560,
        num_layers: int = 12,
        num_heads: int = 16,
        max_audio_tokens: int = 2048,  # Longer sequences for music
        dropout: float = 0.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__(
            n_mels=n_mels,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            max_audio_tokens=max_audio_tokens,
            dropout=dropout,
            device=device,
            dtype=dtype,
        )
        
        # Music-specific feature extractors
        self.harmonic_analyzer = nn.Sequential(
            nn.Conv1d(n_mels, d_model // 2, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.Conv1d(d_model // 2, d_model // 2, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        
        self.rhythm_analyzer = nn.Sequential(
            nn.Conv1d(n_mels, d_model // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(d_model // 2, d_model // 2, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # Combine harmonic and rhythmic features
        self.feature_combiner = nn.Linear(d_model, d_model, device=device, dtype=dtype)
        
    def forward(
        self,
        music_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode music features with harmonic and rhythmic analysis.
        """
        batch_size, n_mels, time_steps = music_features.shape
        
        # Extract harmonic and rhythmic features
        harmonic_features = self.harmonic_analyzer(music_features)
        rhythmic_features = self.rhythm_analyzer(music_features)
        
        # Combine features
        combined_features = torch.cat([harmonic_features, rhythmic_features], dim=1)
        combined_features = combined_features.transpose(1, 2)  # (batch, time, features)
        
        # Project to model dimension
        x = self.feature_combiner(combined_features)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Pass through transformer layers
        for layer in self.transformer_layers:
            x = layer(x, attention_mask=attention_mask)
            
        # Layer normalization and pooling
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.adaptive_pool(x)
        x = x.transpose(1, 2)
        
        return x