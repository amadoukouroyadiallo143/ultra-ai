import torch
import torch.nn as nn
from typing import Optional, List, Union
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutput
from .mamba2_block import Mamba2Block


class Mamba2Config:
    """Configuration class for Mamba-2 model."""
    
    def __init__(
        self,
        vocab_size: int = 50432,
        d_model: int = 2560,
        n_layer: int = 56,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Union[int, str] = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        use_fast_path: bool = True,
        norm_epsilon: float = 1e-5,
        residual_in_fp32: bool = False,
        fused_add_norm: bool = False,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        tie_embeddings: bool = False,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layer = n_layer
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.dt_rank = dt_rank
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        self.conv_bias = conv_bias
        self.bias = bias
        self.use_fast_path = use_fast_path
        self.norm_epsilon = norm_epsilon
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_embeddings = tie_embeddings


class Mamba2Model(nn.Module):
    """
    Mamba-2 Model with selective state space mechanism.
    Implements 70% of the hybrid architecture backbone.
    """
    
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)
        
        # Mamba layers
        self.layers = nn.ModuleList([
            Mamba2Block(
                d_model=config.d_model,
                d_state=config.d_state,
                d_conv=config.d_conv,
                expand=config.expand,
                dt_rank=config.dt_rank,
                dt_min=config.dt_min,
                dt_max=config.dt_max,
                dt_init=config.dt_init,
                dt_scale=config.dt_scale,
                dt_init_floor=config.dt_init_floor,
                conv_bias=config.conv_bias,
                bias=config.bias,
                use_fast_path=config.use_fast_path,
                layer_idx=i,
                norm_epsilon=config.norm_epsilon,
                residual_in_fp32=config.residual_in_fp32,
            )
            for i in range(config.n_layer)
        ])
        
        # Final layer norm
        self.norm_f = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, std=0.02)
            
    def forward(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        inference_params=None,
        **kwargs
    ):
        """
        Forward pass of Mamba-2 model.
        
        Args:
            input_ids: Token indices
            inputs_embeds: Pre-computed embeddings (optional)
            inference_params: Parameters for inference
            
        Returns:
            Hidden states
        """
        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids)
            
        hidden_states = inputs_embeds
        residual = None
        
        # Pass through all layers
        for layer in self.layers:
            hidden_states, residual = layer(
                hidden_states, 
                residual=residual,
                inference_params=inference_params
            )
            
        # Final residual connection and normalization
        if residual is not None:
            hidden_states = hidden_states + residual
            residual = None
            
        hidden_states = self.norm_f(hidden_states)
        
        return hidden_states


class Mamba2ForCausalLM(nn.Module):
    """
    Mamba-2 model for causal language modeling.
    """
    
    def __init__(self, config: Mamba2Config):
        super().__init__()
        self.config = config
        
        # Backbone model
        self.backbone = Mamba2Model(config)
        
        # Language modeling head
        if not config.tie_embeddings:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        else:
            self.lm_head = None
            
    def forward(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inference_params=None,
        **kwargs
    ):
        """
        Forward pass for causal language modeling.
        
        Args:
            input_ids: Token indices
            inputs_embeds: Pre-computed embeddings
            labels: Target labels for loss computation
            inference_params: Parameters for inference
            
        Returns:
            CausalLMOutput with logits and loss
        """
        # Get hidden states from backbone
        hidden_states = self.backbone(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            inference_params=inference_params,
            **kwargs
        )
        
        # Compute logits
        if self.lm_head is not None:
            logits = self.lm_head(hidden_states)
        else:
            # Tied embeddings
            logits = torch.matmul(hidden_states, self.backbone.embeddings.weight.t())
            
        loss = None
        if labels is not None:
            # Shift labels for causal modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
        )
        
    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 50,
        temperature: float = 1.0,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 0.9,
        **kwargs
    ):
        """
        Generate text using the model.
        
        Args:
            input_ids: Input token indices
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_k: Top-k filtering
            top_p: Top-p (nucleus) filtering
            
        Returns:
            Generated token sequences
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize generation
        generated = input_ids
        
        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
            outputs = self.forward(generated)
            next_token_logits = outputs.logits[:, -1, :] / temperature
            
            if do_sample:
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy selection
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for end of sequence
            if (next_token == self.config.eos_token_id).all():
                break
                
        return generated