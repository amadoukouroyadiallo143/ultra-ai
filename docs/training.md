# Training Guide

This comprehensive guide covers training Ultra-AI models from scratch, fine-tuning, and distributed training strategies.

## 🚀 Quick Start Training

### Single GPU Training

```bash
# Basic training on single GPU
python scripts/train.py \
    --config src/config/base.yaml \
    --model-size ultra-3b \
    --batch-size 4 \
    --gradient-accumulation-steps 16 \
    --learning-rate 1e-4 \
    --max-steps 10000

# With mixed precision
python scripts/train.py \
    --config src/config/base.yaml \
    --mixed-precision \
    --gradient-checkpointing \
    --batch-size 8
```

### Multi-GPU Training

```bash
# Data parallel training (recommended for single node)
torchrun --nproc_per_node=8 scripts/train.py \
    --config src/config/base.yaml \
    --distributed \
    --batch-size 2 \
    --gradient-accumulation-steps 32

# Model parallel training for large models
torchrun --nproc_per_node=8 scripts/train.py \
    --config src/config/ultra-52b.yaml \
    --model-parallel \
    --pipeline-parallel-size 4 \
    --tensor-parallel-size 2
```

### DeepSpeed Training

```bash
# DeepSpeed ZeRO Stage 2
deepspeed scripts/train.py \
    --deepspeed \
    --zero-stage 2 \
    --config src/config/base.yaml \
    --batch-size 1

# DeepSpeed ZeRO Stage 3 for largest models
deepspeed scripts/train.py \
    --deepspeed \
    --zero-stage 3 \
    --config src/config/ultra-390b.yaml \
    --cpu-offload \
    --nvme-offload-path /tmp/deepspeed_nvme
```

## 📊 Training Configuration

### Model Configurations

#### Ultra-3B (Development)
```yaml
# src/config/ultra-3b.yaml
model:
  d_model: 2048
  n_layers: 32
  n_heads: 16
  d_ff: 8192
  vocab_size: 50000
  max_seq_length: 1000000  # 1M tokens
  
  # Mamba-2 config
  d_state: 64
  d_conv: 4
  expand_factor: 2
  
  # MoE config
  num_experts: 64
  expert_top_k: 2
  
training:
  batch_size: 8
  learning_rate: 2e-4
  weight_decay: 0.01
  warmup_steps: 2000
  max_steps: 100000
  gradient_accumulation_steps: 8
```

#### Ultra-52B (Production)
```yaml
# src/config/ultra-52b.yaml
model:
  d_model: 6144
  n_layers: 80
  n_heads: 48
  d_ff: 24576
  max_seq_length: 50000000  # 50M tokens
  num_experts: 128
  
training:
  batch_size: 1
  learning_rate: 1e-4
  gradient_accumulation_steps: 128
  max_steps: 500000
  
  # Memory optimizations
  gradient_checkpointing: true
  cpu_offload: true
  mixed_precision: true
```

#### Ultra-390B (Full Scale)
```yaml
# src/config/ultra-390b.yaml
model:
  d_model: 8192
  n_layers: 120
  n_heads: 64
  d_ff: 32768
  max_seq_length: 100000000  # 100M tokens
  num_experts: 256
  
training:
  batch_size: 1
  learning_rate: 5e-5
  gradient_accumulation_steps: 512
  max_steps: 1000000
  
  # Advanced optimizations
  deepspeed_zero_stage: 3
  cpu_offload: true
  nvme_offload: true
  model_parallel: true
  pipeline_parallel_size: 8
  tensor_parallel_size: 8
```

## 🏋️ Training Pipeline

### Data Preparation

```python
# Example data preparation script
from src.training.data_loader import MultimodalDataLoader
from datasets import load_dataset

def prepare_training_data():
    # Text data
    text_datasets = [
        load_dataset("c4", streaming=True),
        load_dataset("the_pile", streaming=True),
        load_dataset("wikipedia", streaming=True)
    ]
    
    # Multimodal data  
    multimodal_datasets = [
        load_dataset("laion-5b", streaming=True),
        load_dataset("yt-bb", streaming=True),
        load_dataset("audiocaps", streaming=True)
    ]
    
    # Create data loader
    data_loader = MultimodalDataLoader(
        text_datasets=text_datasets,
        multimodal_datasets=multimodal_datasets,
        sequence_length=config.max_seq_length,
        batch_size=config.batch_size,
        num_workers=config.num_workers
    )
    
    return data_loader
```

### Training Loop

```python
def train_ultra_ai_model(config):
    """
    Complete training loop for Ultra-AI
    """
    # Initialize model
    model = UltraAIModel(config)
    
    # Setup distributed training
    if config.distributed:
        model = setup_distributed_model(model, config)
    
    # Setup optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_lr_scheduler(optimizer, config)
    
    # Setup data loader
    train_loader = prepare_training_data()
    
    # Training loop
    global_step = 0
    for epoch in range(config.num_epochs):
        for batch in train_loader:
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                multimodal_inputs=batch.get("multimodal_inputs")
            )
            
            # Compute loss
            loss = compute_loss(outputs, batch, config)
            
            # Backward pass
            if config.mixed_precision:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            # Update learning rate
            scheduler.step()
            
            # Logging and checkpointing
            if global_step % config.log_interval == 0:
                log_metrics(loss, global_step)
            
            if global_step % config.checkpoint_interval == 0:
                save_checkpoint(model, optimizer, global_step)
            
            global_step += 1
```

### Advanced Loss Functions

```python
class UltraAILoss(nn.Module):
    """
    Multi-component loss for Ultra-AI training
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.language_loss_weight = 1.0
        self.moe_loss_weight = 0.01
        self.multimodal_loss_weight = 0.1
    
    def forward(self, outputs, targets, batch):
        total_loss = 0
        losses = {}
        
        # 1. Language modeling loss
        lm_loss = F.cross_entropy(
            outputs['logits'].view(-1, outputs['logits'].size(-1)),
            targets['input_ids'].view(-1),
            ignore_index=-100
        )
        losses['lm_loss'] = lm_loss
        total_loss += self.language_loss_weight * lm_loss
        
        # 2. MoE load balancing loss
        if 'aux_loss' in outputs:
            moe_loss = outputs['aux_loss']
            losses['moe_loss'] = moe_loss
            total_loss += self.moe_loss_weight * moe_loss
        
        # 3. Multimodal alignment loss
        if 'multimodal_inputs' in batch:
            multimodal_loss = self.compute_multimodal_loss(
                outputs, batch['multimodal_inputs']
            )
            losses['multimodal_loss'] = multimodal_loss  
            total_loss += self.multimodal_loss_weight * multimodal_loss
        
        # 4. Selective scan regularization
        if hasattr(outputs, 'selective_scan_loss'):
            ss_loss = outputs.selective_scan_loss
            losses['selective_scan_loss'] = ss_loss
            total_loss += 0.001 * ss_loss
        
        return total_loss, losses
```

## ⚙️ Optimization Strategies

### Advanced Optimizers

```python
def create_optimizer(model, config):
    """
    Create advanced optimizer for Ultra-AI training
    """
    if config.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=config.weight_decay,
            eps=1e-8
        )
    
    elif config.optimizer == "lion":
        from lion_pytorch import Lion
        optimizer = Lion(
            model.parameters(),
            lr=config.learning_rate * 0.3,  # Lion uses lower LR
            betas=(0.9, 0.99),
            weight_decay=config.weight_decay * 10  # Lion uses higher WD
        )
    
    elif config.optimizer == "sophia":
        from sophia import SophiaG
        optimizer = SophiaG(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.965, 0.99),
            rho=0.04,
            weight_decay=config.weight_decay
        )
    
    elif config.optimizer == "galore":
        from galore_torch import GaLoreAdamW
        optimizer = GaLoreAdamW(
            model.parameters(),
            lr=config.learning_rate,
            rank=config.galore_rank,
            update_proj_gap=config.galore_update_gap,
            scale=config.galore_scale
        )
    
    return optimizer
```

### Learning Rate Scheduling

```python
def create_lr_scheduler(optimizer, config):
    """
    Advanced learning rate scheduling
    """
    if config.scheduler == "cosine_with_warmup":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.max_steps
        )
    
    elif config.scheduler == "polynomial_decay":
        scheduler = get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=config.warmup_steps,
            num_training_steps=config.max_steps,
            power=0.5
        )
    
    elif config.scheduler == "exponential_decay":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=config.lr_decay_gamma
        )
    
    return scheduler
```

## 🌐 Distributed Training

### Multi-Node Training Setup

```bash
# Node 0 (master node)
torchrun \
    --nnodes=4 \
    --node_rank=0 \
    --master_addr="192.168.1.100" \
    --master_port=29500 \
    --nproc_per_node=8 \
    scripts/train.py \
    --config src/config/ultra-390b.yaml \
    --distributed

# Node 1, 2, 3 (worker nodes)
torchrun \
    --nnodes=4 \
    --node_rank=1 \  # 2 for node 2, 3 for node 3
    --master_addr="192.168.1.100" \
    --master_port=29500 \
    --nproc_per_node=8 \
    scripts/train.py \
    --config src/config/ultra-390b.yaml \
    --distributed
```

### 3D Parallelism Configuration

```python
class DistributedTrainingConfig:
    """
    3D parallelism configuration for Ultra-AI
    """
    
    def __init__(self, total_gpus=64):
        # Calculate optimal parallelism dimensions
        self.data_parallel_size = 8      # Replicate across 8 nodes
        self.pipeline_parallel_size = 4   # 4-stage pipeline
        self.tensor_parallel_size = 2     # Split tensors across 2 GPUs
        
        assert (self.data_parallel_size * 
                self.pipeline_parallel_size * 
                self.tensor_parallel_size == total_gpus)
    
    def setup_parallelism(self, model):
        """
        Setup 3D parallelism for the model
        """
        # Pipeline parallelism
        model = PipelineParallel(
            model, 
            stages=self.pipeline_parallel_size,
            balance=[30, 30, 30, 30]  # Equal stage sizes
        )
        
        # Tensor parallelism  
        model = TensorParallel(
            model,
            world_size=self.tensor_parallel_size
        )
        
        # Data parallelism (handled by DDP)
        model = DistributedDataParallel(model)
        
        return model
```

### Memory Optimization

```python
class MemoryOptimizedTrainer:
    """
    Advanced memory optimization for Ultra-AI training
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
        # Enable gradient checkpointing
        if config.gradient_checkpointing:
            self.enable_gradient_checkpointing()
        
        # Setup CPU offloading
        if config.cpu_offload:
            self.setup_cpu_offload()
        
        # Setup NVMe offloading for very large models
        if config.nvme_offload:
            self.setup_nvme_offload()
    
    def enable_gradient_checkpointing(self):
        """
        Enable gradient checkpointing to trade compute for memory
        """
        self.model.gradient_checkpointing_enable()
        
        # Custom checkpointing for specific layers
        for layer in self.model.layers:
            if hasattr(layer, 'mamba_layer'):
                layer.mamba_layer.gradient_checkpointing = True
    
    def setup_cpu_offload(self):
        """
        Offload unused parameters to CPU
        """
        # Offload optimizer states
        self.optimizer = CPUAdam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
        
        # Offload inactive parameters
        for name, param in self.model.named_parameters():
            if 'expert' in name and not param.requires_grad:
                param.data = param.data.cpu()
    
    def dynamic_memory_management(self):
        """
        Dynamic memory management during training
        """
        # Monitor GPU memory usage
        memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        memory_ratio = memory_used / memory_total
        
        # Adjust batch size dynamically
        if memory_ratio > 0.9:
            self.config.batch_size = max(1, self.config.batch_size // 2)
            print(f"Reduced batch size to {self.config.batch_size}")
        
        # Clear cache when memory is low
        if memory_ratio > 0.95:
            torch.cuda.empty_cache()
```

## 🔍 Monitoring & Logging

### Training Metrics

```python
class TrainingMonitor:
    """
    Comprehensive training monitoring for Ultra-AI
    """
    
    def __init__(self, config):
        self.config = config
        
        # Setup logging backends
        if config.use_wandb:
            import wandb
            wandb.init(project="ultra-ai-training")
        
        if config.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(config.log_dir)
    
    def log_metrics(self, metrics, step):
        """
        Log training metrics
        """
        # Core metrics
        self.log_scalar("loss/total", metrics['total_loss'], step)
        self.log_scalar("loss/lm_loss", metrics['lm_loss'], step)
        self.log_scalar("loss/moe_loss", metrics['moe_loss'], step)
        
        # Learning rate
        self.log_scalar("optimizer/lr", metrics['learning_rate'], step)
        
        # Memory usage
        memory_used = torch.cuda.memory_allocated() / 1024**3
        self.log_scalar("system/memory_gb", memory_used, step)
        
        # Throughput
        self.log_scalar("performance/tokens_per_sec", metrics['tokens_per_sec'], step)
        self.log_scalar("performance/samples_per_sec", metrics['samples_per_sec'], step)
        
        # Model-specific metrics
        if 'expert_utilization' in metrics:
            self.log_histogram("moe/expert_utilization", 
                              metrics['expert_utilization'], step)
        
        if 'attention_entropy' in metrics:
            self.log_scalar("attention/entropy", metrics['attention_entropy'], step)
    
    def log_model_analysis(self, model, step):
        """
        Log model analysis and health metrics
        """
        # Parameter statistics
        for name, param in model.named_parameters():
            if param.requires_grad:
                # Gradient norms
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2)
                    self.log_scalar(f"gradients/{name}_norm", grad_norm, step)
                
                # Parameter norms
                param_norm = param.data.norm(2)
                self.log_scalar(f"parameters/{name}_norm", param_norm, step)
                
                # Parameter histograms (less frequently)
                if step % 1000 == 0:
                    self.log_histogram(f"parameters/{name}", param.data, step)
```

### Performance Profiling

```python
def profile_training_step(model, batch, step):
    """
    Profile a single training step for optimization
    """
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as profiler:
        
        # Forward pass
        outputs = model(batch['input_ids'])
        loss = compute_loss(outputs, batch)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        optimizer.zero_grad()
    
    # Save profiling results
    if step % 100 == 0:
        profiler.export_chrome_trace(f"trace_{step}.json")
        
        # Analyze bottlenecks
        print(profiler.key_averages().table(sort_by="cuda_time_total"))
```

## 🎯 Fine-tuning Strategies

### Instruction Tuning

```python
def instruction_tuning_setup(base_model, config):
    """
    Setup Ultra-AI for instruction tuning
    """
    # Freeze most parameters, only tune specific components
    for param in base_model.parameters():
        param.requires_grad = False
    
    # Unfreeze specific layers for fine-tuning
    # Top layers
    for layer in base_model.layers[-config.num_finetune_layers:]:
        for param in layer.parameters():
            param.requires_grad = True
    
    # Output head
    for param in base_model.lm_head.parameters():
        param.requires_grad = True
    
    # Some MoE experts for specialization
    for i in range(0, base_model.config.num_experts, 4):  # Every 4th expert
        for param in base_model.moe_layers[0].experts[i].parameters():
            param.requires_grad = True
    
    return base_model
```

### LoRA Fine-tuning

```python
from peft import LoraConfig, get_peft_model

def setup_lora_finetuning(model, config):
    """
    Setup LoRA (Low-Rank Adaptation) for efficient fine-tuning
    """
    lora_config = LoraConfig(
        r=config.lora_rank,                    # Rank of adaptation
        lora_alpha=config.lora_alpha,          # Scaling factor
        target_modules=config.lora_target_modules,  # Target layers
        lora_dropout=config.lora_dropout,      # Dropout for LoRA
        bias="none",                           # Bias adaptation
        task_type="CAUSAL_LM"                  # Task type
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    model.print_trainable_parameters()
    
    return model
```

## 🔧 Troubleshooting

### Common Training Issues

#### Out of Memory (OOM) Errors

```python
def handle_oom_error(model, config):
    """
    Handle OOM errors by reducing memory usage
    """
    print("OOM detected! Applying memory optimizations...")
    
    # 1. Reduce batch size
    config.batch_size = max(1, config.batch_size // 2)
    
    # 2. Increase gradient accumulation
    config.gradient_accumulation_steps *= 2
    
    # 3. Enable gradient checkpointing
    config.gradient_checkpointing = True
    
    # 4. Enable CPU offloading
    config.cpu_offload = True
    
    # 5. Reduce sequence length temporarily  
    config.max_seq_length = min(config.max_seq_length, 32768)
    
    print(f"New batch size: {config.batch_size}")
    print(f"New grad accumulation: {config.gradient_accumulation_steps}")
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
```

#### Gradient Explosion

```python
def handle_gradient_explosion(optimizer, max_grad_norm=1.0):
    """
    Handle gradient explosion with clipping
    """
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(
        optimizer.param_groups[0]['params'], 
        max_grad_norm
    )
    
    # Log gradient norms
    total_norm = 0
    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    
    if total_norm > max_grad_norm:
        print(f"Gradient norm {total_norm:.4f} exceeded {max_grad_norm}, clipped")
    
    return total_norm
```

#### Training Instability

```python
def detect_training_instability(loss_history, window=100):
    """
    Detect training instability and suggest fixes
    """
    if len(loss_history) < window:
        return False, []
    
    recent_losses = loss_history[-window:]
    
    # Check for NaN losses
    if any(math.isnan(loss) for loss in recent_losses):
        return True, ["NaN detected - reduce learning rate", "Check for inf gradients"]
    
    # Check for exploding losses
    if recent_losses[-1] > 2 * recent_losses[0]:
        return True, ["Loss explosion - reduce learning rate", "Enable gradient clipping"]
    
    # Check for oscillating losses
    variance = np.var(recent_losses)
    if variance > np.mean(recent_losses):
        return True, ["High variance - reduce learning rate", "Increase warmup steps"]
    
    return False, []
```

### Performance Optimization

```python
def optimize_training_performance(model, config):
    """
    Apply various performance optimizations
    """
    optimizations_applied = []
    
    # 1. Compile model (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        model = torch.compile(model, mode="max-autotune")
        optimizations_applied.append("Model compilation")
    
    # 2. Use fused optimizers
    if config.optimizer == "adamw":
        config.fused = True
        optimizations_applied.append("Fused AdamW")
    
    # 3. Enable optimized attention
    if hasattr(torch.backends.cuda, 'enable_math_sdp'):
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_flash_sdp(True)
        optimizations_applied.append("Optimized attention")
    
    # 4. Set optimal number of threads
    if config.num_workers == -1:
        config.num_workers = min(32, torch.get_num_threads())
        optimizations_applied.append(f"Optimal thread count: {config.num_workers}")
    
    # 5. Enable TensorFloat-32 (TF32)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    optimizations_applied.append("TF32 enabled")
    
    print("Applied optimizations:", ", ".join(optimizations_applied))
    
    return model
```

## 📈 Training Results

### Expected Performance Metrics

| Model Size | Training Time | Peak Memory | Throughput | Final Loss |
|------------|---------------|-------------|-------------|------------|
| **Ultra-3B** | 2-3 days | 48GB | 5K tok/s/GPU | 2.1 |
| **Ultra-13B** | 1-2 weeks | 128GB | 3K tok/s/GPU | 1.8 |
| **Ultra-52B** | 1-2 months | 512GB | 1K tok/s/GPU | 1.5 |
| **Ultra-390B** | 3-6 months | 2TB | 200 tok/s/GPU | 1.2 |

### Training Checkpoints

The training process creates regular checkpoints containing:

- Model state dict
- Optimizer state  
- Learning rate scheduler state
- Random states for reproducibility
- Training configuration
- Performance metrics

## 🎓 Best Practices

1. **Start Small**: Begin with Ultra-3B before scaling to larger models
2. **Monitor Closely**: Use comprehensive monitoring to catch issues early
3. **Validate Frequently**: Run validation every few thousand steps
4. **Save Often**: Create checkpoints every 1000-5000 steps
5. **Profile Regularly**: Identify and eliminate bottlenecks
6. **Test Distributed**: Validate distributed setup before long training runs
7. **Plan Resources**: Ensure adequate compute, storage, and network bandwidth

This guide provides the foundation for successfully training Ultra-AI models at any scale, from development prototypes to production deployments.