import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import os
import logging
from typing import Dict, Any, Optional
import deepspeed
from accelerate import Accelerator

from .trainer import UltraAITrainer


logger = logging.getLogger(__name__)


class DistributedTrainer(UltraAITrainer):
    """
    Distributed trainer supporting multiple parallelism strategies:
    - Data Parallel
    - Model Parallel  
    - Pipeline Parallel
    - Zero Redundancy Optimizer (ZeRO)
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader,
        val_dataloader=None,
        config: Optional[Dict[str, Any]] = None,
        distributed_backend: str = "nccl",
        parallelism_strategy: str = "data_parallel",  # data_parallel, model_parallel, pipeline_parallel, deepspeed
        zero_stage: int = 2,  # For DeepSpeed ZeRO
        **kwargs
    ):
        self.distributed_backend = distributed_backend
        self.parallelism_strategy = parallelism_strategy
        self.zero_stage = zero_stage
        
        # Initialize distributed training
        self._init_distributed()
        
        # Setup model parallelism
        model = self._setup_parallelism(model)
        
        super().__init__(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            config=config,
            **kwargs
        )
        
    def _init_distributed(self):
        """Initialize distributed training environment."""
        # Get distributed training parameters
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))
        
        # Set device
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")
            
        # Initialize process group
        if self.world_size > 1:
            dist.init_process_group(
                backend=self.distributed_backend,
                init_method="env://",
                world_size=self.world_size,
                rank=self.rank,
            )
            
        logger.info(f"Initialized distributed training: rank {self.rank}/{self.world_size}")
        
    def _setup_parallelism(self, model: nn.Module) -> nn.Module:
        """Setup the chosen parallelism strategy."""
        if self.parallelism_strategy == "data_parallel":
            return self._setup_data_parallel(model)
        elif self.parallelism_strategy == "model_parallel":
            return self._setup_model_parallel(model)
        elif self.parallelism_strategy == "pipeline_parallel":
            return self._setup_pipeline_parallel(model)
        elif self.parallelism_strategy == "deepspeed":
            return self._setup_deepspeed(model)
        elif self.parallelism_strategy == "accelerate":
            return self._setup_accelerate(model)
        else:
            raise ValueError(f"Unknown parallelism strategy: {self.parallelism_strategy}")
            
    def _setup_data_parallel(self, model: nn.Module) -> nn.Module:
        """Setup standard data parallelism."""
        model = model.to(self.device)
        
        if self.world_size > 1:
            model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True,  # For complex architectures
            )
            
        logger.info("Set up data parallel training")
        return model
        
    def _setup_model_parallel(self, model: nn.Module) -> nn.Module:
        """Setup model parallelism for large models."""
        # This is a simplified example - actual implementation would be model-specific
        if hasattr(model, 'parallelize'):
            model.parallelize()
        else:
            # Manual model parallelism setup
            if self.world_size > 1:
                # Distribute layers across GPUs
                num_layers = len(model.layers) if hasattr(model, 'layers') else 1
                layers_per_device = num_layers // self.world_size
                
                start_layer = self.rank * layers_per_device
                end_layer = min(start_layer + layers_per_device, num_layers)
                
                # Move specific layers to this device
                for i in range(len(model.layers)):
                    if start_layer <= i < end_layer:
                        model.layers[i] = model.layers[i].to(self.device)
                    else:
                        model.layers[i] = model.layers[i].to(torch.device("cpu"))
                        
        logger.info(f"Set up model parallel training on rank {self.rank}")
        return model
        
    def _setup_pipeline_parallel(self, model: nn.Module) -> nn.Module:
        """Setup pipeline parallelism."""
        try:
            from torch.distributed.pipeline.sync import Pipe
            
            # Convert model to pipeline stages
            if hasattr(model, 'layers'):
                layers_per_stage = len(model.layers) // self.world_size
                stages = []
                
                for stage in range(self.world_size):
                    start = stage * layers_per_stage
                    end = min(start + layers_per_stage, len(model.layers))
                    stage_layers = nn.Sequential(*model.layers[start:end])
                    stages.append(stage_layers)
                    
                # Create pipeline
                model = Pipe(
                    nn.Sequential(*stages),
                    balance=[layers_per_stage] * self.world_size,
                    devices=[f"cuda:{i}" for i in range(self.world_size)],
                    chunks=8,  # Microbatch size
                )
                
            logger.info("Set up pipeline parallel training")
            
        except ImportError:
            logger.warning("Pipeline parallelism not available, falling back to data parallel")
            model = self._setup_data_parallel(model)
            
        return model
        
    def _setup_deepspeed(self, model: nn.Module) -> nn.Module:
        """Setup DeepSpeed for memory-efficient training."""
        # DeepSpeed configuration
        ds_config = {
            "train_batch_size": self.config.get("batch_size", 32),
            "gradient_accumulation_steps": self.config.get("gradient_accumulation_steps", 1),
            "fp16": {
                "enabled": self.config.get("mixed_precision", True),
                "auto_cast": True,
            },
            "zero_optimization": {
                "stage": self.zero_stage,
                "offload_optimizer": {
                    "device": "cpu" if self.zero_stage >= 2 else "none"
                },
                "offload_param": {
                    "device": "cpu" if self.zero_stage == 3 else "none"
                },
            },
            "activation_checkpointing": {
                "partition_activations": True,
                "cpu_checkpointing": True,
                "contiguous_memory_optimization": False,
                "number_checkpoints": None,
                "synchronize_checkpoint_boundary": False,
            },
            "wall_clock_breakdown": False,
        }
        
        # Initialize DeepSpeed
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config,
        )
        
        self.model_engine = model_engine
        self.optimizer = optimizer  # Override optimizer with DeepSpeed optimizer
        
        logger.info(f"Set up DeepSpeed with ZeRO stage {self.zero_stage}")
        return model_engine
        
    def _setup_accelerate(self, model: nn.Module) -> nn.Module:
        """Setup Accelerate for automatic distributed training."""
        # Initialize accelerator
        self.accelerator = Accelerator(
            mixed_precision="fp16" if self.config.get("mixed_precision", True) else "no",
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 1),
            log_with="wandb" if self.config.get("use_wandb", True) else None,
        )
        
        # Prepare model, optimizer, and dataloaders
        model, self.optimizer, self.train_dataloader, self.val_dataloader = self.accelerator.prepare(
            model, self.optimizer, self.train_dataloader, self.val_dataloader
        )
        
        self.device = self.accelerator.device
        logger.info("Set up Accelerate for automatic distributed training")
        
        return model
        
    def train(self, num_epochs: int) -> Dict[str, Any]:
        """Override training for distributed-specific optimizations."""
        if self.parallelism_strategy == "deepspeed":
            return self._train_deepspeed(num_epochs)
        elif self.parallelism_strategy == "accelerate":
            return self._train_accelerate(num_epochs)
        else:
            return super().train(num_epochs)
            
    def _train_deepspeed(self, num_epochs: int) -> Dict[str, Any]:
        """Training loop optimized for DeepSpeed."""
        logger.info(f"Starting DeepSpeed training for {num_epochs} epochs")
        
        training_stats = {
            "total_steps": len(self.train_dataloader) * num_epochs,
            "epochs": num_epochs,
            "losses": [],
            "learning_rates": [],
        }
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            for step, batch in enumerate(self.train_dataloader):
                batch = self._prepare_batch(batch)
                
                # Forward pass
                outputs = self.model_engine(**batch)
                loss_dict = self.loss_fn(outputs, batch)
                loss = loss_dict["total_loss"]
                
                # Backward pass
                self.model_engine.backward(loss)
                self.model_engine.step()
                
                # Update tracking
                self.global_step += 1
                training_stats["losses"].append(loss.item())
                
                # Logging
                if self.global_step % self.logging_steps == 0:
                    lr = self.model_engine.get_lr()[0]
                    training_stats["learning_rates"].append(lr)
                    self._log_training_step(loss_dict, lr, 0.0)
                    
                # Checkpointing
                if self.global_step % self.save_steps == 0:
                    self._save_deepspeed_checkpoint()
                    
        return training_stats
        
    def _train_accelerate(self, num_epochs: int) -> Dict[str, Any]:
        """Training loop optimized for Accelerate."""
        logger.info(f"Starting Accelerate training for {num_epochs} epochs")
        
        training_stats = {
            "total_steps": len(self.train_dataloader) * num_epochs,
            "epochs": num_epochs,
            "losses": [],
            "learning_rates": [],
        }
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            for step, batch in enumerate(self.train_dataloader):
                with self.accelerator.accumulate(self.model):
                    outputs = self.model(**batch)
                    loss_dict = self.loss_fn(outputs, batch)
                    loss = loss_dict["total_loss"]
                    
                    # Backward pass
                    self.accelerator.backward(loss)
                    
                    # Gradient clipping
                    if self.max_grad_norm > 0:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                        
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                # Update tracking
                self.global_step += 1
                training_stats["losses"].append(self.accelerator.gather(loss).mean().item())
                
                # Logging
                if self.global_step % self.logging_steps == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    training_stats["learning_rates"].append(lr)
                    
                    if self.accelerator.is_main_process:
                        self._log_training_step(loss_dict, lr, 0.0)
                        
                # Checkpointing
                if self.global_step % self.save_steps == 0:
                    self._save_accelerate_checkpoint()
                    
        return training_stats
        
    def _save_deepspeed_checkpoint(self):
        """Save DeepSpeed checkpoint."""
        checkpoint_dir = self.output_dir / f"checkpoint-{self.global_step}"
        self.model_engine.save_checkpoint(str(checkpoint_dir))
        
        # Save additional training state
        training_state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
            "config": self.config,
        }
        
        torch.save(training_state, checkpoint_dir / "training_state.pt")
        logger.info(f"Saved DeepSpeed checkpoint to {checkpoint_dir}")
        
    def _save_accelerate_checkpoint(self):
        """Save Accelerate checkpoint."""
        if self.accelerator.is_main_process:
            checkpoint_dir = self.output_dir / f"checkpoint-{self.global_step}"
            checkpoint_dir.mkdir(exist_ok=True)
            
            # Save model and training state
            self.accelerator.save_state(str(checkpoint_dir))
            
            training_state = {
                "global_step": self.global_step,
                "epoch": self.epoch,
                "best_eval_loss": self.best_eval_loss,
                "config": self.config,
            }
            
            torch.save(training_state, checkpoint_dir / "training_state.pt")
            logger.info(f"Saved Accelerate checkpoint to {checkpoint_dir}")
            
    def load_checkpoint(self, checkpoint_path: str):
        """Load distributed checkpoint."""
        if self.parallelism_strategy == "deepspeed":
            self.model_engine.load_checkpoint(checkpoint_path)
            
            # Load additional training state
            training_state_path = os.path.join(checkpoint_path, "training_state.pt")
            if os.path.exists(training_state_path):
                training_state = torch.load(training_state_path)
                self.global_step = training_state["global_step"]
                self.epoch = training_state["epoch"]
                self.best_eval_loss = training_state["best_eval_loss"]
                
        elif self.parallelism_strategy == "accelerate":
            self.accelerator.load_state(checkpoint_path)
            
            # Load additional training state
            training_state_path = os.path.join(checkpoint_path, "training_state.pt")
            if os.path.exists(training_state_path):
                training_state = torch.load(training_state_path)
                self.global_step = training_state["global_step"]
                self.epoch = training_state["epoch"]
                self.best_eval_loss = training_state["best_eval_loss"]
                
        else:
            super().load_checkpoint(checkpoint_path)
            
        logger.info(f"Loaded distributed checkpoint from {checkpoint_path}")
        
    def cleanup(self):
        """Clean up distributed training resources."""
        if self.world_size > 1:
            dist.destroy_process_group()
            
        if hasattr(self, 'accelerator'):
            self.accelerator.free_memory()
            
        logger.info("Cleaned up distributed training resources")


class FSDPTrainer(DistributedTrainer):
    """
    Fully Sharded Data Parallel (FSDP) trainer for ultra-large models.
    """
    
    def __init__(self, model: nn.Module, **kwargs):
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
            
            self.fsdp_available = True
        except ImportError:
            logger.error("FSDP not available in this PyTorch version")
            self.fsdp_available = False
            
        super().__init__(model, parallelism_strategy="fsdp", **kwargs)
        
    def _setup_parallelism(self, model: nn.Module) -> nn.Module:
        """Setup FSDP parallelism."""
        if not self.fsdp_available:
            return self._setup_data_parallel(model)
            
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        
        # Auto-wrap policy for transformer layers
        auto_wrap_policy = transformer_auto_wrap_policy
        
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            cpu_offload=CPUOffload(offload_params=True),
            mixed_precision=None,  # Handle separately
            sharding_strategy=self.config.get("sharding_strategy", "FULL_SHARD"),
            device_id=self.local_rank,
        )
        
        logger.info("Set up FSDP training")
        return model