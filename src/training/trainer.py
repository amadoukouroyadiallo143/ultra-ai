import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, List, Tuple
import logging
import time
import os
import json
from pathlib import Path
import wandb
from transformers import get_scheduler
import torch.nn.functional as F

from .loss import UltraAILoss, MultiObjectiveLoss
from .optimizer import get_optimizer
from ..utils.monitoring import MetricsTracker, PerformanceMonitor
from ..utils.checkpointing import CheckpointManager
from ..utils.memory import MemoryManager


logger = logging.getLogger(__name__)


class UltraAITrainer:
    """
    Advanced trainer for the Ultra-AI multimodal model.
    Supports distributed training, mixed precision, and ultra-long contexts.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        loss_fn: Optional[nn.Module] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        output_dir: str = "./checkpoints",
        logging_steps: int = 100,
        eval_steps: int = 1000,
        save_steps: int = 5000,
        max_grad_norm: float = 1.0,
        mixed_precision: bool = True,
        gradient_checkpointing: bool = True,
        use_wandb: bool = True,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config or {}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training parameters
        self.logging_steps = logging_steps
        self.eval_steps = eval_steps
        self.save_steps = save_steps
        self.max_grad_norm = max_grad_norm
        self.mixed_precision = mixed_precision
        self.gradient_checkpointing = gradient_checkpointing
        
        # Initialize optimizer
        if optimizer is None:
            self.optimizer = get_optimizer(model, self.config)
        else:
            self.optimizer = optimizer
            
        # Initialize scheduler
        if scheduler is None:
            num_training_steps = len(train_dataloader) * self.config.get("num_epochs", 3)
            self.scheduler = get_scheduler(
                "cosine",
                optimizer=self.optimizer,
                num_warmup_steps=int(0.1 * num_training_steps),
                num_training_steps=num_training_steps,
            )
        else:
            self.scheduler = scheduler
            
        # Initialize loss function
        if loss_fn is None:
            self.loss_fn = UltraAILoss(self.config)
        else:
            self.loss_fn = loss_fn
            
        # Mixed precision training
        if self.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
            
        # Gradient checkpointing
        if self.gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
            else:
                logger.info("Model does not support gradient_checkpointing_enable, skipping")
            
        # Monitoring and checkpointing - utilise les instances déjà créées dans le main trainer
        self.metrics_tracker = MetricsTracker(window_size=100, save_frequency=1000)
        self.performance_monitor = PerformanceMonitor(monitor_interval=1.0, gpu_monitoring=torch.cuda.is_available())
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=str(Path(self.output_dir) / "checkpoints"))
        self.memory_manager = MemoryManager(enable_auto_cleanup=True, cleanup_threshold=0.85)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float('inf')
        
        # Initialize wandb
        if use_wandb:
            wandb.init(
                project="ultra-ai-model",
                config=self.config,
                name=f"ultra-ai-{int(time.time())}",
            )
            wandb.watch(self.model, log="all")
            
        logger.info(f"Trainer initialized. Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
    def train(self, num_epochs: int) -> Dict[str, Any]:
        """
        Main training loop with comprehensive monitoring and optimization.
        
        Args:
            num_epochs: Number of epochs to train
            
        Returns:
            Training metrics and statistics
        """
        logger.info(f"Starting training for {num_epochs} epochs")
        
        self.model.train()
        total_steps = len(self.train_dataloader) * num_epochs
        
        training_stats = {
            "total_steps": total_steps,
            "epochs": num_epochs,
            "losses": [],
            "learning_rates": [],
            "eval_metrics": [],
        }
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
            
            epoch_stats = self._train_epoch()
            training_stats["losses"].extend(epoch_stats["losses"])
            training_stats["learning_rates"].extend(epoch_stats["learning_rates"])
            
            # Validation
            if self.val_dataloader is not None:
                eval_metrics = self.evaluate()
                training_stats["eval_metrics"].append(eval_metrics)
                
                # Save best model
                if eval_metrics["eval_loss"] < self.best_eval_loss:
                    self.best_eval_loss = eval_metrics["eval_loss"]
                    self.save_checkpoint(is_best=True)
                    
            # End of epoch checkpoint
            self.save_checkpoint()
            
        logger.info("Training completed!")
        return training_stats
        
    def _train_epoch(self) -> Dict[str, Any]:
        """Train for one epoch."""
        epoch_losses = []
        epoch_lrs = []
        
        self.performance_monitor.start_epoch()
        
        for step, batch in enumerate(self.train_dataloader):
            step_start_time = time.time()
            
            # Move batch to device
            batch = self._prepare_batch(batch)
            
            # Forward pass with mixed precision
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss_dict = self.loss_fn(outputs, batch)
                    loss = loss_dict["total_loss"]
            else:
                outputs = self.model(**batch)
                loss_dict = self.loss_fn(outputs, batch)
                loss = loss_dict["total_loss"]
                
            # Backward pass
            if self.mixed_precision:
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                self.optimizer.step()
                
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Update metrics
            self.global_step += 1
            current_lr = self.scheduler.get_last_lr()[0]
            
            epoch_losses.append(loss.item())
            epoch_lrs.append(current_lr)
            
            # Logging and monitoring
            step_time = time.time() - step_start_time
            
            if self.global_step % self.logging_steps == 0:
                self._log_training_step(loss_dict, current_lr, step_time)
                
            if self.global_step % self.save_steps == 0:
                self.save_checkpoint()
                
            if self.val_dataloader and self.global_step % self.eval_steps == 0:
                eval_metrics = self.evaluate()
                self.model.train()  # Return to training mode
                
            # Memory management
            if self.global_step % 100 == 0:
                self.memory_manager.cleanup()
                
        return {"losses": epoch_losses, "learning_rates": epoch_lrs}
        
    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model on validation data.
        
        Returns:
            Evaluation metrics
        """
        logger.info("Starting evaluation")
        
        self.model.eval()
        eval_losses = []
        eval_metrics = {}
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = self._prepare_batch(batch)
                
                if self.mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                        loss_dict = self.loss_fn(outputs, batch)
                else:
                    outputs = self.model(**batch)
                    loss_dict = self.loss_fn(outputs, batch)
                    
                eval_losses.append(loss_dict["total_loss"].item())
                
                # Collect additional metrics
                for key, value in loss_dict.items():
                    if key != "total_loss":
                        if key not in eval_metrics:
                            eval_metrics[key] = []
                        eval_metrics[key].append(value.item() if torch.is_tensor(value) else value)
                        
        # Compute average metrics
        avg_eval_loss = sum(eval_losses) / len(eval_losses)
        avg_metrics = {
            "eval_loss": avg_eval_loss,
            **{key: sum(values) / len(values) for key, values in eval_metrics.items()}
        }
        
        # Log evaluation metrics
        logger.info(f"Evaluation - Loss: {avg_eval_loss:.4f}")
        
        if wandb.run:
            wandb.log({"eval/" + key: value for key, value in avg_metrics.items()})
            
        return avg_metrics
        
    def _prepare_batch(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device and prepare for training."""
        prepared_batch = {}
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                prepared_batch[key] = value.to(self.device, non_blocking=True)
            elif isinstance(value, dict):
                prepared_batch[key] = {
                    k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                    for k, v in value.items()
                }
            else:
                prepared_batch[key] = value
                
        return prepared_batch
        
    def _log_training_step(
        self, 
        loss_dict: Dict[str, torch.Tensor], 
        learning_rate: float, 
        step_time: float
    ):
        """Log training step metrics."""
        # Extract loss components
        total_loss = loss_dict["total_loss"].item()
        
        # Log to console
        logger.info(
            f"Step {self.global_step} | "
            f"Loss: {total_loss:.4f} | "
            f"LR: {learning_rate:.2e} | "
            f"Time: {step_time:.2f}s"
        )
        
        # Log to wandb
        if wandb.run:
            log_dict = {
                "train/total_loss": total_loss,
                "train/learning_rate": learning_rate,
                "train/step_time": step_time,
                "train/global_step": self.global_step,
                "train/epoch": self.epoch,
            }
            
            # Add component losses
            for key, value in loss_dict.items():
                if key != "total_loss":
                    log_dict[f"train/{key}"] = value.item() if torch.is_tensor(value) else value
                    
            # Add memory usage
            if torch.cuda.is_available():
                log_dict["train/gpu_memory_gb"] = torch.cuda.memory_allocated() / 1e9
                
            wandb.log(log_dict)
            
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_data = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
            "config": self.config,
        }
        
        if self.scaler:
            checkpoint_data["scaler_state_dict"] = self.scaler.state_dict()
            
        self.checkpoint_manager.save(checkpoint_data, self.global_step, is_best)
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        self.global_step = checkpoint["global_step"]
        self.epoch = checkpoint["epoch"]
        self.best_eval_loss = checkpoint["best_eval_loss"]
        
        if self.scaler and "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
            
        logger.info(f"Loaded checkpoint from step {self.global_step}")
        
    def generate(
        self,
        input_text: str,
        max_length: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text using the trained model.
        
        Args:
            input_text: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Generated text
        """
        self.model.eval()
        
        # Tokenize input (this would need proper tokenizer)
        # For now, assume input_ids are provided
        input_ids = kwargs.get("input_ids")
        
        if input_ids is None:
            raise ValueError("input_ids must be provided for generation")
            
        input_ids = input_ids.to(self.device)
        
        with torch.no_grad():
            if hasattr(self.model, 'generate'):
                generated = self.model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    **kwargs
                )
            else:
                # Manual generation loop
                generated = self._manual_generation(
                    input_ids, max_length, temperature, do_sample
                )
                
        return generated
        
    def _manual_generation(
        self,
        input_ids: torch.Tensor,
        max_length: int,
        temperature: float,
        do_sample: bool,
    ) -> torch.Tensor:
        """Manual generation loop for models without generate method."""
        batch_size = input_ids.shape[0]
        generated = input_ids.clone()
        
        for _ in range(max_length - input_ids.shape[1]):
            outputs = self.model(input_ids=generated)
            
            if hasattr(outputs, 'logits'):
                next_token_logits = outputs.logits[:, -1, :] / temperature
            else:
                next_token_logits = outputs[:, -1, :] / temperature
                
            if do_sample:
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
            generated = torch.cat([generated, next_token], dim=1)
            
        return generated