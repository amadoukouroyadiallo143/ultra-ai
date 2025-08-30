import torch
import torch.distributed as dist
import os
import shutil
import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import threading
import pickle
from collections import defaultdict

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Advanced checkpoint manager for ultra-large models.
    Supports incremental checkpointing, compression, and distributed saving.
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        save_optimizer: bool = True,
        save_scheduler: bool = True,
        compress_checkpoints: bool = True,
        async_save: bool = True,
        incremental_save: bool = True,
        shard_checkpoints: bool = True,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_optimizer = save_optimizer
        self.save_scheduler = save_scheduler
        self.compress_checkpoints = compress_checkpoints
        self.async_save = async_save
        self.incremental_save = incremental_save
        self.shard_checkpoints = shard_checkpoints
        
        # Track checkpoints
        self.checkpoints = []
        self.last_checkpoint_state = {}
        
        # Async saving
        self.save_thread = None
        self.save_queue = []
        self.save_lock = threading.Lock()
        
        # Load existing checkpoints
        self._discover_checkpoints()
        
        logger.info(f"CheckpointManager initialized: {self.checkpoint_dir}")
        
    def save(
        self,
        checkpoint_data: Dict[str, Any],
        step: int,
        is_best: bool = False,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Save checkpoint with optional async and incremental saving.
        
        Args:
            checkpoint_data: Data to save
            step: Training step
            is_best: Whether this is the best checkpoint
            metadata: Additional metadata
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint-{step}"
        
        # Prepare save data
        save_data = {
            'checkpoint_data': checkpoint_data,
            'step': step,
            'is_best': is_best,
            'checkpoint_path': checkpoint_path,
            'metadata': metadata or {},
            'timestamp': time.time(),
        }
        
        if self.async_save:
            self._queue_async_save(save_data)
        else:
            self._save_checkpoint(save_data)
            
    def _queue_async_save(self, save_data: Dict[str, Any]):
        """Queue checkpoint for async saving."""
        with self.save_lock:
            self.save_queue.append(save_data)
            
        # Start save thread if not running
        if self.save_thread is None or not self.save_thread.is_alive():
            self.save_thread = threading.Thread(target=self._async_save_worker, daemon=True)
            self.save_thread.start()
            
    def _async_save_worker(self):
        """Worker thread for async checkpoint saving."""
        while True:
            with self.save_lock:
                if not self.save_queue:
                    break
                save_data = self.save_queue.pop(0)
                
            try:
                self._save_checkpoint(save_data)
            except Exception as e:
                logger.error(f"Error saving checkpoint: {e}")
                
    def _save_checkpoint(self, save_data: Dict[str, Any]):
        """Internal checkpoint saving logic."""
        checkpoint_data = save_data['checkpoint_data']
        step = save_data['step']
        is_best = save_data['is_best']
        checkpoint_path = save_data['checkpoint_path']
        metadata = save_data['metadata']
        
        start_time = time.time()
        
        try:
            # Create checkpoint directory
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            
            # Save model state
            if self.shard_checkpoints:
                self._save_sharded_checkpoint(checkpoint_data, checkpoint_path)
            else:
                self._save_unified_checkpoint(checkpoint_data, checkpoint_path)
                
            # Save metadata
            checkpoint_metadata = {
                'step': step,
                'timestamp': save_data['timestamp'],
                'save_duration': time.time() - start_time,
                **metadata
            }
            
            with open(checkpoint_path / "metadata.json", 'w') as f:
                json.dump(checkpoint_metadata, f, indent=2)
                
            # Update checkpoint tracking
            self.checkpoints.append({
                'step': step,
                'path': checkpoint_path,
                'is_best': is_best,
                'timestamp': save_data['timestamp'],
            })
            
            # Copy best checkpoint
            if is_best:
                best_path = self.checkpoint_dir / "best"
                if best_path.exists():
                    shutil.rmtree(best_path)
                shutil.copytree(checkpoint_path, best_path)
                logger.info(f"Saved best checkpoint at step {step}")
                
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
            save_time = time.time() - start_time
            logger.info(f"Saved checkpoint at step {step} in {save_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint at step {step}: {e}")
            if checkpoint_path.exists():
                shutil.rmtree(checkpoint_path)
                
    def _save_sharded_checkpoint(self, checkpoint_data: Dict[str, Any], checkpoint_path: Path):
        """Save checkpoint with sharding for large models."""
        model_state_dict = checkpoint_data.get('model_state_dict', {})
        
        # Group parameters by layer/module
        sharded_states = defaultdict(dict)
        
        for param_name, param_tensor in model_state_dict.items():
            # Determine shard based on parameter name
            parts = param_name.split('.')
            if len(parts) >= 2:
                shard_key = f"{parts[0]}.{parts[1]}"  # e.g., "layers.0"
            else:
                shard_key = "misc"
                
            sharded_states[shard_key][param_name] = param_tensor
            
        # Save each shard
        shard_info = {}
        for shard_key, shard_state in sharded_states.items():
            shard_filename = f"model_shard_{shard_key.replace('.', '_')}.pt"
            shard_path = checkpoint_path / shard_filename
            
            if self.compress_checkpoints:
                torch.save(shard_state, shard_path, pickle_protocol=pickle.HIGHEST_PROTOCOL)
            else:
                torch.save(shard_state, shard_path)
                
            shard_info[shard_key] = {
                'filename': shard_filename,
                'param_count': sum(p.numel() for p in shard_state.values()),
                'size_mb': shard_path.stat().st_size / (1024 * 1024),
            }
            
        # Save shard index
        with open(checkpoint_path / "shard_index.json", 'w') as f:
            json.dump(shard_info, f, indent=2)
            
        # Save non-model components
        other_data = {k: v for k, v in checkpoint_data.items() if k != 'model_state_dict'}
        if other_data:
            torch.save(other_data, checkpoint_path / "training_state.pt")
            
    def _save_unified_checkpoint(self, checkpoint_data: Dict[str, Any], checkpoint_path: Path):
        """Save unified checkpoint file."""
        checkpoint_file = checkpoint_path / "checkpoint.pt"
        
        if self.compress_checkpoints:
            torch.save(checkpoint_data, checkpoint_file, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        else:
            torch.save(checkpoint_data, checkpoint_file)
            
    def load(self, checkpoint_path: str, device: Optional[torch.device] = None) -> Dict[str, Any]:
        """
        Load checkpoint from path.
        
        Args:
            checkpoint_path: Path to checkpoint
            device: Device to load checkpoint to
            
        Returns:
            Loaded checkpoint data
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        start_time = time.time()
        
        try:
            if self.shard_checkpoints and (checkpoint_path / "shard_index.json").exists():
                checkpoint_data = self._load_sharded_checkpoint(checkpoint_path, device)
            else:
                checkpoint_data = self._load_unified_checkpoint(checkpoint_path, device)
                
            load_time = time.time() - start_time
            logger.info(f"Loaded checkpoint in {load_time:.2f}s")
            
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise
            
    def _load_sharded_checkpoint(self, checkpoint_path: Path, device: Optional[torch.device] = None) -> Dict[str, Any]:
        """Load sharded checkpoint."""
        # Load shard index
        with open(checkpoint_path / "shard_index.json", 'r') as f:
            shard_info = json.load(f)
            
        # Load all shards
        model_state_dict = {}
        for shard_key, info in shard_info.items():
            shard_path = checkpoint_path / info['filename']
            shard_state = torch.load(shard_path, map_location=device)
            model_state_dict.update(shard_state)
            
        # Load training state
        training_state_path = checkpoint_path / "training_state.pt"
        if training_state_path.exists():
            training_state = torch.load(training_state_path, map_location=device)
        else:
            training_state = {}
            
        # Combine
        checkpoint_data = {
            'model_state_dict': model_state_dict,
            **training_state
        }
        
        return checkpoint_data
        
    def _load_unified_checkpoint(self, checkpoint_path: Path, device: Optional[torch.device] = None) -> Dict[str, Any]:
        """Load unified checkpoint."""
        checkpoint_file = checkpoint_path / "checkpoint.pt"
        return torch.load(checkpoint_file, map_location=device)
        
    def get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to the latest checkpoint."""
        if not self.checkpoints:
            return None
            
        # Sort by step and return latest
        latest = max(self.checkpoints, key=lambda x: x['step'])
        return latest['path']
        
    def get_best_checkpoint(self) -> Optional[Path]:
        """Get path to the best checkpoint."""
        best_path = self.checkpoint_dir / "best"
        return best_path if best_path.exists() else None
        
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        return sorted(self.checkpoints, key=lambda x: x['step'])
        
    def _discover_checkpoints(self):
        """Discover existing checkpoints in the directory."""
        if not self.checkpoint_dir.exists():
            return
            
        for item in self.checkpoint_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                try:
                    step = int(item.name.split("-")[1])
                    metadata_file = item / "metadata.json"
                    
                    timestamp = item.stat().st_mtime
                    is_best = False
                    
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        timestamp = metadata.get('timestamp', timestamp)
                        
                    self.checkpoints.append({
                        'step': step,
                        'path': item,
                        'is_best': is_best,
                        'timestamp': timestamp,
                    })
                    
                except (ValueError, IndexError):
                    logger.warning(f"Invalid checkpoint directory: {item}")
                    
        logger.info(f"Discovered {len(self.checkpoints)} existing checkpoints")
        
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints limit."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
            
        # Sort by step and keep only recent ones
        sorted_checkpoints = sorted(self.checkpoints, key=lambda x: x['step'])
        checkpoints_to_remove = sorted_checkpoints[:-self.max_checkpoints]
        
        for checkpoint in checkpoints_to_remove:
            if not checkpoint['is_best']:  # Never remove best checkpoints
                try:
                    shutil.rmtree(checkpoint['path'])
                    self.checkpoints.remove(checkpoint)
                    logger.info(f"Removed old checkpoint: {checkpoint['path']}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {checkpoint['path']}: {e}")
                    
    def cleanup_async_saves(self):
        """Wait for all async saves to complete."""
        if self.save_thread and self.save_thread.is_alive():
            self.save_thread.join()
            logger.info("Completed all async checkpoint saves")


class IncrementalCheckpointer:
    """
    Incremental checkpointing to save only changed parameters.
    Reduces checkpoint size and save time for ultra-large models.
    """
    
    def __init__(self, threshold: float = 1e-6):
        self.threshold = threshold  # Minimum change threshold
        self.last_state = {}
        self.parameter_hashes = {}
        
    def save_incremental(
        self,
        current_state: Dict[str, torch.Tensor],
        checkpoint_path: Path,
        full_checkpoint: bool = False,
    ):
        """
        Save only parameters that have changed significantly.
        
        Args:
            current_state: Current model state dict
            checkpoint_path: Path to save checkpoint
            full_checkpoint: Force full checkpoint save
        """
        if full_checkpoint or not self.last_state:
            # Save full checkpoint
            changed_params = current_state
            change_type = "full"
        else:
            # Find changed parameters
            changed_params = self._find_changed_parameters(current_state)
            change_type = "incremental"
            
        # Save changed parameters
        checkpoint_data = {
            'changed_parameters': changed_params,
            'change_type': change_type,
            'parameter_names': list(changed_params.keys()),
            'total_parameters': len(current_state),
            'changed_parameters_count': len(changed_params),
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        
        # Update tracking
        self.last_state = {name: param.clone().detach() for name, param in current_state.items()}
        self._update_parameter_hashes(current_state)
        
        logger.info(f"Saved {change_type} checkpoint: {len(changed_params)}/{len(current_state)} parameters")
        
    def _find_changed_parameters(self, current_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Find parameters that have changed beyond threshold."""
        changed_params = {}
        
        for name, current_param in current_state.items():
            if name not in self.last_state:
                # New parameter
                changed_params[name] = current_param
            else:
                last_param = self.last_state[name]
                
                # Check if parameter has changed significantly
                if current_param.shape != last_param.shape:
                    # Shape change - always include
                    changed_params[name] = current_param
                else:
                    # Check magnitude of change
                    param_diff = torch.abs(current_param - last_param)
                    max_change = torch.max(param_diff).item()
                    
                    if max_change > self.threshold:
                        changed_params[name] = current_param
                        
        return changed_params
        
    def _update_parameter_hashes(self, current_state: Dict[str, torch.Tensor]):
        """Update parameter hashes for faster change detection."""
        for name, param in current_state.items():
            # Simple hash based on parameter statistics
            param_hash = hash((
                param.shape,
                param.mean().item(),
                param.std().item(),
                param.min().item(),
                param.max().item(),
            ))
            self.parameter_hashes[name] = param_hash
            
    def load_incremental(self, checkpoint_paths: List[Path], base_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Load incremental checkpoints and apply changes.
        
        Args:
            checkpoint_paths: List of checkpoint paths in order
            base_state: Base state to apply changes to
            
        Returns:
            Reconstructed full state
        """
        current_state = base_state.copy()
        
        for checkpoint_path in checkpoint_paths:
            checkpoint_data = torch.load(checkpoint_path)
            
            if checkpoint_data['change_type'] == 'full':
                # Full checkpoint - replace everything
                current_state.update(checkpoint_data['changed_parameters'])
            else:
                # Incremental - apply changes
                current_state.update(checkpoint_data['changed_parameters'])
                
        return current_state


class DistributedCheckpointManager:
    """
    Distributed checkpointing for multi-GPU training.
    Coordinates checkpoint saving across multiple processes.
    """
    
    def __init__(self, checkpoint_dir: str, world_size: int, rank: int):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.world_size = world_size
        self.rank = rank
        self.is_main_process = rank == 0
        
        if self.is_main_process:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
    def save_distributed(
        self,
        model_state: Dict[str, torch.Tensor],
        optimizer_state: Optional[Dict[str, Any]] = None,
        step: int = 0,
    ):
        """
        Save checkpoint in distributed manner.
        
        Args:
            model_state: Model state dict (may be sharded)
            optimizer_state: Optimizer state (may be sharded)
            step: Training step
        """
        checkpoint_path = self.checkpoint_dir / f"checkpoint-{step}"
        
        if self.is_main_process:
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            
        # Synchronize all processes
        if dist.is_initialized():
            dist.barrier()
            
        # Each process saves its shard
        rank_checkpoint_path = checkpoint_path / f"rank_{self.rank}.pt"
        
        checkpoint_data = {
            'model_state_dict': model_state,
            'rank': self.rank,
            'world_size': self.world_size,
            'step': step,
        }
        
        if optimizer_state is not None:
            checkpoint_data['optimizer_state_dict'] = optimizer_state
            
        torch.save(checkpoint_data, rank_checkpoint_path)
        
        # Main process saves metadata
        if self.is_main_process:
            metadata = {
                'step': step,
                'world_size': self.world_size,
                'timestamp': time.time(),
                'rank_files': [f"rank_{i}.pt" for i in range(self.world_size)],
            }
            
            with open(checkpoint_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
                
        # Final synchronization
        if dist.is_initialized():
            dist.barrier()
            
        logger.info(f"Saved distributed checkpoint at step {step} (rank {self.rank})")
        
    def load_distributed(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load distributed checkpoint for current rank.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            
        Returns:
            Checkpoint data for current rank
        """
        checkpoint_path = Path(checkpoint_path)
        rank_checkpoint_path = checkpoint_path / f"rank_{self.rank}.pt"
        
        if not rank_checkpoint_path.exists():
            raise FileNotFoundError(f"Rank checkpoint not found: {rank_checkpoint_path}")
            
        checkpoint_data = torch.load(rank_checkpoint_path)
        logger.info(f"Loaded distributed checkpoint (rank {self.rank})")
        
        return checkpoint_data
        
    def gather_full_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """
        Gather full checkpoint from all ranks (main process only).
        
        Args:
            checkpoint_path: Path to checkpoint directory
            
        Returns:
            Full checkpoint data (main process only)
        """
        if not self.is_main_process:
            return None
            
        checkpoint_path = Path(checkpoint_path)
        
        # Load metadata
        with open(checkpoint_path / "metadata.json", 'r') as f:
            metadata = json.load(f)
            
        # Load all rank checkpoints
        full_model_state = {}
        full_optimizer_state = {}
        
        for rank in range(metadata['world_size']):
            rank_file = checkpoint_path / f"rank_{rank}.pt"
            rank_data = torch.load(rank_file)
            
            full_model_state.update(rank_data['model_state_dict'])
            
            if 'optimizer_state_dict' in rank_data:
                full_optimizer_state.update(rank_data['optimizer_state_dict'])
                
        # Combine into full checkpoint
        full_checkpoint = {
            'model_state_dict': full_model_state,
            'step': metadata['step'],
            'world_size': metadata['world_size'],
        }
        
        if full_optimizer_state:
            full_checkpoint['optimizer_state_dict'] = full_optimizer_state
            
        return full_checkpoint