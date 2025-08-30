"""
Système de gradient checkpointing intelligent pour Ultra-AI
- Sélection automatique des couches à checkpoint
- Optimisation basée sur la mémoire et la vitesse
- Stratégies adaptatives par composant
- Monitoring et ajustement dynamique
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from typing import Dict, List, Optional, Tuple, Any, Callable
import math
import psutil
import gc
from dataclasses import dataclass
from collections import defaultdict, deque
import time
import warnings


@dataclass
class CheckpointConfig:
    """Configuration du gradient checkpointing."""
    # Stratégies générales
    enable_checkpointing: bool = True
    adaptive_strategy: bool = True
    memory_threshold: float = 0.85  # Seuil d'utilisation mémoire pour activer
    
    # Stratégies par composant
    mamba_checkpoint_ratio: float = 0.5  # 50% des couches Mamba
    attention_checkpoint_ratio: float = 0.3  # 30% des couches attention
    moe_checkpoint_ratio: float = 0.7  # 70% des couches MoE (plus coûteuses)
    multimodal_checkpoint: bool = True
    
    # Paramètres d'optimisation
    preserve_rng_state: bool = True
    use_reentrant: bool = False  # Plus efficace mais moins compatible
    
    # Seuils de décision
    min_memory_benefit: float = 0.1  # 10% minimum d'économie mémoire
    max_compute_overhead: float = 0.3  # 30% maximum de surcoût calcul
    
    # Monitoring
    profile_layers: bool = False
    adjust_dynamically: bool = True


class LayerProfiler:
    """Profileur pour analyser les couches et décider du checkpointing."""
    
    def __init__(self):
        self.layer_stats = defaultdict(dict)
        self.memory_tracker = MemoryTracker()
        
    def profile_layer(self, layer_name: str, layer: nn.Module, 
                     sample_input: torch.Tensor) -> Dict[str, Any]:
        """Profiler une couche pour analyser mémoire/calcul."""
        stats = {}
        
        # Mesurer la mémoire avant
        initial_memory = self.memory_tracker.get_memory_usage()
        
        # Forward pass sans gradient
        with torch.no_grad():
            start_time = time.time()
            _ = layer(sample_input)
            forward_time = time.time() - start_time
        
        # Forward pass avec gradient
        layer.train()
        sample_input.requires_grad_(True)
        
        start_time = time.time()
        output = layer(sample_input)
        forward_grad_time = time.time() - start_time
        
        peak_memory = self.memory_tracker.get_memory_usage()
        memory_used = peak_memory - initial_memory
        
        # Backward pass
        dummy_loss = output.sum()
        start_time = time.time()
        dummy_loss.backward()
        backward_time = time.time() - start_time
        
        final_memory = self.memory_tracker.get_memory_usage()
        
        stats = {
            'forward_time': forward_time,
            'forward_grad_time': forward_grad_time,
            'backward_time': backward_time,
            'memory_used': memory_used,
            'peak_memory': peak_memory,
            'parameter_count': sum(p.numel() for p in layer.parameters()),
            'activation_size': output.numel() * output.element_size(),
        }
        
        self.layer_stats[layer_name] = stats
        return stats
    
    def recommend_checkpointing(self, layer_name: str, config: CheckpointConfig) -> bool:
        """Recommander si une couche doit être checkpointée."""
        if layer_name not in self.layer_stats:
            # Par défaut, checkpoint les couches coûteuses
            return 'moe' in layer_name.lower() or 'attention' in layer_name.lower()
        
        stats = self.layer_stats[layer_name]
        
        # Critères de décision
        high_memory = stats['memory_used'] > 100 * 1024 * 1024  # > 100MB
        high_activation = stats['activation_size'] > 50 * 1024 * 1024  # > 50MB
        reasonable_compute = stats['backward_time'] < 0.1  # < 100ms
        
        return high_memory and (high_activation or reasonable_compute)


class MemoryTracker:
    """Suivi de l'utilisation mémoire GPU/CPU."""
    
    def __init__(self):
        self.peak_memory = 0
        self.memory_history = deque(maxlen=100)
        
    def get_memory_usage(self) -> float:
        """Obtenir l'utilisation mémoire actuelle en MB."""
        if torch.cuda.is_available():
            memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
        else:
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 ** 2)
        
        self.memory_history.append(memory_mb)
        self.peak_memory = max(self.peak_memory, memory_mb)
        
        return memory_mb
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Obtenir les statistiques mémoire."""
        if not self.memory_history:
            return {'current': 0, 'peak': 0, 'average': 0, 'available': self._get_available_memory()}
        
        current = self.memory_history[-1]
        average = sum(self.memory_history) / len(self.memory_history)
        
        return {
            'current': current,
            'peak': self.peak_memory,
            'average': average,
            'available': self._get_available_memory()
        }
    
    def _get_available_memory(self) -> float:
        """Obtenir la mémoire disponible."""
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        else:
            return psutil.virtual_memory().available / (1024 ** 2)


class SmartCheckpointer:
    """Gestionnaire intelligent du gradient checkpointing."""
    
    def __init__(self, config: CheckpointConfig):
        self.config = config
        self.profiler = LayerProfiler()
        self.memory_tracker = MemoryTracker()
        self.checkpointed_layers = set()
        self.layer_decisions = {}
        self.performance_history = deque(maxlen=50)
        
    def should_checkpoint_layer(self, layer_name: str, layer: nn.Module) -> bool:
        """Décider si une couche doit être checkpointée."""
        if not self.config.enable_checkpointing:
            return False
        
        # Vérifier le seuil mémoire global
        memory_stats = self.memory_tracker.get_memory_stats()
        memory_usage_ratio = memory_stats['current'] / memory_stats['available']
        
        if memory_usage_ratio < self.config.memory_threshold and not self.config.adaptive_strategy:
            return False
        
        # Décision basée sur le profiling si disponible
        if self.config.profile_layers:
            return self.profiler.recommend_checkpointing(layer_name, self.config)
        
        # Stratégie heuristique par type de composant
        return self._heuristic_decision(layer_name, layer)
    
    def _heuristic_decision(self, layer_name: str, layer: nn.Module) -> bool:
        """Décision heuristique basée sur le type de couche."""
        layer_lower = layer_name.lower()
        
        # MoE layers (plus coûteuses)
        if 'moe' in layer_lower or 'expert' in layer_lower:
            layer_hash = hash(layer_name) % 100
            return layer_hash < (self.config.moe_checkpoint_ratio * 100)
        
        # Attention layers
        if 'attention' in layer_lower or 'attn' in layer_lower:
            layer_hash = hash(layer_name) % 100
            return layer_hash < (self.config.attention_checkpoint_ratio * 100)
        
        # Mamba layers
        if 'mamba' in layer_lower:
            layer_hash = hash(layer_name) % 100
            return layer_hash < (self.config.mamba_checkpoint_ratio * 100)
        
        # Multimodal layers
        if 'multimodal' in layer_lower or 'fusion' in layer_lower:
            return self.config.multimodal_checkpoint
        
        # Par défaut, pas de checkpoint pour les autres couches
        return False
    
    def checkpoint_function(self, function: Callable, *args, **kwargs):
        """Appliquer le checkpointing à une fonction."""
        if self.config.enable_checkpointing:
            return checkpoint(
                function, 
                *args, 
                use_reentrant=self.config.use_reentrant,
                preserve_rng_state=self.config.preserve_rng_state,
                **kwargs
            )
        else:
            return function(*args, **kwargs)
    
    def checkpoint_sequential(self, functions: List[Callable], segments: int, *args):
        """Appliquer le checkpointing séquentiel."""
        if self.config.enable_checkpointing and segments > 1:
            return checkpoint_sequential(
                functions, 
                segments, 
                *args,
                preserve_rng_state=self.config.preserve_rng_state
            )
        else:
            # Exécution normale
            input_data = args[0]
            for function in functions:
                input_data = function(input_data)
            return input_data
    
    def profile_and_optimize(self, model: nn.Module, sample_batch: torch.Tensor):
        """Profiler le modèle et optimiser la stratégie de checkpointing."""
        if not self.config.profile_layers:
            return
        
        print("Profiling du modèle pour optimiser le checkpointing...")
        
        model.eval()
        layer_count = 0
        
        # Profiler chaque couche
        for name, layer in model.named_modules():
            if isinstance(layer, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                if layer_count >= 10:  # Limiter le profiling pour la vitesse
                    break
                
                try:
                    print(f"  Profiling: {name}")
                    self.profiler.profile_layer(name, layer, sample_batch)
                    layer_count += 1
                except Exception as e:
                    warnings.warn(f"Erreur profiling {name}: {e}")
                    continue
        
        # Analyser les résultats et ajuster la stratégie
        self._analyze_profiling_results()
    
    def _analyze_profiling_results(self):
        """Analyser les résultats du profiling et ajuster la configuration."""
        if not self.profiler.layer_stats:
            return
        
        # Calculer les statistiques globales
        total_memory = sum(stats['memory_used'] for stats in self.profiler.layer_stats.values())
        total_compute = sum(stats['backward_time'] for stats in self.profiler.layer_stats.values())
        
        print(f"Profiling terminé:")
        print(f"  Mémoire totale analysée: {total_memory / (1024**2):.1f} GB")
        print(f"  Temps de calcul total: {total_compute*1000:.1f} ms")
        
        # Identifier les couches les plus coûteuses
        sorted_layers = sorted(
            self.profiler.layer_stats.items(),
            key=lambda x: x[1]['memory_used'] + x[1]['activation_size'],
            reverse=True
        )
        
        print("  Top 5 des couches les plus coûteuses:")
        for i, (name, stats) in enumerate(sorted_layers[:5]):
            memory_mb = (stats['memory_used'] + stats['activation_size']) / (1024**2)
            print(f"    {i+1}. {name}: {memory_mb:.1f} MB")
        
        # Ajuster la stratégie si adaptatif
        if self.config.adjust_dynamically:
            self._adjust_strategy_from_profiling()
    
    def _adjust_strategy_from_profiling(self):
        """Ajuster automatiquement la stratégie basée sur le profiling."""
        # Calculer les ratios optimaux basés sur l'analyse
        moe_layers = {k: v for k, v in self.profiler.layer_stats.items() if 'moe' in k.lower()}
        attention_layers = {k: v for k, v in self.profiler.layer_stats.items() if 'attention' in k.lower()}
        mamba_layers = {k: v for k, v in self.profiler.layer_stats.items() if 'mamba' in k.lower()}
        
        # Ajuster les ratios basés sur la consommation mémoire
        if moe_layers:
            avg_moe_memory = sum(s['memory_used'] for s in moe_layers.values()) / len(moe_layers)
            if avg_moe_memory > 200 * 1024 * 1024:  # > 200MB
                self.config.moe_checkpoint_ratio = min(0.9, self.config.moe_checkpoint_ratio + 0.1)
        
        if attention_layers:
            avg_attn_memory = sum(s['memory_used'] for s in attention_layers.values()) / len(attention_layers)
            if avg_attn_memory > 100 * 1024 * 1024:  # > 100MB
                self.config.attention_checkpoint_ratio = min(0.7, self.config.attention_checkpoint_ratio + 0.1)
        
        print(f"Stratégie ajustée:")
        print(f"  MoE checkpoint ratio: {self.config.moe_checkpoint_ratio}")
        print(f"  Attention checkpoint ratio: {self.config.attention_checkpoint_ratio}")
        print(f"  Mamba checkpoint ratio: {self.config.mamba_checkpoint_ratio}")
    
    def get_checkpointing_stats(self) -> Dict[str, Any]:
        """Obtenir les statistiques du checkpointing."""
        memory_stats = self.memory_tracker.get_memory_stats()
        
        return {
            'config': self.config,
            'checkpointed_layers': len(self.checkpointed_layers),
            'memory_stats': memory_stats,
            'profiling_available': len(self.profiler.layer_stats) > 0,
            'layer_decisions': dict(self.layer_decisions),
        }
    
    def monitor_performance_impact(self, train_time: float, memory_peak: float):
        """Monitor l'impact sur les performances."""
        self.performance_history.append({
            'train_time': train_time,
            'memory_peak': memory_peak,
            'checkpointed_layers': len(self.checkpointed_layers)
        })
        
        # Ajuster si performance dégradée
        if len(self.performance_history) >= 10:
            recent_avg = sum(h['train_time'] for h in list(self.performance_history)[-5:]) / 5
            older_avg = sum(h['train_time'] for h in list(self.performance_history)[-10:-5]) / 5
            
            if recent_avg > older_avg * 1.2:  # 20% plus lent
                print("Performance dégradée détectée, réduction du checkpointing...")
                self._reduce_checkpointing()
    
    def _reduce_checkpointing(self):
        """Réduire le checkpointing pour améliorer la vitesse."""
        self.config.moe_checkpoint_ratio *= 0.8
        self.config.attention_checkpoint_ratio *= 0.8
        self.config.mamba_checkpoint_ratio *= 0.8
        
        # Nettoyer les décisions précédentes
        self.layer_decisions.clear()
        self.checkpointed_layers.clear()


# Décorateur pour appliquer le checkpointing automatiquement
def smart_checkpoint(config: Optional[CheckpointConfig] = None):
    """Décorateur pour appliquer le checkpointing intelligent."""
    
    def decorator(forward_method):
        def wrapper(self, *args, **kwargs):
            # Initialiser le checkpointer si nécessaire
            if not hasattr(self, '_smart_checkpointer'):
                self._smart_checkpointer = SmartCheckpointer(
                    config or CheckpointConfig()
                )
            
            # Obtenir le nom de la couche
            layer_name = getattr(self, 'layer_name', self.__class__.__name__)
            
            # Décider du checkpointing
            should_checkpoint = self._smart_checkpointer.should_checkpoint_layer(
                layer_name, self
            )
            
            if should_checkpoint:
                self._smart_checkpointer.checkpointed_layers.add(layer_name)
                return self._smart_checkpointer.checkpoint_function(
                    forward_method, self, *args, **kwargs
                )
            else:
                return forward_method(self, *args, **kwargs)
        
        return wrapper
    return decorator


# Utilitaire pour estimer les économies mémoire
def estimate_memory_savings(model: nn.Module, checkpointer: SmartCheckpointer) -> Dict[str, float]:
    """Estimer les économies mémoire du checkpointing."""
    total_activations = 0
    checkpointed_activations = 0
    
    # Estimation basée sur les paramètres et les couches
    for name, layer in model.named_modules():
        if hasattr(layer, 'weight') and layer.weight is not None:
            # Estimer la taille des activations (approximation)
            param_count = layer.weight.numel()
            estimated_activation_size = param_count * 4  # float32
            
            total_activations += estimated_activation_size
            
            if checkpointer.should_checkpoint_layer(name, layer):
                checkpointed_activations += estimated_activation_size
    
    if total_activations == 0:
        return {'savings_mb': 0, 'savings_percentage': 0}
    
    # Les activations checkpointées sont économisées (mais recalculées)
    savings_mb = checkpointed_activations / (1024 ** 2)
    savings_percentage = (checkpointed_activations / total_activations) * 100
    
    return {
        'total_activations_mb': total_activations / (1024 ** 2),
        'checkpointed_activations_mb': checkpointed_activations / (1024 ** 2),
        'savings_mb': savings_mb,
        'savings_percentage': savings_percentage
    }