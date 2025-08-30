"""
Système de cache avancé pour l'inférence Ultra-AI
- Cache KV pour attention
- Cache d'état Mamba
- Cache d'activation MoE
- Gestion mémoire dynamique
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import math
from dataclasses import dataclass
from collections import OrderedDict
import gc


@dataclass
class CacheConfig:
    """Configuration du système de cache."""
    max_batch_size: int = 8
    max_seq_length: int = 4096
    cache_dtype: torch.dtype = torch.float16
    enable_kv_cache: bool = True
    enable_mamba_cache: bool = True
    enable_moe_cache: bool = True
    memory_fraction: float = 0.8  # Fraction de mémoire GPU utilisable


class InferenceCache:
    """Gestionnaire de cache pour l'inférence optimisée."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.kv_cache = KVCache(config) if config.enable_kv_cache else None
        self.mamba_cache = MambaStateCache(config) if config.enable_mamba_cache else None
        self.moe_cache = MoECache(config) if config.enable_moe_cache else None
        self.memory_manager = CacheMemoryManager(config)
        
    def clear_all(self):
        """Vider tous les caches."""
        if self.kv_cache:
            self.kv_cache.clear()
        if self.mamba_cache:
            self.mamba_cache.clear()
        if self.moe_cache:
            self.moe_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Obtenir l'utilisation mémoire des caches."""
        usage = {}
        
        if self.kv_cache:
            usage['kv_cache'] = self.kv_cache.get_memory_usage()
        if self.mamba_cache:
            usage['mamba_cache'] = self.mamba_cache.get_memory_usage()
        if self.moe_cache:
            usage['moe_cache'] = self.moe_cache.get_memory_usage()
            
        usage['total'] = sum(usage.values())
        return usage


class KVCache:
    """Cache Key-Value pour les couches d'attention."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache_k: Dict[str, torch.Tensor] = {}
        self.cache_v: Dict[str, torch.Tensor] = {}
        self.cache_positions: Dict[str, int] = {}
        self.max_positions = {}
        
    def get_kv(self, layer_id: str, batch_size: int, num_heads: int, 
               head_dim: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Obtenir les tenseurs K et V pour une couche."""
        cache_key = f"{layer_id}_{batch_size}_{num_heads}_{head_dim}"
        
        if cache_key not in self.cache_k:
            # Créer de nouveaux tensors de cache
            cache_shape = (batch_size, num_heads, self.config.max_seq_length, head_dim)
            self.cache_k[cache_key] = torch.zeros(
                cache_shape, dtype=self.config.cache_dtype, device=device
            )
            self.cache_v[cache_key] = torch.zeros(
                cache_shape, dtype=self.config.cache_dtype, device=device
            )
            self.cache_positions[cache_key] = 0
            self.max_positions[cache_key] = 0
            
        return self.cache_k[cache_key], self.cache_v[cache_key]
    
    def update_kv(self, layer_id: str, batch_size: int, num_heads: int, head_dim: int,
                  new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mettre à jour le cache KV et retourner les valeurs complètes."""
        cache_key = f"{layer_id}_{batch_size}_{num_heads}_{head_dim}"
        
        cache_k, cache_v = self.get_kv(layer_id, batch_size, num_heads, head_dim, new_k.device)
        
        seq_len = new_k.shape[2]
        start_pos = self.cache_positions[cache_key]
        end_pos = start_pos + seq_len
        
        # Vérifier la capacité
        if end_pos > self.config.max_seq_length:
            # Déplacer les anciennes valeurs et réinitialiser
            self._shift_cache(cache_key, seq_len)
            start_pos = self.config.max_seq_length - seq_len
            end_pos = self.config.max_seq_length
        
        # Mettre à jour le cache
        cache_k[:, :, start_pos:end_pos, :] = new_k.to(self.config.cache_dtype)
        cache_v[:, :, start_pos:end_pos, :] = new_v.to(self.config.cache_dtype)
        
        self.cache_positions[cache_key] = end_pos
        self.max_positions[cache_key] = max(self.max_positions[cache_key], end_pos)
        
        # Retourner la portion utilisée du cache
        return (cache_k[:, :, :end_pos, :].to(new_k.dtype), 
                cache_v[:, :, :end_pos, :].to(new_v.dtype))
    
    def _shift_cache(self, cache_key: str, new_seq_len: int):
        """Déplacer le cache pour faire de la place."""
        shift_amount = new_seq_len
        cache_k = self.cache_k[cache_key]
        cache_v = self.cache_v[cache_key]
        
        # Déplacer vers la gauche
        cache_k[:, :, :-shift_amount, :] = cache_k[:, :, shift_amount:, :]
        cache_v[:, :, :-shift_amount, :] = cache_v[:, :, shift_amount:, :]
        
        # Mettre à jour la position
        self.cache_positions[cache_key] -= shift_amount
        self.cache_positions[cache_key] = max(0, self.cache_positions[cache_key])
    
    def clear(self):
        """Vider le cache KV."""
        self.cache_k.clear()
        self.cache_v.clear()
        self.cache_positions.clear()
        self.max_positions.clear()
    
    def get_memory_usage(self) -> float:
        """Obtenir l'utilisation mémoire en MB."""
        total_bytes = 0
        for k_tensor in self.cache_k.values():
            total_bytes += k_tensor.numel() * k_tensor.element_size()
        for v_tensor in self.cache_v.values():
            total_bytes += v_tensor.numel() * v_tensor.element_size()
        return total_bytes / (1024 ** 2)


class MambaStateCache:
    """Cache d'état pour les couches Mamba-2."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.state_cache: Dict[str, torch.Tensor] = {}
        self.conv_cache: Dict[str, torch.Tensor] = {}  # Cache pour conv1d
        
    def get_state(self, layer_id: str, batch_size: int, d_inner: int, d_state: int, 
                  device: torch.device) -> torch.Tensor:
        """Obtenir l'état Mamba pour une couche."""
        cache_key = f"{layer_id}_{batch_size}_{d_inner}_{d_state}"
        
        if cache_key not in self.state_cache:
            self.state_cache[cache_key] = torch.zeros(
                (batch_size, d_inner, d_state),
                dtype=self.config.cache_dtype, 
                device=device
            )
        
        return self.state_cache[cache_key]
    
    def update_state(self, layer_id: str, batch_size: int, d_inner: int, d_state: int,
                     new_state: torch.Tensor):
        """Mettre à jour l'état Mamba."""
        cache_key = f"{layer_id}_{batch_size}_{d_inner}_{d_state}"
        
        if cache_key in self.state_cache:
            self.state_cache[cache_key] = new_state.to(self.config.cache_dtype)
    
    def get_conv_cache(self, layer_id: str, batch_size: int, d_inner: int, conv_width: int,
                       device: torch.device) -> torch.Tensor:
        """Obtenir le cache de convolution."""
        cache_key = f"conv_{layer_id}_{batch_size}_{d_inner}_{conv_width}"
        
        if cache_key not in self.conv_cache:
            self.conv_cache[cache_key] = torch.zeros(
                (batch_size, d_inner, conv_width - 1),
                dtype=self.config.cache_dtype,
                device=device
            )
        
        return self.conv_cache[cache_key]
    
    def update_conv_cache(self, layer_id: str, batch_size: int, d_inner: int, conv_width: int,
                          new_values: torch.Tensor):
        """Mettre à jour le cache de convolution."""
        cache_key = f"conv_{layer_id}_{batch_size}_{d_inner}_{conv_width}"
        
        if cache_key in self.conv_cache:
            cache = self.conv_cache[cache_key]
            # Décaler et ajouter nouvelles valeurs
            cache[:, :, :-1] = cache[:, :, 1:]
            cache[:, :, -1] = new_values[:, :, 0].to(self.config.cache_dtype)
    
    def clear(self):
        """Vider le cache Mamba."""
        self.state_cache.clear()
        self.conv_cache.clear()
    
    def get_memory_usage(self) -> float:
        """Obtenir l'utilisation mémoire en MB."""
        total_bytes = 0
        for tensor in self.state_cache.values():
            total_bytes += tensor.numel() * tensor.element_size()
        for tensor in self.conv_cache.values():
            total_bytes += tensor.numel() * tensor.element_size()
        return total_bytes / (1024 ** 2)


class MoECache:
    """Cache pour les activations MoE."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.expert_cache: Dict[str, Dict[int, torch.Tensor]] = {}
        self.routing_cache: Dict[str, torch.Tensor] = {}
        self.hit_count = 0
        self.miss_count = 0
        
    def get_expert_output(self, layer_id: str, expert_id: int, 
                         input_hash: int) -> Optional[torch.Tensor]:
        """Obtenir la sortie mise en cache d'un expert."""
        if layer_id not in self.expert_cache:
            return None
            
        if expert_id not in self.expert_cache[layer_id]:
            return None
            
        cached_output = self.expert_cache[layer_id][expert_id].get(input_hash)
        if cached_output is not None:
            self.hit_count += 1
            return cached_output
        else:
            self.miss_count += 1
            return None
    
    def cache_expert_output(self, layer_id: str, expert_id: int, input_hash: int,
                           output: torch.Tensor):
        """Mettre en cache la sortie d'un expert."""
        if layer_id not in self.expert_cache:
            self.expert_cache[layer_id] = {}
            
        if expert_id not in self.expert_cache[layer_id]:
            self.expert_cache[layer_id][expert_id] = OrderedDict()
            
        # Limiter la taille du cache par expert
        expert_cache = self.expert_cache[layer_id][expert_id]
        if len(expert_cache) >= 100:  # Limite arbitraire
            expert_cache.popitem(last=False)  # FIFO
            
        expert_cache[input_hash] = output.to(self.config.cache_dtype).detach()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Obtenir les statistiques du cache."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
        
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'total_cached_entries': sum(
                len(experts) for layer_cache in self.expert_cache.values() 
                for experts in layer_cache.values()
            )
        }
    
    def clear(self):
        """Vider le cache MoE."""
        self.expert_cache.clear()
        self.routing_cache.clear()
        self.hit_count = 0
        self.miss_count = 0
    
    def get_memory_usage(self) -> float:
        """Obtenir l'utilisation mémoire en MB."""
        total_bytes = 0
        for layer_cache in self.expert_cache.values():
            for expert_cache in layer_cache.values():
                for tensor in expert_cache.values():
                    total_bytes += tensor.numel() * tensor.element_size()
        return total_bytes / (1024 ** 2)


class CacheMemoryManager:
    """Gestionnaire de mémoire pour les caches."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.max_memory_mb = self._get_max_memory()
        
    def _get_max_memory(self) -> float:
        """Calculer la mémoire maximale utilisable."""
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            return (total_memory * self.config.memory_fraction) / (1024 ** 2)
        else:
            # Estimer pour CPU (8GB par défaut)
            return 8192 * self.config.memory_fraction
    
    def should_evict(self, current_usage: float) -> bool:
        """Vérifier si on doit évacuer du cache."""
        return current_usage > self.max_memory_mb * 0.9  # 90% de seuil
    
    def suggest_eviction_strategy(self, cache_usage: Dict[str, float]) -> List[str]:
        """Suggérer quelle cache évacuer en premier."""
        # Évacuer dans l'ordre : MoE, KV, Mamba (par ordre d'importance)
        eviction_order = []
        
        if 'moe_cache' in cache_usage and cache_usage['moe_cache'] > 100:  # > 100MB
            eviction_order.append('moe_cache')
            
        if 'kv_cache' in cache_usage and cache_usage['kv_cache'] > 500:  # > 500MB
            eviction_order.append('kv_cache')
            
        if 'mamba_cache' in cache_usage and cache_usage['mamba_cache'] > 200:  # > 200MB
            eviction_order.append('mamba_cache')
            
        return eviction_order


# Décorateur pour cache automatique
def with_inference_cache(cache_config: Optional[CacheConfig] = None):
    """Décorateur pour ajouter le cache d'inférence à un modèle."""
    
    def decorator(model_class):
        original_init = model_class.__init__
        original_forward = model_class.forward
        
        def __init__(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            self.inference_cache = InferenceCache(
                cache_config or CacheConfig()
            )
            self._cache_enabled = False
        
        def forward(self, *args, **kwargs):
            if not self._cache_enabled:
                return original_forward(self, *args, **kwargs)
            
            # Logique de forward avec cache
            # (à implémenter selon le modèle)
            return original_forward(self, *args, **kwargs)
        
        def enable_cache(self):
            self._cache_enabled = True
            
        def disable_cache(self):
            self._cache_enabled = False
            self.inference_cache.clear_all()
        
        # Remplacer les méthodes
        model_class.__init__ = __init__
        model_class.forward = forward
        model_class.enable_cache = enable_cache
        model_class.disable_cache = disable_cache
        
        return model_class
    
    return decorator