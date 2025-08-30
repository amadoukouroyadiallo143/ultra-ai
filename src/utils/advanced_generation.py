"""
Système de génération avancé pour Ultra-AI
- Beam Search optimisé avec cache
- Nucleus sampling amélioré  
- Génération contrainte et guidée
- Optimisations parallèles et batch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Union, Any, Callable
import math
import heapq
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from abc import ABC, abstractmethod


@dataclass
class GenerationConfig:
    """Configuration pour la génération de texte."""
    # Paramètres de base
    max_length: int = 100
    min_length: int = 1
    do_sample: bool = True
    early_stopping: bool = True
    
    # Beam Search
    num_beams: int = 1
    num_beam_groups: int = 1
    diversity_penalty: float = 0.0
    length_penalty: float = 1.0
    
    # Sampling
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    typical_p: float = 1.0
    epsilon_cutoff: float = 0.0
    eta_cutoff: float = 0.0
    
    # Répétition
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    
    # Tokens spéciaux
    pad_token_id: Optional[int] = None
    bos_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None
    
    # Optimisations
    use_cache: bool = True
    batch_size: int = 1
    num_return_sequences: int = 1
    
    # Contraintes
    bad_words_ids: Optional[List[List[int]]] = None
    force_words_ids: Optional[List[List[int]]] = None
    
    # Callbacks
    stopping_criteria: Optional[List[Callable]] = None
    logits_processor: Optional[List[Callable]] = None


class BeamHypothesis:
    """Hypothèse pour le beam search."""
    
    def __init__(self, tokens: List[int], log_prob: float, attention_mask: Optional[torch.Tensor] = None):
        self.tokens = tokens
        self.log_prob = log_prob
        self.attention_mask = attention_mask
        
    def __len__(self):
        return len(self.tokens)
    
    def __lt__(self, other):
        return self.log_prob < other.log_prob
    
    def extend(self, token: int, log_prob: float) -> 'BeamHypothesis':
        """Étendre l'hypothèse avec un nouveau token."""
        new_tokens = self.tokens + [token]
        new_log_prob = self.log_prob + log_prob
        return BeamHypothesis(new_tokens, new_log_prob, self.attention_mask)
    
    def get_score(self, length_penalty: float = 1.0) -> float:
        """Calculer le score normalisé par la longueur."""
        length_penalty = ((5.0 + len(self.tokens)) / 6.0) ** length_penalty
        return self.log_prob / length_penalty


class BeamSearchDecoder:
    """Décodeur Beam Search optimisé avec cache."""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        self.batch_beams: List[List[BeamHypothesis]] = []
        self.finished_beams: List[List[BeamHypothesis]] = []
        
    def search(self, model: nn.Module, input_ids: torch.Tensor, 
               attention_mask: Optional[torch.Tensor] = None) -> List[List[BeamHypothesis]]:
        """Effectuer le beam search."""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialiser les beams
        self._initialize_beams(input_ids, batch_size)
        
        # Variables de cache si supporté
        past_key_values = None
        
        for step in range(self.config.max_length - input_ids.shape[1]):
            # Préparer les entrées pour ce step
            current_input_ids, current_attention_mask = self._prepare_inputs_for_step(
                input_ids, attention_mask, step
            )
            
            # Forward pass du modèle
            with torch.no_grad():
                outputs = model(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask,
                    past_key_values=past_key_values if self.config.use_cache else None,
                    use_cache=self.config.use_cache
                )
            
            # Extraire logits et cache
            if isinstance(outputs, dict):
                logits = outputs['logits']
                past_key_values = outputs.get('past_key_values', None)
            else:
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                past_key_values = getattr(outputs, 'past_key_values', None)
            
            # Traiter les logits et mettre à jour les beams
            next_token_logits = logits[:, -1, :]  # [batch_size * num_beams, vocab_size]
            self._update_beams(next_token_logits, step)
            
            # Vérifier les critères d'arrêt
            if self._should_stop(step):
                break
        
        # Finaliser et retourner les meilleurs beams
        return self._finalize_beams()
    
    def _initialize_beams(self, input_ids: torch.Tensor, batch_size: int):
        """Initialiser les beams pour chaque exemple du batch."""
        self.batch_beams = []
        self.finished_beams = []
        
        for batch_idx in range(batch_size):
            beams = []
            finished = []
            
            for beam_idx in range(self.config.num_beams):
                if beam_idx == 0:
                    # Premier beam: séquence originale
                    tokens = input_ids[batch_idx].tolist()
                    log_prob = 0.0
                else:
                    # Autres beams: copies avec probabilité très faible
                    tokens = input_ids[batch_idx].tolist()
                    log_prob = -float('inf')
                
                beam = BeamHypothesis(tokens, log_prob)
                beams.append(beam)
            
            self.batch_beams.append(beams)
            self.finished_beams.append(finished)
    
    def _prepare_inputs_for_step(self, original_input_ids: torch.Tensor,
                                original_attention_mask: Optional[torch.Tensor],
                                step: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Préparer les entrées pour ce step du beam search."""
        batch_size = len(self.batch_beams)
        device = original_input_ids.device
        
        # Collecter tous les beams actifs
        all_beam_tokens = []
        attention_masks = []
        
        for batch_idx in range(batch_size):
            for beam in self.batch_beams[batch_idx]:
                all_beam_tokens.append(beam.tokens)
                
                # Créer le masque d'attention
                seq_len = len(beam.tokens)
                if original_attention_mask is not None:
                    # Étendre le masque original
                    original_mask = original_attention_mask[batch_idx][:len(beam.tokens)]
                    mask = torch.ones(seq_len, device=device, dtype=original_mask.dtype)
                    mask[:len(original_mask)] = original_mask
                else:
                    mask = torch.ones(seq_len, device=device)
                
                attention_masks.append(mask)
        
        # Convertir en tenseurs avec padding
        max_len = max(len(tokens) for tokens in all_beam_tokens)
        
        padded_input_ids = torch.full(
            (len(all_beam_tokens), max_len),
            self.config.pad_token_id or 0,
            device=device,
            dtype=original_input_ids.dtype
        )
        
        padded_attention_mask = torch.zeros(
            (len(all_beam_tokens), max_len),
            device=device,
            dtype=torch.long
        )
        
        for i, (tokens, mask) in enumerate(zip(all_beam_tokens, attention_masks)):
            seq_len = len(tokens)
            padded_input_ids[i, :seq_len] = torch.tensor(tokens, device=device)
            padded_attention_mask[i, :len(mask)] = mask
        
        return padded_input_ids, padded_attention_mask
    
    def _update_beams(self, next_token_logits: torch.Tensor, step: int):
        """Mettre à jour les beams avec les nouveaux tokens."""
        batch_size = len(self.batch_beams)
        vocab_size = next_token_logits.shape[-1]
        
        # Traitement par batch
        logit_idx = 0
        
        for batch_idx in range(batch_size):
            current_beams = self.batch_beams[batch_idx]
            new_beams = []
            
            # Collecter toutes les extensions possibles
            candidates = []
            
            for beam_idx, beam in enumerate(current_beams):
                if len(beam.tokens) >= self.config.max_length:
                    # Beam terminé par longueur maximale
                    self.finished_beams[batch_idx].append(beam)
                    continue
                
                # Obtenir les logits pour ce beam
                beam_logits = next_token_logits[logit_idx]
                logit_idx += 1
                
                # Appliquer les pénalités de répétition
                beam_logits = self._apply_repetition_penalty(beam_logits, beam.tokens)
                
                # Top-K pour ce beam
                top_k = min(self.config.num_beams * 2, vocab_size)  # Plus de candidats que de beams
                top_logits, top_indices = torch.topk(beam_logits, top_k)
                top_log_probs = F.log_softmax(top_logits, dim=-1)
                
                # Créer les candidats
                for i in range(top_k):
                    token_id = top_indices[i].item()
                    log_prob = top_log_probs[i].item()
                    
                    new_beam = beam.extend(token_id, log_prob)
                    score = new_beam.get_score(self.config.length_penalty)
                    
                    # Vérifier si c'est un token EOS
                    if token_id == self.config.eos_token_id:
                        if len(new_beam.tokens) >= self.config.min_length:
                            self.finished_beams[batch_idx].append(new_beam)
                        continue
                    
                    candidates.append((score, new_beam))
            
            # Sélectionner les meilleurs candidats
            candidates.sort(key=lambda x: x[0], reverse=True)
            new_beams = [candidate[1] for candidate in candidates[:self.config.num_beams]]
            
            self.batch_beams[batch_idx] = new_beams
    
    def _apply_repetition_penalty(self, logits: torch.Tensor, tokens: List[int]) -> torch.Tensor:
        """Appliquer les pénalités de répétition."""
        if self.config.repetition_penalty != 1.0:
            # Pénalité de répétition simple
            for token in set(tokens):
                if logits[token] > 0:
                    logits[token] /= self.config.repetition_penalty
                else:
                    logits[token] *= self.config.repetition_penalty
        
        # N-gram répétition
        if self.config.no_repeat_ngram_size > 0:
            ngram_size = self.config.no_repeat_ngram_size
            if len(tokens) >= ngram_size - 1:
                # Derniers n-1 tokens
                prefix = tokens[-(ngram_size-1):]
                
                # Interdire les tokens qui créeraient des n-grams répétés
                for i in range(len(tokens) - ngram_size + 1):
                    if tokens[i:i+ngram_size-1] == prefix:
                        next_token = tokens[i+ngram_size-1]
                        logits[next_token] = -float('inf')
        
        return logits
    
    def _should_stop(self, step: int) -> bool:
        """Vérifier si le beam search doit s'arrêter."""
        if not self.config.early_stopping:
            return False
        
        # Vérifier si tous les beams sont terminés
        for batch_idx in range(len(self.batch_beams)):
            if len(self.batch_beams[batch_idx]) > 0:
                return False
        
        return True
    
    def _finalize_beams(self) -> List[List[BeamHypothesis]]:
        """Finaliser et trier les beams."""
        final_beams = []
        
        for batch_idx in range(len(self.batch_beams)):
            # Combiner beams actifs et terminés
            all_beams = self.batch_beams[batch_idx] + self.finished_beams[batch_idx]
            
            # Trier par score
            all_beams.sort(key=lambda x: x.get_score(self.config.length_penalty), reverse=True)
            
            # Prendre les meilleurs
            final_beams.append(all_beams[:self.config.num_return_sequences])
        
        return final_beams


class NucleusSampler:
    """Sampler Nucleus (Top-P) amélioré avec optimisations."""
    
    def __init__(self, config: GenerationConfig):
        self.config = config
        
    def sample(self, logits: torch.Tensor, past_tokens: Optional[List[int]] = None) -> torch.Tensor:
        """Échantillonner avec Nucleus sampling."""
        # Appliquer la température
        if self.config.temperature != 1.0:
            logits = logits / self.config.temperature
        
        # Appliquer les pénalités de répétition
        if past_tokens and self.config.repetition_penalty != 1.0:
            logits = self._apply_repetition_penalty(logits, past_tokens)
        
        # Top-K filtering
        if self.config.top_k > 0:
            logits = self._top_k_filter(logits, self.config.top_k)
        
        # Top-P (Nucleus) filtering
        if self.config.top_p < 1.0:
            logits = self._top_p_filter(logits, self.config.top_p)
        
        # Typical sampling
        if self.config.typical_p < 1.0:
            logits = self._typical_p_filter(logits, self.config.typical_p)
        
        # Epsilon/Eta cutoff
        if self.config.epsilon_cutoff > 0.0:
            logits = self._epsilon_filter(logits, self.config.epsilon_cutoff)
        
        if self.config.eta_cutoff > 0.0:
            logits = self._eta_filter(logits, self.config.eta_cutoff)
        
        # Échantillonnage final
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    
    def _apply_repetition_penalty(self, logits: torch.Tensor, past_tokens: List[int]) -> torch.Tensor:
        """Appliquer la pénalité de répétition."""
        for token in set(past_tokens):
            if logits[token] > 0:
                logits[token] /= self.config.repetition_penalty
            else:
                logits[token] *= self.config.repetition_penalty
        return logits
    
    def _top_k_filter(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        """Filtrage Top-K optimisé."""
        top_k = min(k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('inf')
        return logits
    
    def _top_p_filter(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """Filtrage Top-P (Nucleus) optimisé."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Supprimer les tokens avec une probabilité cumulative > p
        sorted_indices_to_remove = cumulative_probs > p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0  # Garder au moins le meilleur token
        
        indices_to_remove = sorted_indices_to_remove.scatter(
            0, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = -float('inf')
        return logits
    
    def _typical_p_filter(self, logits: torch.Tensor, p: float) -> torch.Tensor:
        """Filtrage Typical-P (basé sur l'entropie locale)."""
        probs = F.softmax(logits, dim=-1)
        
        # Calculer l'information surprise pour chaque token
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -torch.sum(probs * log_probs)
        surprises = -log_probs
        
        # Garder les tokens avec surprise proche de l'entropie
        surprise_diff = torch.abs(surprises - entropy)
        _, sorted_indices = torch.sort(surprise_diff)
        
        # Calculer la probabilité cumulative des tokens typiques
        sorted_probs = probs[sorted_indices]
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Supprimer les tokens atypiques
        last_ind = (cumulative_probs < p).sum()
        indices_to_remove = torch.ones_like(probs, dtype=torch.bool)
        indices_to_remove[sorted_indices[:last_ind]] = False
        
        logits[indices_to_remove] = -float('inf')
        return logits
    
    def _epsilon_filter(self, logits: torch.Tensor, epsilon: float) -> torch.Tensor:
        """Filtrage Epsilon (seuil de probabilité absolue)."""
        probs = F.softmax(logits, dim=-1)
        indices_to_remove = probs < epsilon
        logits[indices_to_remove] = -float('inf')
        return logits
    
    def _eta_filter(self, logits: torch.Tensor, eta: float) -> torch.Tensor:
        """Filtrage Eta (seuil de probabilité relative)."""
        probs = F.softmax(logits, dim=-1)
        max_prob = probs.max()
        indices_to_remove = probs < (eta * max_prob)
        logits[indices_to_remove] = -float('inf')
        return logits


class AdvancedGenerator:
    """Générateur avancé combinant toutes les techniques."""
    
    def __init__(self, model: nn.Module, config: GenerationConfig):
        self.model = model
        self.config = config
        self.beam_decoder = BeamSearchDecoder(config) if config.num_beams > 1 else None
        self.nucleus_sampler = NucleusSampler(config)
        
    def generate(self, input_ids: torch.Tensor, 
                 attention_mask: Optional[torch.Tensor] = None,
                 **kwargs) -> torch.Tensor:
        """Génération principale avec sélection automatique de stratégie."""
        
        # Mettre à jour la config avec les kwargs
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Choisir la stratégie de génération
        if self.config.num_beams > 1:
            return self._beam_search_generate(input_ids, attention_mask)
        else:
            return self._sampling_generate(input_ids, attention_mask)
    
    def _beam_search_generate(self, input_ids: torch.Tensor,
                             attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Génération par beam search."""
        if not self.beam_decoder:
            self.beam_decoder = BeamSearchDecoder(self.config)
        
        beam_results = self.beam_decoder.search(self.model, input_ids, attention_mask)
        
        # Convertir les résultats en tenseurs
        batch_outputs = []
        for batch_beams in beam_results:
            batch_sequences = []
            for beam in batch_beams[:self.config.num_return_sequences]:
                sequence = torch.tensor(beam.tokens, device=input_ids.device, dtype=input_ids.dtype)
                batch_sequences.append(sequence)
            
            # Padding si nécessaire
            if len(batch_sequences) < self.config.num_return_sequences:
                batch_sequences.extend([batch_sequences[0]] * 
                                     (self.config.num_return_sequences - len(batch_sequences)))
            
            batch_outputs.append(batch_sequences)
        
        # Convertir en tensor final
        max_len = max(seq.shape[0] for batch in batch_outputs for seq in batch)
        
        final_output = torch.full(
            (len(batch_outputs), self.config.num_return_sequences, max_len),
            self.config.pad_token_id or 0,
            device=input_ids.device,
            dtype=input_ids.dtype
        )
        
        for batch_idx, sequences in enumerate(batch_outputs):
            for seq_idx, sequence in enumerate(sequences):
                final_output[batch_idx, seq_idx, :len(sequence)] = sequence
        
        return final_output.squeeze(1) if self.config.num_return_sequences == 1 else final_output
    
    def _sampling_generate(self, input_ids: torch.Tensor,
                          attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Génération par échantillonnage."""
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialiser
        generated = input_ids.clone()
        past_key_values = None
        
        # Générer token par token
        for step in range(self.config.max_length - input_ids.shape[1]):
            # Forward pass
            with torch.no_grad():
                model_inputs = {
                    'input_ids': generated,
                    'attention_mask': attention_mask,
                }
                
                if self.config.use_cache and past_key_values is not None:
                    model_inputs['past_key_values'] = past_key_values
                    model_inputs['input_ids'] = generated[:, -1:]  # Seulement le dernier token
                
                outputs = self.model(**model_inputs)
            
            # Extraire logits et cache
            if isinstance(outputs, dict):
                logits = outputs['logits']
                past_key_values = outputs.get('past_key_values', None) if self.config.use_cache else None
            else:
                logits = outputs.logits if hasattr(outputs, 'logits') else outputs
                past_key_values = getattr(outputs, 'past_key_values', None) if self.config.use_cache else None
            
            # Obtenir les logits du prochain token
            next_token_logits = logits[:, -1, :]
            
            # Échantillonner pour chaque exemple du batch
            next_tokens = []
            for batch_idx in range(batch_size):
                batch_logits = next_token_logits[batch_idx]
                past_tokens = generated[batch_idx].tolist()
                
                if self.config.do_sample:
                    next_token = self.nucleus_sampler.sample(batch_logits, past_tokens)
                else:
                    # Greedy
                    next_token = torch.argmax(batch_logits, dim=-1, keepdim=True)
                
                next_tokens.append(next_token)
            
            next_tokens = torch.cat(next_tokens, dim=0).unsqueeze(1)
            
            # Ajouter à la séquence générée
            generated = torch.cat([generated, next_tokens], dim=1)
            
            # Mettre à jour le masque d'attention
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones((batch_size, 1), device=device, dtype=attention_mask.dtype)
                ], dim=1)
            
            # Vérifier les critères d'arrêt
            if self.config.eos_token_id is not None:
                if (next_tokens == self.config.eos_token_id).all():
                    break
        
        return generated
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Obtenir les statistiques de génération."""
        stats = {
            'config': self.config,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
        }
        
        if self.beam_decoder:
            stats['beam_search_enabled'] = True
            stats['num_beams'] = self.config.num_beams
        else:
            stats['sampling_enabled'] = True
            stats['temperature'] = self.config.temperature
            stats['top_p'] = self.config.top_p
            stats['top_k'] = self.config.top_k
        
        return stats


# Utilitaires pour la génération contrainte
class ConstrainedGenerator:
    """Générateur avec contraintes et guidage."""
    
    def __init__(self, base_generator: AdvancedGenerator):
        self.base_generator = base_generator
        
    def generate_with_prefix_constraint(self, input_ids: torch.Tensor,
                                       required_prefix: List[int],
                                       **kwargs) -> torch.Tensor:
        """Générer avec un préfixe requis."""
        # Forcer les premiers tokens à être le préfixe
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # Construire la séquence avec préfixe
        prefix_tensor = torch.tensor(required_prefix, device=device, dtype=input_ids.dtype)
        prefix_batch = prefix_tensor.unsqueeze(0).repeat(batch_size, 1)
        
        # Concaténer input + préfixe requis
        forced_input = torch.cat([input_ids, prefix_batch], dim=1)
        
        # Générer normalement à partir de cette base
        return self.base_generator.generate(forced_input, **kwargs)
    
    def generate_with_format_constraint(self, input_ids: torch.Tensor,
                                       format_template: str,
                                       tokenizer,
                                       **kwargs) -> torch.Tensor:
        """Générer en respectant un format spécifique."""
        # Cette fonction nécessiterait un tokenizer pour fonctionner complètement
        # Pour l'instant, générer normalement
        return self.base_generator.generate(input_ids, **kwargs)