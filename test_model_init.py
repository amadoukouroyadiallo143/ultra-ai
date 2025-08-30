#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de test complet pour Ultra-AI
- Initialisation et analyse du modèle
- Tests du forward pass
- Tests de génération
- Benchmarks de performance
- Comparaison entre configurations
"""

import torch
import sys
import os
import time
import json
from pathlib import Path
from typing import Dict, List, Optional

# Ajouter le repertoire racine au path pour les imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from src.models.ultra_ai_model import UltraAIModel
    from src.utils.config import UltraAIConfig, load_config
except ImportError as e:
    print(f"Erreur d'import: {e}")
    print("Assurez-vous que le projet est installe avec: pip install -e .")
    sys.exit(1)

def format_number(num):
    """Formate un nombre avec des separateurs de milliers."""
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return str(num)

def get_model_memory_usage(model):
    """Calcule l'utilisation memoire du modele."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    total_size = param_size + buffer_size
    return total_size / (1024 * 1024)  # Convert to MB

def main(config_name="ultra_edge", skip_generation=False, skip_benchmark=False, verbose=False):
    print(f"Initialisation d'Ultra-AI - Configuration: {config_name}")
    print("=" * 80)
    
    # Charger la configuration demandée
    config = load_config(config_name=config_name)
    
    print("Configuration du modele:")
    print(f"   • Dimension cachee (d_model): {config.d_model}")
    print(f"   • Couches Mamba-2: {config.mamba_layers}")
    print(f"   • Couches Attention: {config.attention_layers}")
    print(f"   • Couches MoE: {config.moe_layers}")
    print(f"   • Tetes d'attention: {config.attention_heads}")
    print(f"   • Taille du vocabulaire: {config.vocab_size}")
    print(f"   • Longueur max de sequence: {config.max_seq_length:,}")
    print(f"   • Nombre d'experts MoE: {config.num_experts}")
    print()
    
    try:
        print("Creation du modele avec optimisations...")
        
        # Creer le modele avec optimisations activées
        model = UltraAIModel(config, enable_optimizations=True)
        
        # Profiler le modèle si possible
        if hasattr(model, 'profile_and_optimize'):
            print("   • Profiling du modèle...")
            sample_input = torch.randint(0, config.vocab_size, (1, 32))
            try:
                model.profile_and_optimize(sample_input)
                print("   • Profiling terminé")
            except Exception as e:
                print(f"   • Profiling échoué: {e}")
        
        # Calculer le nombre de parametres
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Estimer les parametres actifs (approximation pour MoE)
        # Commencer par les parametres non-MoE
        non_moe_params = sum(p.numel() for name, p in model.named_parameters() 
                            if 'moe' not in name.lower() and 'expert' not in name.lower())
        
        active_params = non_moe_params
        if hasattr(model, 'moe_layers') and model.moe_layers:
            # Pour MoE, seulement top_k experts sont actifs par couche
            try:
                expert_params_per_expert = sum(
                    p.numel() for p in model.moe_layers[0].experts[0].parameters()
                ) if len(model.moe_layers) > 0 and len(model.moe_layers[0].experts) > 0 else 0
                
                if expert_params_per_expert > 0:
                    # Paramètres actifs MoE = top_k experts * nombre de couches MoE
                    active_expert_params = expert_params_per_expert * config.moe_top_k * len(model.moe_layers)
                    # Plus les paramètres du routeur
                    router_params = sum(p.numel() for moe_layer in model.moe_layers 
                                      for name, p in moe_layer.named_parameters() 
                                      if 'router' in name.lower())
                    active_params += active_expert_params + router_params
            except (IndexError, AttributeError):
                # Fallback en cas d'erreur d'accès aux experts
                active_params = int(total_params * 0.25)  # Estimation conservative
        
        # Utilisation memoire
        memory_mb = get_model_memory_usage(model)
        
        print("Modele cree avec succes!")
        
        # Afficher les statistiques d'optimisation
        if hasattr(model, 'get_optimization_stats'):
            opt_stats = model.get_optimization_stats()
            if opt_stats.get('optimizations_enabled', False):
                print("   [OK] Optimisations activées:")
                print("     • Smart checkpointing")
                print("     • Dynamic quantization")
                print("     • Advanced generation")
                print("     • Mixed precision")
                print("     • JIT compilation")
            else:
                print("   [INFO] Mode standard (optimisations désactivées)")
        
        print()
        print("Statistiques du modele:")
        print(f"   • Parametres totaux: {format_number(total_params)} ({total_params:,})")
        print(f"   • Parametres entrainables: {format_number(trainable_params)} ({trainable_params:,})")
        print(f"   • Parametres actifs (MoE): {format_number(active_params)} ({active_params:,})")
        print(f"   • Efficacite parametrique: {(active_params/total_params)*100:.1f}%")
        print(f"   • Utilisation memoire: {memory_mb:.2f} MB")
        print()
        
        # Analyser les composants
        print("Analyse des composants:")
        
        component_stats = {}
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                component_params = module.weight.numel()
                if hasattr(module, 'bias') and module.bias is not None:
                    component_params += module.bias.numel()
                
                # Grouper par type de composant
                if 'mamba' in name.lower():
                    component_stats['Mamba-2'] = component_stats.get('Mamba-2', 0) + component_params
                elif 'attention' in name.lower() or 'attn' in name.lower():
                    component_stats['Attention'] = component_stats.get('Attention', 0) + component_params
                elif 'moe' in name.lower() or 'expert' in name.lower():
                    component_stats['MoE'] = component_stats.get('MoE', 0) + component_params
                elif 'multimodal' in name.lower():
                    component_stats['Multimodal'] = component_stats.get('Multimodal', 0) + component_params
                elif 'embed' in name.lower():
                    component_stats['Embeddings'] = component_stats.get('Embeddings', 0) + component_params
                elif 'lm_head' in name.lower():
                    component_stats['Output Head'] = component_stats.get('Output Head', 0) + component_params
        
        for component, params in component_stats.items():
            percentage = (params / total_params) * 100
            print(f"   • {component}: {format_number(params)} ({percentage:.1f}%)")
        
        print()
        
        # Test rapide du forward pass
        print("Test du forward pass...")
        model.eval()
        
        # Creer des donnees d'entree factices
        batch_size, seq_len = 1, 32
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # Test du forward pass
        forward_success = test_forward_pass(model, config, input_ids)
        
        if forward_success:
            if not skip_generation:
                # Test de génération
                print("\nTest de generation de texte...")
                test_text_generation(model, config)
            
            if not skip_benchmark:
                # Benchmarks de performance
                print("\nBenchmarks de performance...")
                run_performance_benchmarks(model, config)
                
                # Tests avec différentes tailles de séquence
                print("\nTests multi-séquences...")
                test_variable_sequence_lengths(model, config)
        
        print()
        print("Configuration recommandee pour differents usages:")
        print("   • Developpement/Test: Configuration actuelle")
        print(f"   • Production Edge: {format_number(total_params)} parametres")
        print(f"   • Production Serveur: Multiplier par ~50x -> {format_number(total_params * 50)}")
        print(f"   • Recherche/Full-scale: Multiplier par ~1000x -> {format_number(total_params * 1000)}")
        
    except Exception as e:
        print(f"Erreur lors de l'initialisation: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


def test_forward_pass(model, config, input_ids):
    """Teste le forward pass du modèle."""
    print("Test du forward pass...")
    
    with torch.no_grad():
        try:
            # Test basique
            start_time = time.time()
            outputs = model(input_ids)
            forward_time = time.time() - start_time
            
            if isinstance(outputs, dict):
                logits = outputs.get('logits')
                hidden_states = outputs.get('hidden_states')
                router_outputs = outputs.get('router_outputs', [])
            else:
                logits = getattr(outputs, 'logits', outputs)
                hidden_states = getattr(outputs, 'hidden_states', None)
                router_outputs = getattr(outputs, 'router_outputs', [])
            
            print(f"   [OK] Forward pass reussi!")
            print(f"   • Temps d'execution: {forward_time*1000:.2f}ms")
            print(f"   • Forme d'entree: {input_ids.shape}")
            print(f"   • Forme de sortie (logits): {logits.shape}")
            
            if hidden_states is not None:
                print(f"   • Forme hidden states: {hidden_states.shape}")
                
            if router_outputs:
                print(f"   • Nombre de sorties MoE: {len(router_outputs)}")
            
            # Verifier la coherence des sorties
            expected_vocab_size = config.vocab_size
            if logits.shape[-1] != expected_vocab_size:
                print(f"   [WARN] Attention: taille vocab attendue {expected_vocab_size}, obtenue {logits.shape[-1]}")
                
            return True
            
        except Exception as e:
            print(f"   [ERR] Erreur lors du forward pass: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_text_generation(model, config, max_length=50):
    """Teste la génération de texte."""
    try:
        # Prompt de test
        prompt = "L'intelligence artificielle est"
        print(f"   Prompt: '{prompt}'")
        
        # Simulation d'un tokenizer simple (indices aléatoires pour le test)
        input_ids = torch.randint(1, min(1000, config.vocab_size), (1, 8))  # Séquence courte
        
        start_time = time.time()
        with torch.no_grad():
            # Utiliser la méthode generate si elle existe
            if hasattr(model, 'generate'):
                generated = model.generate(
                    input_ids,
                    max_length=max_length,
                    temperature=0.8,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9
                )
            else:
                # Génération simple token par token
                generated = simple_generate(model, input_ids, max_length)
                
        generation_time = time.time() - start_time
        
        print(f"   [OK] Generation reussie!")
        print(f"   • Temps de génération: {generation_time:.2f}s")
        print(f"   • Tokens générés: {generated.shape[1] - input_ids.shape[1]}")
        print(f"   • Vitesse: {(generated.shape[1] - input_ids.shape[1])/generation_time:.1f} tokens/s")
        
    except Exception as e:
        print(f"   [ERR] Erreur de generation: {e}")
        import traceback
        traceback.print_exc()


def simple_generate(model, input_ids, max_length, temperature=1.0):
    """Génération simple sans la méthode generate."""
    generated = input_ids.clone()
    
    for _ in range(max_length - input_ids.shape[1]):
        with torch.no_grad():
            outputs = model(generated)
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
            
            # Échantillonnage du prochain token
            next_token_logits = logits[:, -1, :] / temperature
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            # Arrêt si token EOS
            if hasattr(model.config, 'eos_token_id') and next_token.item() == model.config.eos_token_id:
                break
                
    return generated


def run_performance_benchmarks(model, config):
    """Exécute des benchmarks de performance."""
    sequence_lengths = [16, 32, 64, 128]
    batch_sizes = [1, 2]
    
    results = []
    
    for batch_size in batch_sizes:
        for seq_len in sequence_lengths:
            try:
                input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
                
                # Warmup
                with torch.no_grad():
                    _ = model(input_ids)
                
                # Mesure
                times = []
                for _ in range(5):
                    start_time = time.time()
                    with torch.no_grad():
                        _ = model(input_ids)
                    times.append(time.time() - start_time)
                
                avg_time = sum(times) / len(times)
                tokens_per_sec = (batch_size * seq_len) / avg_time
                
                results.append({
                    'batch_size': batch_size,
                    'sequence_length': seq_len,
                    'avg_time_ms': avg_time * 1000,
                    'tokens_per_sec': tokens_per_sec
                })
                
                print(f"   • Batch {batch_size}, Seq {seq_len}: {avg_time*1000:.1f}ms, {tokens_per_sec:.0f} tokens/s")
                
            except Exception as e:
                print(f"   [ERR] Erreur benchmark (batch={batch_size}, seq={seq_len}): {e}")
    
    # Résumé des performances
    if results:
        best_throughput = max(results, key=lambda x: x['tokens_per_sec'])
        print(f"   [BEST] Meilleur debit: {best_throughput['tokens_per_sec']:.0f} tokens/s ")
        print(f"      (batch={best_throughput['batch_size']}, seq={best_throughput['sequence_length']})")


def test_variable_sequence_lengths(model, config):
    """Teste le modèle avec différentes longueurs de séquence."""
    test_lengths = [8, 16, 32, 64, 128, 256]
    
    for seq_len in test_lengths:
        try:
            input_ids = torch.randint(0, config.vocab_size, (1, seq_len))
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model(input_ids)
            process_time = time.time() - start_time
            
            logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
            memory_mb = get_tensor_memory_mb(logits)
            
            print(f"   • Seq {seq_len:3d}: {process_time*1000:6.1f}ms, {memory_mb:5.1f}MB output")
            
        except Exception as e:
            print(f"   [ERR] Erreur seq {seq_len}: {e}")
            break


def get_tensor_memory_mb(tensor):
    """Calcule la mémoire utilisée par un tensor en MB."""
    return tensor.element_size() * tensor.numel() / (1024 ** 2)


def compare_model_configs():
    """Compare différentes configurations de modèles."""
    print("\n" + "="*80)
    print("COMPARAISON DES CONFIGURATIONS ULTRA-AI")
    print("="*80)
    
    configs_to_test = [
        ("ultra_edge", "Ultra-AI Edge (Mobile)"),
        ("ultra_3b", "Ultra-AI 3B (Desktop)"),
        ("ultra_13b", "Ultra-AI 13B (Serveur)")
    ]
    
    comparison_results = []
    
    for config_name, description in configs_to_test:
        try:
            print(f"\n[TEST] {description}...")
            config = load_config(config_name=config_name)
            
            # Analyse théorique
            model_info = config.get_model_size_info()
            memory_info = config.estimate_memory_usage()
            
            comparison_results.append({
                'name': description,
                'total_params': model_info['total_parameters'],
                'active_params': model_info['active_parameters'], 
                'efficiency': model_info['parameter_efficiency'],
                'context_length': model_info['context_capacity'],
                'inference_memory': memory_info['total_inference']
            })
            
            print(f"   • Paramètres: {model_info['total_parameters']} ({model_info['active_parameters']} actifs)")
            print(f"   • Efficacité: {model_info['parameter_efficiency']}")
            print(f"   • Contexte: {model_info['context_capacity']}")
            print(f"   • Mémoire inférence: {memory_info['total_inference']}")
            
        except Exception as e:
            print(f"   [ERR] Erreur config {config_name}: {e}")
    
    # Tableau de comparaison
    if comparison_results:
        print(f"\n[SUMMARY] RESUME COMPARATIF:")
        print(f"{'Configuration':<25} {'Params':<12} {'Actifs':<12} {'Efficacité':<12} {'Mémoire':<10}")
        print("-" * 75)
        for result in comparison_results:
            print(f"{result['name']:<25} {result['total_params']:<12} {result['active_params']:<12} "
                  f"{result['efficiency']:<12} {result['inference_memory']:<10}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test complet d'Ultra-AI")
    parser.add_argument('--config', '-c', default='ultra_edge', 
                       choices=['ultra_edge', 'ultra_3b', 'ultra_13b', 'ultra_52b_active', 'ultra_390b'],
                       help='Configuration de modèle à tester')
    parser.add_argument('--compare', action='store_true', 
                       help='Comparer toutes les configurations')
    parser.add_argument('--no-generation', action='store_true',
                       help='Désactiver les tests de génération')
    parser.add_argument('--no-benchmark', action='store_true', 
                       help='Désactiver les benchmarks')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Mode verbeux')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_model_configs()
    else:
        # Modifier la fonction main pour utiliser les arguments
        exit_code = main(args.config, 
                        skip_generation=args.no_generation,
                        skip_benchmark=args.no_benchmark,
                        verbose=args.verbose)
        sys.exit(exit_code)