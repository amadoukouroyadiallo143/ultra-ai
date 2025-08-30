#!/usr/bin/env python3
"""
Test des données preprocessées pour vérifier leur qualité.
"""

import torch
import json
from pathlib import Path
from transformers import AutoTokenizer

def test_processed_data():
    """Test des données preprocessées."""
    
    print("=== TEST DES DONNÉES PREPROCESSÉES ===\n")
    
    # Charger le manifeste
    manifest_path = Path("data/processed/training_manifest.json")
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    print(f"📊 RÉSUMÉ DU DATASET")
    print(f"   Total échantillons: {manifest['total_samples']:,}")
    print(f"   Tokenizer: {manifest['tokenizer']}")
    print()
    
    # Charger le tokenizer
    tokenizer = AutoTokenizer.from_pretrained(manifest['tokenizer'])
    
    # Tester chaque dataset
    for modality, datasets in manifest['datasets'].items():
        print(f"🔤 MODALITÉ: {modality.upper()}")
        
        for dataset_name, dataset_info in datasets.items():
            print(f"\n📦 Dataset: {dataset_name}")
            print(f"   Échantillons: {dataset_info['total_samples']:,}")
            print(f"   Ignorés: {dataset_info['skipped_samples']:,}")
            print(f"   Longueur moyenne tokens: {dataset_info['avg_token_length']:.1f}")
            print(f"   Chunks: {len(dataset_info['chunk_files'])}")
            
            # Charger et tester le premier chunk
            first_chunk_path = dataset_info['chunk_files'][0]
            try:
                chunk_data = torch.load(first_chunk_path, map_location='cpu')
                print(f"   ✅ Chunk chargé: {len(chunk_data)} échantillons")
                
                # Examiner quelques échantillons
                print(f"\n   📋 ÉCHANTILLONS ({dataset_name}):")
                for i, sample in enumerate(chunk_data[:3]):  # Premier 3
                    print(f"   [{i+1}] Tokens: {sample['token_length']}")
                    print(f"       Preview: {sample['text_preview'][:100]}...")
                    
                    # Décoder les tokens pour vérifier
                    decoded = tokenizer.decode(sample['input_ids'][:50])  # Premier 50 tokens
                    print(f"       Decoded: {decoded[:100]}...")
                    print()
                    
            except Exception as e:
                print(f"   ❌ Erreur chargement chunk: {e}")
        print()
    
    # Test des splits
    splits_path = Path("data/processed/data_splits.json")
    if splits_path.exists():
        with open(splits_path, 'r', encoding='utf-8') as f:
            splits = json.load(f)
        
        print("📂 SPLITS DE DONNÉES")
        for modality, datasets in splits.items():
            for dataset_name, split_info in datasets.items():
                print(f"   {dataset_name}:")
                print(f"     Train chunks: {len(split_info['train_chunks'])}")
                print(f"     Val chunks: {len(split_info['val_chunks'])}")
                print(f"     Test chunks: {len(split_info['test_chunks'])}")
    
    print("\n✅ TEST TERMINÉ - Données prêtes pour l'entraînement!")

if __name__ == "__main__":
    test_processed_data()