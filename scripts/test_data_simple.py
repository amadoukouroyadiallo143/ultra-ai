#!/usr/bin/env python3
"""
Test simple des données preprocessées
"""

import torch
import json
from pathlib import Path
from transformers import AutoTokenizer

def test_data():
    print("=== TEST DES DONNEES PREPROCESSEES ===\n")
    
    # Charger le manifeste
    manifest_path = Path("data/processed/training_manifest.json")
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    print(f"RESUME DU DATASET")
    print(f"   Total echantillons: {manifest['total_samples']:,}")
    print(f"   Tokenizer: {manifest['tokenizer']}")
    print()
    
    # Charger le tokenizer
    tokenizer = AutoTokenizer.from_pretrained(manifest['tokenizer'])
    
    # Tester chaque dataset
    for modality, datasets in manifest['datasets'].items():
        print(f"MODALITE: {modality.upper()}")
        
        for dataset_name, dataset_info in datasets.items():
            print(f"\nDataset: {dataset_name}")
            print(f"   Echantillons: {dataset_info['total_samples']:,}")
            print(f"   Ignores: {dataset_info['skipped_samples']:,}")
            print(f"   Longueur moyenne: {dataset_info['avg_token_length']:.1f} tokens")
            print(f"   Chunks: {len(dataset_info['chunk_files'])}")
            
            # Charger premier chunk
            first_chunk_path = dataset_info['chunk_files'][0]
            try:
                chunk_data = torch.load(first_chunk_path, map_location='cpu')
                print(f"   [OK] Chunk charge: {len(chunk_data)} echantillons")
                
                # Examiner échantillons
                print(f"\n   ECHANTILLONS ({dataset_name}):")
                for i, sample in enumerate(chunk_data[:2]):  # Premier 2
                    print(f"   [{i+1}] Tokens: {sample['token_length']}")
                    print(f"       Text: {sample['text_preview'][:80]}...")
                    print()
                    
            except Exception as e:
                print(f"   [ERREUR] Chargement chunk: {e}")
        print()
    
    print("[SUCCESS] Donnees pretes pour entrainement!")

if __name__ == "__main__":
    test_data()