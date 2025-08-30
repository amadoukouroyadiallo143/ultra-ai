#!/usr/bin/env python3
"""
Préprocessing des données Hugging Face pour Ultra-AI
Adapté pour traiter les données JSON téléchargées.
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np

# Ajouter src au path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def setup_logging():
    """Setup logging."""
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler('preprocess.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

class HFTextPreprocessor:
    """Préprocesseur pour données texte Hugging Face."""
    
    def __init__(self, tokenizer_name: str = "microsoft/DialoGPT-large", max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
        
    def process_json_dataset(self, json_path: Path, output_dir: Path, dataset_name: str) -> Dict[str, Any]:
        """Traite un dataset JSON."""
        logger = logging.getLogger(__name__)
        
        try:
            # Charger le JSONL (JSON Lines)
            data = []
            with open(json_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.warning(f"[SKIP] Invalid JSON line: {e}")
                            continue
            
            logger.info(f"[PROCESS] {dataset_name} - {len(data)} samples")
            
            processed_samples = []
            skipped = 0
            
            for i, sample in enumerate(tqdm(data, desc=f"Processing {dataset_name}")):
                try:
                    # Extraire le texte selon le format
                    if 'text' in sample:
                        text = sample['text']
                    elif 'review' in sample:
                        text = sample['review']
                    else:
                        # Prendre la première valeur string trouvée
                        text = None
                        for key, value in sample.items():
                            if isinstance(value, str) and len(value) > 10:
                                text = value
                                break
                    
                    if not text or len(text.strip()) < 10:
                        skipped += 1
                        continue
                    
                    # Nettoyer le texte
                    text = text.strip()
                    
                    # Tokeniser
                    encoded = self.tokenizer(
                        text,
                        max_length=self.max_length,
                        truncation=True,
                        padding=False,
                        return_tensors=None
                    )
                    
                    # Vérifier longueur minimale
                    if len(encoded['input_ids']) < 10:
                        skipped += 1
                        continue
                    
                    processed_sample = {
                        'input_ids': encoded['input_ids'],
                        'attention_mask': encoded['attention_mask'],
                        'text_preview': text[:200],  # Premier 200 chars
                        'original_length': len(text),
                        'token_length': len(encoded['input_ids']),
                        'sample_id': i
                    }
                    
                    # Ajouter label si présent
                    if 'label' in sample:
                        processed_sample['label'] = sample['label']
                    
                    processed_samples.append(processed_sample)
                    
                except Exception as e:
                    logger.warning(f"[SKIP] Sample {i}: {e}")
                    skipped += 1
                    continue
            
            # Sauvegarder les données traitées
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Diviser en chunks pour éviter les gros fichiers
            chunk_size = 1000
            chunks = [processed_samples[i:i + chunk_size] 
                     for i in range(0, len(processed_samples), chunk_size)]
            
            chunk_files = []
            for chunk_idx, chunk in enumerate(chunks):
                chunk_file = output_dir / f"{dataset_name}_chunk_{chunk_idx}.pt"
                torch.save(chunk, chunk_file)
                chunk_files.append(str(chunk_file))
                logger.info(f"[SAVED] Chunk {chunk_idx}: {len(chunk)} samples -> {chunk_file}")
            
            # Métadonnées
            metadata = {
                'dataset_name': dataset_name,
                'total_samples': len(processed_samples),
                'skipped_samples': skipped,
                'chunk_files': chunk_files,
                'max_length': self.max_length,
                'tokenizer': self.tokenizer.name_or_path,
                'avg_token_length': np.mean([s['token_length'] for s in processed_samples]) if processed_samples else 0
            }
            
            metadata_file = output_dir / f"{dataset_name}_metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[SUCCESS] {dataset_name}: {len(processed_samples)} processed, {skipped} skipped")
            return metadata
            
        except Exception as e:
            logger.error(f"[ERROR] Processing {dataset_name}: {e}")
            return {"error": str(e)}

def process_manifest(manifest_path: Path, output_dir: Path, tokenizer_name: str = "microsoft/DialoGPT-large") -> bool:
    """Traite tous les datasets d'un manifeste."""
    logger = logging.getLogger(__name__)
    
    # Charger manifeste
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
    
    data_root = Path(manifest['data_root'])
    logger.info(f"[START] Processing datasets from {data_root}")
    
    # Créer préprocesseur
    preprocessor = HFTextPreprocessor(tokenizer_name=tokenizer_name)
    
    processed_datasets = {}
    total_processed = 0
    
    # Traiter chaque modalité
    for modality, datasets in manifest['modalities'].items():
        logger.info(f"[MODALITY] {modality}")
        modality_output = output_dir / modality
        modality_output.mkdir(parents=True, exist_ok=True)
        
        processed_datasets[modality] = {}
        
        for dataset_name, dataset_info in datasets.items():
            logger.info(f"[DATASET] {dataset_name} - {dataset_info.get('description', 'N/A')}")
            
            # Trouver le fichier de données
            dataset_dir = data_root / modality / dataset_name
            data_file = dataset_dir / "data.json"
            
            if not data_file.exists():
                logger.error(f"[ERROR] Data file not found: {data_file}")
                continue
            
            # Traiter le dataset
            result = preprocessor.process_json_dataset(
                data_file, 
                modality_output / dataset_name,
                dataset_name
            )
            
            if 'error' not in result:
                processed_datasets[modality][dataset_name] = result
                total_processed += result['total_samples']
            else:
                logger.error(f"[FAILED] {dataset_name}: {result['error']}")
    
    # Créer nouveau manifeste pour l'entraînement
    training_manifest = {
        'processed_data_root': str(output_dir),
        'tokenizer': tokenizer_name,
        'total_samples': total_processed,
        'datasets': processed_datasets,
        'processing_date': str(torch.tensor(0).item())  # Placeholder
    }
    
    training_manifest_path = output_dir / "training_manifest.json"
    with open(training_manifest_path, 'w', encoding='utf-8') as f:
        json.dump(training_manifest, f, indent=2, ensure_ascii=False)
    
    logger.info(f"[MANIFEST] Training manifest: {training_manifest_path}")
    logger.info(f"[SUMMARY] Total processed samples: {total_processed}")
    
    return total_processed > 0

def create_data_splits(processed_dir: Path, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Crée des splits train/val/test."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"[SPLITS] Creating data splits (train: {train_ratio}, val: {val_ratio})")
    
    splits_info = {}
    
    # Scanner les datasets traités
    for modality_dir in processed_dir.iterdir():
        if modality_dir.is_dir():
            modality = modality_dir.name
            splits_info[modality] = {}
            
            for dataset_dir in modality_dir.iterdir():
                if dataset_dir.is_dir():
                    dataset_name = dataset_dir.name
                    
                    # Trouver les chunks
                    chunk_files = list(dataset_dir.glob(f"{dataset_name}_chunk_*.pt"))
                    
                    if not chunk_files:
                        logger.warning(f"[SKIP] No chunks found for {dataset_name}")
                        continue
                    
                    # Calculer splits
                    n_chunks = len(chunk_files)
                    n_train = max(1, int(n_chunks * train_ratio))
                    n_val = max(1, int(n_chunks * val_ratio)) if n_chunks > 2 else 0
                    n_test = n_chunks - n_train - n_val
                    
                    splits_info[modality][dataset_name] = {
                        'total_chunks': n_chunks,
                        'train_chunks': chunk_files[:n_train],
                        'val_chunks': chunk_files[n_train:n_train + n_val] if n_val > 0 else [],
                        'test_chunks': chunk_files[n_train + n_val:] if n_test > 0 else []
                    }
                    
                    logger.info(f"[SPLITS] {dataset_name}: train={n_train}, val={n_val}, test={n_test}")
    
    # Sauvegarder info des splits
    splits_file = processed_dir / "data_splits.json"
    with open(splits_file, 'w', encoding='utf-8') as f:
        json.dump(splits_info, f, indent=2, default=str, ensure_ascii=False)
    
    logger.info(f"[SPLITS] Info saved: {splits_file}")

def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Preprocess Hugging Face datasets")
    parser.add_argument(
        '--manifest',
        default='./data/training_manifest.json',
        help='Path to training manifest'
    )
    parser.add_argument(
        '--output-dir',
        default='./data/processed',
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--tokenizer',
        default='microsoft/DialoGPT-large',
        help='Tokenizer to use'
    )
    parser.add_argument(
        '--create-splits',
        action='store_true',
        help='Create train/val/test splits'
    )
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    
    logger.info("[ULTRA-AI] HF Data Preprocessor")
    logger.info(f"Manifest: {manifest_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Tokenizer: {args.tokenizer}")
    
    # Vérifier manifeste
    if not manifest_path.exists():
        logger.error(f"[ERROR] Manifest not found: {manifest_path}")
        sys.exit(1)
    
    # Traiter les données
    success = process_manifest(manifest_path, output_dir, args.tokenizer)
    
    if success:
        logger.info("[SUCCESS] Data processing completed!")
        
        # Créer splits si demandé
        if args.create_splits:
            create_data_splits(output_dir)
        
        print("\n[NEXT STEPS]")
        print("1. Check processed data:", output_dir)
        print("2. Start training: python scripts/train.py")
        print("3. Or test model: python test_model_init.py")
        
    else:
        logger.error("[ERROR] Data processing failed")
        sys.exit(1)

if __name__ == "__main__":
    main()