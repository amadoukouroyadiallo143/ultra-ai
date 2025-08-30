#!/usr/bin/env python3
"""
Téléchargement de datasets avec Hugging Face Datasets API
Version fiable et optimisée pour Ultra-AI
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List
import json

# Import Hugging Face datasets
try:
    from datasets import load_dataset, DatasetDict
    import torch
    from transformers import AutoTokenizer
except ImportError as e:
    print(f"Erreur d'import: {e}")
    print("Installez les dépendances avec: pip install datasets transformers torch")
    sys.exit(1)

# Configuration des datasets Hugging Face
HF_DATASETS_CONFIG = {
    "test": {
        "description": "Datasets test rapides pour validation",
        "datasets": {
            "wikitext": {
                "hf_name": "wikitext",
                "hf_config": "wikitext-103-raw-v1",
                "modality": "text",
                "split": "train[:1000]",  # Premier 1000 échantillons
                "description": "WikiText-103 échantillon"
            },
            "imdb": {
                "hf_name": "imdb", 
                "hf_config": None,
                "modality": "text",
                "split": "train[:2000]",
                "description": "IMDB reviews échantillon"
            }
        }
    },
    "small": {
        "description": "Datasets pour entraînement local",
        "datasets": {
            "wikitext_full": {
                "hf_name": "wikitext",
                "hf_config": "wikitext-103-raw-v1", 
                "modality": "text",
                "split": "train",
                "description": "WikiText-103 complet"
            },
            "openwebtext": {
                "hf_name": "Skylion007/openwebtext",
                "hf_config": None,
                "modality": "text", 
                "split": "train[:50000]",  # 50K échantillons
                "description": "OpenWebText échantillon"
            },
            "common_voice": {
                "hf_name": "mozilla-foundation/common_voice_11_0",
                "hf_config": "en",
                "modality": "audio",
                "split": "train[:5000]",  # 5K échantillons
                "description": "Common Voice English"
            },
            "coco_captions": {
                "hf_name": "HuggingFaceM4/COCO",
                "hf_config": None,
                "modality": "multimodal",
                "split": "train[:10000]",
                "description": "COCO images avec captions"
            }
        }
    },
    "medium": {
        "description": "Datasets pour entraînement sérieux",
        "datasets": {
            "pile": {
                "hf_name": "EleutherAI/pile",
                "hf_config": None,
                "modality": "text",
                "split": "train[:100000]", # 100K échantillons
                "description": "The Pile échantillon"
            },
            "laion": {
                "hf_name": "laion/laion400m",
                "hf_config": None,
                "modality": "multimodal", 
                "split": "train[:50000]",
                "description": "LAION-400M échantillon"
            }
        }
    }
}

def setup_logging():
    """Setup logging compatible Windows sans emojis."""
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    file_handler = logging.FileHandler('hf_download.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def download_hf_dataset(dataset_name: str, dataset_info: Dict, output_dir: Path) -> bool:
    """Télécharge un dataset Hugging Face."""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"[DOWNLOAD] {dataset_name} - {dataset_info['description']}")
        logger.info(f"[HF_NAME] {dataset_info['hf_name']}")
        logger.info(f"[SPLIT] {dataset_info['split']}")
        
        # Télécharger le dataset
        if dataset_info['hf_config']:
            dataset = load_dataset(
                dataset_info['hf_name'], 
                dataset_info['hf_config'],
                split=dataset_info['split'],
                trust_remote_code=True
            )
        else:
            dataset = load_dataset(
                dataset_info['hf_name'],
                split=dataset_info['split'], 
                trust_remote_code=True
            )
        
        logger.info(f"[SUCCESS] Downloaded {len(dataset)} samples")
        
        # Créer répertoire de sortie
        modality_dir = output_dir / dataset_info['modality']
        dataset_dir = modality_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Sauvegarder le dataset
        if isinstance(dataset, DatasetDict):
            # Si c'est un DatasetDict, sauvegarder chaque split
            for split_name, split_dataset in dataset.items():
                save_path = dataset_dir / f"{split_name}.json"
                split_dataset.to_json(save_path)
                logger.info(f"[SAVED] {split_name}: {save_path}")
        else:
            # Dataset simple
            save_path = dataset_dir / "data.json"
            dataset.to_json(save_path)
            logger.info(f"[SAVED] {save_path}")
        
        # Sauvegarder métadonnées
        metadata = {
            "dataset_name": dataset_name,
            "hf_name": dataset_info['hf_name'],
            "hf_config": dataset_info['hf_config'],
            "modality": dataset_info['modality'],
            "split": dataset_info['split'],
            "num_samples": len(dataset),
            "description": dataset_info['description'],
            "features": str(dataset.features) if hasattr(dataset, 'features') else None
        }
        
        metadata_path = dataset_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
            
        logger.info(f"[METADATA] {metadata_path}")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to download {dataset_name}: {e}")
        return False

def download_preset(preset: str, output_dir: Path) -> bool:
    """Télécharge un preset de datasets."""
    logger = logging.getLogger(__name__)
    
    if preset not in HF_DATASETS_CONFIG:
        logger.error(f"[ERROR] Unknown preset: {preset}")
        return False
        
    config = HF_DATASETS_CONFIG[preset]
    logger.info(f"[PRESET] {preset} - {config['description']}")
    
    success_count = 0
    total_count = len(config['datasets'])
    
    for dataset_name, dataset_info in config['datasets'].items():
        if download_hf_dataset(dataset_name, dataset_info, output_dir):
            success_count += 1
        else:
            logger.error(f"[FAILED] {dataset_name}")
    
    logger.info(f"[RESULT] {success_count}/{total_count} datasets downloaded")
    return success_count > 0

def create_training_manifest(data_dir: Path):
    """Crée un manifeste pour l'entraînement."""
    logger = logging.getLogger(__name__)
    
    manifest = {
        "data_root": str(data_dir),
        "created": str(pd.Timestamp.now()) if 'pd' in globals() else "unknown",
        "modalities": {}
    }
    
    # Scanner les modalités
    for modality_dir in data_dir.iterdir():
        if modality_dir.is_dir():
            modality = modality_dir.name
            datasets_info = {}
            
            for dataset_dir in modality_dir.iterdir():
                if dataset_dir.is_dir():
                    # Charger métadonnées si disponibles
                    metadata_path = dataset_dir / "metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                            datasets_info[dataset_dir.name] = metadata
                    else:
                        # Compter les fichiers de données
                        data_files = list(dataset_dir.glob("*.json"))
                        datasets_info[dataset_dir.name] = {
                            "path": str(dataset_dir),
                            "data_files": [str(f) for f in data_files],
                            "num_files": len(data_files)
                        }
            
            if datasets_info:
                manifest["modalities"][modality] = datasets_info
    
    # Sauvegarder manifeste
    manifest_path = data_dir / "training_manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        
    logger.info(f"[MANIFEST] Created: {manifest_path}")
    
    # Afficher résumé
    total_datasets = sum(len(modal_info) for modal_info in manifest["modalities"].values())
    logger.info(f"[SUMMARY] {total_datasets} datasets in {len(manifest['modalities'])} modalities")

def validate_environment():
    """Valide l'environnement."""
    logger = logging.getLogger(__name__)
    
    # Check CUDA
    if torch.cuda.is_available():
        logger.info(f"[GPU] {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("[GPU] CUDA not available - CPU only")
    
    # Check disk space (approximation)
    import shutil
    free_space_gb = shutil.disk_usage('.').free / (1024**3)
    logger.info(f"[DISK] {free_space_gb:.1f}GB free space")
    
    if free_space_gb < 10:
        logger.warning("[DISK] Low disk space - consider cleanup")
    
    return True

def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Download datasets with Hugging Face")
    parser.add_argument(
        '--preset', 
        choices=['test', 'small', 'medium'],
        default='test',
        help='Dataset preset to download'
    )
    parser.add_argument(
        '--output-dir',
        default='./data',
        help='Output directory for datasets'
    )
    parser.add_argument(
        '--manifest',
        action='store_true',
        help='Create training manifest'
    )
    parser.add_argument(
        '--validate',
        action='store_true', 
        help='Validate environment before download'
    )
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("[ULTRA-AI] Hugging Face Dataset Downloader")
    logger.info(f"Preset: {args.preset}")
    logger.info(f"Output: {output_dir}")
    
    # Validation optionnelle
    if args.validate:
        validate_environment()
    
    # Télécharger
    success = download_preset(args.preset, output_dir)
    
    if success:
        logger.info("[SUCCESS] Download completed!")
        
        # Créer manifeste
        if args.manifest:
            create_training_manifest(output_dir)
        
        print("\n[NEXT STEPS]")
        print("1. Check data:", output_dir)
        print("2. Run preprocessing: python scripts/preprocess_data.py")
        print("3. Start training: python scripts/train.py")
        
    else:
        logger.error("[ERROR] Download failed")
        sys.exit(1)

if __name__ == "__main__":
    main()