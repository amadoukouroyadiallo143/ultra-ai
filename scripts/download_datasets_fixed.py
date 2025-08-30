#!/usr/bin/env python3
"""
Script de téléchargement automatique des datasets pour Ultra-AI
Version corrigée sans emojis pour Windows et URLs valides.
"""

import os
import sys
import argparse
import requests
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List
import logging
from tqdm import tqdm
import hashlib
import shutil

# Configuration des datasets recommandés - URLs valides
DATASETS_CONFIG = {
    "small": {
        "total_size": "~5GB",
        "datasets": {
            "openwebtext_sample": {
                "url": "https://huggingface.co/datasets/Skylion007/openwebtext/resolve/main/subsets/urlsf_subset00-1_data.xz", 
                "size": "3GB",
                "extract_to": "data/text/openwebtext",
                "description": "OpenWebText sample"
            },
            "coco2017_val": {
                "url": "http://images.cocodataset.org/zips/val2017.zip",
                "size": "1GB", 
                "extract_to": "data/images/coco2017",
                "description": "COCO validation images"
            },
            "wikipedia_sample": {
                "url": "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles1.xml-p1p41242.bz2",
                "size": "200MB",
                "extract_to": "data/text/wikipedia", 
                "description": "Wikipedia EN sample"
            }
        }
    },
    "test": {
        "total_size": "~50MB",
        "datasets": {
            "tiny_sample": {
                "url": "https://huggingface.co/datasets/wikitext/resolve/main/wikitext-103-raw/wiki.test.raw",
                "size": "50MB",
                "extract_to": "data/text/wikitext",
                "description": "Wikitext test sample"
            }
        }
    }
}

def setup_logging():
    """Configure le logging sans emojis."""
    # Console handler avec encoding approprié pour Windows
    import sys
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    # File handler avec UTF-8
    file_handler = logging.FileHandler('download.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

def download_file(url: str, destination: Path, expected_size: str = None) -> bool:
    """Télécharge un fichier avec barre de progression."""
    logger = logging.getLogger(__name__)
    
    try:
        # Créer le répertoire de destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Vérifier si le fichier existe déjà
        if destination.exists():
            logger.info(f"[SKIP] Fichier existe: {destination.name}")
            return True
            
        logger.info(f"[DOWNLOAD] URL: {url}")
        logger.info(f"[DOWNLOAD] Destination: {destination}")
        
        # Télécharger avec progression
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as f:
            with tqdm(
                desc=destination.name,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)
                    
        logger.info(f"[SUCCESS] Download completed: {destination.name}")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Download failed {url}: {e}")
        if destination.exists():
            destination.unlink()  # Supprimer fichier partiellement téléchargé
        return False

def extract_archive(archive_path: Path, extract_to: Path) -> bool:
    """Extrait une archive."""
    logger = logging.getLogger(__name__)
    
    try:
        extract_to.mkdir(parents=True, exist_ok=True)
        logger.info(f"[EXTRACT] {archive_path.name} -> {extract_to}")
        
        if archive_path.suffix == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
                
        elif archive_path.suffix in ['.tar', '.gz', '.xz', '.bz2']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        elif archive_path.suffix == '.xz':
            # Pour les fichiers .xz simples
            import lzma
            with lzma.LZMAFile(archive_path, 'rb') as f_in:
                output_file = extract_to / archive_path.stem
                with open(output_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        elif archive_path.suffix in ['.raw', '.txt']:
            # Fichiers texte bruts - copier directement
            output_file = extract_to / archive_path.name
            shutil.copy2(archive_path, output_file)
        else:
            logger.warning(f"[WARNING] Format non supporté: {archive_path.suffix}")
            return False
            
        logger.info(f"[SUCCESS] Extraction completed: {extract_to}")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Extraction failed {archive_path}: {e}")
        return False

def download_preset(preset: str, data_dir: Path, keep_archives: bool = False) -> bool:
    """Télécharge un preset de datasets."""
    logger = logging.getLogger(__name__)
    
    if preset not in DATASETS_CONFIG:
        logger.error(f"[ERROR] Preset inconnu: {preset}")
        return False
        
    config = DATASETS_CONFIG[preset]
    logger.info(f"[START] Téléchargement preset '{preset}' (~{config['total_size']})")
    
    success_count = 0
    total_datasets = len(config["datasets"])
    
    for dataset_name, dataset_info in config["datasets"].items():
        logger.info(f"[DATASET] {dataset_name} - {dataset_info['description']}")
        logger.info(f"[DATASET] Taille: {dataset_info['size']}")
        
        # Chemins
        archive_name = Path(dataset_info["url"]).name
        archive_path = data_dir / "archives" / archive_name
        extract_path = data_dir / dataset_info["extract_to"]
        
        # Télécharger
        if download_file(dataset_info["url"], archive_path, dataset_info["size"]):
            # Extraire
            if extract_archive(archive_path, extract_path):
                success_count += 1
                
                # Supprimer l'archive si demandé
                if not keep_archives:
                    try:
                        archive_path.unlink()
                        logger.info(f"[CLEANUP] Archive supprimée: {archive_name}")
                    except Exception as e:
                        logger.warning(f"[WARNING] Impossible de supprimer {archive_name}: {e}")
            else:
                logger.error(f"[ERROR] Échec extraction: {dataset_name}")
        else:
            logger.error(f"[ERROR] Échec téléchargement: {dataset_name}")
    
    logger.info(f"[SUMMARY] Résultat: {success_count}/{total_datasets} datasets téléchargés")
    return success_count == total_datasets

def create_data_manifest(data_dir: Path):
    """Crée un manifeste des données téléchargées."""
    logger = logging.getLogger(__name__)
    
    manifest = {
        "data_directory": str(data_dir),
        "datasets": {},
        "total_size_gb": 0
    }
    
    for modality_dir in data_dir.iterdir():
        if modality_dir.is_dir() and modality_dir.name != "archives":
            manifest["datasets"][modality_dir.name] = {
                "path": str(modality_dir),
                "subdirectories": [str(d) for d in modality_dir.iterdir() if d.is_dir()]
            }
    
    manifest_path = data_dir / "data_manifest.json"
    import json
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
        
    logger.info(f"[MANIFEST] Créé: {manifest_path}")

def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Téléchargement automatique datasets Ultra-AI")
    parser.add_argument(
        "--preset", 
        type=str, 
        choices=["test", "small"], 
        default="test",
        help="Preset de datasets à télécharger"
    )
    parser.add_argument(
        "--data-dir", 
        type=str, 
        default="./data",
        help="Répertoire de destination"
    )
    parser.add_argument(
        "--keep-archives", 
        action="store_true",
        help="Conserver les archives après extraction"
    )
    parser.add_argument(
        "--create-manifest", 
        action="store_true",
        help="Créer un manifeste des données"
    )
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("[ULTRA-AI] Dataset Downloader")
    logger.info(f"   Preset: {args.preset}")
    logger.info(f"   Destination: {data_dir}")
    
    # Télécharger
    success = download_preset(args.preset, data_dir, args.keep_archives)
    
    if success:
        logger.info("[SUCCESS] Tous les datasets téléchargés!")
        
        if args.create_manifest:
            create_data_manifest(data_dir)
            
        # Recommandations
        logger.info("")
        logger.info("[NEXT STEPS]")
        logger.info("1. Vérifier les données téléchargées")
        logger.info("2. Lancer preprocessing: python scripts/preprocess_data.py") 
        logger.info("3. Commencer entraînement: python scripts/train.py")
        
    else:
        logger.error("[ERROR] Certains téléchargements ont échoué")
        sys.exit(1)

if __name__ == "__main__":
    main()