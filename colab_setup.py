#!/usr/bin/env python3
"""
Script de configuration rapide pour Google Colab
Automatise l'installation et la configuration pour l'entraînement Ultra-AI
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def run_command(cmd, description=""):
    """Exécute une commande avec gestion d'erreur."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} - Terminé")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - Erreur: {e}")
        print(f"Sortie d'erreur: {e.stderr}")
        return None

def setup_colab_environment():
    """Configure l'environnement Colab pour Ultra-AI."""
    
    print("🚀 CONFIGURATION ULTRA-AI POUR GOOGLE COLAB")
    print("=" * 50)
    
    # 1. Vérification GPU
    print("\n📊 Vérification du GPU disponible...")
    gpu_info = run_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits")
    if gpu_info:
        print(f"GPU détecté: {gpu_info.strip()}")
    else:
        print("⚠️ Aucun GPU détecté - Vérifiez Runtime > Change runtime type > GPU")
        
    # 2. Montage Google Drive
    print("\n💾 Montage de Google Drive...")
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Google Drive monté avec succès")
    except ImportError:
        print("⚠️ Pas dans un environnement Colab - Drive non monté")
    
    # 3. Clone du repository (si pas déjà fait)
    if not os.path.exists('/content/ultra_ai'):
        print("\n📥 Clonage du repository Ultra-AI...")
        run_command(
            "git clone https://github.com/amadoukouroyadiallo143/ultra-ai.git /content/ultra_ai_model",
            "Clonage du repository"
        )
    else:
        print("✅ Repository déjà cloné")
    
    # 4. Installation des dépendances
    print("\n📦 Installation des dépendances...")
    os.chdir('/content/ultra_ai_model')
    
    dependencies = [
        "pip install --upgrade pip",
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118",
        "pip install transformers datasets accelerate wandb",
        "pip install -e .",
    ]
    
    for dep in dependencies:
        run_command(dep, f"Installation: {dep.split()[-1]}")
    
    # 5. Création des répertoires nécessaires
    print("\n📁 Création des répertoires...")
    dirs_to_create = [
        "/content/drive/MyDrive/ultra_ai_checkpoints",
        "/content/drive/MyDrive/ultra_ai_logs",
        "/content/ultra_ai_model/data/processed",
        "/content/ultra_ai_model/training_output/logs"
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✅ Répertoire créé: {dir_path}")
    
    # 6. Configuration automatique des chemins
    print("\n⚙️ Configuration des chemins...")
    config_updates = {
        "train_data_path": "/content/ultra_ai_model/data/processed/text",
        "output_dir": "/content/drive/MyDrive/ultra_ai_checkpoints",
        "logging_dir": "/content/drive/MyDrive/ultra_ai_logs"
    }
    
    # 7. Test de l'installation
    print("\n🧪 Test de l'installation...")
    test_cmd = "cd /content/ultra_ai_model && python -c \"from src.models.ultra_ai_model import UltraAIModel; print('✅ Import réussi')\""
    run_command(test_cmd, "Test d'import du modèle")
    
    # 8. Génération du script de lancement
    print("\n📝 Génération du script de lancement...")
    launch_script = '''#!/bin/bash
# Script de lancement Ultra-AI pour Colab
cd /content/ultra_ai_model

echo "🚀 Démarrage de l'entraînement Ultra-AI Mini"
echo "Configuration: colab_mini.yaml"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo ""

python train.py \\
    --config configs/colab_mini.yaml \\
    --auto-config \\
    --output-dir /content/drive/MyDrive/ultra_ai_checkpoints \\
    --logging-steps 25 \\
    --save-steps 200 \\
    --num-epochs 2 \\
    --run-name colab-mini-$(date +%Y%m%d_%H%M%S)

echo "🎉 Entraînement terminé!"
'''
    
    with open('/content/ultra_ai_model/launch_colab.sh', 'w') as f:
        f.write(launch_script)
    
    run_command("chmod +x /content/ultra_ai_model/launch_colab.sh", "Permissions script")
    
    print("\n🎉 CONFIGURATION TERMINÉE!")
    print("=" * 50)
    print("\n📋 RÉSUMÉ DE LA CONFIGURATION:")
    print(f"• Modèle: Ultra-AI Mini (~500M paramètres)")
    print(f"• Configuration: configs/colab_mini.yaml")
    print(f"• Checkpoints: /content/drive/MyDrive/ultra_ai_checkpoints")
    print(f"• Logs: /content/drive/MyDrive/ultra_ai_logs")
    
    print("\n🚀 POUR DÉMARRER L'ENTRAÎNEMENT:")
    print("Option 1 - Script automatique:")
    print("  !bash /content/ultra_ai_model/launch_colab.sh")
    print("\nOption 2 - Commande manuelle:")
    print("  !cd /content/ultra_ai_model && python train.py --config configs/colab_mini.yaml --auto-config")
    
    print("\n⚠️ CONSEILS COLAB:")
    print("• Activez le GPU: Runtime > Change runtime type > GPU")
    print("• Gardez l'onglet ouvert pour éviter la déconnexion")
    print("• Les checkpoints sont sauvés sur Google Drive")
    print("• Temps d'entraînement estimé: 2-4h pour 2 époques")

def check_colab_resources():
    """Vérifie les ressources disponibles sur Colab."""
    print("\n🔍 VÉRIFICATION DES RESSOURCES:")
    print("-" * 30)
    
    # RAM
    ram_info = run_command("free -h | grep Mem", "Vérification RAM")
    if ram_info:
        print(f"RAM: {ram_info.strip()}")
    
    # GPU Memory
    gpu_mem = run_command("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits")
    if gpu_mem:
        used, total = gpu_mem.strip().split(', ')
        print(f"GPU Memory: {used}MB / {total}MB utilisés")
    
    # Disk space
    disk_info = run_command("df -h /content", "Vérification espace disque")
    if disk_info:
        lines = disk_info.strip().split('\n')
        if len(lines) > 1:
            print(f"Espace disque: {lines[1]}")

def create_sample_data():
    """Crée des données d'exemple pour tester l'entraînement."""
    print("\n📝 Création de données d'exemple...")
    
    sample_data = []
    sample_texts = [
        "L'intelligence artificielle révolutionne notre monde moderne.",
        "Les modèles de langage permettent de comprendre et générer du texte.",
        "L'architecture Mamba-2 offre une efficacité remarquable pour les séquences longues.",
        "L'entraînement sur GPU nécessite une optimisation minutieuse de la mémoire.",
        "Les transformers ont changé le domaine du traitement du langage naturel.",
    ]
    
    for i, text in enumerate(sample_texts):
        sample_data.append({
            "id": i,
            "text": text
        })
    
    # Sauvegarde des données d'exemple
    sample_dir = Path("/content/ultra_ai_model/data/text/sample")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    with open(sample_dir / "data.json", "w", encoding="utf-8") as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ {len(sample_data)} échantillons créés dans {sample_dir}")

if __name__ == "__main__":
    try:
        setup_colab_environment()
        check_colab_resources()
        create_sample_data()
        
        print("\n🎯 PRÊT POUR L'ENTRAÎNEMENT!")
        print("Exécutez la cellule suivante pour démarrer:")
        print("!bash /content/ultra_ai_model/launch_colab.sh")
        
    except Exception as e:
        print(f"\n❌ Erreur lors de la configuration: {e}")
        sys.exit(1)