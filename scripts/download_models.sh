#!/bin/bash
set -e

echo "=== Model İndirme ==="

# HuggingFace mirror (daha hızlı)
export HF_ENDPOINT=https://hf-mirror.com

# Linly-Talker checkpointları (SadTalker dahil)
echo "→ Linly-Talker modelleri indiriliyor..."
pip install huggingface_hub -q
python3 - <<'PYEOF'
from huggingface_hub import snapshot_download
import os

# SadTalker modelleri
snapshot_download(
    repo_id="Kedreamix/Linly-Talker",
    local_dir="Linly-Talker/checkpoints",
    ignore_patterns=["*.bin.index.json", "*.safetensors.index.json"],
)
print("✓ Linly-Talker modelleri indirildi")
PYEOF

# GPT-SoVITS temel modelleri
echo "→ GPT-SoVITS modelleri indiriliyor..."
python3 - <<'PYEOF'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="lj1995/GPT-SoVITS",
    local_dir="Linly-Talker/GPT_SoVITS/pretrained_models",
)
print("✓ GPT-SoVITS modelleri indirildi")
PYEOF

echo ""
echo "✓ Tüm modeller indirildi"
echo "→ Çalıştırmak için: python run_avatar.py --help"
