#!/bin/bash
set -e

echo "=== Kerem Avatar Kurulum ==="

# Linly-Talker klonla
if [ ! -d "Linly-Talker" ]; then
    git clone --depth=1 https://github.com/Kedreamix/Linly-Talker.git
fi

# Python bağımlılıkları
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r Linly-Talker/requirements_app.txt

# GPT-SoVITS bağımlılıkları
cd Linly-Talker/GPT_SoVITS
pip install -r requirements.txt 2>/dev/null || true
cd ../..

# SadTalker bağımlılıkları
pip install face_alignment dlib
pip install gfpgan

# Proje bağımlılıkları
pip install pyyaml librosa soundfile ffmpeg-python gradio

echo "✓ Kurulum tamamlandı"
echo "→ Modelleri indirmek için: bash scripts/download_models.sh"
