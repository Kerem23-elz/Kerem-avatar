"""
Kerem Avatar — Dijital İkiz Pipeline
Kullanım: python run_avatar.py --video ses.mp4 --image yuz.jpg --text "Merhaba!"
"""

import argparse
import os
import sys
import yaml
from pathlib import Path

from voice_clone import prepare_voice_clone, clone_voice_and_speak
from talking_head import generate_talking_head_sadtalker, generate_talking_head_musetalk


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run_pipeline(
    video_path: str,
    image_path: str,
    text: str,
    output_path: str = "outputs/avatar.mp4",
    ref_text: str = "",
    avatar_model: str = "sadtalker",
    config_path: str = "configs/config.yaml",
):
    """Tam dijital ikiz pipeline: ses klonu + konuşan yüz."""
    cfg = load_config(config_path)
    os.makedirs("outputs", exist_ok=True)

    name = Path(video_path).stem
    cloned_audio = f"outputs/{name}_klonses.wav"

    # 1. Ses klonlama
    print("\n[1/2] Ses klonlanıyor...")
    voice_info = prepare_voice_clone(video_path)
    clone_voice_and_speak(
        ref_audio=voice_info["ref_audio"],
        ref_text=ref_text,
        target_text=text,
        output_audio=cloned_audio,
        language=cfg["voice"]["language"],
    )

    # 2. Konuşan yüz oluşturma
    print("\n[2/2] Avatar videosu oluşturuluyor...")
    device = cfg["pipeline"]["device"]

    if avatar_model == "musetalk":
        generate_talking_head_musetalk(
            image_path=image_path,
            audio_path=cloned_audio,
            output_path=output_path,
            device=device,
        )
    else:
        generate_talking_head_sadtalker(
            image_path=image_path,
            audio_path=cloned_audio,
            output_path=output_path,
            still_mode=cfg["avatar"]["still_mode"],
            enhancer=cfg["avatar"]["enhancer"],
            preprocess=cfg["avatar"]["preprocess"],
            size=cfg["avatar"]["size"],
            device=device,
        )

    print(f"\n✓ Dijital ikiz hazır: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Kerem Avatar — Dijital İkiz")
    parser.add_argument("--video", required=True, help="Ses örneği için video (mp4)")
    parser.add_argument("--image", required=True, help="Yüz fotoğrafı (jpg/png)")
    parser.add_argument("--text", required=True, help="Avatarın söyleyeceği metin")
    parser.add_argument("--output", default="outputs/avatar.mp4", help="Çıktı videosu")
    parser.add_argument("--ref-text", default="", help="Ses örneğindeki metin (opsiyonel)")
    parser.add_argument("--model", default="sadtalker", choices=["sadtalker", "musetalk"],
                        help="Avatar modeli")
    parser.add_argument("--config", default="configs/config.yaml", help="Konfigürasyon dosyası")
    args = parser.parse_args()

    run_pipeline(
        video_path=args.video,
        image_path=args.image,
        text=args.text,
        output_path=args.output,
        ref_text=args.ref_text,
        avatar_model=args.model,
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
