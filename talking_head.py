"""
Konuşan Yüz Modülü — SadTalker / MuseTalk tabanlı
Fotoğraf + ses → konuşan avatar videosu
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, "Linly-Talker")


def generate_talking_head_sadtalker(
    image_path: str,
    audio_path: str,
    output_path: str,
    still_mode: bool = False,
    enhancer: str = "gfpgan",
    preprocess: str = "crop",
    size: int = 512,
    device: str = "cuda",
) -> str:
    """
    SadTalker ile konuşan yüz videosu oluştur.

    Args:
        image_path:  Yüz fotoğrafı (JPG/PNG)
        audio_path:  Konuşma sesi (WAV)
        output_path: Çıktı MP4 dosyası
        still_mode:  True = az hareket (profesyonel görünüm)
        enhancer:    "gfpgan" | "RestoreFormer" | None
        preprocess:  "crop" | "full" | "resize"
        size:        256 veya 512 (kalite)
    """
    from TFG import SadTalker

    sadtalker = SadTalker(
        checkpoint_path="Linly-Talker/checkpoints",
        config_path="Linly-Talker/src/config",
        device=device,
        lazy_load=True,
    )

    result = sadtalker.test(
        source_image=image_path,
        driven_audio=audio_path,
        still_mode=still_mode,
        use_enhancer=enhancer is not None,
        enhancer=enhancer,
        preprocess=preprocess,
        size=size,
        result_dir=str(Path(output_path).parent),
    )

    if result and os.path.exists(result):
        os.rename(result, output_path)

    print(f"✓ Avatar videosu oluşturuldu: {output_path}")
    return output_path


def generate_talking_head_musetalk(
    image_path: str,
    audio_path: str,
    output_path: str,
    device: str = "cuda",
) -> str:
    """
    MuseTalk ile konuşan yüz videosu oluştur (daha gerçekçi dudak hareketi).
    """
    from Musetalk.inference import inference as musetalk_inference

    musetalk_inference(
        source_image=image_path,
        driven_audio=audio_path,
        result_dir=str(Path(output_path).parent),
        device=device,
    )

    print(f"✓ MuseTalk avatar videosu oluşturuldu: {output_path}")
    return output_path
