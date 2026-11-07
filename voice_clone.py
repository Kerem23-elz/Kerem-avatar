"""
Ses Klonlama Modülü — GPT-SoVITS tabanlı
Videodan ses çıkarır ve klon model oluşturur.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

sys.path.insert(0, "Linly-Talker")
sys.path.insert(0, "Linly-Talker/GPT_SoVITS")


def extract_audio_from_video(video_path: str, output_wav: str, sample_rate: int = 22050) -> str:
    """Videodan ses çıkar, WAV olarak kaydet."""
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", str(sample_rate), "-ac", "1",
        output_wav, "-y"
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(f"✓ Ses çıkarıldı: {output_wav}")
    return output_wav


def clone_voice_and_speak(
    ref_audio: str,
    ref_text: str,
    target_text: str,
    output_audio: str,
    gpt_model: str = "Linly-Talker/GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt",
    sovits_model: str = "Linly-Talker/GPT_SoVITS/pretrained_models/s2G488k.pth",
    language: str = "auto",
) -> str:
    """
    GPT-SoVITS ile ses klonla ve metni o sesle oku.

    Args:
        ref_audio:    Klonlanacak ses örneği (WAV, min 3s)
        ref_text:     Ses örneğindeki söylenen metin (boş bırakılabilir)
        target_text:  Klonlanmış sesle söylenecek metin
        output_audio: Çıktı WAV dosyası
        language:     "tr", "zh", "en", "auto"
    """
    from GPT_SoVITS.inference_webui import get_tts_wav

    result_gen = get_tts_wav(
        ref_wav_path=ref_audio,
        prompt_text=ref_text,
        prompt_language=language,
        text=target_text,
        text_language=language,
        how_to_cut="凑四句一切",
        top_k=15,
        top_p=1.0,
        temperature=1.0,
        ref_free=True if not ref_text else False,
    )

    import soundfile as sf
    import numpy as np

    chunks = []
    sample_rate = None
    for sr, audio_chunk in result_gen:
        sample_rate = sr
        chunks.append(audio_chunk)

    audio = np.concatenate(chunks)
    sf.write(output_audio, audio, sample_rate)
    print(f"✓ Klonlanmış ses oluşturuldu: {output_audio}")
    return output_audio


def prepare_voice_clone(video_path: str, work_dir: str = "outputs") -> dict:
    """Video dosyasından ses klonlama için hazırlık yap."""
    os.makedirs(work_dir, exist_ok=True)
    video_name = Path(video_path).stem
    ref_audio = os.path.join(work_dir, f"{video_name}_ref.wav")
    extract_audio_from_video(video_path, ref_audio)
    return {
        "ref_audio": ref_audio,
        "video_name": video_name,
    }
