# Kerem Avatar — Dijital İkiz

Linly-Talker tabanlı tam entegre dijital ikiz sistemi.  
Bir video ve fotoğraftan **ses klonu + konuşan avatar** oluşturur.

## Nasıl Çalışır

```
Video (ses örneği)  ──►  GPT-SoVITS  ──►  Klonlanmış Ses
Fotoğraf (yüz)      ──►  SadTalker   ──►  Konuşan Avatar
                                     ──►  Dijital İkiz Video
```

## Hızlı Başlangıç (Google Colab — Ücretsiz GPU)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Kerem23-elz/Kerem-avatar/blob/main/colab_avatar.ipynb)

## Yerel Kurulum (GPU Gerekli)

```bash
# 1. Kurulum
bash scripts/setup.sh

# 2. Modelleri indir
bash scripts/download_models.sh

# 3. Avatar oluştur
python run_avatar.py \
  --video inputs/ses_videosu.mp4 \
  --image inputs/yuz_fotografi.jpg \
  --text "Merhaba, ben dijital bir ikizim!" \
  --output outputs/avatar.mp4
```

## Dosya Yapısı

```
kerem-avatar/
├── run_avatar.py          # Ana pipeline
├── voice_clone.py         # Ses klonlama (GPT-SoVITS)
├── talking_head.py        # Konuşan yüz (SadTalker)
├── configs/config.yaml    # Ayarlar
├── scripts/
│   ├── setup.sh           # Kurulum
│   └── download_models.sh # Model indirme
├── inputs/                # Video ve fotoğraf buraya
└── outputs/               # Oluşturulan videolar buraya
```

## Gereksinimler

- Python 3.10+
- CUDA 11.8+ (GPU)
- 16GB+ RAM
- 20GB+ disk
