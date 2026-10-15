"""Build dataset for training the binary Skin vs. Non-Skin Presence Gate."""

import os
import ssl
import json
import random
import urllib.request
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_IMAGES_DIR = DATA_DIR / "raw" / "images"
GATE_DATA_DIR = DATA_DIR / "skin_gate"

GATE_TRAIN_SKIN = GATE_DATA_DIR / "train" / "skin"
GATE_TRAIN_NONSKIN = GATE_DATA_DIR / "train" / "non_skin"
GATE_VAL_SKIN = GATE_DATA_DIR / "val" / "skin"
GATE_VAL_NONSKIN = GATE_DATA_DIR / "val" / "non_skin"

for d in [GATE_TRAIN_SKIN, GATE_TRAIN_NONSKIN, GATE_VAL_SKIN, GATE_VAL_NONSKIN]:
    d.mkdir(parents=True, exist_ok=True)

ssl_ctx = ssl._create_unverified_context()


def download_picsum_images(count: int = 150):
    """Download diverse real-world everyday non-skin images."""
    print(f"Downloading {count} real-world non-skin images from Picsum...")
    saved = 0
    for i in range(count):
        seed = 1000 + i * 37
        url = f"https://picsum.photos/seed/{seed}/256/256"
        target_dir = GATE_TRAIN_NONSKIN if i < int(count * 0.8) else GATE_VAL_NONSKIN
        target_path = target_dir / f"picsum_{i:04d}.jpg"
        if target_path.exists():
            saved += 1
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ssl_ctx, timeout=6) as resp:
                target_path.write_bytes(resp.read())
                saved += 1
        except Exception as e:
            pass
    print(f"Downloaded {saved}/{count} real-world non-skin images.")


def generate_synthetic_non_skin(count: int = 100):
    """Generate synthetic non-skin patterns (screens, text, noise, geometry)."""
    print(f"Generating {count} synthetic non-skin patterns...")
    for i in range(count):
        target_dir = GATE_TRAIN_NONSKIN if i < int(count * 0.8) else GATE_VAL_NONSKIN
        target_path = target_dir / f"synth_{i:04d}.jpg"
        if target_path.exists():
            continue

        ptype = i % 5
        if ptype == 0:
            # Solid / gradient flat colors (digital UI, walls)
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            img = Image.new("RGB", (256, 256), color=color)
        elif ptype == 1:
            # Gray screen with digital text / code
            bg = random.randint(30, 220)
            img = Image.new("RGB", (256, 256), color=(bg, bg, bg))
            draw = ImageDraw.Draw(img)
            fg = (255 - bg, 255 - bg, 255 - bg)
            for line_y in range(15, 240, 18):
                draw.line([(15, line_y), (random.randint(80, 240), line_y)], fill=fg, width=random.randint(2, 4))
        elif ptype == 2:
            # Checkerboard / geometric pattern
            block_size = random.choice([16, 32, 64])
            arr = np.zeros((256, 256, 3), dtype=np.uint8)
            c1 = [random.randint(0, 255) for _ in range(3)]
            c2 = [random.randint(0, 255) for _ in range(3)]
            for r in range(256 // block_size):
                for c in range(256 // block_size):
                    val = c1 if (r + c) % 2 == 0 else c2
                    arr[r*block_size:(r+1)*block_size, c*block_size:(c+1)*block_size] = val
            img = Image.fromarray(arr)
        elif ptype == 3:
            # Random gaussian noise / textures
            noise = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
            img = Image.fromarray(noise)
        else:
            # High-contrast geometric shapes
            img = Image.new("RGB", (256, 256), color=(240, 240, 240))
            draw = ImageDraw.Draw(img)
            for _ in range(6):
                x1, y1 = random.randint(10, 200), random.randint(10, 200)
                x2, y2 = x1 + random.randint(20, 60), y1 + random.randint(20, 60)
                fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        img.save(target_path, "JPEG", quality=90)
    print("Synthetic non-skin images generated.")


def sample_skin_images(count: int = 250):
    """Sample diverse real skin images from HAM10000 dataset across all classes."""
    print(f"Sampling {count} skin images from HAM10000...")
    all_images = sorted(list(RAW_IMAGES_DIR.glob("*.jpg")))
    if not all_images:
        print("Error: No raw images found!")
        return

    random.seed(42)
    selected = random.sample(all_images, min(count, len(all_images)))
    for i, src in enumerate(selected):
        target_dir = GATE_TRAIN_SKIN if i < int(len(selected) * 0.8) else GATE_VAL_SKIN
        target_path = target_dir / src.name
        if not target_path.exists():
            img = Image.open(src)
            img.save(target_path, "JPEG", quality=90)
    print(f"Sampled {len(selected)} skin images.")


if __name__ == "__main__":
    download_picsum_images(150)
    generate_synthetic_non_skin(100)
    sample_skin_images(250)
    print("\nDataset summary:")
    print("Train skin:", len(list(GATE_TRAIN_SKIN.glob("*.jpg"))))
    print("Train non-skin:", len(list(GATE_TRAIN_NONSKIN.glob("*.jpg"))))
    print("Val skin:", len(list(GATE_VAL_SKIN.glob("*.jpg"))))
    print("Val non-skin:", len(list(GATE_VAL_NONSKIN.glob("*.jpg"))))
