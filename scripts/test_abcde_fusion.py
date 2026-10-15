import pandas as pd
from PIL import Image
from pathlib import Path
from app.server import compute_abcde_metrics

def main():
    df = pd.read_csv("data/splits/test.csv")
    mel_df = df[df["dx"] == "mel"].head(10)
    nv_df = df[df["dx"] == "nv"].head(10)

    print("--- MELANOMA TEST IMAGES ABCDE ---")
    for _, r in mel_df.iterrows():
        img_id = r["image_id"]
        p = Path("data/raw/images") / f"{img_id}.jpg"
        if p.exists():
            im = Image.open(p).convert("RGB")
            m = compute_abcde_metrics(im)
            comp = m["composite_score"]
            col = m["color"]["score"]
            bor = m["border"]["score"]
            print(f"  {img_id}: Composite={comp:.1f}, Color={col:.1f}, Border={bor:.1f}")

    print("\n--- NEVUS TEST IMAGES ABCDE ---")
    for _, r in nv_df.iterrows():
        img_id = r["image_id"]
        p = Path("data/raw/images") / f"{img_id}.jpg"
        if p.exists():
            im = Image.open(p).convert("RGB")
            m = compute_abcde_metrics(im)
            comp = m["composite_score"]
            col = m["color"]["score"]
            bor = m["border"]["score"]
            print(f"  {img_id}: Composite={comp:.1f}, Color={col:.1f}, Border={bor:.1f}")

if __name__ == "__main__":
    main()
