import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from src.constants import CLASS_NAMES
from src.preprocessing import build_evaluation_transform
from src.model import build_efficientnet_b0_classifier, build_efficientnet_b0_cbam_classifier
from src.utils import resolve_device

def main():
    device = resolve_device()
    curated_ids = {
        "akiec": "ISIC_0030826",
        "bcc": "ISIC_0026902",
        "bkl": "ISIC_0033539",
        "df": "ISIC_0031271",
        "mel": "ISIC_0033999",
        "nv": "ISIC_0026683",
        "vasc": "ISIC_0028885",
    }
    transform = build_evaluation_transform(224)

    models = {}
    m_base = build_efficientnet_b0_classifier(num_classes=7, pretrained=False).to(device)
    m_base.load_state_dict(torch.load("checkpoints/baseline_efficientnet_b0_cross_entropy/best_model.pt", map_location=device)["model_state_dict"])
    m_base.eval()
    models["baseline"] = m_base

    m_b = build_efficientnet_b0_classifier(num_classes=7, pretrained=False).to(device)
    m_b.load_state_dict(torch.load("checkpoints/ablation_b_targeted_augmentation_cross_entropy/best_model.pt", map_location=device)["model_state_dict"])
    m_b.eval()
    models["ablation_b"] = m_b

    m_c = build_efficientnet_b0_classifier(num_classes=7, pretrained=False).to(device)
    m_c.load_state_dict(torch.load("checkpoints/ablation_c_targeted_augmentation_focal_loss/best_model.pt", map_location=device)["model_state_dict"])
    m_c.eval()
    models["ablation_c"] = m_c

    m_d = build_efficientnet_b0_cbam_classifier(num_classes=7, pretrained=False, reduction_ratio=16).to(device)
    m_d.load_state_dict(torch.load("checkpoints/ablation_d_cbam_targeted_augmentation_cross_entropy/best_model.pt", map_location=device)["model_state_dict"])
    m_d.eval()
    models["ablation_d"] = m_d

    m_e = build_efficientnet_b0_cbam_classifier(num_classes=7, pretrained=False, reduction_ratio=16).to(device)
    m_e.load_state_dict(torch.load("checkpoints/efficientnet_b0_cbam_focal_targeted_augmentation/best_model.pt", map_location=device)["model_state_dict"])
    m_e.eval()
    models["proposed_e"] = m_e

    # Previous server ensemble weights:
    w_old = {"proposed_e": 2.2, "ablation_b": 1.5, "ablation_d": 1.2, "ablation_c": 1.1, "baseline": 0.4}
    # Optimal ensemble weights:
    w_opt = {"baseline": 3.0, "ablation_c": 1.0, "ablation_b": 0.5, "ablation_d": 0.5, "proposed_e": 0.0}

    print(f"{'Target DX':<10} | {'Image ID':<13} | {'Baseline':<10} | {'Old Ensemble':<15} | {'Optimal Ensemble':<18} | {'Opt Top Prob'}")
    print("-" * 80)

    raw_images = Path("data/raw/images")
    for true_dx, img_id in curated_ids.items():
        img_path = raw_images / f"{img_id}.jpg"
        if not img_path.exists():
            print(f"Missing {img_path}")
            continue
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(device)

        probs_all = {}
        with torch.no_grad():
            for k, m in models.items():
                probs_all[k] = F.softmax(m(tensor), dim=1).cpu().numpy()[0]

        # baseline
        base_pred = CLASS_NAMES[probs_all["baseline"].argmax()]

        # Old ensemble
        p_old = sum(probs_all[k] * w for k, w in w_old.items()) / sum(w_old.values())
        old_pred = CLASS_NAMES[p_old.argmax()]

        # Optimal ensemble
        p_opt = sum(probs_all[k] * w for k, w in w_opt.items()) / sum(w_opt.values())
        opt_pred = CLASS_NAMES[p_opt.argmax()]
        opt_conf = p_opt.max()

        match = "MATCH" if opt_pred == true_dx else "DIFF"
        print(f"{true_dx:<10} | {img_id:<13} | {base_pred:<10} | {old_pred:<15} | {opt_pred:<10} ({match:4s}) | {opt_conf*100:.1f}%")

if __name__ == "__main__":
    main()
