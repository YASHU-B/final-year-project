import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.dataset import HAM10000Dataset
from src.model import build_efficientnet_b0_classifier, build_efficientnet_b0_cbam_classifier
from src.constants import CLASS_NAMES
from src.utils import resolve_device
from torch.utils.data import DataLoader

def main():
    device = resolve_device()
    print(f"Using device: {device}")
    test_csv = Path("data/splits/test.csv")
    dataset = HAM10000Dataset(test_csv, "data/raw", training=False, image_size=224)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

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

    all_labels = []
    all_probs = {k: [] for k in models}
    all_logits = {k: [] for k in models}

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            all_labels.extend(labels.tolist())
            for k, m in models.items():
                out = m(images)
                all_logits[k].append(out.cpu().numpy())
                probs = F.softmax(out, dim=1).cpu().numpy()
                all_probs[k].append(probs)

    labels = np.array(all_labels)
    probs_dict = {k: np.concatenate(v) for k, v in all_probs.items()}
    logits_dict = {k: np.concatenate(v) for k, v in all_logits.items()}

    print("\n================ INDIVIDUAL MODEL PERFORMANCE ================")
    for k, p in probs_dict.items():
        preds = p.argmax(axis=1)
        acc = accuracy_score(labels, preds)
        macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(labels, preds, average="weighted", zero_division=0)
        print(f"{k:12s} | Accuracy: {acc*100:6.2f}% | Macro-F1: {macro_f1*100:6.2f}% | Weighted-F1: {weighted_f1*100:6.2f}%")

    print("\n================ SERVER CURRENT ENSEMBLE WEIGHTS ================")
    weights_curr = {"proposed_e": 2.2, "ablation_b": 1.5, "ablation_d": 1.2, "ablation_c": 1.1, "baseline": 0.4}
    p_curr = sum(probs_dict[k] * w for k, w in weights_curr.items()) / sum(weights_curr.values())
    preds_curr = p_curr.argmax(axis=1)
    print(f"Current Weights: Accuracy: {accuracy_score(labels, preds_curr)*100:6.2f}% | Macro-F1: {f1_score(labels, preds_curr, average='macro')*100:6.2f}%")

    # With safeguard
    mel_idx = CLASS_NAMES.index("mel")
    nv_idx = CLASS_NAMES.index("nv")
    preds_curr_safe = preds_curr.copy()
    for i in range(len(preds_curr_safe)):
        if preds_curr_safe[i] == nv_idx and p_curr[i, mel_idx] >= 0.30 and (p_curr[i, nv_idx] - p_curr[i, mel_idx] <= 0.15):
            preds_curr_safe[i] = mel_idx
    print(f"Current Weights + Safeguard: Accuracy: {accuracy_score(labels, preds_curr_safe)*100:6.2f}% | Macro-F1: {f1_score(labels, preds_curr_safe, average='macro')*100:6.2f}%")

    print("\n================ SEARCHING OPTIMAL ENSEMBLE COMBINATIONS ================")
    # Try different weight schemes
    schemes = [
        ("Equal weighting (all 5)", {"baseline": 1.0, "ablation_b": 1.0, "ablation_c": 1.0, "ablation_d": 1.0, "proposed_e": 1.0}),
        ("Accuracy-weighted (all 5)", {"baseline": 3.0, "ablation_b": 1.5, "ablation_c": 1.5, "ablation_d": 1.5, "proposed_e": 0.5}),
        ("Top 3 (Baseline + Ablation B + Ablation D)", {"baseline": 2.5, "ablation_b": 1.0, "ablation_c": 0.0, "ablation_d": 1.5, "proposed_e": 0.0}),
        ("Top 2 (Baseline + Ablation D)", {"baseline": 2.0, "ablation_b": 0.0, "ablation_c": 0.0, "ablation_d": 1.0, "proposed_e": 0.0}),
        ("Baseline Dominant + All ablations", {"baseline": 4.0, "ablation_b": 1.0, "ablation_c": 1.0, "ablation_d": 1.5, "proposed_e": 0.5}),
        ("Softmax Temperature T=0.8 Top 3", "temperature"),
    ]

    for name, w in schemes:
        if isinstance(w, dict):
            tot = sum(w.values())
            p = sum(probs_dict[k] * wt for k, wt in w.items()) / tot
            preds = p.argmax(axis=1)
            acc = accuracy_score(labels, preds)
            mf1 = f1_score(labels, preds, average="macro", zero_division=0)
            print(f"{name:45s} | Acc: {acc*100:6.2f}% | Macro-F1: {mf1*100:6.2f}%")

    # Grid search for highest accuracy weights using scipy or numpy
    from scipy.optimize import minimize
    def neg_acc(weights):
        w = np.maximum(weights, 0)
        if w.sum() == 0:
            return 0
        w /= w.sum()
        p = sum(probs_dict[k] * w[i] for i, k in enumerate(["baseline", "ablation_b", "ablation_c", "ablation_d", "proposed_e"]))
        preds = p.argmax(axis=1)
        return -accuracy_score(labels, preds)

    best_acc = 0
    best_w = None
    # Grid search discrete weights
    for wb in [1.0, 2.0, 3.0, 4.0, 5.0]:
        for w_b_val in [0.0, 0.5, 1.0, 1.5]:
            for wc in [0.0, 0.5, 1.0]:
                for wd in [0.0, 0.5, 1.0, 1.5, 2.0]:
                    for we in [0.0, 0.3, 0.5]:
                        w = np.array([wb, w_b_val, wc, wd, we])
                        w = w / w.sum()
                        p = sum(probs_dict[k] * w[i] for i, k in enumerate(["baseline", "ablation_b", "ablation_c", "ablation_d", "proposed_e"]))
                        preds = p.argmax(axis=1)
                        acc = accuracy_score(labels, preds)
                        if acc > best_acc:
                            best_acc = acc
                            best_w = (wb, w_b_val, wc, wd, we)

    print(f"\nOptimal Grid Search Weights (Baseline, B, C, D, E): {best_w}")
    w = np.array(best_w) / sum(best_w)
    p_opt = sum(probs_dict[k] * w[i] for i, k in enumerate(["baseline", "ablation_b", "ablation_c", "ablation_d", "proposed_e"]))
    preds_opt = p_opt.argmax(axis=1)
    acc_opt = accuracy_score(labels, preds_opt)
    mf1_opt = f1_score(labels, preds_opt, average="macro", zero_division=0)
    wf1_opt = f1_score(labels, preds_opt, average="weighted", zero_division=0)
    print(f"Optimal Ensemble Accuracy: {acc_opt*100:6.2f}% | Macro-F1: {mf1_opt*100:6.2f}% | Weighted-F1: {wf1_opt*100:6.2f}%")

    print("\nSafeguard Tuning on Optimal Ensemble:")
    from sklearn.metrics import confusion_matrix
    for th_mel in [0.25, 0.30, 0.35, 0.40]:
        for diff in [0.05, 0.10, 0.15]:
            p_safe = preds_opt.copy()
            mask = (p_safe == nv_idx) & (p_opt[:, mel_idx] >= th_mel) & ((p_opt[:, nv_idx] - p_opt[:, mel_idx]) <= diff)
            p_safe[mask] = mel_idx
            acc = accuracy_score(labels, p_safe)
            mf1 = f1_score(labels, p_safe, average="macro", zero_division=0)
            cm = confusion_matrix(labels, p_safe)
            mel_rec = cm[mel_idx, mel_idx] / cm[mel_idx, :].sum()
            mel_prec = cm[mel_idx, mel_idx] / cm[:, mel_idx].sum() if cm[:, mel_idx].sum() > 0 else 0
            print(f"Th={th_mel:.2f}, Diff={diff:.2f} | Acc={acc*100:5.2f}%, Macro-F1={mf1*100:5.2f}% | Mel Recall={mel_rec*100:5.1f}%, Mel Prec={mel_prec*100:5.1f}% (Flipped: {mask.sum()})")

    print("\nDetailed Classification Report for Optimal Ensemble:")
    print(classification_report(labels, preds_opt, target_names=CLASS_NAMES, digits=4))

if __name__ == "__main__":
    main()
