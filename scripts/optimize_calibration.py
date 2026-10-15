import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.dataset import HAM10000Dataset
from src.model import build_efficientnet_b0_classifier, build_efficientnet_b0_cbam_classifier
from src.utils import resolve_device
from src.constants import CLASS_NAMES
from torch.utils.data import DataLoader

def get_loader(split_name):
    csv_path = Path("data/splits") / f"{split_name}.csv"
    dataset = HAM10000Dataset(csv_path, "data/raw", training=False, image_size=224)
    return DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

def main():
    device = resolve_device()
    val_loader = get_loader("validation")
    test_loader = get_loader("test")

    models = {}
    for k, path, cbam in [
        ("baseline", "checkpoints/baseline_efficientnet_b0_cross_entropy/best_model.pt", False),
        ("ablation_b", "checkpoints/ablation_b_targeted_augmentation_cross_entropy/best_model.pt", False),
        ("ablation_c", "checkpoints/ablation_c_targeted_augmentation_focal_loss/best_model.pt", False),
        ("ablation_d", "checkpoints/ablation_d_cbam_targeted_augmentation_cross_entropy/best_model.pt", True),
        ("proposed_e", "checkpoints/efficientnet_b0_cbam_focal_targeted_augmentation/best_model.pt", True),
    ]:
        if cbam:
            m = build_efficientnet_b0_cbam_classifier(num_classes=7, pretrained=False, reduction_ratio=16).to(device)
        else:
            m = build_efficientnet_b0_classifier(num_classes=7, pretrained=False).to(device)
        m.load_state_dict(torch.load(path, map_location=device)["model_state_dict"])
        m.eval()
        models[k] = m

    def extract_logits_and_labels(loader):
        all_labels = []
        all_logits = {k: [] for k in models}
        with torch.no_grad():
            for imgs, lbls, _ in loader:
                imgs = imgs.to(device)
                all_labels.extend(lbls.tolist())
                for k, m in models.items():
                    all_logits[k].append(m(imgs).cpu().numpy())
        lbls_arr = np.array(all_labels)
        lgts_arr = {k: np.concatenate(v) for k, v in all_logits.items()}
        return lgts_arr, lbls_arr

    print("Extracting validation set features...")
    val_logits, val_labels = extract_logits_and_labels(val_loader)
    print("Extracting test set features...")
    test_logits, test_labels = extract_logits_and_labels(test_loader)

    # Let's find optimal weights strictly on VALIDATION set so there is ZERO leakage!
    from scipy.optimize import minimize

    # Softmax on val
    val_probs = {k: np.exp(val_logits[k]) / np.sum(np.exp(val_logits[k]), axis=1, keepdims=True) for k in models}
    test_probs = {k: np.exp(test_logits[k]) / np.sum(np.exp(test_logits[k]), axis=1, keepdims=True) for k in models}

    model_names = ["baseline", "ablation_b", "ablation_c", "ablation_d", "proposed_e"]

    best_val_score = 0
    best_weights = None

    for wb in [1.0, 2.0, 3.0, 4.0]:
        for w_b in [0.0, 0.5, 1.0]:
            for wc in [0.0, 0.5, 1.0]:
                for wd in [0.0, 0.5, 1.0, 1.5]:
                    for we in [0.0, 0.3]:
                        w = np.array([wb, w_b, wc, wd, we])
                        w /= w.sum()
                        p_val = sum(val_probs[m] * w[i] for i, m in enumerate(model_names))
                        preds = p_val.argmax(axis=1)
                        acc = accuracy_score(val_labels, preds)
                        mf1 = f1_score(val_labels, preds, average="macro", zero_division=0)
                        score = 0.5 * acc + 0.5 * mf1
                        if score > best_val_score:
                            best_val_score = score
                            best_weights = w

    print("\nBest Validation-Selected Weights (zero test leakage):")
    for m, weight in zip(model_names, best_weights):
        print(f"  {m}: {weight:.4f}")

    # Now evaluate this strictly selected model on TEST set
    p_test = sum(test_probs[m] * best_weights[i] for i, m in enumerate(model_names))
    preds_test = p_test.argmax(axis=1)
    acc_test = accuracy_score(test_labels, preds_test)
    mf1_test = f1_score(test_labels, preds_test, average="macro", zero_division=0)
    wf1_test = f1_score(test_labels, preds_test, average="weighted", zero_division=0)
    print(f"\nTEST EVALUATION (Validation-Tuned Weights):")
    print(f"Accuracy: {acc_test*100:.2f}% | Macro-F1: {mf1_test*100:.2f}% | Weighted-F1: {wf1_test*100:.2f}%")

    # Now tune safeguard on validation set
    mel_idx = CLASS_NAMES.index("mel")
    nv_idx = CLASS_NAMES.index("nv")
    p_val_opt = sum(val_probs[m] * best_weights[i] for i, m in enumerate(model_names))
    preds_val_opt = p_val_opt.argmax(axis=1)

    best_safe_score = 0
    best_th = 0.30
    best_diff = 0.15

    for th in [0.20, 0.25, 0.30, 0.35, 0.40]:
        for df in [0.05, 0.10, 0.15, 0.20]:
            p_val_s = preds_val_opt.copy()
            mask = (p_val_s == nv_idx) & (p_val_opt[:, mel_idx] >= th) & ((p_val_opt[:, nv_idx] - p_val_opt[:, mel_idx]) <= df)
            p_val_s[mask] = mel_idx
            acc = accuracy_score(val_labels, p_val_s)
            mf1 = f1_score(val_labels, p_val_s, average="macro", zero_division=0)
            score = 0.5 * acc + 0.5 * mf1
            if score > best_safe_score:
                best_safe_score = score
                best_th = th
                best_diff = df

    print(f"\nBest Validation Safeguard: th={best_th:.2f}, diff={best_diff:.2f}")

    # Apply on test set
    preds_test_s = preds_test.copy()
    mask_test = (preds_test_s == nv_idx) & (p_test[:, mel_idx] >= best_th) & ((p_test[:, nv_idx] - p_test[:, mel_idx]) <= best_diff)
    preds_test_s[mask_test] = mel_idx
    acc_test_s = accuracy_score(test_labels, preds_test_s)
    mf1_test_s = f1_score(test_labels, preds_test_s, average="macro", zero_division=0)
    print(f"TEST with Validation Safeguard:")
    print(f"Accuracy: {acc_test_s*100:.2f}% | Macro-F1: {mf1_test_s*100:.2f}%")

if __name__ == "__main__":
    main()
