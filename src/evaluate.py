import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from tqdm import tqdm

from vlm_clip import CLIPBadGoodClassifier, list_images


def load_images_with_labels(good_dir: Path, bad_dir: Path) -> Tuple[List[Path], List[int]]:
    if not good_dir.exists():
        raise FileNotFoundError(f"GOOD directory not found: {good_dir}")
    if not bad_dir.exists():
        raise FileNotFoundError(f"BAD directory not found: {bad_dir}")

    good_images = list_images(good_dir)
    bad_images = list_images(bad_dir)

    if not good_images:
        raise RuntimeError(f"No images found in GOOD dir: {good_dir}")
    if not bad_images:
        raise RuntimeError(f"No images found in BAD dir: {bad_dir}")

    paths = good_images + bad_images
    labels = [0] * len(good_images) + [1] * len(bad_images)
    return paths, labels


def pick_best_f1_threshold(y_true: List[int], probs_bad: List[float]) -> float:
    thresholds = sorted(set(probs_bad + [0.0, 1.0]))
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        preds = [int(p >= t) for p in probs_bad]
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, preds, pos_label=1, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate zero-shot CLIP BAD vs GOOD classifier.")
    parser.add_argument("--good_dir", default="data/success_image_edit", type=str)
    parser.add_argument("--bad_dir", default="data/bad_images", type=str)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--save_csv", type=str, default="outputs/results.csv")
    args = parser.parse_args()

    classifier = CLIPBadGoodClassifier()
    image_paths, labels = load_images_with_labels(Path(args.good_dir), Path(args.bad_dir))

    probs_bad: List[float] = []
    for img_path in tqdm(image_paths, desc="Scoring images"):
        prob = classifier.predict_proba_bad(img_path)
        probs_bad.append(prob)

    if args.threshold is None:
        threshold = pick_best_f1_threshold(labels, probs_bad)
        print(
            f"[Info] Using threshold={threshold:.4f} chosen to maximize F1 on this dataset (optimistic)."
        )
    else:
        threshold = args.threshold

    preds = [int(p >= threshold) for p in probs_bad]

    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, pos_label=1, average="binary", zero_division=0
    )

    roc_auc = None
    if len(set(labels)) > 1 and len(set(probs_bad)) > 1:
        try:
            roc_auc = roc_auc_score(labels, probs_bad)
        except ValueError:
            roc_auc = None

    cm = confusion_matrix(labels, preds, labels=[0, 1])

    print("\nMetrics (BAD=positive class):")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC:   {roc_auc:.4f}")
    else:
        print("ROC-AUC:   undefined (needs both classes and varying scores)")
    print(f"Confusion matrix [[TN FP],[FN TP]]:\n{cm.tolist()}")

    top_bad = sorted(zip(image_paths, probs_bad), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 most-likely BAD images:")
    for path, prob in top_bad:
        print(f"  {prob:.4f}  {path}")

    save_path = Path(args.save_csv)
    if save_path.parent:
        os.makedirs(save_path.parent, exist_ok=True)

    with open(save_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "label", "prob_bad", "pred", "threshold_used"])
        for path, label, prob, pred in zip(image_paths, labels, probs_bad, preds):
            writer.writerow([str(path), label, prob, pred, threshold])

    print(f"\nSaved per-image results to: {save_path}")


if __name__ == "__main__":
    main()
