import argparse
import csv
import os
from pathlib import Path
from typing import List

from vlm_clip import CLIPBadGoodClassifier, list_images


def collect_inputs(input_path: Path) -> List[Path]:
    if input_path.is_file():
        files = list_images(input_path.parent)
        # Retain only the requested file if extension matches
        files = [input_path] if input_path in files or input_path.is_file() else []
        if not files:
            raise RuntimeError(f"Input file is not a supported image: {input_path}")
        return files

    if input_path.is_dir():
        files = list_images(input_path)
        if not files:
            raise RuntimeError(f"No images found under folder: {input_path}")
        return files

    raise FileNotFoundError(f"Input path not found: {input_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict BAD vs GOOD for images using CLIP.")
    parser.add_argument("--input", required=True, help="Image file or folder path.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out_csv", type=str, default="outputs/predictions.csv")
    args = parser.parse_args()

    input_path = Path(args.input)
    images = collect_inputs(input_path)

    classifier = CLIPBadGoodClassifier()

    results = []
    for img in images:
        prob_bad = classifier.predict_proba_bad(img)
        pred = int(prob_bad >= args.threshold)
        results.append((img, prob_bad, pred))
        print(f"{prob_bad:.4f}\t{pred}\t{img}")

    out_csv = Path(args.out_csv)
    if out_csv.parent:
        os.makedirs(out_csv.parent, exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "prob_bad", "pred"])
        for img, prob, pred in results:
            writer.writerow([str(img), prob, pred])

    print(f"\nSaved predictions to: {out_csv}")


if __name__ == "__main__":
    main()
