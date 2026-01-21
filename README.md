## VLM Bad-vs-Good Image Classifier

Zero-shot BAD (1) vs GOOD (0) classifier using CLIP (ViT-B-32, pretrained `openai`) via `open-clip-torch`. Prompts contrast “bad composition / distorted / incoherent” vs “good composition / coherent / clean”. No training required.

### Setup
- Python 3.9+ recommended.
- Create venv: `python -m venv .venv && source .venv/bin/activate`
- Install deps: `pip install -r requirements.txt`

### Data layout
- BAD images: `data/bad_images/` (e.g., `image_1.png`, `image_2.png`, ...)
- GOOD images: `data/success_image_edit/` (e.g., `image_2_corrected.png`, ...)

### Evaluate (prints metrics + CSV)
```
python src/evaluate.py \
  --good_dir data/success_image_edit \
  --bad_dir data/bad_images \
  --save_csv outputs/results.csv
```
- If `--threshold` is omitted, the script chooses the threshold that maximizes F1 on this dataset (optimistic on small sets; for real use, pick a fixed threshold or tune on a separate split).

### Inference (file or folder)
```
python src/predict.py --input data/bad_images --threshold 0.5 --out_csv outputs/predictions.csv
```
- `--input` can be a single image or a folder (recursively scans common extensions).
- Outputs a CSV with `path,prob_bad,pred` into `outputs/`.

### How to check it’s working
1) Run evaluation (above). You should see metrics and a CSV in `outputs/results.csv`.  
2) Spot-check single images:  
   - Bad example: `python src/predict.py --input data/bad_images/image_2.png --threshold 0.5`  
   - Good example: `python src/predict.py --input data/success_image_edit/image_2_corrected.png --threshold 0.5`  
   Higher `prob_bad` should align with true bad images.  
3) Tune threshold for recall/precision trade-off, e.g., `--threshold 0.25` for more recall.

### Example run (current dataset)
- Command: `python src/evaluate.py --good_dir data/success_image_edit --bad_dir data/bad_images --save_csv outputs/results.csv`
- Auto F1-tuned threshold: `0.3136` (optimistic on same data)
- Metrics (BAD positive): Accuracy 0.8750; Precision 1.0000; Recall 0.5000; F1 0.6667; ROC-AUC 0.6042
- Confusion matrix [[TN FP],[FN TP]]: [[12, 0], [2, 2]]
- Top P(bad): `data/bad_images/image_2.png (0.528)`, `data/bad_images/image_5.png (0.314)`, followed by some GOOD images the model is less confident about.
- Per-image results saved to `outputs/results.csv`.

### Notes
- Zero-shot CLIP; no finetuning. Prompt edits or threshold tweaks can improve recall/precision trade-offs.
- For production, hold out data or use grouped splits to pick a stable threshold.