## Bad-vs-Good Image Classifier Report (Template)

### Summary
- Goal: Zero-shot BAD (1) vs GOOD (0) using CLIP ViT-B-32 (`openai`).
- Dataset: BAD=`data/bad_images` (# images: ___), GOOD=`data/success_image_edit` (# images: ___).
- Threshold used: ___ (note if auto-tuned on eval set; optimistic).

### Model & Prompts
- Model: `open-clip-torch` ViT-B-32, pretrained `openai`.
- Bad prompts (edit if changed):
  - "a bad photo, distorted subject, incoherent composition, low quality"
  - "obstructed view, messed up perspective, chaotic scene, blurry"
  - "image with artifacts, corrupted details, wrong proportions, unusable"
  - "failed edit, incoherent result, warped shapes, messy output"
- Good prompts (edit if changed):
  - "a clear, sharp, coherent photo with good composition"
  - "well-framed, clean image, correct proportions, visually pleasing"
  - "high quality photo, coherent subject, accurate details, readable"
  - "successful edit, clean output, consistent shapes, natural look"

### Evaluation (copy from `python src/evaluate.py ...`)
- Command: `python src/evaluate.py --good_dir data/success_image_edit --bad_dir data/bad_images --save_csv outputs/results.csv --threshold ___`
- Metrics (BAD positive):
  - Accuracy: ___
  - Precision: ___
  - Recall: ___
  - F1: ___
  - ROC-AUC: ___ (if available)
  - Confusion matrix [[TN FP],[FN TP]]: [[__, __], [__, __]]
- Top 10 most-likely BAD images (path, P(bad)):
  1. ___ (___)
  2. ___ (___)
  3. ___ (___)
  4. ___ (___)
  5. ___ (___)
  6. ___ (___)
  7. ___ (___)
  8. ___ (___)
  9. ___ (___)
  10. ___ (___)
- Per-image CSV: `outputs/results.csv`

### Inference checks
- Sample bad: `python src/predict.py --input data/bad_images/<file> --threshold ___`
- Sample good: `python src/predict.py --input data/success_image_edit/<file> --threshold ___`
- Observations: ___

### Risks / Next steps
- Threshold tuned on same data is optimistic; use held-out or grouped splits for production.
- Consider prompt refinement or ensembling for higher recall without losing precision.
- If many GOODs rank high P(bad), raise threshold or refine prompts. If many BADs missed, lower threshold or adjust prompts toward failure cues.
