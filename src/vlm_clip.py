import os
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import open_clip
import torch
from PIL import Image
from torch import nn


def _default_bad_prompts() -> List[str]:
    return [
        "a bad photo, distorted subject, incoherent composition, low quality",
        "obstructed view, messed up perspective, chaotic scene, blurry",
        "image with artifacts, corrupted details, wrong proportions, unusable",
        "failed edit, incoherent result, warped shapes, messy output",
    ]


def _default_good_prompts() -> List[str]:
    return [
        "a clear, sharp, coherent photo with good composition",
        "well-framed, clean image, correct proportions, visually pleasing",
        "high quality photo, coherent subject, accurate details, readable",
        "successful edit, clean output, consistent shapes, natural look",
    ]


def list_images(input_path: Path, extensions: Optional[Sequence[str]] = None) -> List[Path]:
    """List image files recursively for common extensions."""
    if extensions is None:
        extensions = [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff"]

    input_path = Path(input_path)
    files: List[Path] = []

    def _is_image(p: Path) -> bool:
        return p.suffix.lower() in extensions

    if input_path.is_file() and _is_image(input_path):
        return [input_path]

    if input_path.is_dir():
        for root, _, filenames in os.walk(input_path):
            for name in filenames:
                candidate = Path(root) / name
                if _is_image(candidate):
                    files.append(candidate)

    files.sort()
    return files


class CLIPBadGoodClassifier:
    def __init__(
        self,
        device: Optional[str] = None,
        bad_prompts: Optional[Iterable[str]] = None,
        good_prompts: Optional[Iterable[str]] = None,
    ) -> None:
        self.device = device or self._choose_device()
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")

        bad_list = list(bad_prompts) if bad_prompts is not None else _default_bad_prompts()
        good_list = list(good_prompts) if good_prompts is not None else _default_good_prompts()
        if not bad_list or not good_list:
            raise ValueError("Both bad and good prompt lists must be non-empty.")

        with torch.no_grad():
            bad_feat = self._encode_texts(bad_list)
            good_feat = self._encode_texts(good_list)

        self.class_text_features = torch.stack([good_feat, bad_feat], dim=0)  # order: good, bad
        self.logit_scale = self.model.logit_scale.exp().detach()

    def _choose_device(self) -> str:
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _encode_texts(self, prompts: Sequence[str]) -> torch.Tensor:
        tokens = self.tokenizer(prompts).to(self.device)
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        pooled = text_features.mean(dim=0)
        return pooled / pooled.norm()

    def _encode_image(self, image_path: Path) -> torch.Tensor:
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            img_feat = self.model.encode_image(image_tensor)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        return img_feat

    def predict_proba_bad(self, image_path: Path) -> float:
        img_feat = self._encode_image(Path(image_path))
        logits = self.logit_scale * img_feat @ self.class_text_features.t()
        probs = nn.functional.softmax(logits, dim=-1)
        return float(probs[0, 1].item())  # index 1 corresponds to BAD

    def predict_label(self, image_path: Path, threshold: float = 0.5) -> int:
        prob_bad = self.predict_proba_bad(image_path)
        return int(prob_bad >= threshold)
