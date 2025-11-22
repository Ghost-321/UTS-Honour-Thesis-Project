from typing import Dict, List, Tuple
import torch, torch.nn.functional as F
import open_clip
from PIL import Image
from base import ZSLModel

class CLIPZeroShot(ZSLModel):
    name = "clip"

    def __init__(self, device="cuda"):
        self.device = device
        # Store model + preprocessing safely
        self.model, _, preprocess_fn = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        self._preprocess_fn = preprocess_fn  # avoid overwriting bug
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        self.model = self.model.to(device).eval()

    def preprocess(self, pil: Image.Image) -> torch.Tensor:
        """Convert PIL → tensor batch"""
        return self._preprocess_fn(pil).unsqueeze(0).to(self.device)

    def embed_images(self, batch: torch.Tensor) -> torch.Tensor:
        """Encode a batch of images"""
        with torch.no_grad():
            z = self.model.encode_image(batch.to(self.device))
            return F.normalize(z, dim=-1)

    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode text prompts"""
        tok = self.tokenizer(texts)
        with torch.no_grad():
            z = self.model.encode_text(tok.to(self.device))
            return F.normalize(z, dim=-1)

    @torch.no_grad()
    def classify_pil(self, pil: Image.Image, class_texts: Dict[str, List[str]]) -> Tuple[str, Dict[str, float]]:
        """
        Classify a single image against prompt sets.
        Aggregates multiple prompts per class via average similarity.
        """
        labs, txts = [], []
        for lab, alts in class_texts.items():
            for t in alts:
                labs.append(lab)
                txts.append(t)

        # Embed text and image
        T = self.embed_texts(txts)               # (M, D)
        x = self.preprocess(pil)                 # (1, 3, H, W)
        z = self.embed_images(x)                 # (1, D)

        # Similarities
        sims = (z @ T.T).squeeze(0)              # (M,)
        from collections import defaultdict
        agg = defaultdict(list)
        for s, lab in zip(sims.tolist(), labs):
            agg[lab].append(s)

        scores = {k: float(sum(v) / len(v)) for k, v in agg.items()}
        pred = max(scores.items(), key=lambda kv: kv[1])[0]
        return pred, scores
