"""
RETCLIPZeroShot
---------------
A drop-in zero-shot classifier based on OpenCLIP, optionally
fine-tuned with retina or ophthalmology-focused weights.

Design:
- Uses OpenCLIP ViT-B/32 by default.
- Accepts an optional ckpt_path to load a RET-CLIP checkpoint.
- Behaves exactly like CLIPZeroShot / MedCLIPZeroShot.
"""

from typing import Dict, List, Tuple, Optional
from PIL import Image
import torch
import torch.nn.functional as F
import open_clip

from base import ZSLModel


def _load_state_dict_flex(model: torch.nn.Module, state: dict, strict: bool = False):
    """Load checkpoint dict, handling typical containers and prefixes."""
    for key in ("state_dict", "model", "params_ema", "params"):
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break
    fixed = {}
    for k, v in state.items():
        fixed[k[7:]] = v if k.startswith("module.") else v
    missing, unexpected = model.load_state_dict(fixed, strict=strict)
    if missing or unexpected:
        print(f"[retclip] load_state_dict: missing={list(missing)}, unexpected={list(unexpected)}")


class RETCLIPZeroShot(ZSLModel):
    name = "retclip"

    def __init__(
        self,
        device: str = "cuda",
        backbone: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        ckpt_path: Optional[str] = None,
        strict_load: bool = False,
    ):
        self.device = device
        self.backbone = backbone
        self.pretrained = pretrained

        # Instantiate OpenCLIP model + preprocess
        self.model, _, preprocess_fn = open_clip.create_model_and_transforms(backbone, pretrained=pretrained)
        self.model = self.model.to(device).eval()
        self._preprocess_fn = preprocess_fn
        self._tokenizer = open_clip.get_tokenizer(backbone)

        # Optionally load retina-tuned checkpoint
        if ckpt_path:
            try:
                sd = torch.load(ckpt_path, map_location=device)
                _load_state_dict_flex(self.model, sd, strict=strict_load)
                print(f"[retclip] Loaded checkpoint: {ckpt_path}")
            except Exception as e:
                raise RuntimeError(f"Failed to load RET-CLIP checkpoint: {ckpt_path}") from e

    # --- Required abstract methods ---
    def preprocess(self, pil: Image.Image) -> torch.Tensor:
        """Standard CLIP preprocessing, returns (1,3,H,W) normalized tensor."""
        return self._preprocess_fn(pil).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def embed_images(self, batch: torch.Tensor) -> torch.Tensor:
        z = self.model.encode_image(batch.to(self.device))
        return F.normalize(z, dim=-1)

    @torch.no_grad()
    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        tokens = self._tokenizer(texts).to(self.device)
        z = self.model.encode_text(tokens)
        return F.normalize(z, dim=-1)

    @torch.no_grad()
    def classify_pil(self, pil: Image.Image, class_texts: Dict[str, List[str]]) -> Tuple[str, Dict[str, float]]:
        # Build text embeddings
        labs, prompts = [], []
        for lab, alts in class_texts.items():
            for p in alts:
                labs.append(lab)
                prompts.append(p)

        T = self.embed_texts(prompts)
        X = self.preprocess(pil)
        Z = self.embed_images(X)
        sims = (Z @ T.T).squeeze(0).tolist()

        from collections import defaultdict
        agg = defaultdict(list)
        for s, lab in zip(sims, labs):
            agg[lab].append(s)
        scores = {k: float(sum(v) / len(v)) for k, v in agg.items()}
        pred = max(scores.items(), key=lambda kv: kv[1])[0]
        return pred, scores
