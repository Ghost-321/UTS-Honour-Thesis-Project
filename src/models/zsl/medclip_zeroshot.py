from typing import Dict, List, Tuple
import torch, torch.nn.functional as F
from PIL import Image
from .base import ZSLModel


class MedCLIPZeroShot(ZSLModel):
    """
    MedCLIPZeroShot
    ----------------
    Drop-in replacement for a medical-domain CLIP-style zero-shot classifier.

    - Wraps the Hugging Face MedCLIP implementation if available.
    - Falls back gracefully to default CLIP preprocessing if not.
    - Fully compatible with your ZSL evaluation pipeline.
    """
    name = "medclip"

    def __init__(self, device: str = "cuda"):
        self.device = device
        try:
            try:
                # ---- Try new import paths first ----
                from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPTextModel
                self.vision = MedCLIPVisionModelViT().to(device)
                self.text   = MedCLIPTextModel().to(device)
                self.model  = MedCLIPModel(self.vision, self.text).to(device)
                self._preprocess_fn = self.vision.preprocess
                self._tokenize_fn   = self.text.tokenize
            except ImportError:
                # ---- Fallback for older medclip installs ----
                from medclip import MedCLIPModel
                from medclip.vision_encoder import MedCLIPVisionModelViT
                from medclip.text_encoder import MedCLIPTextModel
                vision = MedCLIPVisionModelViT().to(device)
                text   = MedCLIPTextModel().to(device)
                self.model = MedCLIPModel(vision, text).to(device)
                self._preprocess_fn = vision.preprocess
                self._tokenize_fn   = text.tokenize
            self.model.eval()
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialise MedCLIP. "
                f"Please reinstall from GitHub: "
                f"pip install git+https://github.com/UCSD-AI4H/MedCLIP.git"
            ) from e

    # --------------------------------------------------------------
    # Abstract base method compatibility
    # --------------------------------------------------------------
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """
        Compatibility shim to satisfy the abstract base class.
        Returns a normalized (1,3,H,W) tensor.
        """
        if callable(getattr(self, "_preprocess_fn", None)):
            return self._preprocess_fn(image)
        else:
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            return transform(image)

    # --------------------------------------------------------------
    # Core embeddings
    # --------------------------------------------------------------
    def embed_images(self, batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.model.encode_image(batch.to(self.device))
            return F.normalize(z, dim=-1)

    def embed_texts(self, texts: List[str]) -> torch.Tensor:
        if callable(getattr(self, "_tokenize_fn", None)):
            tok = self._tokenize_fn(texts).to(self.device)
        else:
            # fallback to simple tokenizer if not available
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
                tok = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            except Exception:
                raise RuntimeError("No tokenizer found for MedCLIP — please install `transformers`.")
        with torch.no_grad():
            z = self.model.encode_text(tok)
            return F.normalize(z, dim=-1)

    # --------------------------------------------------------------
    # Inference: classify a single PIL image
    # --------------------------------------------------------------
    @torch.no_grad()
    def classify_pil(self, pil: Image.Image, class_texts: Dict[str, List[str]]) -> Tuple[str, Dict[str, float]]:
        # Flatten prompts and labels
        labs, txts = [], []
        for k, alts in class_texts.items():
            for t in alts:
                labs.append(k)
                txts.append(t)

        # Compute text + image embeddings
        T = self.embed_texts(txts)
        x = self.preprocess(pil).unsqueeze(0).to(self.device)
        z = self.embed_images(x)

        # Compute cosine similarity scores
        sims = (z @ T.T).squeeze(0).tolist()

        from collections import defaultdict
        agg = defaultdict(list)
        for s, lab in zip(sims, labs):
            agg[lab].append(s)

        scores = {k: float(sum(v) / len(v)) for k, v in agg.items()}
        pred = max(scores.items(), key=lambda kv: kv[1])[0]
        return pred, scores
