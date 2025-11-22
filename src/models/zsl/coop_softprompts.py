"""
CoOp-lite: Learn scalar weights over a small bank of text prompts per class
using a tiny labeled set (few-shot). This is NOT full continuous prompt tuning,
but works well as a quick, safe baseline without modifying CLIP internals.

Usage:
    tuner = CoOpLite(zsl_model=clip_model, class_prompts=prompt_dict, device="cuda")
    tuner.fit(train_paths, train_labels, epochs=5)
    pred, scores = tuner.classify(pil_image)
"""
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from collections import defaultdict

class CoOpLite(nn.Module):
    def __init__(self, zsl_model, class_prompts: Dict[str, List[str]], device="cuda"):
        super().__init__()
        self.zsl = zsl_model
        self.device = device
        # flatten prompts -> list with mapping to class
        self.labels, self.prompts = [], []
        for lab, plist in class_prompts.items():
            for p in plist:
                self.labels.append(lab); self.prompts.append(p)
        # text embeddings fixed
        with torch.no_grad():
            self.T = self.zsl.embed_texts(self.prompts).to(device)  # (M,D)
        # learn scalar weights per prompt (initialized equal)
        self.alpha = nn.Parameter(torch.ones(len(self.prompts), device=device))
        self.classes = sorted(set(self.labels))
        # map indices per class for fast aggregation
        self.class2idxs = defaultdict(list)
        for i, lab in enumerate(self.labels):
            self.class2idxs[lab].append(i)

    def forward(self, img_emb: torch.Tensor) -> Dict[str, float]:
        # img_emb: (1,D) normalized
        sims = (img_emb @ self.T.T).squeeze(0) * self.alpha  # (M,)
        # aggregate per class (weighted average)
        scores = {}
        for c in self.classes:
            idxs = self.class2idxs[c]
            s = sims[idxs]
            scores[c] = float(s.mean())
        return scores

    @torch.no_grad()
    def classify(self, pil: Image.Image) -> Tuple[str, Dict[str,float]]:
        x = self.zsl.preprocess(pil).to(self.device)
        z = self.zsl.embed_images(x)  # (1,D)
        scores = self.forward(z)
        pred = max(scores.items(), key=lambda kv: kv[1])[0]
        return pred, scores

    def fit(self, train_paths: List[str], train_labels: List[str], epochs=5, lr=1e-2, batch=16):
        opt = torch.optim.Adam([self.alpha], lr=lr)
        n = len(train_paths)
        order = list(range(n))
        for ep in range(1, epochs+1):
            import random; random.shuffle(order)
            total = 0.0
            for i in range(0, n, batch):
                idxs = order[i:i+batch]
                xs = [self.zsl.preprocess(Image.open(train_paths[j]).convert("RGB")) for j in idxs]
                x = torch.cat(xs, dim=0).to(self.device)
                z = self.zsl.embed_images(x)         # (B,D)
                sims = (z @ self.T.T) * self.alpha   # (B,M)

                # aggregate per class (mean over its prompts)
                logits = []
                targets = []
                for b, j in enumerate(idxs):
                    lab = train_labels[j]
                    cls_means = []
                    for c in self.classes:
                        idx = torch.tensor(self.class2idxs[c], device=self.device)
                        cls_means.append(sims[b, idx].mean())
                    logits.append(torch.stack(cls_means))
                    targets.append(self.classes.index(lab))
                logits = torch.stack(logits)                 # (B,C)
                y = torch.tensor(targets, device=self.device)
                loss = F.cross_entropy(logits, y)

                opt.zero_grad(); loss.backward(); opt.step()
                total += float(loss.item())
            print(f"[CoOpLite] epoch {ep}/{epochs} loss={total/max(1,(n//batch)):.4f}")
