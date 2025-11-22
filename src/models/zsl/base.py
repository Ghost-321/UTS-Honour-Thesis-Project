from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from PIL import Image
import torch

class ZSLModel(ABC):
    name: str = "zsl_base"

    @abstractmethod
    def preprocess(self, pil: Image.Image) -> torch.Tensor: ...
    @abstractmethod
    def embed_images(self, batch: torch.Tensor) -> torch.Tensor: ...
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> torch.Tensor: ...
    @abstractmethod
    def classify_pil(self, pil: Image.Image, class_texts: Dict[str, List[str]]) -> Tuple[str, Dict[str,float]]: ...
