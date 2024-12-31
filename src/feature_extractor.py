from collections import namedtuple

import torch
from torch import nn
from torchvision.models import vgg16


class VGGFeatureExtractor(nn.Module):
    def __init__(
        self,
        requires_grad: bool = False,
        pretrained_weights: str = "DEFAULT",
    ):
        super().__init__()
        self.slices = nn.ModuleList([nn.Sequential() for _ in range(4)])
        self._initialize_slices(pretrained_weights)
        self.features = namedtuple("Outputs", [f"layer{i}" for i in range(4)])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def _initialize_slices(self, pretrained_weights: str = "DEFAULT") -> None:
        features = vgg16(weights=pretrained_weights).features
        slice_indices = [(0, 4), (4, 9), (9, 16), (16, 23)]

        for slice_idx, (start, end) in enumerate(slice_indices):
            for i in range(start, end):
                self.slices[slice_idx].add_module(str(i), features[i])

    def forward(self, x: torch.Tensor) -> namedtuple:
        outputs = []
        for slice_model in self.slices:
            x = slice_model(x)
            outputs.append(x)
        return self.features(*outputs)
