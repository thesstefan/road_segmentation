import torch

# TODO(thesstefan): Replace this import with typing if upgrading Python to 3.11+
from typing_extensions import NotRequired, TypedDict


class SegmentationItem(TypedDict):
    image: torch.Tensor
    labels: NotRequired[torch.Tensor]
    image_filename: str


class SegmentationBatch(TypedDict):
    image: torch.Tensor
    labels: NotRequired[torch.Tensor]
    image_filename: list[str]
