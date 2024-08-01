# TODO(thesstefan): Use typing.Self instead when/if upgrading to Python 3.11
from __future__ import annotations

from pathlib import Path  # noqa: TCH003

import torch
from torch.utils.data import Dataset
from torchvision import io  # type: ignore[import]
import numpy as np

from road_segmentation.dataset.segmentation_datapoint import SegmentationItem
from road_segmentation.utils.transforms import ImageAndMaskTransform




class MassDataset(Dataset[SegmentationItem]):
    image_paths: list[dict[str, Path]]
    transform: ImageAndMaskTransform | None

    def __init__(
        self,
        image_paths: list[dict[str, Path]],
        transform: ImageAndMaskTransform | None = None,
    ) -> None:
        self.image_paths = image_paths
        self.transform = transform

    # TODO(thesstefan): Get rid of duplication?
    @classmethod
    def train_dataset(
        cls,
        root: Path,
        transform: ImageAndMaskTransform | None = None,
    ) -> MassDataset:
        if not root.exists():
            error_message = f"Massachusetts Dataset not found at {root!s}"
            raise FileNotFoundError(error_message)

        image_paths = [
            {
                "image_path": image_path,
                "mask_path": root / "labels" / image_path.name,
            }
            for image_path in (root / "images").iterdir()
            if image_path.suffix == ".png"
        ]

        return cls(image_paths, transform=transform)


    def __len__(self) -> int:
        return len(self.image_paths)

    def get_image_path(self, idx: int) -> Path:
        return self.image_paths[idx]["image_path"]

    def __getitem__(self, idx: int) -> SegmentationItem:
        image = io.read_image(
            str(self.image_paths[idx]["image_path"]),
            mode=io.ImageReadMode.RGB,
        )
        image = torch.squeeze(image)

        mask = io.read_image(
            str(self.image_paths[idx]["mask_path"]),
            mode=io.ImageReadMode.GRAY,
        )
        mask = mask > 200
        mask = mask.int()
        
        if self.transform:
            image, mask = self.transform(image, mask)

        return SegmentationItem(
            image=image,
            image_filename=self.image_paths[idx]["image_path"].name,
            labels=mask,
        )
