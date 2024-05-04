# TODO(thesstefan): Use typing.Self instead when/if upgrading to Python 3.11
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path  # noqa: TCH003

import torch
from torch.utils.data import Dataset
from torchvision import io  # type: ignore[import]

from road_segmentation.dataset.segmentation_datapoint import SegmentationItem
from road_segmentation.utils.transforms import from_color_to_labels

COLOR_1D_TO_LABEL: dict[tuple[int, ...], int] = {
    (0,): 0,
    (255,): 1,
}

ImageAndMaskTransform = Callable[
    [torch.Tensor, torch.Tensor | None],
    tuple[torch.Tensor, torch.Tensor | None],
]


class ETHZDataset(Dataset[SegmentationItem]):
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
    ) -> ETHZDataset:
        if not root.exists():
            error_message = f"ETHZ CIL Dataset not found at {root!s}"
            raise FileNotFoundError(error_message)

        image_paths = [
            {
                "image_path": image_path,
                "mask_path": root / "groundtruth" / image_path.name,
            }
            for image_path in (root / "images").iterdir()
        ]

        return cls(image_paths, transform=transform)

    @classmethod
    def test_dataset(
        cls,
        root: Path,
        transform: ImageAndMaskTransform | None = None,
    ) -> ETHZDataset:
        if not root.exists():
            error_message = f"ETHZ CIL Dataset not found at {root!s}"
            raise FileNotFoundError(error_message)

        image_paths = [
            {
                "image_path": image_path,
            }
            for image_path in (root / "images").iterdir()
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

        mask = None
        mask_path = self.image_paths[idx].get("mask_path")
        if mask_path:
            mask = io.read_image(str(mask_path), mode=io.ImageReadMode.GRAY)
            mask = from_color_to_labels(mask, COLOR_1D_TO_LABEL)
            mask = torch.squeeze(mask)

        if self.transform:
            image, mask = self.transform(image, mask)

        if mask is None:
            return SegmentationItem(
                image=image,
                image_filename=self.image_paths[idx]["image_path"].name,
            )

        return SegmentationItem(
            image=image,
            image_filename=self.image_paths[idx]["image_path"].name,
            labels=mask,
        )
