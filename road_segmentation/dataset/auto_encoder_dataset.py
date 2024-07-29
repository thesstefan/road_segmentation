import torch
from torch.utils.data import Dataset
from pathlib import Path  # noqa: TCH003
from torchvision import io  # type: ignore[import]

from road_segmentation.utils.transforms import (
    ImageAndMaskTransform,
    from_color_to_labels,
)
from road_segmentation.dataset.segmentation_datapoint import SegmentationItem

COLOR_1D_TO_LABEL: dict[tuple[int, ...], int] = {
    (0,): 0,
    (255,): 1,
}


class AEDataset(Dataset):

    segments_paths: list[dict[str, Path]]
    transform: ImageAndMaskTransform | None

    def __init__(
        self,
        segments_paths: list[dict[str, Path]],
        transform: ImageAndMaskTransform | None = None,
    ) -> None:
        self.segments_paths = segments_paths
        self.transform = transform
        self.pretraining = False
        
    @classmethod
    def train_dataset(
        cls,
        root: Path,
        transform: ImageAndMaskTransform | None = None,
        folders: [Path] = None,
    ) -> Dataset:
        if not root.exists():
            error_message = f"AEDataset not found at {root!s}"
            raise FileNotFoundError(error_message)
        if folders is None:
            segments_paths = [
                {
                    "segment_path": segment_path,
                    "mask_path": root / "groundtruth" / segment_path.name,
                }
                for segment_path in (root / "segments").iterdir() if str(segment_path).endswith(".png")
            ]
        else:
            segments_paths = []
            for folder in folders:
                segments_paths.extend(
                    [
                        {
                            "segment_path": segment_path,
                            "mask_path": root / folder / "groundtruth" / segment_path.name,
                        }
                        for segment_path in (root / folder / "segments").iterdir() if str(segment_path).endswith(".png")
                    ]
                )

        return cls(segments_paths, transform)

    @classmethod
    def test_dataset(
        cls,
        root: Path,
        transform: ImageAndMaskTransform | None = None,
    ) -> Dataset:
        if not root.exists():
            error_message = f"AEDataset not found at {root!s}"
            raise FileNotFoundError(error_message)

        segments_paths = [
            {
                "segment_path": segment_path,
            }
            for segment_path in (root / "segments").iterdir() if str(segment_path).endswith(".png")
        ]
        
        return cls(segments_paths, transform)

    def __len__(self) -> int:
        return len(self.segments_paths)

    def get_image_path(self, idx: int) -> Path:
        return self.segments_paths[idx]["segment_path"]

    def set_pretaining(self, pretraining: bool) -> None:
        self.pretraining = pretraining

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.pretraining:
            image = io.read_image(
                str(self.segments_paths[idx]["mask_path"]),
                mode=io.ImageReadMode.GRAY,
            )
            image = torch.squeeze(image)
            pos_indices = torch.where(image > 127)
            neg_indices = torch.where(image <= 127)
            image[pos_indices] = torch.full_like(image[pos_indices], 1.0)
            image[neg_indices] = torch.full_like(image[neg_indices], 0.0)

            image = image.float()

        else:
            image = io.read_image(
                str(self.segments_paths[idx]["segment_path"]),
                mode=io.ImageReadMode.GRAY,
            )
            image = torch.squeeze(image)
        
        mask = None
        mask_path = self.segments_paths[idx].get("mask_path")
        
            
        image = torch.squeeze(image)
            
        if mask_path:
            mask = io.read_image(
                str(self.segments_paths[idx]["mask_path"]),
                mode=io.ImageReadMode.GRAY,
            )
            mask = torch.squeeze(mask)
            pos_indices = torch.where(mask > 127)
            neg_indices = torch.where(mask <= 127)
            mask[pos_indices] = torch.full_like(mask[pos_indices], 1.0)
            mask[neg_indices] = torch.full_like(mask[neg_indices], 0.0)

            mask = mask.float()

        if self.transform:
            image, mask = self.transform(image, mask)

        if mask is None:
            return SegmentationItem(
                image=image,
                image_filename=self.segments_paths[idx]["segment_path"].name,
            )

        return SegmentationItem(
            image=image,
            image_filename=self.segments_paths[idx]["segment_path"].name,
            labels=mask,
        )

class PretrainingDataset(Dataset):
    segments_paths: list[dict[str, Path]]
    transform: ImageAndMaskTransform | None

    def __init__(
        self,
        segments_paths: list[dict[str, Path]],
        transform: ImageAndMaskTransform | None = None,
    ) -> None:
        self.segments_paths = segments_paths
        self.transform = transform

    @classmethod
    def train_dataset(
        cls,
        root: Path,
        transform: ImageAndMaskTransform | None = None,
        folders: [Path] = None,
    ) -> Dataset:
        if not root.exists():
            error_message = f"AEDataset not found at {root!s}"
            raise FileNotFoundError(error_message)
        if folders is None:
            segments_paths = [
                {
                    "segment_path": segment_path,
                    "mask_path": segment_path,
                }
                for segment_path in (root / "segments").iterdir()
            ]
        else:
            segments_paths = []
            for folder in folders:
                segments_paths.extend(
                    [
                        {
                            "segment_path": segment_path,
                            "mask_path": segment_path,
                        }
                        for segment_path in (root / folder / "segments").iterdir()
                    ]
                )

        return cls(segments_paths, transform)

    @classmethod
    def test_dataset(
        cls,
        root: Path,
        transform: ImageAndMaskTransform | None = None,
    ) -> Dataset:
        if not root.exists():
            error_message = f"AEDataset not found at {root!s}"
            raise FileNotFoundError(error_message)

        segments_paths = [
            {
                "segment_path": segment_path,
            }
            for segment_path in (root / "segments").iterdir()
        ]

        return cls(segments_paths, transform)

    def __len__(self) -> int:
        return len(self.segments_paths)

    def get_image_path(self, idx: int) -> Path:
        return self.segments_paths[idx]["segment_path"]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor | None]:
        image = io.read_image(
            str(self.segments_paths[idx]["segment_path"]),
            mode=io.ImageReadMode.GRAY,
        )
        image = torch.squeeze(image)
        mask = None
        mask_path = self.segments_paths[idx].get("mask_path")

        if mask_path:
            mask = io.read_image(
                str(self.segments_paths[idx]["mask_path"]),
                mode=io.ImageReadMode.GRAY,
            )
            mask = torch.squeeze(mask)
            pos_indices = torch.where(mask > 127)
            neg_indices = torch.where(mask <= 127)
            mask[pos_indices] = torch.full_like(mask[pos_indices], 1.0)
            mask[neg_indices] = torch.full_like(mask[neg_indices], 0.0)

            mask = mask.float()

        if self.transform:
            image, mask = self.transform(image, mask)

        if mask is None:
            return SegmentationItem(
                image=image,
                image_filename=self.segments_paths[idx]["segment_path"].name,
            )

        return SegmentationItem(
            image=image,
            image_filename=self.segments_paths[idx]["segment_path"].name,
            labels=mask,
        )