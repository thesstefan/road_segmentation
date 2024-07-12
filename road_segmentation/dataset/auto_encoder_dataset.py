import torch
from torch.utils.data import Dataset
from pathlib import Path  # noqa: TCH003
from torchvision import io  # type: ignore[import]

from road_segmentation.utils.transforms import ImageAndMaskTransform, from_color_to_labels

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

    @classmethod
    def train_dataset(
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
                "mask_path": root / "groundtruth" / segment_path.name,
            }
            for segment_path in (root / "segments").iterdir()
        ]

        return cls(segments_paths, transform)

    def __len__(self) -> int:
        return len(self.segments_paths)

    def get_image_path(self, idx: int) -> Path:
        return self.segments_paths[idx]["segment_path"]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = io.read_image(
            str(self.segments_paths[idx]["segment_path"]),
            mode=io.ImageReadMode.GRAY,
        )
        image = torch.squeeze(image)

        mask = io.read_image(
            str(self.segments_paths[idx]["mask_path"]),
            mode=io.ImageReadMode.GRAY,
        )
        mask = torch.squeeze(mask)
        mask = from_color_to_labels(mask, COLOR_1D_TO_LABEL)
        mask = mask.float()
        if self.transform:
            image, mask = self.transform(image, mask)
        return image, mask
