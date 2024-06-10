# TODO(thesstefan): Use typing.Self instead when/if upgrading to Python 3.11
from __future__ import annotations
from pathlib import Path  # noqa: TCH003

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

from road_segmentation.dataset.segmentation_datapoint import SegmentationItem
from road_segmentation.utils.transforms import ImageAndMaskTransform
from PIL import Image


class ChesaDataset(Dataset[SegmentationItem]):
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
    ) -> ChesaDataset:
        if not root.exists():
            error_message = f"Chesa CIL Dataset not found at {root!s}"
            raise FileNotFoundError(error_message)

        image_paths = []

        for image_file in root.glob("*_image.tif"):
            # Construct the mask filename by replacing '_sat.jpg' with '_mask.png'
            mask_filename = image_file.name.replace("_image.tif", "_mask.tif")
            mask_path = root / mask_filename
            
            # Check if the corresponding mask file exists
            if mask_path.exists():
                # Add the image and mask paths as a dictionary to the list
                image_paths.append({
                    "image_path": image_file,
                    "mask_path": mask_path,
                })
            else:
                print(f"Warning: Mask not found for image {image_file}")

        return cls(image_paths, transform=transform)


    def __len__(self) -> int:
        return len(self.image_paths)

    def get_image_path(self, idx: int) -> Path:
        return self.image_paths[idx]["image_path"]

    def __getitem__(self, idx: int) -> SegmentationItem:
        image = Image.open(self.image_paths[idx]["image_path"])
        image = image.convert('RGB')
        image = np.array(image)
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)


        mask = Image.open(self.image_paths[idx]["mask_path"])
        mask = np.array(mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        mask = mask > 8
        mask = torch.from_numpy(mask)
        mask = mask.int()
        
        if self.transform:
            image, mask = self.transform(image, mask)

        return SegmentationItem(
            image=image,
            image_filename=self.image_paths[idx]["image_path"].name,
            labels=mask,
        )
