import torch
from collections.abc import Callable
import cv2

ImageAndMaskTransform = Callable[
    [torch.Tensor, torch.Tensor | None],
    tuple[torch.Tensor, torch.Tensor | None],
]


def from_color_to_labels(
    values: torch.Tensor,
    color_to_label_map: dict[tuple[int, ...], int],
) -> torch.Tensor:
    for color, label in color_to_label_map.items():
        if len(color) > 1:
            indices = torch.where(
                torch.all(values == torch.tensor(color), dim=-1),
            )
        else:
            indices = torch.where(values == torch.tensor(color[0]))

        values[indices] = torch.full_like(values[indices], label)

    return values


def clahe_transform(
    image: torch.Tensor, mask: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    img = image.numpy().transpose((1, 2, 0))  # HWC format (for OpenCV)

    # CLAHE application
    if img.ndim == 3:  # RGB Image
        channels = cv2.split(img)
        clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))

        eq_channels = [
            clahe.apply(channel) 
            for channel in channels
        ]
        img = cv2.merge(eq_channels)
    else:  # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

    return torch.from_numpy(img.transpose((2, 0, 1))).float(), mask
