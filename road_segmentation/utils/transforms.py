import torch
from collections.abc import Callable

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
