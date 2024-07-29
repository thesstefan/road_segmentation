from collections.abc import Sequence
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from PIL import Image  # noqa: TCH002
from torchvision.transforms.functional import (  # type: ignore[import]
    to_pil_image,  # type: ignore[reportUnkonwnVariableType]
)

from road_segmentation.dataset.segmentation_datapoint import SegmentationBatch


class OnBatchImageOutputWriter(BasePredictionWriter):
    def __init__(
        self,
        output_dir: Path,
    ) -> None:
        super().__init__("batch")

        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_on_batch_end(  # noqa: PLR0913
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        prediction: torch.Tensor,
        batch_indices: Sequence[int] | None,
        batch: SegmentationBatch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        del trainer
        del pl_module
        del batch_indices
        del batch_idx
        del dataloader_idx

        prediction = (prediction.squeeze(0) * 255).to(torch.uint8)
        for idx, pred in enumerate(prediction):
            image: Image.Image = to_pil_image(pred)  # type: ignore[reportUnkonwnMemberType]
            image.save(self.output_dir / batch["image_filename"][idx])  # type: ignore[reportUnkonwnMemberType]
