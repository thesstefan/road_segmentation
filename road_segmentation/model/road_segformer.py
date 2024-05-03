import lightning.pytorch as pl
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics.collections import MetricCollection
from transformers import (  # type: ignore[import]
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

ID2LABEL = {0: "background", 1: "road"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def segformer_feature_extractor(
    image: torch.Tensor,
    labels: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = SegformerImageProcessor()(image, labels, return_tensors="pt")

    for values in inputs.values():  # type: ignore[reportUnkonwnMemberType]
        values.squeeze_()  # type: ignore[reportUnkonwnMemberType]

    return inputs["pixel_values"], inputs["labels"]


def upsample_logits(logits: torch.Tensor, size: torch.Size) -> torch.Tensor:
    upsampled_logits: torch.Tensor = nn.functional.interpolate(  # type: ignore[reportUnknownMemberType]
        logits,
        size=size,
        mode="bilinear",
        align_corners=False,
    )

    return upsampled_logits.argmax(dim=1)  # type: ignore[reportUnkonwnMemberType]


class RoadSegformer(pl.LightningModule):
    segformer_ckpt: str
    segformer: SegformerForSemanticSegmentation  # type: ignore[no-any-unimported]

    dataloaders: dict[str, DataLoader[dict[str, torch.Tensor]]]
    metrics: dict[str, MetricCollection]

    lr: float
    batch_size: int | None = None
    metrics_interval: int

    train_dataset_name: str

    def __init__(  # noqa: PLR0913
        self,
        segformer_ckpt: str,
        dataloaders: dict[str, DataLoader[dict[str, torch.Tensor]]],
        metrics: MetricCollection,
        lr: float,
        metrics_interval: int,
        train_dataset_name: str = "Unknown",
    ) -> None:
        super().__init__()

        train_dataloader = dataloaders.get("train")
        if train_dataloader:
            self.batch_size = train_dataloader.batch_size

        self.segformer_ckpt = segformer_ckpt
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            segformer_ckpt,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        self.dataloaders = dataloaders
        self.metrics = {
            "train": metrics.clone(prefix="train/"),
            "val": metrics.clone(prefix="val/"),
            "test": metrics.clone(prefix="test/"),
        }
        self.lr = lr
        self.train_data_name = train_dataset_name
        self.metrics_interval = metrics_interval

        self.save_hyperparameters(
            "lr",
            "train_dataset_name",
            "segformer_ckpt",
        )

    def _compute_loss_and_update_metrics(
        self,
        batch: dict[str, torch.Tensor],
        phase: str,
    ) -> torch.Tensor:
        images, labels = batch["image"], batch["labels"]

        (loss, logits) = self.segformer(
            pixel_values=images,
            labels=labels,
            return_dict=False,
        )

        predicted = upsample_logits(logits, labels.shape[-2:])

        self.metrics[phase].update(predicted, labels)

        return loss  # type: ignore[no-any-return]

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        phase: str,
    ) -> torch.Tensor:
        loss = self._compute_loss_and_update_metrics(batch, phase)
        self.log(f"{phase}/loss", loss)

        if phase == "test" or (
            batch_idx and batch_idx % self.metrics_interval == 0
        ):
            metric_results = self.metrics[phase].compute()
            self.log_dict(metric_results)

        return loss

    def training_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._step(batch, batch_idx, "train")

    def validation_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        self._step(batch, batch_idx, "val")

    def test_step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        self._step(batch, batch_idx, "test")

    def configure_optimizers(self) -> optim.Optimizer:
        return torch.optim.Adam(self.segformer.parameters(), lr=self.lr)

    def train_dataloader(self) -> DataLoader[dict[str, torch.Tensor]] | None:
        return self.dataloaders.get("train")

    def val_dataloader(self) -> DataLoader[dict[str, torch.Tensor]] | None:
        return self.dataloaders.get("val")

    def test_dataloader(self) -> DataLoader[dict[str, torch.Tensor]] | None:
        return self.dataloaders.get("test")
