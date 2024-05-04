import lightning.pytorch as pl
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics.collections import MetricCollection
from transformers import (  # type: ignore[import]
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)

from road_segmentation.dataset.segmentation_datapoint import SegmentationItem

ID2LABEL = {0: "background", 1: "road"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def segformer_feature_extractor(
    image: torch.Tensor,
    labels: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    inputs = SegformerImageProcessor()(image, labels, return_tensors="pt")

    for values in inputs.values():  # type: ignore[reportUnkonwnMemberType]
        values.squeeze_()  # type: ignore[reportUnkonwnMemberType]

    # TODO(thesstefan): Fix interface so that returning empty is not necessary
    if labels is None:
        return inputs["pixel_values"], None

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

    dataloaders: dict[str, DataLoader[SegmentationItem]]
    metrics: dict[str, MetricCollection] | None

    lr: float
    batch_size: int
    metrics_interval: int

    train_dataset_name: str

    def __init__(  # noqa: PLR0913
        self,
        segformer_ckpt: str,
        lr: float,
        batch_size: int,
        dataloaders: dict[str, DataLoader[SegmentationItem]] | None = None,
        metrics: MetricCollection | None = None,
        metrics_interval: int = 20,
        train_dataset_name: str = "Unknown",
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.segformer_ckpt = segformer_ckpt
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(  # type: ignore[reportUnknwonMemberType]
            segformer_ckpt,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )
        self.dataloaders = dataloaders or {}
        self.metrics = (
            {
                "train": metrics.clone(prefix="train/"),
                "val": metrics.clone(prefix="val/"),
                "test": metrics.clone(prefix="test/"),
            }
            if metrics
            else None
        )
        self.lr = lr
        self.train_data_name = train_dataset_name
        self.metrics_interval = metrics_interval

        self.save_hyperparameters(
            "batch_size",
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

        if self.metrics:
            self.metrics[phase].update(predicted, labels)

        return loss  # type: ignore[no-any-return]

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        phase: str,
    ) -> torch.Tensor:
        loss = self._compute_loss_and_update_metrics(batch, phase)
        self.log(f"{phase}/loss", value=loss, batch_size=self.batch_size)  # type: ignore[reportUnkonwnMemberType]

        if self.metrics and (
            phase == "test"
            or (batch_idx and batch_idx % self.metrics_interval == 0)
        ):
            metric_results = self.metrics[phase].compute()
            self.log_dict(metric_results, batch_size=self.batch_size)  # type: ignore[reportUnkonwnMemberType]

        return loss

    # TODO(thesstefan): Fix type errors using ParamSpec
    def training_step(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        return self._step(batch, batch_idx, "train")

    def validation_step(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        self._step(batch, batch_idx, "val")

    def test_step(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> None:
        self._step(batch, batch_idx, "test")

    def predict_step(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        images = batch["image"]
        logits = self.segformer(pixel_values=images, return_dict=False)[0]

        return upsample_logits(logits, images.shape[-2:])

    def on_train_start(self) -> None:
        if self.logger:
            self.logger.log_hyperparams(self.hparams)  # type: ignore  # noqa: PGH003

    def configure_optimizers(self) -> optim.Optimizer:
        return torch.optim.Adam(self.segformer.parameters(), lr=self.lr)

    def train_dataloader(self) -> DataLoader[SegmentationItem] | None:
        return self.dataloaders.get("train")

    def val_dataloader(self) -> DataLoader[SegmentationItem] | None:
        return self.dataloaders.get("val")

    def test_dataloader(self) -> DataLoader[SegmentationItem] | None:
        return self.dataloaders.get("test")
