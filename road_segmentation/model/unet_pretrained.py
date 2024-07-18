import lightning.pytorch as pl
import torch
import cv2
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics.collections import MetricCollection
from torchvision.transforms.functional import resize, normalize
from torchvision.transforms import InterpolationMode, ToTensor

from road_segmentation.dataset.segmentation_datapoint import SegmentationItem
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

def upsample_logits(logits: torch.Tensor, size: torch.Size) -> torch.Tensor:
    upsampled_logits: torch.Tensor = nn.functional.interpolate(  # type: ignore[reportUnknownMemberType]
        logits,
        size=size,
        mode="bilinear",
        align_corners=False,
    )

    # type: ignore[reportUnkonwnMemberType]
    return upsampled_logits.argmax(dim=1)


def unet_transforms(
    image: torch.Tensor,
    labels: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:

    # image = image.float()
    transforms = get_preprocessing_fn('efficientnet-b4', pretrained='imagenet')
    image = transforms(image.permute(1, 2, 0))
    image = image.permute(2, 0, 1)

    # Resize the image and labels to 384x380
    image = resize(image, size= (384, 384), interpolation=InterpolationMode.BICUBIC)
    if labels is not None:
        # labels = labels.float()
        labels = labels.unsqueeze(0)  # Add channel dimension if missing
        labels = resize(
            labels, size=(384, 384), interpolation=InterpolationMode.NEAREST
        )

    return image.double(), labels.double()


class UNetTrained(pl.LightningModule):
    unet: nn.Module  # type: ignore[no-any-unimported]

    dataloaders: dict[str, DataLoader[SegmentationItem]]
    metrics: dict[str, MetricCollection] | None

    lr: float
    batch_size: int
    metrics_interval: int

    train_dataset_name: str

    def __init__(  # noqa: PLR0913
        self,
        lr: float,
        batch_size: int,
        dataloaders: dict[str, DataLoader[SegmentationItem]] | None = None,
        metrics: MetricCollection | None = None,
        metrics_interval: int = 20,
        train_dataset_name: str = "Unknown",
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.unet = smp.Unet(
            encoder_name="efficientnet-b4",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,                      # model output channels (number of classes in your dataset)
        ).double()
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
        )

    def _compute_loss_and_update_metrics(
        self,
        batch: dict[str, torch.Tensor],
        phase: str,
    ) -> torch.Tensor:
        images, labels = batch["image"], batch["labels"]
        logits = self.unet(images)
        loss = nn.BCEWithLogitsLoss()(logits, labels)

        if self.metrics:
            self.metrics[phase].update(logits, labels)

        return loss  # type: ignore[no-any-return]

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
        phase: str,
    ) -> torch.Tensor:
        loss = self._compute_loss_and_update_metrics(batch, phase)
        # type: ignore[reportUnkonwnMemberType]
        self.log(f"{phase}/loss", value=loss, batch_size=self.batch_size)

        if self.metrics and (
            phase == "test" or (batch_idx and batch_idx %
                                self.metrics_interval == 0)
        ):
            metric_results = self.metrics[phase].compute()
            # type: ignore[reportUnkonwnMemberType]
            self.log_dict(metric_results, batch_size=self.batch_size)

        return loss

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
        logits = self.unet(images)

        return upsample_logits(logits, images.shape[-2:])

    def on_train_start(self) -> None:
        if self.logger:
            self.logger.log_hyperparams(self.hparams)  # type: ignore  # noqa: PGH003

    def configure_optimizers(self) -> optim.Optimizer:
        return torch.optim.Adam(self.unet.parameters(), lr=self.lr)

    def train_dataloader(self) -> DataLoader[SegmentationItem] | None:
        return self.dataloaders.get("train")

    def val_dataloader(self) -> DataLoader[SegmentationItem] | None:
        return self.dataloaders.get("val")

    def test_dataloader(self) -> DataLoader[SegmentationItem] | None:
        return self.dataloaders.get("test")
