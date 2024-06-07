import lightning.pytorch as pl
import torch
import cv2
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics.collections import MetricCollection
from transformers import (  # type: ignore[import]
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
)
from torchvision.transforms.functional import resize, to_tensor, normalize
from torchvision.transforms import InterpolationMode

from road_segmentation.dataset.segmentation_datapoint import SegmentationItem

class Block(nn.Module):
    # a repeating structure composed of two convolutional layers with batch normalization and ReLU activations
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
                                   nn.ReLU(),
                                   nn.BatchNorm2d(out_ch),
                                   nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1),
                                   nn.ReLU())

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    # UNet-like architecture for single class semantic segmentation.
    def __init__(self, chs=(3,64,128,256,512,1024)):
        super().__init__()
        enc_chs = chs  # number of channels in the encoder
        dec_chs = chs[::-1][:-1]  # number of channels in the decoder
        self.enc_blocks = nn.ModuleList([Block(in_ch, out_ch) for in_ch, out_ch in zip(enc_chs[:-1], enc_chs[1:])])  # encoder blocks
        self.pool = nn.MaxPool2d(2)  # pooling layer (can be reused as it will not be trained)
        self.upconvs = nn.ModuleList([nn.ConvTranspose2d(in_ch, out_ch, 2, 2) for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])])  # deconvolution
        self.dec_blocks = nn.ModuleList([Block(in_ch, out_ch) for in_ch, out_ch in zip(dec_chs[:-1], dec_chs[1:])])  # decoder blocks
        self.head = nn.Sequential(nn.Conv2d(dec_chs[-1], 1, 1), nn.Sigmoid()) # 1x1 convolution for producing the output

    def forward(self, x):
        # encode
        enc_features = []
        for block in self.enc_blocks[:-1]:
            x = block(x)  # pass through the block
            enc_features.append(x)  # save features for skip connections
            x = self.pool(x)  # decrease resolution
        x = self.enc_blocks[-1](x)
        # decode
        for block, upconv, feature in zip(self.dec_blocks, self.upconvs, enc_features[::-1]):
            x = upconv(x)  # increase resolution
            x = torch.cat([x, feature], dim=1)  # concatenate skip features
            x = block(x)  # pass through the block
        return self.head(x)  # reduce to 1 channel


def unet_transforms(
    image: torch.Tensor,
    labels: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    image = image.float()
    labels = labels.float()
    image_np = image.numpy().transpose(1, 2, 0)  # Assuming image is CxHxW, convert to HxWxC for OpenCV

    # Resize the image and labels to 384x384
    image_np = cv2.resize(image_np, (384, 384), interpolation=cv2.INTER_AREA)
    if labels is not None:
        labels = labels.unsqueeze(0)  # Add channel dimension if missing
        labels = resize(labels, size=(384, 384), interpolation=InterpolationMode.NEAREST)
        # labels = labels.squeeze(1)  # Remove channel dimension after resizing

    # Apply CLAHE to each channel of the RGB image
    clahe = cv2.createCLAHE(tileGridSize=(16, 16))
    image_np[:, :, 0] = clahe.apply((image_np[:, :, 0] * 255).astype('uint8'))  # Apply on R channel
    image_np[:, :, 1] = clahe.apply((image_np[:, :, 1] * 255).astype('uint8'))  # Apply on G channel
    image_np[:, :, 2] = clahe.apply((image_np[:, :, 2] * 255).astype('uint8'))  # Apply on B channel

    # Convert back to tensor and normalize
    image_tensor = to_tensor(image_np / 255.0)  # Normalize to 0-1 range
    image_tensor = normalize(image_tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return image_tensor, labels


def upsample_logits(logits: torch.Tensor, size: torch.Size) -> torch.Tensor:
    upsampled_logits: torch.Tensor = nn.functional.interpolate(  # type: ignore[reportUnknownMemberType]
        logits,
        size=size,
        mode="bilinear",
        align_corners=False,
    )

    return upsampled_logits.argmax(dim=1)  # type: ignore[reportUnkonwnMemberType]



class UNet_simple(pl.LightningModule):
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
        self.unet = UNet()
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
        loss = nn.BCELoss()(logits, labels)

        # predicted = upsample_logits(logits, labels.shape[-2:])   # Maybe not needed for UNet?

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
        self.log(f"{phase}/loss", value=loss, batch_size=self.batch_size)  # type: ignore[reportUnkonwnMemberType]

        if self.metrics and (
            phase == "test"
            or (batch_idx and batch_idx % self.metrics_interval == 0)
        ):
            metric_results = self.metrics[phase].compute()
            self.log_dict(metric_results, batch_size=self.batch_size)  # type: ignore[reportUnkonwnMemberType]

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
