import lightning.pytorch as pl
import torch
import cv2
from torch import nn, optim
from torch.utils.data import DataLoader
from torchmetrics.collections import MetricCollection
from torchvision.transforms.functional import resize, normalize
from torchvision.transforms import InterpolationMode

from road_segmentation.dataset.segmentation_datapoint import SegmentationItem
from road_segmentation.model.impl.umamba_bot_2d import UMambaBot
from road_segmentation.model.impl.umamba_loss import build_umamba_loss
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm


def umamba_transforms(
    image: torch.Tensor,
    labels: torch.Tensor | None
) -> tuple[torch.Tensor, list[torch.Tensor] | None]:
    
    # Normalize the image
    image = image.float()
    image = image / 255.0
    image = normalize(
        image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    target_sizes = [(448, 448), (224, 224), (112, 112), (56, 56), (28, 28), (14, 14)]
    
    # Process image resizing
    image = resize(image, target_sizes[0], interpolation=InterpolationMode.BICUBIC)
    
    # Process labels if provided
    labels_list = []
    if labels is not None:
        if labels.ndim == 2:
            labels = labels.unsqueeze(0)  # Add channel dimension if missing
        for size in target_sizes:
            # Decide interpolation based on size comparison
            resized_label = resize(labels.float(), size=size, interpolation=InterpolationMode.NEAREST)
            labels_list.append(resized_label)  # Remove batch dim added by interpolate

        return image, labels_list
    else:
        return image, None


def upsample_logits(logits: torch.Tensor, size: torch.Size) -> torch.Tensor:
    upsampled_logits: torch.Tensor = nn.functional.interpolate(  # type: ignore[reportUnknownMemberType]
        logits,
        size=size,
        mode="bilinear",
        align_corners=False,
    )

    # type: ignore[reportUnkonwnMemberType]
    return upsampled_logits.argmax(dim=1)

class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)

def get_umamba_bot_2d(
        deep_supervision: bool = True
    ):
    
    conv_kernel_sizes = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3]]
    num_stages = len(conv_kernel_sizes)
    UNet_base_num_features = 32
    unet_max_num_features = 512
    conv_op = nn.Conv2d
    pool_op_kernel_sizes = [[1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]
    n_conv_per_stage_encoder = [2, 2, 2, 2, 1, 1, 1] 
    n_conv_per_stage_decoder = [2, 2, 2, 2, 1, 1]
    num_classes = 2
    norm_op = get_matching_instancenorm(conv_op)
    norm_op_kwargs = {'eps': 1e-5, 'affine': True}
    nonlin_kwargs = {'inplace': True}
    
    model = UMambaBot(
        input_channels = 3,
        n_stages = num_stages,
        features_per_stage=[min(UNet_base_num_features * 2 ** i,
                                unet_max_num_features) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes = conv_kernel_sizes,
        strides = pool_op_kernel_sizes,
        n_conv_per_stage = n_conv_per_stage_encoder,
        num_classes = num_classes,
        n_conv_per_stage_decoder = n_conv_per_stage_decoder,
        conv_bias = True,
        norm_op = norm_op,
        norm_op_kwargs = norm_op_kwargs,
        dropout_op = None,
        dropout_op_kwargs = None,
        nonlin = nn.LeakyReLU,
        nonlin_kwargs = nonlin_kwargs,
        deep_supervision = deep_supervision,
        stem_channels = None,
    )
    
    model.apply(InitWeights_He(1e-2))

    return model


class UMamba(pl.LightningModule):
    umamba: nn.Module  # type: ignore[no-any-unimported]

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
        self.unet = get_umamba_bot_2d()
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
        
        self.loss = build_umamba_loss()

    def _compute_loss_and_update_metrics(
        self,
        batch: dict[str, torch.Tensor],
        phase: str,
    ) -> torch.Tensor:
        images, labels = batch["image"], batch["labels"]
        
        logits = self.unet(images)
        loss = self.loss

        # Compute the loss by feeding the paired logits and labels
        loss = loss(logits, labels)
        
        logits_binary = torch.argmax(logits[0], dim=1, keepdim=True)

        if self.metrics:
            self.metrics[phase].update(logits_binary, labels[0])

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
