# MIT License

# Original work Copyright (c) 2018 Joris (https://github.com/jvanvugt/pytorch-unet)
# Modified work Copyright (C) 2022 Canon Medical Systems Corporation.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from math import sqrt

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from road_segmentation.model.impl.swish import CustomSwish
from road_segmentation.model.impl.ws_conv import WNConv2d
import lightning.pytorch as pl
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
from road_segmentation.dataset.segmentation_datapoint import SegmentationItem

def ae_transform_factory(
    input_height: int,
    output_height: int,
) -> callable:

    def transform(
        image: torch.Tensor,
        labels: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor | None]:

        image = image.view(1, *image.shape).float()
        image_rs = resize(
            image,
            size=(input_height, input_height),
            interpolation=InterpolationMode.NEAREST,
        )

        if labels is not None:
            labels = labels.view(1, *labels.shape)
            labels_rs = resize(
                labels,
                size=(output_height, output_height),
                interpolation=InterpolationMode.NEAREST,
            )
            return image_rs, labels_rs

        return image_rs, None

    return transform

def segmentation_transform_factory(
    input_height: int,
    output_height: int,
) -> callable:

    def transform(
        image: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        image = image.view(1, *image.shape).float()
        image_rs = resize(
            image,
            size=(input_height, input_height),
            interpolation=InterpolationMode.BICUBIC,
        )
        image_rs = image_rs.squeeze()
        image_rs = image_rs.float()

        if labels is not None:
            labels = labels.view(1, *labels.shape)
            labels_rs = resize(
                labels,
                size=(output_height, output_height),
                interpolation=InterpolationMode.NEAREST,
            )
            labels_rs = labels_rs.float()
            return image_rs, labels_rs

        return image_rs, None

    return transform

def get_groups(channels: int) -> int:
    """:param channels:
    :return: return a suitable parameter for number of groups in GroupNormalisation'.
    """  # noqa: D205
    divisors = []
    for i in range(1, int(sqrt(channels)) + 1):
        if channels % i == 0:
            divisors.append(i)
            other = channels // i
            if i != other:
                divisors.append(other)
    return sorted(divisors)[len(divisors) // 2]


class UNet(pl.LightningModule):
    def __init__(
            self,
            batch_size: int,
            lr: float = 1e-3,
            in_channels=1,
            depth=5,
            wf=4,
            padding=True,
            norm="group",
            up_mode='upconv',
            metrics=None,
            metrics_interval: int = 20,
            dataloaders = None,
            ):
        """A modified U-Net implementation [1].

        [1] U-Net: Convolutional Networks for Biomedical Image Segmentation
            Ronneberger et al., 2015 https://arxiv.org/abs/1505.04597

        Args:
        ----
            in_channels (int): number of input channels
            n_classes (int): number of output channels
            depth (int): depth of the network
            wf (int): number of filters in the first layer is 2**wf
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
            norm (str): one of 'batch' and 'group'.
                        'batch' will use BatchNormalization.
                        'group' will use GroupNormalization.
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for learned 
                           upsampling.
                           'upsample' will use bilinear upsampling.

        """  # noqa: D401
        super().__init__()
        assert up_mode in ("upconv", "upsample")
        self.lr = lr
        self.padding = padding
        self.depth = depth
        self.in_channels = in_channels
        self.depth = depth
        self.wf = wf
        self.norm = norm
        self.up_mode = up_mode
        self.batch_size = batch_size
        self.save_hyperparameters(
            "batch_size",
            "lr",
            "in_channels",
            "depth",
            "wf",
            "padding",
            "norm",
            "up_mode",
        )
        self.metrics_interval = metrics_interval
        self.dataloaders = dataloaders
        self.metrics = (
            {
                "train": metrics.clone(prefix="train/"),
                "val": metrics.clone(prefix="val/"),
                "test": metrics.clone(prefix="test/"),
            }
            if metrics
            else None
        )
        prev_channels = in_channels
        self.down_path = nn.ModuleList()
        for i in range(depth):
            self.down_path.append(
                UNetConvBlock(prev_channels, 2 ** (wf + i), padding, norm=norm)
            )
            prev_channels = 2 ** (wf + i)

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path.append(
                UNetUpBlock(
                    prev_channels,
                    2 ** (wf + i),
                    up_mode,
                    padding,
                    norm=norm,
                ),
            )
            prev_channels = 2 ** (wf + i)

        self.last = nn.Sequential(
            nn.Conv2d(prev_channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward_down(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:

        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            blocks.append(x)
            if i != len(self.down_path) - 1:
                x = F.avg_pool2d(x, 2)
        return x, blocks

    def forward_up_without_last(
        self,
        x: torch.Tensor,
        blocks: list[torch.Tensor],
    ) -> torch.Tensor:
        for i, up in enumerate(self.up_path):
            skip = blocks[-i - 2]
            x = up(x, skip)
        return x

    def forward_without_last(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x, blocks = self.forward_down(x)
        return self.forward_up_without_last(x, blocks)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.get_features(x)
        return self.last(x)

    def get_features(
        self,
        x: torch.Tensor,
        ) -> torch.Tensor:
        return self.forward_without_last(x)

    def _compute_loss_and_update_metrics(
        self,
        batch: dict[str, torch.Tensor],
        phase: str,
    ) -> torch.Tensor:
        images, labels = batch["image"], batch["labels"]

        probs = self.forward(images)
        loss = F.binary_cross_entropy(probs, labels)

        predicted = (probs >= 0.5).float()  # noqa: PLR2004

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
        probs = self.forward(images)

        return (probs >= 0.5).float()

    def on_train_start(self) -> None:
        if self.logger:
            self.logger.log_hyperparams(self.hparams)
    def train_dataloader(self) -> DataLoader[SegmentationItem] | None:
        return self.dataloaders.get("train")

    def val_dataloader(self) -> DataLoader[SegmentationItem] | None:
        return self.dataloaders.get("val")

    def test_dataloader(self) -> DataLoader[SegmentationItem] | None:
        return self.dataloaders.get("test")

    def configure_optimizers(self) -> DataLoader[SegmentationItem] | None:
        return torch.optim.Adam(self.parameters(), lr=self.lr)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, norm="group", kernel_size=3):
        super().__init__()
        block = []
        if padding:
            block.append(nn.ReflectionPad2d(1))

        block.append(WNConv2d(in_size, out_size, kernel_size=kernel_size))
        block.append(CustomSwish())

        if norm == "batch":
            block.append(nn.BatchNorm2d(out_size))
        elif norm == "group":
            block.append(nn.GroupNorm(get_groups(out_size), out_size))

        if padding:
            block.append(nn.ReflectionPad2d(1))

        block.append(WNConv2d(out_size, out_size, kernel_size=kernel_size))
        block.append(CustomSwish())

        if norm == "batch":
            block.append(nn.BatchNorm2d(out_size))
        elif norm == "group":
            block.append(nn.GroupNorm(get_groups(out_size), out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, norm="group"):
        super().__init__()
        if up_mode == "upconv":
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        elif up_mode == "upsample":
            self.up = nn.Sequential(
                nn.Upsample(mode="bilinear", scale_factor=2),
                nn.Conv2d(in_size, out_size, kernel_size=1),
            )

        self.conv_block = UNetConvBlock(in_size, out_size, padding, norm=norm)

    def center_crop(
        self,
        layer: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])]

    def forward(
        self,
        x: torch.Tensor,
        bridge: torch.Tensor,
    ) -> torch.Tensor:
        up = self.up(x)
        crop1 = self.center_crop(bridge, up.shape[2:])
        out = torch.cat([up, crop1], 1)
        return self.conv_block(out)

