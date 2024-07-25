from pl_bolts.models.autoencoders import AE
import torch.nn as nn
from torch.nn import functional as F, Conv2d
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
import torch
from road_segmentation.model.impl.AE_components import ResNetDecoderTop, DecoderBlock

def ae_transform_factory(input_height: int):
    def transform(image, labels=None):
        image = image.view(1, *image.shape).float()
        image_rs = resize(
            image,
            size=(input_height, input_height),
            interpolation=InterpolationMode.NEAREST,
        )

        if not labels is None:
            labels = labels.view(1, *labels.shape)
            labels_rs = resize(
                labels,
                size=(input_height, input_height),
                interpolation=InterpolationMode.NEAREST,
            )
            return image_rs, labels_rs
        else:
            return image_rs, None

    return transform


class AutoEncoder(AE):
    def __init__(
        self,
        input_height: int,
        enc_type: str = "resnet18",
        first_conv: bool = False,
        maxpool1: bool = False,
        enc_out_dim: int = 512,
        latent_dim: int = 256,
        lr: float = 1e-4,
        base_pretrained: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            input_height,
            enc_type,
            first_conv,
            maxpool1,
            enc_out_dim,
            latent_dim,
            lr,
            **kwargs,
        )
        self.input_layer = nn.Sequential(
            Conv2d(
                1,
                3,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(3),
        )
        self.rgb_AE = AE(
            input_height=input_height,
            enc_type=enc_type,
            first_conv=first_conv,
            maxpool1=maxpool1,
            enc_out_dim=enc_out_dim,
            latent_dim=latent_dim,
            lr=lr,
            )
        self.output_layer = nn.Sequential(
            Conv2d(
                3,
                1,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.Sigmoid(),
        )
        
        if base_pretrained:
            self.rgb_AE = self.rgb_AE.from_pretrained("cifar10-resnet18")
            self.rgb_AE.decoder.conv1 = ResNetDecoderTop(DecoderBlock, [2,2,2,2,2], 512)

    def forward(self, x) -> torch.Tensor:
        model_input = self.input_layer(x)
        x_hat = self.rgb_AE(model_input)
        probs = self.output_layer(x_hat)
        return probs

    def step(self, batch, batch_idx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = batch["image"]
        y = batch["labels"]
        model_input = self.input_layer(x)
        x_hat = self.rgb_AE(model_input)
        probs = self.output_layer(x_hat)
        loss = nn.BCELoss()(probs, y)
        return loss, {"loss": loss}

    def predict_step(self, batch: torch.Tensor, idx: int) -> torch.Tensor:
        x = batch["image"]
        output = self.forward(x)
        return (output >= 0.5).float()
