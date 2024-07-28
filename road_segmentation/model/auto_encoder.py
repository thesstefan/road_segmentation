from pl_bolts.models.autoencoders import AE
import torch.nn as nn
from torch.nn import functional as F, Conv2d
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
import torch


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
        ae_ckpt: str | None = None,
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
        self.output_layer = nn.Sequential(
            Conv2d(
                3,
                1,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(),
            nn.BatchNorm2d(1),
        )
        self.head = nn.Sequential(nn.Conv2d(1, 1, 1), nn.Sigmoid())
        if ae_ckpt:
            self.load_from_checkpoint(ae_ckpt)

    def forward(self, x) -> torch.Tensor:
        model_input = self.input_layer(x)
        feats = self.encoder(model_input)
        z = self.fc(feats)
        x_hat = self.decoder(z)
        logits = self.output_layer(x_hat)
        return self.head(logits)

    def step(self, batch, batch_idx) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = batch["image"]
        y = batch["labels"]
        model_input = self.input_layer(x)
        feats = self.encoder(model_input)
        z = self.fc(feats)
        x_hat = self.decoder(z)
        output = self.output_layer(x_hat)
        probs = self.head(output)
        loss = nn.BCELoss()(probs, y)
        return loss, {"loss": loss}

    def predict_step(self, batch: torch.Tensor, idx: int) -> torch.Tensor:
        x = batch["image"]
        output = self.forward(x)
        return (output >= 0.5).float()
