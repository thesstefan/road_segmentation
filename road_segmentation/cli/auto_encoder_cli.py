import argparse
import logging
from pathlib import Path

import torch
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
)
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger  # type: ignore[import]
from torch.utils.data import DataLoader, Subset, random_split
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryJaccardIndex,
)
from torchmetrics.collections import MetricCollection

from road_segmentation.dataset.auto_encoder_dataset import AEDataset
from road_segmentation.model.auto_encoder import AutoEncoder, ae_transform_factory
import warnings




warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
subparser = parser.add_subparsers(dest="mode", required=True)

train_parser = subparser.add_parser("train")
predict_parser = subparser.add_parser("predict")

predict_parser.add_argument("--model_ckpt_path", type=str, required=True)
train_parser.add_argument("--dataset_dir", type=str, required=True)
train_parser.add_argument("--lr", type=str, default=6e-5)
train_parser.add_argument("--epochs", type=int, default=50)
train_parser.add_argument("--batch_size", type=int, default=2)
train_parser.add_argument("--ae_base", type=str, default=None)
train_parser.add_argument("--train_val_split_ratio", type=float, default=0.9)
train_parser.add_argument(
    "--early_stop", action=argparse.BooleanOptionalAction, type=bool, default=True
)
train_parser.add_argument("--ckpt_save_dir", type=str, required=True)
train_parser.add_argument("--ckpt_save_top_k", type=int, default=1)
train_parser.add_argument("--ckpt_monitor", type=str, default="val_loss")
train_parser.add_argument("--resume_checkpoint", type=str, default=None)
train_parser.add_argument("--tb_logdir", type=str, default="tb_logs")
train_parser.add_argument("--experiment_name", type=str, default=None)
train_parser.add_argument(
    "--pretrained", action=argparse.BooleanOptionalAction, type=bool, default=True
)
train_parser.add_argument("--image_height", type=int, default=400)
train_parser.add_argument("--enc_type", type=str, default="resnet18")
train_parser.add_argument("--enc_out_dim", type=int, default=512)
train_parser.add_argument("--latent_dim", type=int, default=256)


def split_train_val(
    dataset: AEDataset,
    train_ratio: float,
) -> tuple[Subset[tuple], Subset[tuple]]:
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_subset, val_subset = random_split(dataset, [train_size, val_size])
    return train_subset, val_subset


def train(  # noqa: PLR0913
    device: torch.device,
    experiment_name: str | None,
    dataset_dir: str,
    lr: float,
    epochs: int,
    batch_size: int,
    ae_base: str | None,
    train_val_split_ratio: float,
    tb_logdir: Path,
    early_stop: bool,  # noqa: FBT001
    ckpt_save_top_k: int,
    ckpt_save_dir: Path,
    ckpt_monitor: str,
    resume_checkpoint: Path | None,
    pretrained: bool = True,
    image_height: int = 400,
    enc_type: str = "resnet18",
    enc_out_dim: int = 512,
    latent_dim: int = 256,
) -> None:

    dataset = AEDataset.train_dataset(
        Path(dataset_dir), ae_transform_factory(image_height)
    )

    train_dataset, val_dataset = split_train_val(
        dataset,
        train_val_split_ratio,
    )

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
        ),
    }
    model = AutoEncoder(
        ae_ckpt=ae_base,
        input_height=image_height,
        enc_type=enc_type,
        enc_out_dim=enc_out_dim,
        latent_dim=latent_dim,
        lr=lr,
    )

    #model = AE(input_height=image_height)
    if pretrained and not ae_base:
        model.from_pretrained("cifar10-resnet18")

    logger = TensorBoardLogger(
        tb_logdir,
        name=experiment_name or "AutoEncoder_Imputation",
        default_hp_metric=False,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_save_dir,
        save_top_k=ckpt_save_top_k,
        monitor=ckpt_monitor,
    )

    callbacks: list[Callback] = [  # type: ignore[no-any-unimported]
        checkpoint_callback,
    ]

    if early_stop:
        early_stop_callback = EarlyStopping(
            monitor=ckpt_monitor,
            min_delta=0.00,
            patience=10,
            verbose=False,
            mode="min",
        )
        callbacks.append(early_stop_callback)

    trainer = Trainer(
        accelerator=str(device),
        max_epochs=epochs,
        val_check_interval=len(dataloaders["train"]),
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(model, ckpt_path=resume_checkpoint, train_dataloaders=dataloaders["train"], val_dataloaders=dataloaders["val"])


def main() -> None:

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "predict":
        raise NotImplementedError("Predict mode not implemented yet")
    else:
        train(
            device,
            args.experiment_name,
            args.dataset_dir,
            args.lr,
            args.epochs,
            args.batch_size,
            args.ae_base,
            args.train_val_split_ratio,
            Path(args.tb_logdir),
            args.early_stop,
            args.ckpt_save_top_k,
            Path(args.ckpt_save_dir),
            args.ckpt_monitor,
            Path(args.resume_checkpoint) if args.resume_checkpoint else None,
            args.pretrained,
            args.image_height,
            args.enc_type,
            args.enc_out_dim,
            args.latent_dim,
        )


if __name__ == "__main__":
    main()
