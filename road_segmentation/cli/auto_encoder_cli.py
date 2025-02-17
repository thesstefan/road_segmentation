import argparse
import logging
from pathlib import Path

import lightning.pytorch as pl  # type: ignore[import]
import torch
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import TensorBoardLogger  # type: ignore[import]
from torch.utils.data import DataLoader, Subset, random_split
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryJaccardIndex,
)
from torchmetrics.collections import MetricCollection

from road_segmentation.utils.prediction_writer import OnBatchImageOutputWriter

import warnings

from road_segmentation.dataset.auto_encoder_dataset import AEDataset
from road_segmentation.model.unet import UNet, ae_transform_factory
from lightning.pytorch.callbacks import ModelSummary



warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
subparser = parser.add_subparsers(dest="mode", required=True)

train_parser = subparser.add_parser("train")
predict_parser = subparser.add_parser("predict")

predict_parser.add_argument("--model_ckpt_path", type=str, required=True)
predict_parser.add_argument("--input_dir", type=str, required=True)
predict_parser.add_argument("--prediction_output_dir", type=str, required=True)
predict_parser.add_argument("--image_height", type=int, default=512)


train_parser.add_argument("--dataset_dir", type=str, required=True)
train_parser.add_argument("--dataset_folders", type=str, nargs="*", default=None)
train_parser.add_argument("--lr", type=float, default=6e-5)
train_parser.add_argument("--epochs", type=int, default=50)
train_parser.add_argument("--batch_size", type=int, default=2)
train_parser.add_argument("--train_val_split_ratio", type=float, default=0.9)
train_parser.add_argument(
    "--early_stop", action=argparse.BooleanOptionalAction, type=bool, default=True
)
train_parser.add_argument("--ckpt_save_dir", type=str, required=True)
train_parser.add_argument("--ckpt_save_top_k", type=int, default=1)
train_parser.add_argument("--ckpt_monitor", type=str, default="val/loss")
train_parser.add_argument("--resume_checkpoint", type=str, default=None)
train_parser.add_argument("--metrics_interval", type=int, default=5)
train_parser.add_argument("--tb_logdir", type=str, default="tb_logs")
train_parser.add_argument("--experiment_name", type=str, default=None)

train_parser.add_argument("--image_height", type=int, default=512)
train_parser.add_argument("--depth", type=int, default=5)


def split_train_val(
    dataset: AEDataset,
    train_ratio: float,
) -> tuple[Subset[tuple], Subset[tuple]]:
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_subset, val_subset = random_split(dataset, [train_size, val_size])
    return train_subset, val_subset

def predict(
    device: torch.device,
    model_ckpt_path: Path,
    input_dir: Path,
    prediction_output_dir: Path,
    image_height: int,
) -> None:

    model = UNet.load_from_checkpoint(  # type: ignore[reportUnkonwnMemberType]
        checkpoint_path=model_ckpt_path,
    ).to(device)

    predict_dataset = AEDataset.test_dataset(
        input_dir,
        transform=ae_transform_factory(image_height, image_height),
    )
    dataloader = DataLoader(
        predict_dataset,
        shuffle=False,
    )
    predictor = pl.Trainer(
        accelerator=str(device),
        logger=False,
        callbacks=[OnBatchImageOutputWriter(prediction_output_dir) ],
    )
    predictor.predict(
        model,
        dataloaders=dataloader,
        return_predictions=False,
    )

def train(  # noqa: PLR0913
    device: torch.device,
    experiment_name: str | None,
    dataset_dir: str,
    data_set_folders: list[Path] | None, #if multiple dataset_dir contains multiple folders
    lr: float,
    epochs: int,
    batch_size: int,
    train_val_split_ratio: float,
    tb_logdir: Path,
    early_stop: bool,  # noqa: FBT001
    ckpt_save_top_k: int,
    ckpt_save_dir: Path,
    ckpt_monitor: str,
    resume_checkpoint: Path | None,
    image_height: int,
    depth: int,
    metrics_interval: int,
    ) -> None:

    dataset = AEDataset.train_dataset(
        Path(dataset_dir),
        ae_transform_factory(image_height, image_height),
        data_set_folders,
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

    metrics = MetricCollection(
        {
            "accuracy": BinaryAccuracy(),
            "f1": BinaryF1Score(),
            "jaccard": BinaryJaccardIndex(),
        },
    ).to(device)

    model = UNet(
        batch_size=batch_size,
        lr=lr,
        depth=depth,
        metrics=metrics,
        metrics_interval=metrics_interval,
        dataloaders=dataloaders,
    )

    logger = TensorBoardLogger(
        tb_logdir,
        name=experiment_name or "Unet_Denoising",
        default_hp_metric=False,
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_save_dir,
        save_top_k=ckpt_save_top_k,
        monitor=ckpt_monitor,
    )
    callbacks: list[Callback] = [  # type: ignore[no-any-unimported]
        checkpoint_callback,
        ModelSummary(max_depth=-1),
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

    trainer = pl.Trainer(
        accelerator=str(device),
        max_epochs=epochs,
        val_check_interval=len(dataloaders["train"]),
        logger=logger,
        callbacks=callbacks,
    )
    trainer.fit(model, ckpt_path=resume_checkpoint)



def main() -> None:

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "predict":
        predict(
            device,
            Path(args.model_ckpt_path),
            Path(args.input_dir),
            Path(args.prediction_output_dir),
            args.image_height,
        )
    else:
        train(
            device,
            args.experiment_name,
            args.dataset_dir,
            [Path(folder) for folder in args.dataset_folders] if args.dataset_folders else None,
            args.lr,
            args.epochs,
            args.batch_size,
            args.train_val_split_ratio,
            Path(args.tb_logdir),
            args.early_stop,
            args.ckpt_save_top_k,
            Path(args.ckpt_save_dir),
            args.ckpt_monitor,
            Path(args.resume_checkpoint) if args.resume_checkpoint else None,
            args.image_height,
            args.depth,
            args.metrics_interval,
        )


if __name__ == "__main__":
    main()
