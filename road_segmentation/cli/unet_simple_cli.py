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
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryJaccardIndex,
)
from torchmetrics.collections import MetricCollection

from road_segmentation.dataset.ethz_cil_dataset import ETHZDataset
from road_segmentation.dataset.merged_datasets import get_datasets
from road_segmentation.dataset.segmentation_datapoint import SegmentationItem
from road_segmentation.model.unet_simple import UNetSimple, unet_transforms
from road_segmentation.utils.prediction_writer import OnBatchImageOutputWriter

import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
subparser = parser.add_subparsers(dest="mode", required=True)

train_parser = subparser.add_parser("train")
predict_parser = subparser.add_parser("predict")

predict_parser.add_argument("--model_ckpt_path", type=str, required=True)
predict_parser.add_argument("--ethz_input_dir", type=str, required=True)
predict_parser.add_argument("--prediction_output_dir", type=str, required=True)
train_parser.add_argument("--ckpt_save_dir", type=str, required=True)

train_parser.add_argument("--dataset_dir", type=str, required=True)
train_parser.add_argument("--epfl_dataset_dir", type=str, default= None)
train_parser.add_argument("--deepglobe_dataset_dir", type=str, default= None)
train_parser.add_argument("--chesa_dataset_dir", type=str, default= None)
train_parser.add_argument("--mass_dataset_dir", type=str, default= None)

train_parser.add_argument("--lr", type=str, default=6e-5)
train_parser.add_argument("--epochs", type=int, default=50)
train_parser.add_argument("--batch_size", type=int, default=2)
train_parser.add_argument("--metrics_interval", type=int, default=5)
train_parser.add_argument("--train_val_split_ratio", type=float, default=0.9)
train_parser.add_argument("--early_stop", action=argparse.BooleanOptionalAction, type=bool, default=True,)

train_parser.add_argument("--ckpt_save_top_k", type=int, default=1)
train_parser.add_argument("--ckpt_monitor", type=str, default="val/loss")
train_parser.add_argument("--resume_checkpoint", type=str, default=None)
train_parser.add_argument("--tb_logdir", type=str, default="tb_logs")
train_parser.add_argument("--experiment_name", type=str, default=None)


def split_train_val(
    dataset: Dataset,
    train_ratio: float,
) -> tuple[Subset[SegmentationItem], Subset[SegmentationItem]]:
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_subset, val_subset = random_split(dataset, [train_size, val_size])
    return train_subset, val_subset


def predict(
    device: torch.device,
    model_ckpt_path: Path,
    input_dir: Path,
    prediction_output_dir: Path,
) -> None:
    model = UNetSimple.load_from_checkpoint(  # type: ignore[reportUnkonwnMemberType]
        checkpoint_path=model_ckpt_path,
    ).to(device)

    predict_dataset = ETHZDataset.test_dataset(
        input_dir,
        transform=unet_transforms,
    )
    dataloader = DataLoader(
        predict_dataset,
        batch_size=model.batch_size,
        shuffle=False,
    )
    predictor = pl.Trainer(
        accelerator=str(device),
        logger=False,
        callbacks=[OnBatchImageOutputWriter(prediction_output_dir)],
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
    epfl_dataset_dir: str | None,
    deepglobe_dataset_dir: str | None,
    chesa_dataset_dir: str | None,
    mass_dataset_dir: str | None,
    lr: float,
    epochs: int,
    batch_size: int,
    metrics_interval: int,
    train_val_split_ratio: float,
    tb_logdir: Path,
    early_stop: bool,  # noqa: FBT001
    ckpt_save_top_k: int,
    ckpt_save_dir: Path,
    ckpt_monitor: str,
    resume_checkpoint: Path | None,
) -> None:
    dataset = get_datasets(
        dataset_dir,
        epfl_dataset_dir,
        deepglobe_dataset_dir,
        chesa_dataset_dir,
        mass_dataset_dir,
        unet_transforms
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

    model = UNetSimple(
        batch_size=batch_size,
        dataloaders=dataloaders,
        metrics=metrics,
        lr=lr,
        metrics_interval=metrics_interval,
        train_dataset_name="ETHZDataset",
    )

    logger = TensorBoardLogger(
        tb_logdir,
        name=experiment_name or "UNetSimple",
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
            Path(args.ethz_input_dir),
            Path(args.prediction_output_dir),
        )
    else:
        train(
            device,
            args.experiment_name,
            args.dataset_dir,
            args.epfl_dataset_dir,
            args.deepglobe_dataset_dir,
            args.chesa_dataset_dir,
            args.mass_dataset_dir,
            args.lr,
            args.epochs,
            args.batch_size,
            args.metrics_interval,
            args.train_val_split_ratio,
            Path(args.tb_logdir),
            args.early_stop,
            args.ckpt_save_top_k,
            Path(args.ckpt_save_dir),
            args.ckpt_monitor,
            Path(args.resume_checkpoint) if args.resume_checkpoint else None,
        )


if __name__ == "__main__":
    main()
