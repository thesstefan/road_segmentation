import argparse
from pathlib import Path

import lightning.pytorch as pl  # type: ignore[import]
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger  # type: ignore[import]
from torch.utils.data import DataLoader, Subset, random_split
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryJaccardIndex,
)
from torchmetrics.collections import MetricCollection

from road_segmentation.dataset.ethz_cil_dataset import ETHZDataset
from road_segmentation.model.road_segformer import (
    RoadSegformer,
    segformer_feature_extractor,
)

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_dir", type=str)
parser.add_argument("--lr", type=str, default=6e-5)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--segformer_base", type=str, default="nvidia/mit-b3")
parser.add_argument("--metrics_interval", type=int, default=5)
parser.add_argument("--train_val_split_ratio", type=float, default=0.9)
parser.add_argument("--tb_logdir", type=str, default="tb_logs")
parser.add_argument(
    "--early_stop",
    action=argparse.BooleanOptionalAction,
    type=bool,
    default=True,
)
parser.add_argument("--ckpt_save_top_k", type=int, default=1)
parser.add_argument("--ckpt_monitor", type=str, default="val/loss")
parser.add_argument("--resume_checkpoint", type=str, default=None)


def split_train_val(
    dataset: ETHZDataset,
    train_ratio: float,
) -> tuple[Subset[dict[str, torch.Tensor]], Subset[dict[str, torch.Tensor]]]:
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size

    train_subset, val_subset = random_split(dataset, [train_size, val_size])
    return train_subset, val_subset


def main() -> None:
    args = parser.parse_args()

    dataset = ETHZDataset.train_dataset(
        Path(f"{args.dataset_dir}/training"),
        transform=segformer_feature_extractor,
    )
    train_dataset, val_dataset = split_train_val(
        dataset,
        args.train_val_split_ratio,
    )
    test_dataset = ETHZDataset.test_dataset(
        Path(f"{args.dataset_dir}/test"),
        transform=segformer_feature_extractor,
    )

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        ),
    }

    metrics = MetricCollection(
        {
            "accuracy": BinaryAccuracy(),
            "f1": BinaryF1Score(),
            "jaccard": BinaryJaccardIndex(),
        },
    )

    model = RoadSegformer(
        segformer_ckpt=args.segformer_base,
        dataloaders=dataloaders,
        metrics=metrics,
        lr=args.lr,
        metrics_interval=args.metrics_interval,
    )

    logger = TensorBoardLogger(
        args.tb_logdir,
        name="RoadSegformer_ETHZDataset",
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=args.ckpt_save_top_k,
        monitor=args.ckpt_monitor,
    )

    callbacks: list[Callback] = [  # type: ignore[no-any-unimported]
        checkpoint_callback,
    ]

    if args.early_stop:
        early_stop_callback = EarlyStopping(
            monitor=args.ckpt_monitor,
            min_delta=0.00,
            patience=10,
            verbose=False,
            mode="min",
        )
        callbacks.append(early_stop_callback)

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        val_check_interval=len(dataloaders["train"]),
        logger=logger,
        callbacks=callbacks,
    )

    trainer.fit(model, ckpt_path=args.resume_checkpoint)
    trainer.test(model, dataloaders["test"], ckpt_path="best")


if __name__ == "__main__":
    main()
