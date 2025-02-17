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

from road_segmentation.dataset.ethz_cil_dataset import ETHZDataset
from road_segmentation.dataset.segmentation_datapoint import SegmentationItem
from road_segmentation.dataset.merged_datasets import get_datasets, copy_dataset
from road_segmentation.model.road_segformer import (
    RoadSegformer,
    segformer_feature_extractor,
)
from road_segmentation.utils.prediction_writer import OnBatchImageOutputWriter
from road_segmentation.utils.transforms import clahe_transform

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
predict_parser.add_argument(
    "--clahe",
    action=argparse.BooleanOptionalAction,
    type=bool,
    default=True,
)
predict_parser.add_argument("--epfl_dataset_dir", type=str, default=None)
predict_parser.add_argument("--deepglobe_dataset_dir", type=str, default=None)
predict_parser.add_argument("--chesa_dataset_dir", type=str, default=None)
predict_parser.add_argument("--mass_dataset_dir", type=str, default=None)
predict_parser.add_argument("--create_autoencoder_dataset", type=bool, default=False)

train_parser.add_argument("--ckpt_save_dir", type=str, required=True)
train_parser.add_argument("--dataset_dir", type=str, required=True)
train_parser.add_argument("--epfl_dataset_dir", type=str, default=None)
train_parser.add_argument("--deepglobe_dataset_dir", type=str, default=None)
train_parser.add_argument("--chesa_dataset_dir", type=str, default=None)
train_parser.add_argument("--mass_dataset_dir", type=str, default=None)

train_parser.add_argument("--lr", type=str, default=6e-5)
train_parser.add_argument("--epochs", type=int, default=50)
train_parser.add_argument("--batch_size", type=int, default=2)
train_parser.add_argument("--segformer_base", type=str, default="nvidia/mit-b3")
train_parser.add_argument("--metrics_interval", type=int, default=5)
train_parser.add_argument("--train_val_split_ratio", type=float, default=0.9)
train_parser.add_argument(
    "--early_stop",
    action=argparse.BooleanOptionalAction,
    type=bool,
    default=True,
)
train_parser.add_argument("--ckpt_save_top_k", type=int, default=1)
train_parser.add_argument("--ckpt_monitor", type=str, default="val/loss")
train_parser.add_argument("--resume_checkpoint", type=str, default=None)
train_parser.add_argument("--tb_logdir", type=str, default="tb_logs")
train_parser.add_argument("--experiment_name", type=str, default=None)
train_parser.add_argument(
    "--clahe",
    action=argparse.BooleanOptionalAction,
    type=bool,
    default=True,
)
train_parser.add_argument("--tversky_loss_factor", type=float, default=0.0)
train_parser.add_argument("--tversky_alpha", type=float, default=0.5)
train_parser.add_argument("--tversky_beta", type=float, default=0.5)

train_parser.add_argument("--focal_loss_factor", type=float, default=0.0)
train_parser.add_argument("--focal_alpha", type=float, default=None)
train_parser.add_argument("--focal_gamma", type=float, default=2.0)

train_parser.add_argument("--bce_loss_factor", type=float, default=1.0)


def clahe_segformer_data_transform(
    image: torch.Tensor,
    labels: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    image, labels = clahe_transform(image, labels)
    image, labels = segformer_feature_extractor(image, labels)
    return image, labels


def split_train_val(
    dataset: ETHZDataset,
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
    clahe: bool,
    epfl_dataset_dir: str | None,
    deepglobe_dataset_dir: str | None,
    chesa_dataset_dir: str | None,
    mass_dataset_dir: str | None,
    create_autoencoder_dataset: bool,
) -> None:

    model = RoadSegformer.load_from_checkpoint(  # type: ignore[reportUnkonwnMemberType]
        checkpoint_path=model_ckpt_path,
    ).to(device)

    if create_autoencoder_dataset:
        predict_dataset = get_datasets(
            input_dir,
            epfl_dataset_dir,
            deepglobe_dataset_dir,
            chesa_dataset_dir,
            mass_dataset_dir,
            clahe_segformer_data_transform
            if clahe
            else segformer_feature_extractor,
        )
        copy_dataset(
            input_dir,
            epfl_dataset_dir,
            deepglobe_dataset_dir,
            chesa_dataset_dir,
            mass_dataset_dir,
            prediction_output_dir / Path("groundtruth"),
        )
    else:
        predict_dataset = ETHZDataset.test_dataset(
            input_dir,
            transform=clahe_segformer_data_transform
            if clahe
            else segformer_feature_extractor,
        )

    dataloader = DataLoader(
        predict_dataset,
        batch_size=model.batch_size,
        shuffle=False,
    )
    predictor = pl.Trainer(
        accelerator=str(device),
        logger=False,
        callbacks=[OnBatchImageOutputWriter(prediction_output_dir / Path("segments"))],
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
    segformer_base: str,
    metrics_interval: int,
    train_val_split_ratio: float,
    tb_logdir: Path,
    early_stop: bool,  # noqa: FBT001
    ckpt_save_top_k: int,
    ckpt_save_dir: Path,
    ckpt_monitor: str,
    resume_checkpoint: Path | None,
    clahe: bool,
    tversky_loss_factor: float = 0.0,
    tversky_alpha: float = 0.5,
    tversky_beta: float = 0.5,
    focal_loss_factor: float = 0.0,
    focal_alpha: float | None = None,
    focal_gamma: float = 2.0,
    bce_loss_factor: float = 1.0,
) -> None:
    dataset = get_datasets(
        dataset_dir,
        epfl_dataset_dir,
        deepglobe_dataset_dir,
        chesa_dataset_dir,
        mass_dataset_dir,
        clahe_segformer_data_transform
        if clahe
        else segformer_feature_extractor,
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

    model = RoadSegformer(
        segformer_ckpt=segformer_base,
        batch_size=batch_size,
        dataloaders=dataloaders,
        metrics=metrics,
        lr=lr,
        metrics_interval=metrics_interval,
        train_dataset_name="ETHZDataset",
        tversky_loss_factor=tversky_loss_factor,
        tversky_alpha=tversky_alpha,
        tversky_beta=tversky_beta,
        focal_loss_factor=focal_loss_factor,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        bce_loss_factor=bce_loss_factor,
    )

    logger = TensorBoardLogger(
        tb_logdir,
        name=experiment_name or "RoadSegformer_ETHZDataset",
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
            args.clahe,
            args.epfl_dataset_dir,
            args.deepglobe_dataset_dir,
            args.chesa_dataset_dir,
            args.mass_dataset_dir,
            args.create_autoencoder_dataset,
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
            args.segformer_base,
            args.metrics_interval,
            args.train_val_split_ratio,
            Path(args.tb_logdir),
            args.early_stop,
            args.ckpt_save_top_k,
            Path(args.ckpt_save_dir),
            args.ckpt_monitor,
            Path(args.resume_checkpoint) if args.resume_checkpoint else None,
            args.clahe,
            args.tversky_loss_factor,
            args.tversky_alpha,
            args.tversky_beta,
            args.focal_loss_factor,
            args.focal_alpha,
            args.focal_gamma,
            args.bce_loss_factor,
        )


if __name__ == "__main__":
    main()
