import argparse
import logging
from pathlib import Path

import lightning.pytorch as pl  # type: ignore[import]
import torch

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
from road_segmentation.model.road_segformer import (
    RoadSegformer,
    segformer_feature_extractor,
)
from road_segmentation.model.seg_imputator import SegImputator
from road_segmentation.model.unet_simple import UNetSimple, unet_transforms
from road_segmentation.utils.prediction_writer import OnBatchImageOutputWriter

import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()

parser.add_argument("--base_segmenter", type=str, default="segformer")
parser.add_argument("--imputer", type=str, default="bolts_ae")
parser.add_argument("--base_segmenter_ckpt", type=str, required=True)
parser.add_argument("--imputer_ckpt", type=str, required=True)

parser.add_argument("--dataset_dir", type=str, required=True)

parser.add_argument("--prediction_output_dir", type=str, required=True)


TRANSFORMS = {
    "unet": (unet_transforms),
    "segformer": (segformer_feature_extractor),
}


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
    base_segmenter: str,
    imputer: str,
    base_segmenter_ckpt: Path,
    imputer_ckpt: Path,
    dataset_dir: Path,
    prediction_output_dir: Path,
) -> None:
    model = SegImputator(
        device, base_segmenter, imputer, base_segmenter_ckpt, imputer_ckpt
    )

    predict_dataset = ETHZDataset.test_dataset(
        dataset_dir,
        transform=TRANSFORMS[base_segmenter],
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


def main() -> None:

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    predict(
        device,
        args.base_segmenter,
        args.imputer,
        Path(args.base_segmenter_ckpt),
        Path(args.imputer_ckpt),
        Path(args.dataset_dir),
        Path(args.prediction_output_dir),
    )


if __name__ == "__main__":
    main()
