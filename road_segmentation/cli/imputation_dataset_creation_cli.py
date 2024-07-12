import argparse
from road_segmentation.dataset.merged_datasets import get_datasets
import torch
from torch.utils.data import DataLoader
from PIL import Image
import lightning.pytorch as pl  # type: ignore[import]
import os
from road_segmentation.model.road_segformer import (
    RoadSegformer,
    segformer_feature_extractor,
)
from road_segmentation.model.unet_simple import UNetSimple, unet_transforms
from road_segmentation.utils.prediction_writer import OnBatchImageOutputWriter

from pathlib import Path

MODELS = {
    "segformer": RoadSegformer,
    "unet": UNetSimple,
}
TRANSFORMS = {
    "segformer": segformer_feature_extractor,
    "unet": unet_transforms,
}
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--dataset_dir", type=str, required=True)
parser.add_argument("--model_ckpt_path", type=str, required=True)
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--epfl_dataset_dir", type=str, default=None)
parser.add_argument("--deepglobe_dataset_dir", type=str, default=None)
parser.add_argument("--chesa_dataset_dir", type=str, default=None)


def create_imputation_dataset(
    model: str,
    dataset_dir: Path,
    model_ckpt_path: str,
    output_dir: Path,
    device: torch.device,
    epfl_dataset_dir: str | None,
    deepglobe_dataset_dir: str | None,
    chesa_dataset_dir: str | None,
) -> None:
    dataset = get_datasets(
        dataset_dir,
        epfl_dataset_dir,
        deepglobe_dataset_dir,
        chesa_dataset_dir,
        TRANSFORMS[model],
    )

    model = RoadSegformer.load_from_checkpoint(
        checkpoint_path=model_ckpt_path,
    ).to(device)

    dataloader = DataLoader(
        dataset,
        batch_size=model.batch_size,
        shuffle=False,
    )
    predictor = pl.Trainer(
        accelerator=str(device),
        logger=False,
        callbacks=[OnBatchImageOutputWriter(output_dir / "segments")],
    )
    predictor.predict(
        model,
        dataloaders=dataloader,
        return_predictions=False,
    )

    output_dir = Path(output_dir)
    groundtruth_dir = output_dir / "groundtruth"
    os.makedirs(groundtruth_dir, exist_ok=True)

    for sample in dataset:
        label = sample["labels"]
        label_array = label.numpy()
        label_array = (label_array * 255).astype("uint8")
        image = Image.fromarray(label_array)
        image.save(groundtruth_dir / sample["image_filename"])


def main() -> None:
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    create_imputation_dataset(
        args.model,
        Path(args.dataset_dir),
        Path(args.model_ckpt_path),
        Path(args.output_dir),
        device,
        args.epfl_dataset_dir,
        args.deepglobe_dataset_dir,
        args.chesa_dataset_dir,
    )


if __name__ == "__main__":
    main()
