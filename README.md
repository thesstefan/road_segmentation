# The Cartographers - Road Segmentation

This project provides a solution for road segmentation using various models such as U-Net, U-Mamba, and RoadSegFormer.

## Prerequisites

Ensure you have Docker installed on your machine. If not, follow the instructions on the [official Docker website](https://docs.docker.com/get-docker/).

## Getting Started

1. Clone this repository to your local machine.
2. Navigate to the directory of the cloned repository.
3. Build and start the Docker container using the following commands:

```bash
docker build -t cartographers-road-segmentation .
docker run -d -p 8080:8080 cartographers-road-segmentation
```

## Datasets

You can download the training datasets from the following sources:

- [ETHZ CIL Road Segmentation](https://www.kaggle.com/competitions/ethz-cil-road-segmentation-2024/data)
- [EPFL Road Segmentation](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation#dataset)
- [Massachusetts Road Dataset](https://www.kaggle.com/datasets/balraj98/massachusetts-roads-dataset)
- [DeepGlobe Road Extraction Dataset](https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset)
- [ChesapeakeRSC](https://huggingface.co/datasets/torchgeo/ChesapeakeRSC/tree/main)

## Training

To train the segmentation models (U-Net, U-Mamba, or RoadSegFormer), run the specific command for your desired model. Replace `{Dataset Folder}` with the appropriate dataset directory.

### RoadSegformer

```bash
road_segformer train --ckpt_save_dir model_ckpt/ --dataset_dir road_segmentation/dataset/{ETHZ CIL Road Segmentation Folder}  --epfl_dataset_dir road_segmentation/dataset/{EPFL Road Segmentation Folder} --deepglobe_dataset_dir road_segmentation/dataset/{DeepGlobe Folder} --chesa_dataset_dir road_segmentation/dataset/{ChesapeakeRSC Folder} --mass_dataset_dir road_segmentation/dataset/{Massachusetts Road Dataset folder}
```

(Note: In order to play with the combination of Tversky, Focal and Binary Cross Entropy loss you can specify the args: --tversky_loss_facto, --tversky_alpha, --tversky_beta, --focal_loss_factor, --focal_alpha, --focal_gamma, --bce_loss_factor )

### U-Mamba

(Note: You need to Cuda to run U-Mamba, make sure you have access to a GPU and CUDA installed)

```bash
umamba train --ckpt_save_dir model_ckpt/ --dataset_dir road_segmentation/dataset/{ETHZ CIL Road Segmentation Folder}  --epfl_dataset_dir road_segmentation/dataset/{EPFL Road Segmentation Folder} --deepglobe_dataset_dir road_segmentation/dataset/{DeepGlobe Folder} --chesa_dataset_dir road_segmentation/dataset/{ChesapeakeRSC Folder} --mass_dataset_dir road_segmentation/dataset/{Massachusetts Road Dataset folder}
```

### U-Net

```bash
unet train --ckpt_save_dir model_ckpt/ --dataset_dir road_segmentation/dataset/{ETHZ CIL Road Segmentation Folder}  --epfl_dataset_dir road_segmentation/dataset/{EPFL Road Segmentation Folder} --deepglobe_dataset_dir road_segmentation/dataset/{DeepGlobe Folder} --chesa_dataset_dir road_segmentation/dataset/{ChesapeakeRSC Folder} --mass_dataset_dir road_segmentation/dataset/{Massachusetts Road Dataset folder}
```

## Auto encoder refined predictions

To refine your predictions, you can train an autoencoder (U-Net) on top of the segmentation model.

1. Generate a new dataset containing the predictions from the dataset we trained the segmentation model together with the ground truth labels:

```bash
segformer predict --model_ckpt_path model_ckpt/{your_model_ckpt.ckpt} --ethz_input_dir road_segmentation/dataset/{ETHZ CIL Road Segmentation Folder}  --epfl_dataset_dir road_segmentation/dataset/{EPFL Road Segmentation Folder} --deepglobe_dataset_dir road_segmentation/dataset/{DeepGlobe Folder} --chesa_dataset_dir road_segmentation/dataset/{ChesapeakeRSC Folder} --mass_dataset_dir road_segmentation/dataset/{Massachusetts Road Dataset folder} --prediction_output_dir road_segmentation/dataset/AE_dataset/ --create_autoencoder_dataset True
```

(Note analogous for U-Net and U-Mamba, replace segformer with unet or umamba respectively)

2. Train the autoencoder (U-Net) on these predictions:

```bash
unet_ae train --dataset_dir road_segmentation/dataset/AE_dataset/ --ckpt_save_dir model_ckpt/AE
```

## Prediction

To run prediction on the ETHZ CIL Road Segmentation test dataset:

```bash
segformer predict --model_ckpt_path model_ckpt/{your_segmentation_model_ckpt.ckpt}  --ethz_input_dir road_segmentation/dataset/{ETHZ CIL Road Segmentation Folder} --prediction_output_dir predictions
```

(Note analogous for U-Net and U-Mamba, replace segformer with unet or umamba respectively)

If you have trained the AE (U-Net) on top of the segmentation model and you would like to refine your prediction:

```bash
unet_ae predict --model_ckpt_path model_ckpt/AE/{your_AE_model_ckpt.ckpt} --input_dir predictions --prediction_output_dir AE_predictions
```
