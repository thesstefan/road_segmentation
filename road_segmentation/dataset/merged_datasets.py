from pathlib import Path

from road_segmentation.dataset.ethz_cil_dataset import ETHZDataset
from road_segmentation.dataset.epfl_dataset import EPFLDataset
from road_segmentation.dataset.deepglobe_dataset import DeepGlobeDataset
from road_segmentation.dataset.chesa_dataset import ChesaDataset
from road_segmentation.dataset.mass_dataset import MassDataset

from torch.utils.data import Dataset, ConcatDataset




def get_datasets(
    ethz_dataset_dir: str,
    epfl_dataset_dir: str | None,
    deepglobe_dataset_dir: str | None,
    chesa_dataset_dir: str | None,
    mass_dataset_dir: str | None,
    transform,
) -> Dataset:
    
    datasets = []
    
    ethz_dataset = ETHZDataset.train_dataset(
        Path(f"{Path(ethz_dataset_dir)}/training"),
        transform=transform,
    )

    datasets.append(ethz_dataset)

    if epfl_dataset_dir:
        epfl_dataset = EPFLDataset.train_dataset(
            Path(f"{Path(epfl_dataset_dir)}/training"),
            transform=transform,
        )
        datasets.append(epfl_dataset)
        
    if deepglobe_dataset_dir:
        deepglobe_dataset = DeepGlobeDataset.train_dataset(
            Path(f"{Path(deepglobe_dataset_dir)}/train_mod"),
            transform=transform,
        )
        datasets.append(deepglobe_dataset)

    if chesa_dataset_dir:
        chesa_dataset = ChesaDataset.train_dataset(
            Path(f"{Path(chesa_dataset_dir)}/data"),
            transform=transform,
        )
        datasets.append(chesa_dataset)
        
        
    if mass_dataset_dir:
        mass_dataset = MassDataset.train_dataset(
            Path(f"{Path(mass_dataset_dir)}/train_mod"),
            transform=transform,
        )
        datasets.append(mass_dataset)
        
    
    
    return ConcatDataset(datasets)