import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from road_segmentation.model.road_segformer import RoadSegformer
from road_segmentation.model.auto_encoder import AutoEncoder
from road_segmentation.model.unet_simple import UNetSimple

SEGMENTER = {
    "unet": UNetSimple,
    "segformer": RoadSegformer,
}
IMPUTATOR = {
    "bolts_ae": AutoEncoder,
}


class SegImputator(pl.LightningModule):
    def __init__(
        self,
        device: torch.device,
        base_segmenter: pl.LightningModule,
        base_imputer: pl.LightningModule,
        base_segmenter_ckpt: str,
        base_imputer_ckpt: str,
        
    ) -> None:
        super().__init__()
        self.base_segmenter = (
            SEGMENTER[base_segmenter]
            .load_from_checkpoint(base_segmenter_ckpt)
            .to(device)
        )
        self.base_imputer = (
            IMPUTATOR[base_imputer]
            .load_from_checkpoint(base_imputer_ckpt)
            .to(device)
        )
        self.batch_size = self.base_segmenter.batch_size
        

        

    def _step(
        self,
        batch: dict[str, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        images, labels = batch["image"], batch["labels"]
        segments = self.base_segmenter(images)
        logits = self.base_imputer(segments)
        loss = F.cross_entropy(logits, labels)

        return loss

    

    def predict_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        segments = self.base_segmenter.predict_step(batch)
        segments = segments.unsqueeze(1).float()
        logits = self.base_imputer.predict_step(segments)

        return logits

    
