import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt

class PredictionCallback(pl.Callback):
    def __init__(self, output_dir='../test_dataset', every_n_epochs=1):
        super().__init__()
        self.output_dir = output_dir  # Directory to save the output images
        self.every_n_epochs = every_n_epochs  # Save output every N epochs

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Check if the epoch is one of the ones where we want to save the prediction
        if trainer.current_epoch % self.every_n_epochs == 0:
            # Set model to evaluation mode and disable gradient calculation
            pl_module.eval()
            with torch.no_grad():
                # Get the model prediction for the sample input
                sample_prediction = pl_module(batch['image'])[0]
                sample_prediction = sample_prediction.squeeze(0).cpu()

            # Convert the tensor to a numpy array and save it as an image
            prediction_np = sample_prediction.squeeze(0).numpy()
            # Save the prediction using matplotlib
            plt.imshow(prediction_np, cmap='gray')
            plt.title(f'Epoch {trainer.current_epoch}')
            plt.axis('off')
            plt.savefig(f'./predictions/prediction_{trainer.current_epoch}.png')
            plt.close()
            
            plt.imshow(batch['image'][0].squeeze(0).squeeze(0).numpy(), cmap='gray')
            plt.axis('off')
            plt.savefig(f'x.png')
            plt.close()
            
            plt.imshow(batch['labels'][0].squeeze(0).squeeze(0).numpy(), cmap='gray')
            plt.axis('off')
            plt.savefig(f'y.png')
            plt.close()
            
            # Reset model to training mode
            pl_module.train()