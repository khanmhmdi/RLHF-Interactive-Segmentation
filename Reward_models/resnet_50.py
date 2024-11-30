import os
import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim import lr_scheduler

# Dataset definition
class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None, mask_transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.images = os.listdir(images_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.images[idx])

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path.replace('.png', '.jpg')).convert('L')

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask


# Define directories
DATA_DIR = '/home/kasra/Desktop/New Folder/PosEstimation/UNET/Data/'
x_train_dir = os.path.join(DATA_DIR, 'test')
y_train_dir = os.path.join(DATA_DIR, 'test_labels')
x_valid_dir = os.path.join(DATA_DIR, 'train')
y_valid_dir = os.path.join(DATA_DIR, 'train_labels')

# Define transforms
image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

mask_transform = transforms.Compose([
    transforms.ToTensor()
])

# Create datasets and dataloaders
n_cpu = os.cpu_count()

train_dataset = SegmentationDataset(x_train_dir, y_train_dir, transform=image_transform, mask_transform=mask_transform)
valid_dataset = SegmentationDataset(x_valid_dir, y_valid_dir, transform=image_transform, mask_transform=mask_transform)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=n_cpu)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=n_cpu)

# Define the Lightning module
class PetModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs)
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, image):
        image = (image - self.mean) / self.std
        return self.model(image)

    def shared_step(self, batch, stage):
        images, masks = batch
        images = F.interpolate(images, size=(256, 256), mode='bilinear', align_corners=False)
        masks = F.interpolate(masks, size=(256, 256), mode='bilinear', align_corners=False)
        images = (images - self.mean) / self.std
        logits_mask = self(images)
        loss = self.loss_fn(logits_mask, masks)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), masks.long(), mode="binary")
        return {"loss": loss, "tp": tp, "fp": fp, "fn": fn, "tn": tn}

    def shared_epoch_end(self, outputs, stage):
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        self.log(f"{stage}_dataset_iou", dataset_iou, prog_bar=True)

    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch, "train")
        self.training_step_outputs.append(output)
        return output

    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(output)
        return output

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * len(train_dataloader), eta_min=1e-5)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

# Set up logging and callbacks
log_dir = "tensorboard_logs"
checkpoint_dir = "checkpoints"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

logger = TensorBoardLogger(log_dir, name="pet_model")
checkpoint_callback = ModelCheckpoint(
    dirpath=checkpoint_dir,
    filename='pet_model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=-1,
    every_n_epochs=1,
)

# Trainer and model initialization
EPOCHS = 31
model = PetModel("Unet", "efficientnet-b0", in_channels=3, out_classes=1)

trainer = pl.Trainer(
    max_epochs=EPOCHS,
    log_every_n_steps=1,
    callbacks=[checkpoint_callback],
    logger=logger,
)

# Train the model
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=valid_dataloader)
