import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch
import os
import pandas as pd
import numpy as np
import torch
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = CBR(in_channels, 64)
        self.encoder2 = CBR(64, 128)
        self.encoder3 = CBR(128, 256)
        self.encoder4 = CBR(256, 512)
        self.encoder5 = CBR(512, 1024)

        self.pool = nn.MaxPool2d(2)

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = CBR(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = CBR(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = CBR(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = CBR(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))
        enc5 = self.encoder5(self.pool(enc4))

        dec4 = self.upconv4(enc5)
        dec4 = torch.cat((enc4, dec4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((enc3, dec3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((enc2, dec2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((enc1, dec1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.final_conv(dec1)


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
        img_filename = self.images[idx]
        img_path = os.path.join(self.images_dir, img_filename)
        mask_path = os.path.join(self.masks_dir, img_filename)  # Adjusted to match mask file extension

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path.replace('.png', '.jpg')).convert('L')

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask, img_filename


def save_image(image, path):
    """Function to save an image to a specified path."""
    cv2.imwrite(path, image)


import os
import numpy as np
import torch
import h5py
from tqdm import tqdm
import asyncio

def save_to_hdf5(f, filename, pred_image, coord_1, coord_minus1):
    f.create_dataset(f'{filename}_pred_image', data=pred_image, compression='gzip')
    f.create_dataset(f'{filename}_coordinates_1', data=coord_1, compression='gzip')
    f.create_dataset(f'{filename}_coordinates_-1', data=coord_minus1, compression='gzip')

async def evaluate_model_async(model, dataloader, device, output_dir, mode='train', threshold=0.5):
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    # Use synchronous context manager for h5py.File
    f = h5py.File(os.path.join(output_dir, 'results_{}.hdf5'.format(mode)), 'w')

    tasks = []

    with torch.no_grad():
        for images, masks, filenames in tqdm(dataloader):
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            predictions = (outputs >= threshold).float()

            predictions_np = predictions.cpu().numpy()
            masks_np = masks.cpu().numpy()
            diff = masks_np - predictions_np

            filenames = [filename.split('.')[0] for filename in filenames]

            for i in range(predictions_np.shape[0]):
                pred_image = predictions_np[i].squeeze()
                pred_image = (pred_image * 255).astype(np.uint8) if pred_image.max() <= 1 else pred_image.astype(np.uint8)
                coord_1 = np.argwhere(diff[i] == 1)
                coord_minus1 = np.argwhere(diff[i] == -1)

                save_to_hdf5(f, filenames[i], pred_image, coord_1, coord_minus1)

    f.close()  # Ensure the file is properly closed

def evaluate_model(model, dataloader, device, output_dir, mode='train', threshold=0.5):
    asyncio.run(evaluate_model_async(model, dataloader, device, output_dir, mode, threshold))





def generate_prediction(Data_path, output_saving_path, model_path='best_model.pth'):

    x_train_dir = os.path.join(Data_path, 'train')
    y_train_dir = os.path.join(Data_path, 'train_labels')

    x_valid_dir = os.path.join(Data_path, 'val')
    y_valid_dir = os.path.join(Data_path, 'val_labels')

    x_test_dir = os.path.join(Data_path, 'test')
    y_test_dir = os.path.join(Data_path, 'test_labels')

    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = SegmentationDataset(x_train_dir, y_train_dir, transform=image_transform,
                                        mask_transform=mask_transform)
    val_dataset = SegmentationDataset(x_valid_dir, y_valid_dir, transform=image_transform,
                                      mask_transform=mask_transform)
    test_dataset = SegmentationDataset(x_test_dir, y_test_dir, transform=image_transform, mask_transform=mask_transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(in_channels=3, out_channels=1).to(device)

    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Make sure to create the output directory if it does not exist
    # output_dir = 'pred_UNET'
    os.makedirs(output_saving_path, exist_ok=True)

    evaluate_model(model=model, dataloader=test_loader, device=device, output_dir=output_saving_path, mode='test')
    evaluate_model(model, val_loader, device, output_saving_path, 'val')
    # evaluate_model(model, train_loader, device, output_saving_path, 'train')
