import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import h5py
import os

# class KeyPointDataset(Dataset):
#     def __init__(self, csv_filepath,mode='train', transform=None):
#         self.df = pd.read_csv(csv_filepath)
#         self.hdf5_filepath = '/home/kasra/Desktop/New Folder/External PosEstimation Files/pred_test/results_{}.hdf5'.format(mode)
#         self.transform = transform
#         self.h5_file = h5py.File(self.hdf5_filepath, 'r')
# 
#     def __len__(self):
#         return len(self.df)
# 
#     def __getitem__(self, idx):
#         # Retrieve filename from DataFrame
#         input_image_dir = self.df.iloc[idx, 0]
#         filename = os.path.basename(input_image_dir).split('.')[0]
# 
#         # Load image and apply transformation
#         image1 = Image.open(input_image_dir).convert('RGB')
#         if self.transform:
#             image1 = self.transform(image1)
# 
#         # Load datasets from HDF5
#         pred_image = self.h5_file[f'{filename}_pred_image'][:]
#         coordinates_1 = self.h5_file[f'{filename}_coordinates_1'][:,1:]
#         coordinates_minus1 = self.h5_file[f'{filename}_coordinates_-1'][:,1:]
# 
#         num_points = 5  # Number of points to sample
# 
#         def sample_and_pad(coords, num_points, seed):
#             np.random.seed(seed)
#             n = coords.shape[0]
# 
#             if n >= num_points:
#                 indices = np.random.choice(n, num_points, replace=False)
#                 sampled_coords = coords[indices]
#             else:
#                 pad_width = num_points - n
#                 sampled_coords = np.pad(coords, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)
# 
#             return sampled_coords
# 
#         # Ensure both coordinates_1 and coordinates_minus1 have at least 10 points
#         coordinates_1 = sample_and_pad(coordinates_1, num_points, seed=idx) / 240
#         coordinates_minus1 = sample_and_pad(coordinates_minus1, num_points, seed=idx) / 240
# 
#         # Convert pred_image to tensor
#         pred_image = torch.from_numpy(pred_image).float()
# 
#         return image1, pred_image, coordinates_1, coordinates_minus1
# 
#     def __del__(self):
#         # Close the HDF5 file when the dataset object is deleted
#         self.h5_file.close()

#In this modified code the dataloader return the name of the images too.
class KeyPointDataset(Dataset):
    def __init__(self, csv_filepath, mode='train', transform=None):
        self.df = pd.read_csv(csv_filepath)
        self.hdf5_filepath = '/home/kasra/Desktop/New Folder/External PosEstimation Files/pred_test/results_{}.hdf5'.format(mode)
        self.transform = transform
        self.h5_file = h5py.File(self.hdf5_filepath, 'r')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Retrieve filename from DataFrame
        input_image_dir = self.df.iloc[idx, 0]
        filename = os.path.basename(input_image_dir).split('.')[0]

        # Load image and apply transformation
        image1 = Image.open(input_image_dir).convert('RGB')
        if self.transform:
            image1 = self.transform(image1)

        # Load datasets from HDF5
        pred_image = self.h5_file[f'{filename}_pred_image'][:]
        coordinates_1 = self.h5_file[f'{filename}_coordinates_1'][:, 1:]
        coordinates_minus1 = self.h5_file[f'{filename}_coordinates_-1'][:, 1:]

        num_points = 5  # Number of points to sample

        def sample_and_pad(coords, num_points, seed):
            np.random.seed(seed)
            n = coords.shape[0]

            if n >= num_points:
                indices = np.random.choice(n, num_points, replace=False)
                sampled_coords = coords[indices]
            else:
                pad_width = num_points - n
                sampled_coords = np.pad(coords, ((0, pad_width), (0, 0)), mode='constant', constant_values=0)

            return sampled_coords

        # Ensure both coordinates_1 and coordinates_minus1 have at least 10 points
        coordinates_1 = sample_and_pad(coordinates_1, num_points, seed=idx) / 240
        coordinates_minus1 = sample_and_pad(coordinates_minus1, num_points, seed=idx) / 240

        # Convert pred_image to tensor
        pred_image = torch.from_numpy(pred_image).float()

        # Return the image data along with the filename
        return image1, pred_image, coordinates_1, coordinates_minus1, filename

    def __del__(self):
        # Close the HDF5 file when the dataset object is deleted
        self.h5_file.close()
