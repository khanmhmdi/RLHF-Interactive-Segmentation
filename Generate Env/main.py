"""
Environment generation module for interactive segmentation.

This module handles the generation of training environments for reinforcement learning
based interactive segmentation.
"""

import os
import sys
from typing import List, Tuple, Optional, Union, Any
from argparse import Namespace
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

# Add necessary paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append('/home/kasra/Desktop/New Folder/PosEstimation/SAM/')

from DDPG.Dice import DiceCoefficient
from SAM.segment_anything.predictor_sammed import SammedPredictor
from SAM.segment_anything import sam_model_registry
from Pos_Network.Model import KeyPointEstimator
from Pos_Network.dataloader import KeyPointDataset


# Setup image transformations
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to training size
    transforms.ToTensor(),  # Convert to PyTorch tensor
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet mean
        std=[0.229, 0.224, 0.225]  # ImageNet std
    ),
])

# Create test dataset
test_dataset = KeyPointDataset(
    csv_filepath='/home/kasra/Desktop/Kasra-khan/UNET/Data/test_image_paths.csv',
    mode='test',
    transform=test_transform
)

# Create test dataloader
test_dataloader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4
)

# Initialize model and training components
model = KeyPointEstimator(num_keypoints=5)
model_load_path = '/home/kasra/Desktop/Kasra-khan/Pos_Network/model_15.pth'
model.load_state_dict(torch.load(model_load_path))
model.eval()
model.to(device='cuda')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

class EnvironmentGenerator:
    """Handles the generation of training environments for interactive segmentation."""

    def __init__(self, device: str = 'cuda'):
        """
        Initialize the environment generator.

        Args:
            device: Device to run computations on ('cuda' or 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.setup_transforms()
        self.setup_model()
        self.setup_sam()
        self.dice_calculator = DiceCoefficient(multi_class=False)

    def setup_transforms(self) -> None:
        """Set up image transformations."""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def setup_model(self) -> None:
        """Set up the keypoint estimation model."""
        self.model = KeyPointEstimator(num_keypoints=5)
        model_path = '/home/kasra/Desktop/Kasra-khan/Pos_Network/model_15.pth'
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.model.to(self.device)
        self.criterion = nn.MSELoss()

    def setup_sam(self) -> None:
        """Set up the SAM model."""
        args = Namespace(
            image_size=256,
            encoder_adapter=True,
            sam_checkpoint="/home/kasra/Desktop/Kasra-khan/SAM/Model/sam-med2d_b.pth"
        )
        model_sam = sam_model_registry["vit_b"](args).to(self.device)
        self.predictor = SammedPredictor(model_sam)

    @staticmethod
    def show_mask(mask: np.ndarray, ax: plt.Axes, random_color: bool = False) -> None:
        """Display a segmentation mask on a matplotlib axis.

        Args:
            mask: The segmentation mask to display
            ax: Matplotlib axis object
            random_color: Whether to use random colors for visualization
        """
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30/255, 144/255, 255/255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    @staticmethod
    def show_points(
        coords: np.ndarray,
        labels: np.ndarray,
        ax: plt.Axes,
        marker_size: int = 375
    ) -> None:
        """Display points on a matplotlib axis.

        Args:
            coords: Point coordinates to display
            labels: Labels for the points (0 or 1)
            ax: Matplotlib axis object
            marker_size: Size of the markers
        """
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(
            pos_points[:, 0], pos_points[:, 1],
            color='green', marker='*',
            s=marker_size, edgecolor='white',
            linewidth=1.25
        )
        ax.scatter(
            neg_points[:, 0], neg_points[:, 1],
            color='red', marker='*',
            s=marker_size, edgecolor='white',
            linewidth=1.25
        )

    @staticmethod
    def show_box(box: np.ndarray, ax: plt.Axes) -> None:
        """Display a bounding box on a matplotlib axis.

        Args:
            box: Bounding box coordinates [x0, y0, x1, y1]
            ax: Matplotlib axis object
        """
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle(
            (x0, y0), w, h,
            edgecolor='green',
            facecolor=(0, 0, 0, 0),
            lw=2
        ))

    @staticmethod
    def denormalize(tensor: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
        """Denormalize a tensor using given mean and standard deviation.

        Args:
            tensor: Input tensor to denormalize
            mean: Mean values for each channel
            std: Standard deviation values for each channel

        Returns:
            Denormalized tensor
        """
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor

    def generate_one_state(
        self,
        image1: torch.Tensor,
        pred_image: torch.Tensor,
        outputs: torch.Tensor,
        predictor: SammedPredictor,
        img_in_batch_idx: int,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate one state for the environment.

        Args:
            image1: Input image tensor
            pred_image: Predicted image tensor
            outputs: Model outputs
            predictor: SAM predictor instance
            img_in_batch_idx: Index in the batch
            mask: Optional input mask

        Returns:
            Tuple of masks, scores, and logits
        """
        denormalized_image = self.denormalize(
            image1[img_in_batch_idx].clone(),
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        image = denormalized_image.permute(1, 2, 0).cpu().numpy()

        predictor.set_image(image)

        input_points = outputs[img_in_batch_idx].cpu().numpy()
        input_points[:, 0] *= 240
        input_points[:, 1] *= 240
        input_points = input_points[:, [1, 0]]  # Swap x and y

        input_labels = np.ones(input_points.shape[0], dtype=int)
        predict_kwargs = {
            'point_coords': input_points,
            'point_labels': input_labels,
            'multimask_output': True
        }
        if isinstance(mask, np.ndarray):
            predict_kwargs['mask_input'] = mask

        return predictor.predict(**predict_kwargs)

    def generate_env(self, test_dataloader: DataLoader, max_batches: int = 5) -> None:
        """
        Generate the environment data.

        Args:
            test_dataloader: DataLoader for test data
            max_batches: Maximum number of batches to process
        """
        with torch.no_grad():
            self._process_batches(test_dataloader, max_batches)

    def _process_batches(self, dataloader: DataLoader, max_batches: int) -> None:
        """
        Process batches of data to generate environment.

        Args:
            dataloader: DataLoader instance
            max_batches: Maximum number of batches to process
        """
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Processing batches")):
            if batch_idx > max_batches:
                break

            image1, pred_image, coords_1, coords_minus1, filename, label, low_res_mask = batch_data
            
            # Process valid samples only
            valid_indices = [idx for idx in range(label.shape[0]) if (label[idx] == 1).any()]
            if not valid_indices:
                continue

            # Filter batch data
            filtered_data = self._filter_batch_data(
                valid_indices, image1, pred_image, coords_1, coords_minus1, 
                filename, label, low_res_mask
            )
            
            # Generate and save environment data
            self._generate_environment_data(*filtered_data)

    def _filter_batch_data(self, valid_indices: List[int], *batch_data: Any) -> Tuple[Any, ...]:
        """Filter batch data based on valid indices."""
        filtered = []
        for data in batch_data:
            if isinstance(data, (torch.Tensor, np.ndarray)):
                filtered.append(data[valid_indices])
            elif isinstance(data, list):
                filtered.append([data[i] for i in valid_indices])
            else:
                filtered.append(data)
        return tuple(filtered)

    def _generate_environment_data(self, *batch_data: Any) -> None:
        """Generate environment data from filtered batch data."""
        memory = [[], [], [], []]
        
        for state_idx in range(4):
            self._process_state(state_idx, memory, *batch_data)
        
        self._save_environment_data(memory)

    def _save_environment_data(self, memory: List[List[Any]]) -> None:
        """Save generated environment data."""
        target_directory = Path('/home/kasra/Desktop/New Folder/PosEstimation/Generate Env/env_data')
        target_directory.mkdir(parents=True, exist_ok=True)

        number_of_elements = len(memory[0])
        for n in tqdm(range(number_of_elements), desc="Saving data"):
            nth_elements = [memory[i][n] for i in range(4)]
            
            for i, element in enumerate(nth_elements):
                filename = target_directory / f'{nth_elements[0][0]}_{i}.pkl'
                with open(filename, 'wb') as f:
                    pickle.dump(element, f)


def main():
    """Main function to run the environment generation."""
    # Initialize environment generator
    env_generator = EnvironmentGenerator()

    # Generate environment
    env_generator.generate_env(test_dataloader)


if __name__ == "__main__":
    main()
