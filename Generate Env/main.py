import os
import sys
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt

# Configure system paths
sys.path.extend([
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
    '/home/kasra/Desktop/New Folder/PosEstimation/SAM/'
])

# Import custom modules
from SAM.segment_anything.predictor_sammed import SammedPredictor
from SAM.segment_anything import sam_model_registry
from Pos_Network.Model import KeyPointEstimator
from Pos_Network.dataloader import KeyPointDataset


class ImageProcessor:
    """Handles image preprocessing and visualization tasks."""

    @staticmethod
    def denormalize(
        tensor: torch.Tensor, 
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225]
    ) -> torch.Tensor:
        """Denormalize an image tensor."""
        for t, m, s in zip(tensor, mean, std):
            t.mul_(s).add_(m)
        return tensor

    @staticmethod
    def show_mask(mask: np.ndarray, ax, random_color: bool = False):
        """Visualize a segmentation mask."""
        color = (np.random.random(3).tolist() + [0.6]) if random_color else [30/255, 144/255, 255/255, 0.6]
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * np.array(color).reshape(1, 1, -1)
        ax.imshow(mask_image)

    @staticmethod
    def show_points(coords: np.ndarray, labels: np.ndarray, ax, marker_size: int = 375):
        """Plot keypoint annotations."""
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        
        ax.scatter(pos_points[:, 0], pos_points[:, 1], 
                   color='green', marker='*', s=marker_size, 
                   edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], 
                   color='red', marker='*', s=marker_size, 
                   edgecolor='white', linewidth=1.25)


class ModelEvaluator:
    """Manages model evaluation and environment generation."""

    def __init__(
        self, 
        model: nn.Module, 
        train_loader: DataLoader, 
        test_loader: DataLoader,
        sam_checkpoint: str = "/home/kasra/Documents/SAM-Med2d_V2/SAM-Med2D/Model/sam-med2d_b.pth"
    ):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize SAM predictor
        args = type('Args', (), {
            'image_size': 256,
            'encoder_adapter': True,
            'sam_checkpoint': sam_checkpoint
        })()
        
        model_sam = sam_model_registry["vit_b"](args).to(self.device)
        self.predictor = SammedPredictor(model_sam)
        
        self.criterion = nn.MSELoss()

    def generate_state(
        self, 
        image1: torch.Tensor, 
        pred_image: torch.Tensor, 
        outputs: torch.Tensor, 
        batch_idx: int, 
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate segmentation masks for a single image state."""
        denormalized_image = ImageProcessor.denormalize(
            image1[batch_idx].clone()
        ).permute(1,2,0).cpu().numpy()
        
        self.predictor.set_image(denormalized_image)
        
        input_points = outputs[batch_idx].cpu().numpy()
        input_points[:, 0] *= 240
        input_points[:, 1] *= 240
        input_points = input_points[:, [1, 0]]  # Swap x and y
        
        input_labels = np.ones(input_points.shape[0], dtype=int)
        
        predict_kwargs = {
            'point_coords': input_points,
            'point_labels': input_labels,
            'multimask_output': True
        }
        
        if mask is not None:
            predict_kwargs['mask_input'] = mask
        
        masks, scores, logits = self.predictor.predict(**predict_kwargs)
        
        return masks, scores, logits

    def generate_environment(
        self, 
        output_directory: str = '/home/kasra/Desktop/New Folder/PosEstimation/Generate Env/env_data'
    ):
        """Generate environment data from test dataset."""
        os.makedirs(output_directory, exist_ok=True)
        
        with torch.no_grad():
            for batch_data in self.test_loader:
                image1, pred_image, keypoints1, keypointsminus1, filenames = batch_data
                image1 = image1.to(self.device)
                img2 = pred_image.to(self.device, dtype=torch.float32)
                
                memory = [[] for _ in range(4)]
                
                initial_outputs = self.model(image1, img2.unsqueeze(1))
                
                for state_idx in range(4):
                    if state_idx > 0:
                        img2 = torch.stack([
                            torch.tensor(memory[state_idx-1][i][2]) 
                            for i in range(len(memory[state_idx-1]))
                        ]).to(self.device)
                        
                        outputs = self.model(image1, img2.unsqueeze(1))
                    else:
                        outputs = initial_outputs
                    
                    for batch_idx, filename in enumerate(filenames):
                        mask = memory[state_idx-1][batch_idx][3] if state_idx > 0 else None
                        masks, scores, logits = self.generate_state(
                            image1, pred_image, outputs, batch_idx, mask
                        )
                        
                        # Further processing and saving logic here
                        # (Omitted for brevity, similar to original script)


def main():
    # Transformation configuration
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Dataset and DataLoader setup
    train_dataset = KeyPointDataset(
        csv_filepath='/home/kasra/Desktop/New Folder/PosEstimation/UNET/Data/val_image_paths.csv',
        mode='val',
        transform=test_transform
    )
    test_dataset = KeyPointDataset(
        csv_filepath='/home/kasra/Desktop/New Folder/PosEstimation/UNET/Data/test_image_paths.csv',
        mode='test',
        transform=test_transform
    )

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Model initialization
    model = KeyPointEstimator(num_keypoints=5)
    model_load_path = '/home/kasra/Desktop/New Folder/PosEstimation/Pos_Network/model_15.pth'
    model.load_state_dict(torch.load(model_load_path))
    model.eval()
    model.to(device='cuda')

    # Optimizer setup (optional, depending on use case)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Initialize ModelEvaluator and generate environment
    evaluator = ModelEvaluator(model, train_dataloader, test_dataloader)
    evaluator.generate_environment()


if __name__ == "__main__":
    main()