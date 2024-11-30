import os
import sys
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Append parent directory to system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Pos_Network.Model import KeyPointEstimator

class KeyPointInference:
    """
    A class for performing keypoint inference on images.
    
    Attributes:
        model (torch.nn.Module): Trained keypoint estimation model
        device (torch.device): Device to run inference on (CPU/GPU)
        transform (transforms.Compose): Image preprocessing transformations
    """

    def __init__(self, model_path, num_keypoints=5, device=None):
        """
        Initialize the KeyPointInference class.
        
        Args:
            model_path (str): Path to the saved model weights
            num_keypoints (int, optional): Number of keypoints to estimate. Defaults to 5.
            device (str or torch.device, optional): Device to run inference on
        """
        # Initialize model
        self.model = KeyPointEstimator(num_keypoints=num_keypoints)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        # Set device
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Set image transformation
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def infer(self, image, pred_image):
        """
        Run inference on an image.
        
        Args:
            image (PIL.Image or torch.Tensor): Input image
            pred_image (np.ndarray or torch.Tensor): Predicted image mask
        
        Returns:
            np.ndarray: Estimated keypoints
        """
        # Preprocess image
        if isinstance(image, Image.Image):
            image = self.transform(image).unsqueeze(0)
        elif not isinstance(image, torch.Tensor):
            raise ValueError("Image must be a PIL Image or torch Tensor")

        # Preprocess prediction mask
        if isinstance(pred_image, np.ndarray):
            pred_image = torch.from_numpy(pred_image).float()
        elif not isinstance(pred_image, torch.Tensor):
            raise ValueError("Prediction mask must be a NumPy array or torch Tensor")

        # Move tensors to device
        image = image.to(self.device)
        pred_image = pred_image.unsqueeze(0).unsqueeze(1).to(self.device, dtype=torch.float32)

        # Run inference
        with torch.no_grad():
            outputs = self.model(image, pred_image)

        # Convert outputs to numpy and reshape
        return outputs.cpu().numpy().reshape(-1, 2)

    @staticmethod
    def visualize_keypoints(image, keypoints, title=None):
        """
        Visualize keypoints on an image.
        
        Args:
            image (PIL.Image or np.ndarray): Input image
            keypoints (np.ndarray): Array of keypoint coordinates
            title (str, optional): Plot title
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        
        for x, y in keypoints:
            plt.scatter(x * 240, y * 240, s=20, marker='o', c='red', edgecolor='black')
        
        if title:
            plt.title(title)
        plt.axis('off')
        plt.show()

def main():
    """
    Main function to demonstrate keypoint inference.
    """
    # Configuration
    MODEL_PATH = '/home/kasra/Desktop/New Folder/PosEstimation/Pos_Network/model_15.pth'
    HDF5_FILEPATH = '/home/kasra/Desktop/New Folder/External PosEstimation Files/pred_test/results_train.hdf5'
    IMAGE_DIR = '/home/kasra/Desktop/New Folder/PosEstimation/UNET/Data/train'

    # Sample image IDs for demonstration
    image_ids = ['00604_t1_111', '01137_t1_7', '00512_t1_40']

    # Initialize inference
    inference = KeyPointInference(MODEL_PATH)

    # Open HDF5 file
    with h5py.File(HDF5_FILEPATH, 'r') as h5_file:
        for image_id in image_ids:
            # Load image and prediction mask
            image_path = os.path.join(IMAGE_DIR, f'{image_id}.png')
            image = Image.open(image_path).convert('RGB')
            pred_image = h5_file[f'{image_id}_pred_image'][:]

            # Perform inference
            keypoints = inference.infer(image, pred_image)

            # Visualize results
            inference.visualize_keypoints(image, keypoints, title=f'Keypoints for {image_id}')

if __name__ == "__main__":
    main()