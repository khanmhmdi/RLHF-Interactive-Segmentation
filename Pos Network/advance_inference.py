import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# Append project root to system path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Local imports
from Pos_Network.Model import KeyPointEstimator
from dataloader import KeyPointDataset

class PoseEstimationPredictor:
    def __init__(self, 
                 train_data_path: str, 
                 test_data_path: str, 
                 model_load_path: str, 
                 num_keypoints: int = 6, 
                 batch_size: int = 64):
        """
        Initialize the pose estimation predictor.
        
        Args:
            train_data_path (str): Path to training data CSV
            test_data_path (str): Path to test data CSV
            model_load_path (str): Path to pre-trained model weights
            num_keypoints (int, optional): Number of keypoints to estimate. Defaults to 6.
            batch_size (int, optional): Batch size for data loading. Defaults to 64.
        """
        # Define transformations
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        # Create datasets
        self.train_dataset = KeyPointDataset(
            dir=train_data_path,
            transform=self.transform
        )
        self.test_dataset = KeyPointDataset(
            dir=test_data_path,
            transform=self.transform
        )
        
        # Create dataloaders
        self.train_dataloader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4
        )
        self.test_dataloader = DataLoader(
            self.test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4
        )
        
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = KeyPointEstimator(num_keypoints=num_keypoints)
        
        # Load pre-trained weights
        self.model.load_state_dict(torch.load(model_load_path))
        self.model.to(self.device)
        self.model.eval()
        
        # Define loss function
        self.criterion = nn.MSELoss()
    
    def evaluate_model(self) -> float:
        """
        Evaluate the model on the test dataset.
        
        Returns:
            float: Average test loss
        """
        self.model.eval()
        total_test_loss = 0.0
        
        with torch.no_grad():
            for image1, pred_image, coordinates_1, coordinates_minus1 in self.test_dataloader:
                # Move data to device
                img1 = image1.to(self.device)
                img2 = pred_image.to(self.device, dtype=torch.float32)
                keypoints1 = coordinates_1.to(self.device)
                keypointsminus1 = coordinates_minus1.to(self.device)
                
                # Forward pass
                outputs = self.model(img1, img2.unsqueeze(1))
                
                # Compute loss
                loss = self.criterion(
                    outputs, 
                    torch.concat((keypoints1.float(), keypointsminus1.float()), 1)
                )
                
                # Accumulate loss
                total_test_loss += loss.item()
        
        # Compute average loss
        average_test_loss = total_test_loss / len(self.test_dataloader)
        return average_test_loss

def main():
    # Paths - consider using configuration or environment variables
    TRAIN_DATA_PATH = '/home/kasra/Desktop/New Folder/PosEstimation/UNET/Data/val_image_paths.csv'
    TEST_DATA_PATH = '/home/kasra/Desktop/New Folder/PosEstimation/UNET/Data/val_image_paths.csv'
    MODEL_LOAD_PATH = '/home/kasra/Desktop/New Folder/PosEstimation/Pos_Network/model.pth'
    
    # Create predictor and evaluate
    predictor = PoseEstimationPredictor(
        train_data_path=TRAIN_DATA_PATH,
        test_data_path=TEST_DATA_PATH,
        model_load_path=MODEL_LOAD_PATH
    )
    
    # Run evaluation
    avg_test_loss = predictor.evaluate_model()
    print(f"Average Test Loss: {avg_test_loss}")

if __name__ == "__main__":
    main()