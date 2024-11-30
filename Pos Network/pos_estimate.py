import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import sys

# Add parent directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Pos_Network.Model import KeyPointEstimator
from dataloader import KeyPointDataset

# Define transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Create datasets and dataloaders
train_dataset = KeyPointDataset(
    csv_filepath='/home/kasra/Desktop/New Folder/PosEstimation/UNET/Data/test_image_paths.csv',
    mode='test',
    transform=train_transform
)
test_dataset = KeyPointDataset(
    csv_filepath='/home/kasra/Desktop/New Folder/PosEstimation/UNET/Data/train_image_paths.csv',
    mode='train',
    transform=test_transform
)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

# Initialize model, optimizer, and loss function
model = KeyPointEstimator(num_keypoints=5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

num_epochs = 20

# Evaluate model
def evaluate_model():
    model.eval()
    total_test_loss = 0.0
    with torch.no_grad():
        for image1, pred_image, coordinates_1, _ in test_dataloader:
            img1 = image1.to(device)
            img2 = pred_image.to(device, dtype=torch.float32)
            keypoints1 = coordinates_1.to(device)

            outputs = model(img1, img2.unsqueeze(1))
            loss = criterion(outputs, keypoints1)

            total_test_loss += loss.item()
    return total_test_loss / len(test_dataloader)

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (image1, pred_image, coordinates_1, _) in enumerate(train_dataloader):
        img1 = image1.to(device)
        img2 = pred_image.to(device, dtype=torch.float32)
        keypoints1 = coordinates_1.to(device)

        optimizer.zero_grad()
        outputs = model(img1, img2.unsqueeze(1))
        loss = criterion(outputs, keypoints1.float())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:  # Print every 10 batches
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(train_dataloader)}], Loss: {running_loss / 10:.4f}')
            running_loss = 0.0

    # Save model and evaluate every 5 epochs
    if epoch % 5 == 0:
        model_save_path = f'/home/kasra/Desktop/New Folder/PosEstimation/Pos_Network/model_{epoch}.pth'
        torch.save(model.state_dict(), model_save_path)
        print(f'Model saved to {model_save_path}')
        
    test_loss = evaluate_model()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Test Loss: {test_loss:.4f}')

# Final model save
final_model_path = '/home/kasra/Desktop/New Folder/PosEstimation/Pos_Network/model.pth'
torch.save(model.state_dict(), final_model_path)
print(f'Model saved to {final_model_path}')
print('Training finished.')
