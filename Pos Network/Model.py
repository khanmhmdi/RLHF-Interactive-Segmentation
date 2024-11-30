import torch
import torch.nn as nn
import torch.nn.functional as F

class KeyPointEstimator(nn.Module):
    def __init__(self, num_keypoints=10):
        super(KeyPointEstimator, self).__init__()
        self.num_keypoints = num_keypoints

        # Convolutional layers for the first image
        self.conv1_1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Convolutional layers for the second image
        self.conv2_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv2_3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Fully connected layers
        self.fc1 = nn.Linear(215552, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_keypoints * 2)

    def forward(self, img1, img2):
        # Process the first image
        x1 = F.relu(self.conv1_1(img1))
        x1 = F.max_pool2d(x1, kernel_size=2, stride=2)
        x1 = F.relu(self.conv1_2(x1))
        x1 = F.max_pool2d(x1, kernel_size=2, stride=2)
        x1 = F.relu(self.conv1_3(x1))
        x1 = F.max_pool2d(x1, kernel_size=2, stride=2)

        # Process the second image
        x2 = F.relu(self.conv2_1(img2))
        x2 = F.max_pool2d(x2, kernel_size=2, stride=2)
        x2 = F.relu(self.conv2_2(x2))
        x2 = F.max_pool2d(x2, kernel_size=2, stride=2)
        x2 = F.relu(self.conv2_3(x2))
        x2 = F.max_pool2d(x2, kernel_size=2, stride=2)

        # Flatten the feature maps
        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)

        # Concatenate the features from both images
        x = torch.cat((x1, x2), dim=1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        keypoints = self.fc3(x).view(img1.size(0), self.num_keypoints, 2)

        return keypoints
