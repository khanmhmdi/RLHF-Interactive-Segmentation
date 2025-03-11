"""
Deep Deterministic Policy Gradients (DDPG) implementation with PyTorch.
Original paper: https://arxiv.org/abs/1509.02971
"""

import argparse
import os
import pickle
import random
import sys
from itertools import count
from typing import List, Tuple, Optional

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.distributions import Normal
from tqdm import tqdm

# Add the directory containing Pos_Network to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Pos_Network.Model import KeyPointEstimator
from SAM.segment_anything import sam_model_registry
from SAM.segment_anything.predictor_sammed import SammedPredictor

# Constants
SOURCE_DIR = "/home/kasra/Desktop/Kasra-khan/"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MIN_VALUE = torch.tensor(1e-7).float().to(DEVICE)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument("--env_name", default="Pendulum-v1")
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--target_update_interval', default=1, type=int)
    parser.add_argument('--test_iteration', default=10, type=int)
    parser.add_argument('--learning_rate', default=1e-4, type=float)
    parser.add_argument('--gamma', default=0.99, type=int)
    parser.add_argument('--capacity', default=1000000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--seed', default=False, type=bool)
    parser.add_argument('--random_seed', default=9527, type=int)
    parser.add_argument('--sample_frequency', default=2000, type=int)
    parser.add_argument('--render', default=False, type=bool)
    parser.add_argument('--log_interval', default=50, type=int)
    parser.add_argument('--load', default=False, type=bool)
    parser.add_argument('--render_interval', default=100, type=int)
    parser.add_argument('--exploration_noise', default=0.1, type=float)
    parser.add_argument('--max_episode', default=100000, type=int)
    parser.add_argument('--print_log', default=5, type=int)
    parser.add_argument('--update_iteration', default=200, type=int)
    return parser.parse_args()

def visualize_points_on_image(
    image_batch: torch.Tensor,
    points: torch.Tensor,
    batch_index: int,
    marker: str = 'ro',
    figsize: Tuple[int, int] = (5, 5)
) -> None:
    """
    Visualize points on a specific image in the batch.

    Args:
        image_batch: Batch of images (batch_size, channels, height, width)
        points: Points to plot
        batch_index: Index of image in batch to display
        marker: Marker style for points
        figsize: Figure size for display
    """
    image = image_batch[batch_index].cpu()
    image = image.permute(1, 2, 0).numpy()

    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis('off')

    if isinstance(points, torch.Tensor):
        points = points.cpu().numpy()
    elif isinstance(points, list):
        points = np.array(points)

    for point in points[batch_index]:
        x, y = point
        plt.plot(y, x, marker, markersize=5)

    plt.show()

def dice_coefficient_torch(
    pred: torch.Tensor,
    label: torch.Tensor,
    epsilon: float = 1e-6
) -> torch.Tensor:
    """
    Compute Dice coefficient for segmentation tasks.

    Args:
        pred: Predicted segmentation mask
        label: Ground truth segmentation mask
        epsilon: Small constant to avoid division by zero

    Returns:
        Dice coefficient averaged over batch
    """
    if pred.shape != label.shape:
        raise ValueError("Shape mismatch: pred and label must have the same shape.")

    pred = pred.reshape(pred.size(0), -1)
    label = label.reshape(label.size(0), -1)

    intersection = (pred * label.to(DEVICE)).sum(dim=1)
    union = pred.sum(dim=1) + label.to(DEVICE).sum(dim=1)

    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice

class ReplayBuffer:
    """Experience replay buffer for DDPG."""

    def __init__(self, env_path: str):
        """Initialize replay buffer with environment data path."""
        self.env_path = env_path
        files = os.listdir(self.env_path)
        self.pkl_file_paths = [
            os.path.join(self.env_path, file)
            for file in files
            if file.endswith('.pkl') and '3' not in file
        ]

    def load_pkl_file(self, pkl_path: str):
        """Load pickle file data."""
        with open(pkl_path, 'rb') as file:
            return pickle.load(file)

    def get_next_iteration_paths(self, selected_paths: List[str]) -> List[str]:
        """Get paths for next iteration based on selected paths."""
        next_paths = []
        for path in selected_paths:
            parts = path.split('_')
            iteration_and_extension = parts[-1].split('.')
            current_iteration = int(iteration_and_extension[0])
            next_iteration = current_iteration + 1
            new_iteration_part = f"{next_iteration}.pkl"
            next_path = '_'.join(parts[:-1] + [new_iteration_part])
            next_paths.append(next_path)
        return next_paths

    def sample(self, batch_size: int) -> Tuple:
        """Sample batch of experiences from buffer."""
        # Initialize data containers
        (image_name, input_image, pred_image, logits, input_points,
         padded_mask, scores, label, low_res_mask) = [], [], [], [], [], [], [], [], []
        
        (image_name_plus_1, input_image_plus_1, pred_image_plus_1,
         logits_plus_1, input_points_plus_1, padded_mask_plus_1,
         scores_plus_1, label_plus_1, low_res_mask_plus_1) = [], [], [], [], [], [], [], [], []

        selected_paths = random.sample(self.pkl_file_paths, batch_size)
        next_iteration_paths = self.get_next_iteration_paths(selected_paths)

        loaded_state_i = [self.load_pkl_file(path) for path in selected_paths]
        loaded_state_i_plus_1 = [self.load_pkl_file(path) for path in next_iteration_paths]

        # Populate current state data
        for state in loaded_state_i:
            (image_name.append(state[0]), input_image.append(state[1]),
             pred_image.append(state[2]), logits.append(state[3]),
             input_points.append(state[4]), padded_mask.append(state[5]),
             scores.append(state[6]), label.append(state[7]),
             low_res_mask.append(state[8]))

        # Populate next state data
        for state in loaded_state_i_plus_1:
            (image_name_plus_1.append(state[0]), input_image_plus_1.append(state[1]),
             pred_image_plus_1.append(state[2]), logits_plus_1.append(state[3]),
             input_points_plus_1.append(state[4]), padded_mask_plus_1.append(state[5]),
             scores_plus_1.append(state[6]), label_plus_1.append(state[7]),
             low_res_mask_plus_1.append(state[8]))

        return (image_name, input_image, pred_image, logits, input_points,
                padded_mask, scores, label, low_res_mask,
                image_name_plus_1, input_image_plus_1, pred_image_plus_1,
                logits_plus_1, input_points_plus_1, padded_mask_plus_1,
                scores_plus_1, label_plus_1, low_res_mask_plus_1)

class Actor(nn.Module):
    """Neural network model for the actor in DDPG."""

    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        """Initialize the actor network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            max_action: Maximum action value
        """
        super(Actor, self).__init__()
        self.model = KeyPointEstimator(num_keypoints=5)
        self.model_load_path = SOURCE_DIR + 'PosEstimation/Pos_Network/model_15.pth'
        self.model.load_state_dict(torch.load(self.model_load_path))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.model(x)


class Critic(nn.Module):
    """Neural network model for the critic in DDPG."""

    def __init__(self, state_dim: int, action_dim: int):
        """Initialize the critic network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
        """
        super(Critic, self).__init__()

        args = argparse.Namespace()
        args.image_size = 256
        args.encoder_adapter = True
        args.sam_checkpoint = "/home/kasra/sam-med2d_b.pth"
        
        self.model_sam = sam_model_registry["vit_b"](args).to(DEVICE)
        self.predictor = SammedPredictor(self.model_sam)

    def forward(self, image: torch.Tensor, prompt: torch.Tensor, mask: torch.Tensor) -> Tuple[List, List]:
        """Forward pass through the network.
        
        Args:
            image: Input image tensor
            prompt: Input prompt tensor
            mask: Input mask tensor
            
        Returns:
            Tuple of masks and scores
        """
        all_masks = []
        all_scores = []
        image = image.permute(0, 3, 2, 1)
        
        for batch_idx in tqdm(range(image.shape[0])):
            img = image[batch_idx].to(DEVICE)
            input_points = prompt[batch_idx]

            input_points[:, 0] *= 240
            input_points[:, 1] *= 240

            input_labels = torch.from_numpy(np.ones(input_points.shape[0], dtype=int)).to(DEVICE)
            if isinstance(input_points, np.ndarray):
                input_points = torch.from_numpy(input_points).to(DEVICE)

            img = cv2.cvtColor(img.detach().cpu().numpy(), cv2.COLOR_BGR2RGB)
            input_image_torch = torch.as_tensor(img, device=DEVICE)
            transformed_image = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

            self.predictor.set_image(input_image_torch.detach().cpu().numpy())

            masks, scores_point, logits = self.predictor.predict(
                point_coords=input_points.detach().cpu().numpy(),
                point_labels=input_labels.detach().cpu().numpy(),
                box=None,
                multimask_output=True,
            )

        return all_masks, all_scores


class DDPG:
    """Deep Deterministic Policy Gradient algorithm implementation."""

    def __init__(self, state_dim: int, action_dim: int, max_action: float):
        """Initialize DDPG agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            max_action: Maximum action value
        """
        actor_model_load_path = SOURCE_DIR + 'Pos_Network/model_15.pth'

        # Initialize actor networks
        self.actor = KeyPointEstimator(num_keypoints=5)
        self.actor.load_state_dict(torch.load(actor_model_load_path))
        self.actor_target = KeyPointEstimator(num_keypoints=5)
        self.actor_target.load_state_dict(torch.load(actor_model_load_path))
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        # Initialize critic networks
        self.critic = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target = Critic(state_dim, action_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())

        try:
            self.critic_optimizer = optim.Adam(
                self.critic.model_sam.mask_decoder.parameters(),
                lr=1e-3
            )
        except AttributeError as e:
            print(f"Error accessing parameters: {e}")

        # Initialize replay buffer and writer
        self.replay_buffer = ReplayBuffer('/home/kasra/Desktop/New Folder/PosEstimation/Generate Env/env_data/')
        self.writer = SummaryWriter(directory)

        # Enable gradient computation for mask decoder
        for param in self.critic.model_sam.mask_decoder.parameters():
            param.requires_grad = True

        # Debug parameter gradients
        for name, param in self.critic.model_sam.mask_decoder.named_parameters():
            if param.grad is not None:
                print(f"Parameter: {name}, grad_fn: {param.grad_fn}")
            else:
                print(f"Parameter: {name} has no gradient")

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, input_image: List[torch.Tensor], initial_mask: List[torch.Tensor]) -> torch.Tensor:
        """Select an action based on current state.
        
        Args:
            input_image: List of input image tensors
            initial_mask: List of initial mask tensors
            
        Returns:
            Selected action tensor
        """
        initial_masks_tensor = torch.stack([torch.tensor(i) for i in initial_mask])
        input_image_tensor = torch.stack([torch.tensor(i) for i in input_image])
        action = self.actor(input_image_tensor, initial_masks_tensor.unsqueeze(1))
        return action

    def update(self):
        """Update the actor and critic networks."""
        for it in range(args.update_iteration):
            # Sample batch from replay buffer
            (image_name, input_image, pred_image, logits, input_points,
             padded_mask, scores, label, low_res_mask,
             image_name_plus_1, input_image_plus_1, pred_image_plus_1,
             logits_plus_1, input_points_plus_1, padded_mask_plus_1,
             scores_plus_1, label_plus_1, low_res_mask_plus_one) = self.replay_buffer.sample(args.batch_size)

            # Convert tensors
            padded_mask_plus_1_tensor = torch.stack([torch.from_numpy(arr) for arr in padded_mask_plus_1])
            input_image_plus_1_tensor = torch.stack(input_image_plus_1)
            padded_mask_tensor = torch.stack([torch.from_numpy(arr) for arr in padded_mask])
            input_image_tensor = torch.stack(input_image)

            # Get target Q value
            target_Q_masks, target_Q_scores = self.critic_target(
                input_image_plus_1_tensor,
                self.actor_target(input_image_plus_1_tensor, padded_mask_plus_1_tensor.unsqueeze(1)),
                input_points_plus_1
            )

            target_Q_scores = dice_coefficient_torch(
                torch.stack(target_Q_masks).squeeze()[:, 0, :240, :240],
                torch.stack(label_plus_1)
            )

            scores_plus_1 = torch.stack([torch.tensor(arr) for arr in scores_plus_1]).to(DEVICE)
            target_Q = scores_plus_1 + (args.gamma * target_Q_scores)

            # Get current Q estimate
            current_Q_masks, current_Q_scores = self.critic(
                torch.stack(input_image),
                input_points,
                padded_mask_tensor
            )
            current_Q_scores = dice_coefficient_torch(
                torch.stack(current_Q_masks).squeeze()[:, 0, :240, :240],
                torch.stack(label)
            )

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q_scores.reshape(-1), target_Q.reshape(-1)).to(DEVICE)
            critic_loss.requires_grad = True
            print(critic_loss.item())
            self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)

            # Optimize critic
            self.critic_optimizer.zero_grad()

            # Compute actor loss
            a, b = self.critic(
                torch.stack(input_image),
                self.actor_target(input_image_tensor, padded_mask_tensor.unsqueeze(1)),
                padded_mask_tensor
            )
            b = dice_coefficient_torch(
                torch.stack(a).squeeze()[:, 0, :240, :240],
                torch.stack(label)
            )
            b = 1 - b
            actor_loss = b.mean()
            self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

            # Optimize actor
            actor_loss.requires_grad = True
            self.actor_optimizer.zero_grad()

            # Update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self, episode: Optional[int] = None):
        """Save the model parameters.
        
        Args:
            episode: Optional episode number to include in filename
        """
        actor_filename = f'actor_{episode}.pth' if episode is not None else 'actor.pth'
        critic_filename = f'critic_{episode}.pth' if episode is not None else 'critic.pth'
        torch.save(self.actor.state_dict(), os.path.join(directory, actor_filename))
        torch.save(self.critic.state_dict(), os.path.join(directory, critic_filename))
        print(f"Models saved for episode {episode}")

    def load(self):
        """Load the model parameters."""
        self.actor.load_state_dict(torch.load(os.path.join(directory, 'actor.pth')))
        self.critic.load_state_dict(torch.load(os.path.join(directory, 'critic.pth')))
        print("Model has been loaded successfully")


def main():
    """Main training loop."""
    # Initialize environment and agent
    args = parse_arguments()
    env = gym.make(args.env_name)
    
    if args.seed:
        env.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = DDPG(state_dim, action_dim, max_action)
    batch_size = 4

    if args.load:
        agent.load()

    total_step = 0
    replay_buffer = ReplayBuffer('/home/kasra/Desktop/New Folder/PosEstimation/Generate Env/env_data/')

    for i in tqdm(range(args.max_episode)):
        total_reward = 0
        step = 0

        # Sample batch from replay buffer
        (image_name, input_image, pred_image, logits, input_points,
         padded_mask, scores, label, low_res_mask,
         image_name_plus_1, input_image_plus_1, pred_image_plus_1,
         logits_plus_1, input_points_plus_1, padded_mask_plus_1,
         scores_plus_1, label_plus_1, low_res_mask) = replay_buffer.sample(batch_size)

        # Select and perform action
        action = agent.select_action(input_image, pred_image)
        step += 1
        total_reward += scores_plus_1[0]
        total_step += step + 1

        print(f"Total T: {total_step} Episode: {i} Total Reward: {total_reward:0.2f}")
        agent.update()

        # Save models periodically
        if i % 10 == 0:
            agent.save(episode=i)


if __name__ == '__main__':
    main()

