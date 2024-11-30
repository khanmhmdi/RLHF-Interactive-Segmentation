import argparse
import pickle
import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from itertools import count
from tensorboardX import SummaryWriter
from tqdm import tqdm

from Pos_Network.Model import KeyPointEstimator
from SAM.segment_anything import sam_model_registry
from SAM.segment_anything.predictor_sammed import SammedPredictor

class ReplayBuffer:
    def __init__(self, env_path):
        self.env_path = env_path
        self.pkl_file_paths = [
            os.path.join(env_path, file) 
            for file in os.listdir(env_path) 
            if file.endswith('.pkl') and '3' not in file
        ]

    def load_pkl_file(self, pkl_path):
        with open(pkl_path, 'rb') as file:
            return pickle.load(file)

    def get_next_iteration_paths(self, selected_paths):
        next_paths = []
        for path in selected_paths:
            parts = path.split('_')
            current_iteration = int(parts[-1].split('.')[0])
            new_path = '_'.join(parts[:-1] + [f"{current_iteration + 1}.pkl"])
            next_paths.append(new_path)
        return next_paths

    def sample(self, batch_size):
        selected_paths = random.sample(self.pkl_file_paths, batch_size)
        next_iteration_paths = self.get_next_iteration_paths(selected_paths)

        def load_data(paths):
            loaded_data = []
            for path in paths:
                data = self.load_pkl_file(path)
                loaded_data.append(data)
            return list(zip(*loaded_data))

        state_i = load_data(selected_paths)
        state_i_plus_1 = load_data(next_iteration_paths)

        return state_i + state_i_plus_1

class Actor(nn.Module):
    def __init__(self, num_keypoints=5):
        super(Actor, self).__init__()
        self.model = KeyPointEstimator(num_keypoints=num_keypoints)
        model_load_path = '/home/kasra/Desktop/New Folder/PosEstimation/Pos_Network/model_15.pth'
        self.model.load_state_dict(torch.load(model_load_path))

    def forward(self, x, initial_masks):
        return self.model(x, initial_masks)

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args = argparse.Namespace()
        args.image_size = 256
        args.encoder_adapter = True
        args.sam_checkpoint = "/home/kasra/Documents/SAM-Med2d_V2/SAM-Med2D/Model/sam-med2d_b.pth"
        
        model_sam = sam_model_registry["vit_b"](args).to(device)
        self.predictor = SammedPredictor(model_sam)

    def forward(self, image, prompt, mask=None):
        for batch_idx in range(image.shape[0]):
            self.predictor.set_image(image[batch_idx])

            input_points = prompt[batch_idx].cpu().numpy()
            input_points[:, 0] *= 240
            input_points[:, 1] *= 240
            input_points = input_points[:, [1, 0]]  # Swap x and y

            input_labels = np.ones(input_points.shape[0], dtype=int)
            
            if mask is not None and isinstance(mask, np.ndarray):
                masks, scores, logits = self.predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    mask_input=mask,
                    multimask_output=True
                )
            else:
                masks, scores, logits = self.predictor.predict(
                    point_coords=input_points,
                    point_labels=input_labels,
                    multimask_output=True
                )
        
        return masks, scores, logits

class DDPG:
    def __init__(self, state_dim, action_dim, max_action):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        actor_model_path = '/home/kasra/Desktop/New Folder/PosEstimation/Pos_Network/model_15.pth'
        
        self.actor = Actor().to(self.device)
        self.actor_target = Actor().to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic().to(self.device)
        self.critic_target = Critic().to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.replay_buffer = ReplayBuffer('/home/kasra/Desktop/New Folder/PosEstimation/Generate Env')
        self.writer = SummaryWriter('./exp')

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0

    def select_action(self, input_image, initial_mask):
        input_image_tensor = torch.stack([torch.tensor(i) for i in input_image])
        initial_masks_tensor = torch.stack([torch.tensor(i) for i in initial_mask])
        return self.actor(input_image_tensor, initial_masks_tensor.unsqueeze(1))

    def update(self, args):
        for _ in range(args.update_iteration):
            # Sample from replay buffer
            (image_name, input_image, pred_image, logits, input_points, padded_mask, scores,
             image_name_plus_1, input_image_plus_1, pred_image_plus_1, logits_plus_1, 
             input_points_plus_1, padded_mask_plus_1, scores_plus_1) = self.replay_buffer.sample(args.batch_size)

            # Compute losses and update networks (Note: This part needs more careful implementation)
            # The current implementation seems incomplete and would require debugging

    def save(self, directory):
        torch.save(self.actor.state_dict(), os.path.join(directory, 'actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(directory, 'critic.pth'))

    def load(self, directory):
        self.actor.load_state_dict(torch.load(os.path.join(directory, 'actor.pth')))
        self.critic.load_state_dict(torch.load(os.path.join(directory, 'critic.pth')))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='train', choices=['train', 'test'])
    parser.add_argument('--env_name', default='Pendulum-v1')
    parser.add_argument('--max_episode', default=100000, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--update_iteration', default=200, type=int)
    parser.add_argument('--load', default=False, type=bool)
    args = parser.parse_args()

    # Place initialization and training logic here
    # Note: This script requires careful integration with your existing environment setup

if __name__ == '__main__':
    main()