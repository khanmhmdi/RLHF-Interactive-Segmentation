# DDPG with SAM-based Critic

This repository contains the implementation of a Deep Deterministic Policy Gradient (DDPG) agent with a SAM-based (Segment Anything Model) critic. The code leverages replay buffers, actor-critic networks, and SAM's capabilities for segmentation-based tasks.

## Features
- **Actor-Critic Architecture**: Combines actor and critic networks to learn optimal policies.
- **Replay Buffer**: Efficiently samples batches of environment interactions for training.
- **SAM Integration**: Utilizes SAM for mask-based segmentation tasks as part of the critic's decision-making process.

## Prerequisites
- Python 3.8+
- PyTorch
- `tensorboardX`
- `tqdm`
- `numpy`
- `SAM` library (Segment Anything Model)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name/ddpg-sam-critic.git
   cd ddpg-sam-critic
```
.
├── main.py                  # Entry point for training/testing
├── ReplayBuffer.py          # Replay buffer implementation
├── Actor.py                 # Actor network definition
├── Critic.py                # Critic network using SAM
├── utils/                   # Utility functions and helper files
└── README.md                # Project documentation
```

   ```bash
python main.py --mode train --env_name Pendulum-v1 --max_episode 100000 --batch_size 100 --update_iteration 200```
python main.py --load True
```
   ```bash
python main.py --mode train --env_name BRATS1 --max_episode 50000 --batch_size 64 --update_iteration 100
```
You can directly copy this into your `README.md` file. Let me know if any section requires additional information or modifications!
