# Positional Estimation and Segmentation Framework

This repository implements a framework for keypoint estimation and environment generation using SAM (Segment Anything Model) and a custom KeyPointEstimator model. The framework integrates image processing, neural network inference, and segmentation tasks to generate meaningful insights from input images.

---

## Table of Contents
1. [Features](#features)
2. [Structure](#project-structure)
---

## Features

- **Keypoint Estimation**: Predicts keypoint positions on input images using a pre-trained `KeyPointEstimator`.
- **Segmentation using SAM**: Generates segmentation masks with SAM for input images and keypoints.
- **Iterative State Generation**: Creates sequential segmentation states based on predicted outputs.
- **Environment Data Generation**: Saves processed outputs into a specified directory for downstream tasks.
- **Visualization Utilities**: Includes tools for image normalization and mask/keypoint visualization.

---

## Installation

### Prerequisites
- Python >= 3.8
- PyTorch >= 1.10
- CUDA (if using GPU)
