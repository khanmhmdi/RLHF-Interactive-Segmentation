import torch
import matplotlib.pyplot as plt
import cv2


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def predict_mask(image_path, sam_path, )
# os.chdir(f'{CODE_DIR}')
image = cv2.imread('data_demo/images/amos_0507_31.png')
image.shape

plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.show()

from SAM.segment_anything import sam_model_registry
from SAM.segment_anything.predictor_sammed import SammedPredictor
from argparse import Namespace

args = Namespace()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
args.image_size = 256
args.encoder_adapter = True
args.sam_checkpoint = "/home/kasra/Documents/SAM-Med2d_V2/SAM-Med2D/Model/sam-med2d_b.pth"
model = sam_model_registry["vit_b"](args).to(device)
predictor = SammedPredictor(model)

predictor.set_image(image)

ori_h, ori_w, _ = image.shape
import numpy as np

# Define the range for x and y coordinates
x_range = np.arange(109, 120)
y_range = np.arange(109, 110)

# Generate a grid of x and y values
x_grid, y_grid = np.meshgrid(x_range, y_range)

# Flatten the grid arrays and combine them into coordinate pairs
input_point = np.vstack([x_grid.ravel(), y_grid.ravel()]).T

# Create an array of labels, all set to 1
input_label = np.ones(input_point.shape[0], dtype=int)

plt.figure(figsize=(10, 10))
plt.imshow(image)
show_points(input_point, input_label, plt.gca())
plt.axis('on')
plt.show()

masks, scores, logits = predictor.predict(
    point_coords=input_point,
    point_labels=input_label,
    multimask_output=True,
)
masks.shape  # (number_of_masks) x H x W

plt.figure(figsize=(10, 10))
plt.imshow(image)
show_mask(masks, plt.gca())
show_points(input_point, input_label, plt.gca())
plt.axis('off')
plt.show()

# class Mask_Prediction_SAM:
#     def __init__(self, sam_path, images_dir):
#         self.sam_path = sam_path
#         self.image_dir = images_dir
#     def