# OpenPilot White-box Adversarial Attack

This guide introduces how to implement white-box adversarial attacks against the OpenPilot autonomous driving system, specifically targeting the Supercombo model. The focus will be on crafting adversarial patches that trick the model into making incorrect predictions. This attack will be conducted on both the **0.8.3** and **0.9.4** versions of Supercombo, using **ONNX** and **PyTorch** models.

## Table of Contents
- [Introduction](#introduction)
- [Setting Up the Environment](#setting-up-the-environment)
- [Defining Important Functions](#defining-important-functions)
- [Adversarial Iterative Algorithm](#adversarial-iterative-algorithm)
- [Running the Attack](#running-the-attack)
- [Visualizing the Results](#visualizing-the-results)

## Introduction

OpenPilot is an open-source autonomous driving system that uses deep neural networks to process sensor inputs and make real-time driving decisions. At the core of its decision-making is the **Supercombo model**, which integrates lane detection, object detection, and end-to-end driving tasks into a single neural network.

In this guide, we aim to exploit the weaknesses of the Supercombo model through a **white-box attack**, where we have complete access to the model's architecture, weights, and gradients. The attack will involve generating an **adversarial patch**—a small image perturbation designed to trick the Supercombo model into misinterpreting its surroundings, potentially causing unsafe driving behavior.

## Setting Up the Environment

Before starting, you will need to set up an environment to load and manipulate the Supercombo model. This involves converting the OpenPilot **ONNX** model to **PyTorch** and preparing the necessary libraries for running attacks.

### Required Libraries:
1. **ONNX**: To load the OpenPilot models.
2. **PyTorch**: For building the neural network and optimizing the adversarial patches.
3. **ONNX Runtime**: To run inferences on the ONNX model.
4. **Other Libraries**: `matplotlib`, `numpy`, `opencv-python` for image processing.

```python
import sys
import os
import re
import json

import onnx
import onnxruntime
from onnx2torch import convert
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd

import cv2
from matplotlib import pyplot as plt
```

Once the environment is set up, the ONNX model is loaded and converted into PyTorch for easier manipulation during the adversarial attack:
```python
# Load Supercombo ONNX model
model_name = "supercombo_0.8.3.onnx" # For 0.9.4: supercombo_0.9.4.onnx
onnx_model = onnx.load(model_name)

# Conver to PyTorch model
torch_model = convert(onnx_model)
if torch.cuda.is_available():
    torch_model.cuda()
#torch_model.half() # Make the model use float16 dtype
torch_model.eval()
#print(torch_model)

# Session ONNX
session = onnxruntime.InferenceSession(model_name, providers=['CPUExecutionProvider'])
```

## Defining Important Functions

To run the attack effectively, we need several utility functions. Some key functionalities include:

1. **Converting BGR to YUV**: The Supercombo model operates on YUV images rather than RGB. You'll need a function to convert images to the YUV color space before passing them to the model.
```python
def rgb_to_yuv(rgb_tensor):
    # Ensure tensor is in (N, C, H, W) format
    assert rgb_tensor.dim() == 4 and rgb_tensor.size(1) == 3, "Input tensor must be in (N, C, H, W) format with 3 channels"

    # Convert RGB to YUV
    R = rgb_tensor[:, 0, :, :]
    G = rgb_tensor[:, 1, :, :]
    B = rgb_tensor[:, 2, :, :]

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    U = -0.14713 * R - 0.28886 * G + 0.436 * B
    V = 0.614 * R - 0.51498 * G - 0.10001 * B

    yuv_tensor = torch.stack([Y, U, V], dim=1)
    return yuv_tensor
```
2. **Parsing Images**: The Supercombo model expects images in a specific format, with six channels for YUV encoding. Parsing the image into this format is essential for accurate predictions.
```python
def parse_image(frame):
    # Ensure frame is a tensor of shape (1, 3, H, W)
    assert frame.dim() == 4 and frame.size(1) == 3, "Input tensor must be of shape (1, 3, H, W)"
    
    H = frame.size(2)
    W = frame.size(3)
    
    # Initialize the parsed tensor with shape (6, H//2, W//2)
    parsed = torch.zeros((6, H//2, W//2), dtype=torch.uint8)
    
    # Extract the channels from the input tensor
    Y = frame[0, 0, :, :]
    U = frame[0, 1, :, :]
    V = frame[0, 2, :, :]

    # Populate the parsed tensor
    parsed[0] = Y[0:H:2, 0::2]
    parsed[1] = Y[1:H:2, 0::2]
    parsed[2] = Y[0:H:2, 1::2]
    parsed[3] = Y[1:H:2, 1::2]
    parsed[4] = U[0:H//2, 0::2]
    parsed[5] = V[0:H//2, 0::2]
    
    return parsed.unsqueeze(0)
```
3. **Preprocessing Frames**: Frames captured from the simulation environment need to be cropped, resized, and normalized before being fed into the model. This step ensures that the input format is compatible with Supercombo's architecture.
```python
def preprocess_frame(frame_tensor, roi_area=None, resize_dim=(128,256)):
    x, y, w, h = roi_area
    # Extract ROI (Region Of Interest) area of an image
    roi_tensor = frame_tensor[:, :, y:y+h, x:x+w]
    # Resize the images to the required dimensions
    roi_tensor_resized = F.interpolate(roi_tensor, size=resize_dim, mode='bilinear', align_corners=False)
    # Convert to YUV
    roi_tensor_resized_yuv = rgb_to_yuv(roi_tensor_resized)
    # Parse YUV with 6 channels: YUV_4:2:0
    parsed_frame = parse_image(roi_tensor_resized_yuv)
    return parsed_frame
```
4. **Image display**: Some functions to display images and see the patch or frames.
```python
def display_image(image):
	plt.imshow(image)
	plt.show()
	plt.clf()

def display_img(image):
	plt.imshow(image)
	plt.axis('off')
	plt.show()
	plt.clf()

def subplot(img1, img2):
	# Create a figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Display the frame in the first subplot
    ax1.imshow(img1)
    ax1.set_title('Original Frame')
    ax1.axis('off')

    # Display the adversarial patch in the second subplot
    ax2.imshow(img2)
    ax2.set_title('Adversarial Patch')
    ax2.axis('off')

    # Show the plot
    plt.tight_layout()
    plt.show()
```
5. **Format conversion**: ???
```python
def numpy_to_tensor(array):
	# Convert image from BGR to RGB as PyTorch uses RGB by default
	frame_rgb = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
	# Convert to float32 for precision, then to float16
	tensor = torch.tensor(frame_rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
	return tensor.to(torch.float16)

def tensor_to_numpy(tensor):
	# Convert back to float32 to avoid overflow when converting to uint8
	tensor_float32 = tensor.squeeze(0).permute(1, 2, 0).to(torch.float32)
	image_back = tensor_float32.detach().numpy().astype(np.uint8)
	# Convert RGB back to BGR
	return cv2.cvtColor(image_back, cv2.COLOR_RGB2BGR)

def check_images_conversion(frame):
	# Step 1: Load the image using OpenCV
	#frame = cv2.imread(data_dir + frame_name)

	# Step 2: Convert the image to a PyTorch tensor in float16
	# Convert image from BGR to RGB as PyTorch uses RGB by default
	frame_rgb = numpy_to_tensor(frame)

	# Step 3: Convert the tensor back to a NumPy array
	# Convert back to float32 to avoid overflow when converting to uint8
	image_back_bgr = tensor_to_numpy(frame_rgb)

	# Check if both images are the same
	assert np.array_equal(frame, image_back_bgr), "The images are not the same!"
	display_img(frame)
	display_img(image_back_bgr)
```
6. **Expectation over Transform (EoT)**: ???
```python
def place_patch(frames, patch, patch_size=(50, 50), eot_locations=[], eot_rotations=[], eot_scales=[]):
	"""
	Places a patch on 2 consecutive frames with Expectation over Transform (EoT).

	Parameters:
	- frames: List of 2 tensors of shape (N, C, H, W), the batch of frames.
	- patch: Tensor of shape (N, C, H_patch, W_patch), the patch to place.
	- patch_size: Tuple (H_patch, W_patch), the size of the patch.
	- eot_locations: List of tuples [(x, y)], locations to place the patch.
	- eot_rotations: List of angles in degrees to rotate the patch.
	- eot_scales: List of scale factors to resize the patch.

	Returns:
	- frames_patches: List of 2 lists of transformed frames with patches applied for consecutive frames.
	"""
	frames_patches = []
	for frame in frames:
		frame_transforms = []
		for (x, y) in eot_locations:
			for rotation in eot_rotations:
				for scale in eot_scales:
					# Clone the frame
					frame_with_patch = frame.clone()
					
					# Resize (scale) the patch
					scaled_patch = F.interpolate(patch, scale_factor=scale, mode='bilinear', align_corners=False)

					# Calculate new patch size after scaling
					new_H_patch, new_W_patch = scaled_patch.shape[2], scaled_patch.shape[3]
					
					# Create an affine transformation matrix for rotation
					theta = torch.tensor([
						[torch.cos(torch.tensor(rotation)), -torch.sin(torch.tensor(rotation)), 0],
						[torch.sin(torch.tensor(rotation)), torch.cos(torch.tensor(rotation)), 0]
					], dtype=torch.float32)
					
					# Grid for sampling
					grid = F.affine_grid(theta.unsqueeze(0), scaled_patch.size(), align_corners=False)
					
					# Apply the affine transformation (rotation)
					rotated_patch = F.grid_sample(scaled_patch, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
					
					# Place the rotated and scaled patch onto the frame
					frame_with_patch[:, :, y:y + new_H_patch, x:x + new_W_patch] = rotated_patch
					
					# Append the transformed frame to the list
					frame_transforms.append(frame_with_patch)
		
		frames_patches.append(frame_transforms)

	return frames_patches
```
7. **Disappearance loss**: ???
```python
def disappearance_loss(patch, conf, patchDist, realDist, l1=0.01, l2=0.001):
	Lconf = -torch.log(1 - conf) # 1-conf ya que se busca minimizar conf
	Ldist = -torch.abs(patchDist/realDist)
	# Compute differences along height and width
	diff_h = patch[:, :, 1:, :] - patch[:, :, :-1, :]
	diff_w = patch[:, :, :, 1:] - patch[:, :, :, :-1]
	Ltv = torch.sum(torch.abs(diff_h)) + torch.sum(torch.abs(diff_w))
	return Lconf + l1*Ldist + l2*Ltv
```

## Adversarial Iterative Algorithm

### Overview

The attack we are conducting is a **disappearance attack**, where an adversarial patch is placed in the driving scene to make critical objects, such as vehicles or lane markings, disappear from the model's perception.

The patch will be optimized to reduce the confidence of the model's object detection, causing it to ignore important objects. The patch will be placed in various locations in the input frames and transformed using **Expectation Over Transformation (EoT)** to make the attack robust against different angles, scales, and rotations.

### Expectation Over Transformation

**EoT** is a technique used to generate adversarial examples that are robust to different transformations. In our case, we will apply several transformations (such as rotation, scaling, and translation) to the adversarial patch, ensuring that it remains effective even under different viewing conditions.

For each frame, the patch will be placed at different locations, rotated by varying degrees, and scaled to different sizes. This ensures that the attack remains effective across a range of scenarios in the driving environment.

# TODO: comment for placing adversarial patches with EoT transformations

## Supercombo Model Interaction

After generating the adversarial patch, we need to interact with the **Supercombo model** to evaluate the effects of the patch. This step involves feeding the patched images into the Supercombo model and measuring how the patch affects the model's predictions.

The **Supercombo** model processes two consecutive frames, along with additional inputs (like desire, traffic convention, and feature buffers). We will need to preprocess the input frames and pass them through the model, first with and then without the adversarial patch, to observe the differences in predictions.

# TODO: comment for interacting with Supercombo model and evaluating predictions

## Running the Attack

To run the attack, we will:
1. **Initialize the adversarial patch** with random values.
2. **Apply the patch** to a sequence of frames from a driving scene.
3. **Run inference** on the patched frames using the Supercombo model.
4. **Optimize the patch** by minimizing a loss function that penalizes high confidence in detecting critical objects (such as vehicles or lanes).

The optimization will be carried out using the **Adam optimizer**, which adjusts the patch to minimize the model's confidence in detecting objects near the patch.

# TODO: comment for initializing and optimizing the adversarial patch

## Visualizing the Results

Finally, we will visualize the results of the attack by comparing:
1. The original frames without the adversarial patch.
2. The frames with the adversarial patch applied.
3. The patch itself.

These visualizations will help us understand how the patch affects the Supercombo model's perception and how subtle the patch can be while still being effective.

# TODO: comment for visualizing the original, patched, and difference images

This guide provides a step-by-step approach to implementing a white-box adversarial attack on the OpenPilot system. By carefully crafting adversarial patches and applying them to driving scenes, we can observe how small perturbations can mislead even advanced neural network models like Supercombo, highlighting the importance of robustness in autonomous driving systems.
