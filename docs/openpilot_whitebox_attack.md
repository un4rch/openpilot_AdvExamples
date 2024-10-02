# OpenPilot White-box Adversarial Attack

This guide introduces how to implement white-box adversarial attacks against the OpenPilot autonomous driving system, specifically targeting the Supercombo model. The focus will be on crafting adversarial patches that trick the model into making incorrect predictions. This attack will be conducted on both the **0.8.3** and **0.9.4** versions of Supercombo, using **ONNX** and **PyTorch** models.

## Table of Contents
- [Introduction](#introduction)
- [Setting Up the Environment](#setting-up-the-environment)
- [Defining Important Functions](#defining-important-functions)
- [Comparison of ONNX and PyTorch Models](#comparison-of-onnx-and-pytorch-models)
- [Adversarial Patch Generation](#adversarial-patch-generation)
  - [Expectation Over Transformation](#expectation-over-transformation)
- [Supercombo Model Interaction](#supercombo-model-interaction)
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
   
2. **Parsing Images**: The Supercombo model expects images in a specific format, with six channels for YUV encoding. Parsing the image into this format is essential for accurate predictions.

3. **Preprocessing Frames**: Frames captured from the simulation environment need to be cropped, resized, and normalized before being fed into the model. This step ensures that the input format is compatible with Supercombo's architecture.

```pyhton
#TODO
```

## Comparison of ONNX and PyTorch Models

When working with adversarial attacks, it is critical to ensure that the PyTorch-converted model behaves identically to the ONNX version. A mismatch in outputs could indicate inconsistencies in the model conversion, leading to inaccurate attack results.

In this step, we will:
1. **Load the ONNX model** using **ONNX Runtime**.
2. **Run inference** on both the ONNX and PyTorch versions of the model.
3. **Compare the outputs** to ensure they are equivalent.

This comparison guarantees that our adversarial examples, generated via PyTorch, are valid and will affect the OpenPilot system.

# TODO: comment for comparing ONNX and PyTorch outputs

## Adversarial Patch Generation

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
