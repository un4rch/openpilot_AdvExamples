# Carlini & Wagner (CW) L2 Attack

This guide aims to introduce the Carlini & Wagner adversarial attack to the reader, focusing on its formulation as an optimization problem and how it can be applied to neural networks. The CW attack is a white-box adversarial attack that provides a fine-grained, highly effective means of generating adversarial examples. In this guide, we will demonstrate how to apply the CW L2 attack on two models: a fine-tuned ResNet-50 and a custom CNN, both trained on the CIFAR-10 dataset.

## Table of Contents
- [Introduction](#introduction)
- [Optimization Problem](#optimization-problem)
- [Norms](#norms)
- [ResNet-50 Attack](#resnet-50-attack)
- [Custom CNN Attack](#custom-cnn-attack)

## Introduction

The **Carlini & Wagner (CW) attack** is a powerful white-box adversarial attack that seeks to minimize the perturbation added to an input image while successfully fooling the target model. The CW attack focuses on making small, often imperceptible changes to the image that mislead a deep neural network (DNN) into making incorrect classifications.

The attack is framed as an optimization problem where the objective is to find the smallest possible perturbation that leads the model to misclassify the input. The **L2 norm** is used to measure this perturbation, which represents the Euclidean distance between the original image and the adversarial image.

In this implementation, we will target two models:
- **ResNet-50**: A fine-tuned, pre-trained model on the CIFAR-10 dataset.
- **Custom CNN**: A CNN built and trained from scratch on the CIFAR-10 dataset.

## Optimization Problem

The Carlini & Wagner attack is essentially an optimization problem. The goal is to solve for an adversarial perturbation that minimizes the following objective function:

\[
\min ||\delta||_2^2 + c \cdot f(x + \delta)
\]

Where:
- \( ||\delta||_2^2 \) is the **L2 norm** (Euclidean distance) of the perturbation.
- \( f(x + \delta) \) is a loss function that penalizes the model for classifying \( x + \delta \) as the original class.
- \( c \) is a constant that balances between minimizing the perturbation and ensuring the adversarial example misleads the model.

The attack iteratively adjusts the perturbation \( \delta \) using gradient descent to minimize this objective, making the perturbation as small as possible while still causing the model to misclassify the image.

### Key Parameters
- **Learning rate**: Controls how quickly the perturbation is adjusted at each iteration.
- **Max iterations**: Determines how many times the optimization will run.
- **Kappa (confidence)**: Ensures the adversarial example is classified with a certain confidence. A higher kappa value makes the attack more aggressive.

### The Tanh Trick
To ensure that pixel values remain valid (between 0 and 1), the CW attack uses a transformation called the **tanh trick**. This transformation bounds the pixel values within a valid range during the optimization process.

## Norms

### L2 Norm
In the Carlini & Wagner attack, the **L2 norm** is used to measure the size of the perturbation. The L2 norm represents the Euclidean distance between the original and perturbed images:

\[
L2 = \sqrt{\sum (x_{i} - x'_{i})^2}
\]

This ensures that the perturbation remains small and imperceptible to the human eye. In contrast to simpler methods like FGSM (which uses the L∞ norm), the CW attack's use of the L2 norm allows for more subtle perturbations that are harder to detect.

### Other Norms (For Context)
- **L∞ Norm**: Measures the maximum change to any individual pixel. This norm is used in attacks like FGSM.
- **L1 Norm**: Measures the sum of the absolute differences between the original and perturbed images.

## ResNet-50 Attack

The first model we target with the CW attack is a **fine-tuned ResNet-50**, pre-trained on ImageNet and fine-tuned for the CIFAR-10 dataset. Below are the key steps:

### 1. Fine-tuning ResNet-50
We begin by fine-tuning the pre-trained ResNet-50 model for CIFAR-10 by freezing all layers except the final fully connected layer, which we replace to output 10 classes (for CIFAR-10).

```python
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models

# Load pre-trained ResNet-50 model
model = models.resnet50(pretrained=True)

# Freeze all layers except the final fully connected layer
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer for CIFAR-10
num_classes = 10
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)
