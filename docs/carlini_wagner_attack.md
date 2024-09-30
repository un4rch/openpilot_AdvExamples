# Carlini & Wagner (CW) L2 Attack

This guide aims to introduce the Carlini & Wagner adversarial attack to the reader, focusing on its formulation as an optimization problem and how it can be applied to neural networks. The CW attack is a white-box adversarial attack that provides a fine-grained, highly effective means of generating adversarial examples. In this guide, we will demonstrate how to apply the CW L2 attack on two models: a fine-tuned ResNet-50 and a custom CNN, both trained on the CIFAR-10 dataset.

## Table of Contents
- [Introduction](#introduction)
- [Key parameters](#key-parameters)
- [Optimization Problem](#optimization-problem)
- [Norms](#norms)
- [Adversarial Attacks](#adversarial-attacks)
    - [ResNet-50 Attack](#resnet-50-attack)
    - [Custom CNN Attack](#custom-cnn-attack)

## Introduction

The [**Carlini & Wagner (CW) attack**](https://arxiv.org/pdf/1608.04644) is a powerful white-box adversarial attack that seeks to minimize the perturbation added to an input image while successfully fooling the target model. The CW attack focuses on making small, often imperceptible changes to the image that mislead a deep neural network (DNN) into making incorrect classifications.

The attack is framed as an optimization problem where the objective is to find the smallest possible perturbation that leads the model to misclassify the input. The **L2 norm** is used to measure this perturbation, which represents the Euclidean distance between the original image and the adversarial image.

In this implementation, we will target two models:
- **ResNet-50**: A fine-tuned, pre-trained model on the CIFAR-10 dataset.
- **Custom CNN**: A CNN built and trained from scratch on the CIFAR-10 dataset.

First, we are going to understand the background behind the algorithm.

## Key Parameters
- **Learning rate**: Controls how quickly the perturbation is adjusted at each iteration.
- **Max iterations**: Determines how many times the optimization will run.
- **Kappa (confidence)**: Ensures the adversarial example is classified with a certain confidence. A higher kappa value makes the attack more aggressive.

## Optimization Problem

The Carlini & Wagner attack is essentially an optimization problem. The goal is to solve for an adversarial perturbation that minimizes the following objective function.
Here the adversary optimizes the objective function iteratively, adjusting the perturbation to minimize the magnitude while maximizing the attack’s success rate

![Objective function](/images/cw_original_optimization.png)

Being D the function used to quantify the distance between the original input and the perturbed one, C the classification function used. And lastly, ensuring that the adversarial example produced is valid (in the case of images, all the values need to be constrained between 0 and 1 for all the color channels). As the classification function C(x + δ) = t is highly non-linear, making the optimization problem is really difficult to solve with previously existing algorithms.

That is why the authors proposed a different approach:

![CW Proposed approach](/images/cw_proposed_approach.png)

By changing the classification function to an objective function, the non-lineality problem is solved. Then they proceed to reformulate the problem introducing the function on the minimization problem and multiplying it by a constant c to ensure that when performing the gradient descent, none of the terms takes precedence over the other:

![CW Objective function](/images/cw_objective_function.png)

Lastly, they replace the distance metric D with the Lp norm of δ:

![Replace D by Lnorm](/images/cw_replace_d_Lnorm.png)

The authors proposed and tested numerous objective functions:

![CW tested objective functions](/images/cw_all_objective_functions.png)

They made a comparison among the proposed objective functions and finally realized that the 6th function was the best and they chose that for their attacks.

![CW objective functions comparison](/images/cw_objectivesF_comparison.png)

The perturbation is mapped to the hyperbolic tangent space to ensure that the generated images are within a valid input range. They decided to use this space for two main reasons:
- **Constraint satisfaction**: The tanh function naturally satisfies the constraints applied to the perturbation on CW attacks, such as L2-norm or L∞-norm bounds, ensuring that the perturbation remains within the desired range. The tanh function restricts its output values between -1 and 1, which aligns with the constraints and prevents the perturbation from becoming excessively large.
- **Smooth mapping**: The tanh function provides a continuous and smooth mapping between the perturbation and input spaces. The function’s differentiability facilitates the gradient computation and enables efficient optimization of the objective function.

![CW Tanh function](/images/cw_tanh.png)

## Norms

### L2 Norm
They presented three possible approaches for the attack using different Lp norms:
- **L0-norm**: Measures the number of non-zero elements in a vector. In this context, it represents the number of pixels modified in the perturbation. By minimizing the L0-norm, the attack seeks to change as few pixels as possible, focusing on a few critical pixels. It makes it so that a few pixels go under a significant modification, thus making it clear to the human eye.
- **L2-norm**: Computes the Euclidean distance between two vectors, which in this case, represents the pixel-wise differences between the original input and the adversarial example. By minimizing the L2-norm the attack aims to make small changes across the overall pixels of the initial image. It distributes the perturbation more evenly across the input, minimizing the visual impact on the human eye.
- **L∞-norm**: Calculates the maximum absolute difference between corresponding elements of two vectors. This method constrains the perturbation such that the absolute maximum change in any individual pixel value does not exceed a certain threshold. By minimizing the L∞-norm, the attack aims to keep the largest pixel change across all pixels within the allowed limit.

A comparison between generated AEs with the three norms proposed:
![CW Norms comparison](/images/cw_norms_comparison.png)

## Adversarial attacks
For the adversarial attacks we are going to start importing all necessary libraries and also preparing our CIFAR-10 dataset:
```python
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from torch.utils.data import random_split
import torch.nn as nn
import torchvision.models as models

# Training dataset (80% train / 20% validation)
dataset_train = CIFAR10('./data', train=True, download=True, transform = transforms.Compose([transforms.ToTensor()]))
random_seed = 42
torch.manual_seed(random_seed)
val_size = int(0.2*len(dataset_train)) # Validation size
train_size = int(len(dataset_train)-val_size) # Training size
train_ds, val_ds = random_split(dataset_train, [train_size, val_size]) # Dataset for training phase
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=128*2, shuffle=True, num_workers=2) # Training data loader
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=128*2, shuffle=True, num_workers=2) # Validation data loader

# Testing dataset
test_ds = CIFAR10('./data', train=False, download=True,transform=transforms.Compose([transforms.ToTensor()])) # Testing dataset
test_dl = torch.utils.data.DataLoader(test_ds,batch_size=128*2, shuffle=True, num_workers=2) # Testing data loader
```

Now, we are going to define the L2-Norm Carlini & Wagner algorithm:
```python
def cw_l2_attack(model, original_images, labels, targeted=False, c=1e-4, kappa=0, max_iter=1000, learning_rate=0.01):
    perturbed_images = torch.zeros_like(original_images, requires_grad=True).to("cpu")
    optimizer = optim.Adam([perturbed_images], lr=learning_rate)
    
    for iteration in range(max_iter):
        perturbed_images_tanh = 1/2*(nn.Tanh()(perturbed_images) + 1)
        outputs = model(perturbed_images_tanh)
        labels_one_hot = torch.eye(len(outputs[0]))[labels].to(original_images.device)
        
        i, _ = torch.max((1 - labels_one_hot) * outputs, dim=1)
        j = torch.masked_select(outputs, labels_one_hot.bool())
        loss = torch.clamp(j - i, min=-kappa)
        
        l2dist = torch.norm(perturbed_images_tanh - original_images, p=2)
        loss = l2dist + torch.sum(c * loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % (max_iter // 10) == 0:
            print(f'Iteration {iteration}, Loss: {loss.item()}')
    
    perturbed_images = 1/2*(nn.Tanh()(perturbed_images) + 1)
    return perturbed_images
```

### ResNet-50 Attack

The first model we target with the CW attack is a **fine-tuned ResNet-50**, pre-trained on ImageNet and fine-tuned for the CIFAR-10 dataset. Below are the key steps:

#### 1. Loading ResNet-50
We begin by loading the pre-trained ResNet-50 model for CIFAR-10 and freezing all layers except the final fully connected layer, which we replace to output 10 classes (for CIFAR-10).

```python
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
```

#### 2. Fine-tuning ResNet-50
Then, we are going to train our last layer with the CIFAR-10 dataset, for the model to be able to predict the 10 classes taking advantage of the previous knowledge:
```python
def train_model(model, train_dl, val_dl, criterion, optimizer, num_epochs=10):
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_dl:
            # Move data to GPU if available
            images, labels = images, labels

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dl.dataset)
        train_losses.append(epoch_loss)
        
        # Validation loss
        model.eval()
        with torch.no_grad():
            val_running_loss = 0.0
            for images, labels in val_dl:
                images, labels = images, labels
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)

            val_loss = val_running_loss / len(val_dl.dataset)
            val_losses.append(val_loss)

        print(f'Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}')

    print('Finished Training')
    return train_losses, val_losses

train_losses, val_losses = train_model(model, train_dl, val_dl, criterion, optimizer, num_epochs=20)

# Save finetuned resnet_50 model
torch.save(model, './resnet_50.pth')

# Load resnet_50 model
model = torch.load('./resnet_50.pth')
model.eval()  # Set the model to evaluation mode
```

#### 3. Applying the CW Attack to ResNet-50
After fine-tuning the model, we apply the Carlini & Wagner attack to generate adversarial examples.

```python
specific_index = 45  # Example index you want to access
subset = Subset(test_ds, [specific_index])
loader = DataLoader(subset, batch_size=1)  # Ensure batch size matches what you need

# Example of running the attack
dataiter = iter(loader)
original_image, label = next(dataiter)

# Run the attack
perturbed_image = cw_l2_attack(model, original_image, label, c=1, targeted=False, max_iter=10000)

# Evaluate model on perturbed image
output = model(perturbed_image)
pred_label = output.max(1, keepdim=True)[1]
print('Predicted Label after attack:', pred_label.item())
```

#### 3. Visualizing the Results
Once the attack is completed, we can visualize the original, perturbed, and difference images.

```python
import matplotlib.pyplot as plt

def show_images(original_image, perturbed_image):
    images = [original_image, perturbed_image, perturbed_image - original_image]
    titles = ["Original Image", "Perturbed Image", "Perturbation"]
    for i, img in enumerate(images):
        plt.subplot(1, 3, i+1)
        imshow(torchvision.utils.make_grid(img))
        plt.title(titles[i])
    plt.show()

images = [original_image, perturbed_image.detach(), perturbed_image.detach() -  original_image + 0.5]
titles = [f"Original Image (class {label.item()})", f"Perturbed Image (class {pred_label.item()})", "Perturbation"]
show_images(images, titles)
```
![Resnet-50 mislead](/images/cw_resnet50_mislead.png)

### Custom CNN Attack
We also apply the CW attack to a custom CNN trained from scratch on the CIFAR-10 dataset.

#### 1. Building and Training the Custom CNN
We create a simple convolutional neural network to classify CIFAR-10 images.

```python
import torch.nn as nn
import torch.optim as optim

class Cifar10CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(Cifar10CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64*8*8, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(2)(nn.ReLU()(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = nn.ReLU()(self.fc1(x))
        x = self.fc2(x)
        return x

model = Cifar10CNN()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
```

#### 2. Applying the CW Attack to Custom CNN
The CW attack can be applied to the custom CNN similarly to how we applied it to ResNet-50.

```python
perturbed_image = cw_l2_attack(model, original_image, label, c=1e-4, max_iter=1000)
show_images(original_image, perturbed_image)
```
