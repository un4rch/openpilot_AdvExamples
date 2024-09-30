# Carlini & Wagner (CW) L2 Attack

## Introduction

The **Carlini & Wagner (CW) attack** is a powerful white-box adversarial attack that seeks to minimize the amount of perturbation added to an input image while successfully fooling the target model. The CW attack focuses on making small, often imperceptible changes to the image to mislead a deep neural network (DNN) into making incorrect classifications.

The attack is formulated as an optimization problem where the goal is to find the smallest perturbation to fool the model, measured by the **L2 norm** (Euclidean distance). This attack is particularly effective against models that are highly robust to simpler adversarial attacks like FGSM.

For this implementation, we will target two models:
- **ResNet-50**: A fine-tuned pre-trained model with the CIFAR-10 dataset.
- **Custom CNN**: A CNN built and trained from scratch on the CIFAR-10 dataset.

---

## Key Concepts

Before diving into the code, it is essential to understand the key components that make the Carlini & Wagner attack effective.

### 1. **White-Box Attack**
White-box attacks assume full knowledge of the target model, including access to its architecture, weights, gradients, and training data. This allows the attacker to compute precise adversarial perturbations using the model’s internal structure.

### 2. **L2 Norm**
The L2 norm measures the Euclidean distance between the original image and the adversarial image. The goal is to minimize this distance, ensuring that the adversarial perturbations are as small as possible while still misleading the model.

### 3. **Optimization Problem**
The CW attack is framed as an optimization problem. The objective is to modify the input image such that:
- The model's confidence in the original label decreases.
- The confidence in an incorrect label (or target label for targeted attacks) increases.
The attack solves this by adjusting the input image iteratively using gradient descent.

### 4. **Kappa Parameter**
The **kappa** parameter controls the confidence of the attack. A higher kappa forces the model to classify the adversarial image with higher confidence, making the attack more aggressive.

### 5. **Tanh Trick**
To ensure that the perturbed image remains valid (i.e., pixel values are between 0 and 1), the **tanh trick** is used. This transforms the image into a space where the pixel values are bounded, allowing the attack to operate within valid image constraints.

---

## Step-by-Step Methodology

### Step 1: Dataset Preparation

We begin by loading the **CIFAR-10** dataset, which contains 60,000 32x32 color images in 10 different classes. We split the dataset into training, validation, and test sets.

```python
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# Load CIFAR-10 dataset
dataset_train = CIFAR10('./data', train=True, download=True, transform=transforms.ToTensor())
val_size = int(0.2 * len(dataset_train))
train_size = len(dataset_train) - val_size
train_ds, val_ds = random_split(dataset_train, [train_size, val_size])

train_dl = DataLoader(train_ds, batch_size=128, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=128)
test_ds = CIFAR10('./data', train=False, download=True, transform=transforms.ToTensor())
test_dl = DataLoader(test_ds, batch_size=128)
Step 2: Fine-Tuning ResNet-50
We load a pre-trained ResNet-50 model and fine-tune it for the CIFAR-10 dataset. The final fully connected layer is replaced to accommodate the 10 classes in CIFAR-10.

python
Copiar código
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
Explanation:
Fine-Tuning: We freeze the weights of all the layers in ResNet-50 except for the last fully connected layer. This approach is commonly used in transfer learning, where we leverage pre-trained models and adjust only the final layers to fit our new dataset.
Optimizer: We use the Adam optimizer to minimize the loss and update the model weights.
Step 3: Training the Models
Next, we train both the fine-tuned ResNet-50 and the custom CNN. The training process involves multiple epochs, during which the model learns to minimize the classification error on the CIFAR-10 dataset.

python
Copiar código
def train_model(model, train_dl, val_dl, criterion, optimizer, num_epochs=10):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_dl:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_dl.dataset):.4f}')
    print('Training Completed')

# Train the model
train_model(model, train_dl, val_dl, criterion, optimizer)
Step 4: Carlini & Wagner L2 Attack
Here, we implement the Carlini & Wagner attack using the L2 norm. The attack iteratively perturbs the input image by optimizing the adversarial loss, minimizing the perturbation while misleading the model.

python
Copiar código
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
Explanation:
Tanh Transformation: We use the tanh function to ensure that the pixel values of the perturbed image remain within valid bounds (0, 1).
Optimization: The Adam optimizer is used to iteratively adjust the perturbations, minimizing the adversarial loss.
L2 Distance: The perturbation is minimized based on the L2 norm, which measures the Euclidean distance between the original and perturbed images.
Step 5: Visualizing the Results
Finally, we visualize the original image, the perturbed image, and the perturbation (difference between the two).

python
Copiar código
import matplotlib.pyplot as plt

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show_images(original_image, perturbed_image):
    images = [original_image, perturbed_image, perturbed_image - original_image]
    titles = ["Original Image", "Perturbed Image", "Perturbation"]
    for i, img in enumerate(images):
        plt.subplot(1, 3, i+1)
        imshow(torchvision.utils.make_grid(img))
        plt.title(titles[i])
    plt.show()

# Example of attack
dataiter = iter(test_dl)
original_image, label = next(dataiter)

# Run the attack
perturbed_image = cw_l2_attack(model, original_image, label)

# Show the images
show_images(original_image, perturbed_image)
Conclusion
The Carlini & Wagner L2 attack is a highly effective adversarial attack that minimizes perturbations while ensuring misclassification. By using the L2 norm and solving the optimization problem iteratively, we can craft adversarial examples that are nearly imperceptible but still cause the model to make incorrect predictions.

yaml
Copiar código

---

This methodology is now ready to help users understand the theory and practical implementation of the Carlini
