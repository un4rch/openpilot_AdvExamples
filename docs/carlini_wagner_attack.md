# Carlini & Wagner (CW) L2 Attack

## Table of contents
- [Introduction](#introduction)

## Introduction

The **Carlini & Wagner (CW) attack** is a powerful white-box adversarial attack that seeks to minimize the amount of perturbation added to an input image while successfully fooling the target model. The CW attack focuses on making small, often imperceptible changes to the image to mislead a deep neural network (DNN) into making incorrect classifications.

The attack is formulated as an optimization problem where the goal is to find the smallest perturbation to fool the model, measured by the **L2 norm** (Euclidean distance). This attack is particularly effective against models that are highly robust to simpler adversarial attacks like FGSM.

For this implementation, we will target two models:
- **ResNet-50**: A fine-tuned pre-trained model with the CIFAR-10 dataset.
- **Custom CNN**: A CNN built and trained from scratch on the CIFAR-10 dataset.
