# Adversarial Examples against OpenPilot 0.9.4

This repository explains methodologies to attack the OpenPilot 0.9.4 self-driving software using adversarial examples in both **white-box** and **black-box** settings. This user guide facilitates knowledge for new users who want to learn about adversarial examples and also provides new information to the field based on research of scientific papers and analysis of source code performed for the development of the algorithms.

## Table of Contents
- [Introduction](#introduction)
- [White-Box Attacks](#white-box-attacks)
- [Black-Box Attacks](#black-box-attacks)
- [Methodology](#methodology)
- [Installation and Usage](#installation-and-usage)
- [References](#references)

## Introduction

OpenPilot is an open-source software for autonomous vehicles. In this project, we explore how Adversarial Examples can trick the perception system of OpenPilot, causing it to make incorrect driving decisions, such as accelerating unsafely and causing a rear-end collision.

A [Physical Adversarial Example](https://github.com/weihui1308/PAA?tab=readme-ov-file) is a carefully crafted input designed to mislead the prediction of DNN-based models.

This guide employs both **white-box** (where the attacker has complete knowledge of the model) and **black-box** (where the attacker has no knowledge of the model) attack strategies.

## White-Box Attacks

White-box attacks have full access to the target model, including its architecture, parameters, and weights. In this section, an algorithm is developed to craft an Adversarial Example, exploring how to manage data for the Supercombo model:
- Input data types: [YUV 4:2:0](https://github.com/peter-popov/unhack-openpilot) format
- Model Output: [Array](https://github.com/commaai/openpilot/blob/fa310d9e2542cf497d92f007baec8fd751ffa99c/selfdrive/modeld/models/driving.h#L239) of 6120 floats
- Iterative algorithm: Train an Adversarial Example by making small changes
- Loss function: [Disappearance attack](https://iccv21-adv-workshop.github.io/short_paper/yanan_arow.pdf)
- Gradients: Optimization vector indicating the direction in which the loss function increases the most
- Expectation over Transform: [EoT](https://proceedings.mlr.press/v80/athalye18b/athalye18b.pdf) consists of applying transformations to the Adversarial Example to make more robust the effect under different conditions

Read more in [White-Box Attacks](docs/white-box.md).

## Black-Box Attacks

In black-box attacks, the attacker only has access to the inputs and outputs of the model. This means that the Supercombo model cannot be used, therefore Evolution Strategies and Gaussian mutations are implemented.

Learn more in [Black-Box Attacks](docs/black-box.md).

## Methodology

A detailed explanation of the methodology used for the attacks, including the setup, data collection, and experiment results, can be found in [Methodology](docs/methodology.md).

## Installation and Usage

### Prerequisites
- Python 3.x
- OpenPilot 0.9.4
- TensorFlow/PyTorch

### Setup
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/openpilot-adversarial-attacks.git
    cd openpilot-adversarial-attacks
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Follow the instructions in the `attacks` folder to execute white-box and black-box attacks.

## References

- [OpenPilot Documentation](https://github.com/commaai/openpilot)
- [Adversarial Attacks on Neural Networks](https://arxiv.org/abs/1412.6572)
