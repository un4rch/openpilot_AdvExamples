# Adversarial Examples against OpenPilot autonomous driving system

This repository explains methodologies to attack the OpenPilot 0.9.4 self-driving software using adversarial examples in both **white-box** and **black-box** settings. This user guide facilitates knowledge for new users who want to learn about adversarial examples and also provides new information to the field based on research of scientific papers and analysis of source code performed for the development of the algorithms.

## Table of Contents
- [Introduction](#introduction)
- [Openpilot Version History](#openpilot-version-history)
- [Accidents Related to Openpilot](accidents-related-to-openpilot)
- [Methodology](#methodology)
- [White-Box Attacks](#white-box-attacks)
- [Black-Box Attacks](#black-box-attacks)
- [Installation and Usage](#installation-and-usage)
- [References](#references)

## Introduction

OpenPilot is an open-source software for autonomous vehicles. In this project, we explore how Adversarial Examples can trick the perception system of OpenPilot, causing it to make incorrect driving decisions, such as accelerating unsafely and causing a rear-end collision.

A [Physical Adversarial Example](https://github.com/weihui1308/PAA?tab=readme-ov-file) is a carefully crafted input designed to mislead the prediction of DNN-based models.

This guide employs both **white-box** (where the attacker has complete knowledge of the model) and **black-box** (where the attacker has no knowledge of the model) attack strategies.

## Openpilot Version History

This section will provide a brief summary of the key features and improvements in different versions of Openpilot since november 2021.

### [Openpilot 0.8.10](https://blog.comma.ai/0810release/)
- Introduced a new driving model trained on over 1 million minutes of driving, improving localization and cut-in prediction. Updated driver monitoring model with wider FOV for comma three.

### [Openpilot 0.8.11](https://blog.comma.ai/0811release/)
- Introduced smoother acceleration trajectories to improve user comfort and support for CAN FD for more vehicles. Added six new car ports.

### [Openpilot 0.8.12](https://blog.comma.ai/0812release/)
- Improved longitudinal control, redesigned the alert system, and introduced new sounds for alerts. Enhanced stopping behavior and reduced follow distance.

### [Openpilot 0.8.13](https://blog.comma.ai/0813release/)
- Implemented improvements in laneline detection and enhanced longitudinal control, especially for stop-and-go traffic.

### [Openpilot 0.8.14](https://blog.comma.ai/0814release/)
- Further enhancements to longitudinal and lateral control systems, as well as bug fixes for braking disengagements in supported cars.

### [Openpilot 0.8.15](https://blog.comma.ai/0815release/)
- Improved model performance by reducing unplanned disengagements and introduced the ability to export and share video clips of drives. UI updates and stability improvements.

### [Openpilot 0.8.16](https://blog.comma.ai/0816release/)
- Introduced a new stop-and-go longitudinal control and several user interface updates. Enhanced braking accuracy and responsiveness.

### [Openpilot 0.9.0](https://blog.comma.ai/090release/)
- Major improvements to driving models, with updates that included refined steering control and new platforms support.

### [Openpilot 0.9.2](https://blog.comma.ai/092release/)
- New dataset used for the driving model, expanding supported platforms. Improved path visualization for better trajectory understanding.

### [Openpilot 0.9.3](https://blog.comma.ai/093release/)
- Introduced fuzzy fingerprinting for Hyundai, Kia, and Genesis models, improving first-time setup. Driving personality settings were added to control how aggressively Openpilot drives.

### [Openpilot 0.9.4](https://blog.comma.ai/094release/)
- Enhanced driving performance in different environments and fixed several bugs related to steering control and braking.

### [Openpilot 0.9.5](https://blog.comma.ai/095release/)
- Implemented smoother handling of long turns, with more precise lateral control, focusing on user comfort during high-speed driving.

### [Openpilot 0.9.6](https://blog.comma.ai/096release/)
- Introduced a new driving model and improved driver monitoring, along with a neural-based steering control model. Fuzzy fingerprinting was further improved, and support was added for new Toyota models. The update also included a new format for log segment management and bug fixes in the CAN parser.

### [Openpilot 0.9.7](https://blog.comma.ai/097release/)
- New updates focused on long-term stability of the driving models, with further refinements to trajectory control.

## Accidents Related to Openpilot

### [1. Adversarial Perturbation Research (2018)](https://people.csail.mit.edu/madry/lab/adversarial/examples/)
- **Incident**: In research by the MIT CSAIL, subtle perturbations were introduced into road sign images, tricking autonomous systems into misinterpreting them. Stop signs, for instance, were classified as speed limit signs by the system.
- **Impact on Openpilot**: Though no specific accidents involving Openpilot were reported, this research raised concerns about how easily ADAS systems relying on image classification can be fooled by adversarial inputs. This finding highlights a potential vulnerability in any system using vision-based models, including Openpilot.

### [2. Tesla Adversarial Image Attack (2020)](https://arxiv.org/abs/2003.01265)
- **Incident**: Researchers demonstrated how small stickers placed on the road could mislead Tesla’s Autopilot into changing lanes or slowing down unexpectedly. The adversarial inputs tricked the system into seeing non-existent obstacles or misreading lane lines.
- **Impact on Openpilot**: While the attack targeted Tesla’s Autopilot, the principle of this adversarial attack could be extended to Openpilot’s camera and vision-based systems, raising concerns about similar vulnerabilities.

### [3. Adversarial Attack on Camera-based Perception Systems (2020)](https://keenlab.tencent.com/en/2020/03/30/Exploring-Security-Implications-of-AI-in-Autonomous-Driving-%E2%80%93-Case-Studies-on-Tesla/)
- **Incident**: Researchers from Tencent’s Keen Security Lab successfully executed adversarial attacks on Tesla's Autopilot by projecting altered images onto a car’s path. These images caused the system to misinterpret lane markings and traffic signs, posing a potential crash risk.
- **Impact on Openpilot**: The attack method demonstrated a generalized threat to any camera-based ADAS, including Openpilot, especially in scenarios involving manipulated road markings or projected adversarial images.

### [4. Image Classifier Misinterpretation by Adversarial Attacks (2021)](https://arxiv.org/abs/2101.04232)
- **Incident**: Research conducted on deep neural networks used in ADAS showed how small, imperceptible noise added to images could cause significant misinterpretation of traffic scenes. For example, adding minor noise to images of traffic signs or pedestrians could lead to fatal decisions by the ADAS.
- **Impact on Openpilot**: Systems like Openpilot, which rely on convolutional neural networks for perception, could be susceptible to such attacks, especially in their earlier versions. This demonstrates a critical risk of adversarial attacks in real-world driving, although no specific accidents were attributed to Openpilot.

### [5. Adversarial Examples Leading to Over- or Under-Braking (Theoretical Impact)](https://arxiv.org/abs/1807.00459)
- **Incident**: Research in adversarial machine learning has demonstrated how adversarial perturbations can be introduced to alter the car's perception of nearby obstacles, potentially leading to false-positive or false-negative braking scenarios.
- **Potential Openpilot Impact**: Adversarial perturbations could mislead Openpilot’s obstacle detection system, causing it to either brake unnecessarily or fail to brake when needed, posing a significant risk in real-world driving. While this remains theoretical, the possibility of such attacks could lead to severe accidents if exploited in practice.

## Methodology

A generical step-by-step methodology for researching and developing attacks, including the information gathering, tools setup, data collection, algorithm structures, and experiment results, can be found in [Methodology](docs/methodology.md).

Then, **white-box** attack and **black-box** strategies are explained more deeply in the following sections.

## White-Box Attacks

White-box attacks have full access to the target model, including its architecture, parameters, and weights. In this section, an algorithm is developed to craft an Adversarial Example, exploring how to manage data for the Supercombo model:
- Input data types: [YUV 4:2:0](https://github.com/peter-popov/unhack-openpilot) format
- Model Output: [Array](https://github.com/commaai/openpilot/blob/fa310d9e2542cf497d92f007baec8fd751ffa99c/selfdrive/modeld/models/driving.h#L239) of 6120 floats
- Iterative algorithm: Train an Adversarial Example by making small changes (e.g. [FGSM tutorial](https://www.tensorflow.org/tutorials/generative/adversarial_fgsm))
- Loss function: [Disappearance attack](https://iccv21-adv-workshop.github.io/short_paper/yanan_arow.pdf)
- Gradients: Optimization vector indicating the direction in which the loss function increases the most
- Expectation over Transform: [EoT](https://proceedings.mlr.press/v80/athalye18b/athalye18b.pdf) consists of applying transformations to the Adversarial Example to make more robust the effect under different conditions

Learn more in [White-Box Attacks](docs/white-box.md).

## Black-Box Attacks

In black-box attacks, the attacker only has access to the inputs and outputs of the model. This means that the Supercombo model cannot be used, therefore Evolution Strategies and Gaussian mutations are implemented.

Learn more in [Black-Box Attacks](docs/black-box.md).

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
