# Methodology Overview

This guide serves as a comprehensive, hands-on methodology for generating adversarial examples, providing users with both theoretical insights and practical implementations. It covers attacks on a wide range of machine learning systems, from image classification models to complex autonomous driving systems like **Openpilot**.

The guide is designed for users with a basic understanding of machine learning and neural networks, and it provides clear, step-by-step instructions for implementing adversarial attacks in various environments. Throughout the guide, users will explore both **white-box** and **black-box** adversarial strategies, applying them to different machine learning models, including:

- **Image classification models** such as custom-built neural networks and fine-tuned models like **ResNet-50**.
- **Autonomous driving systems** like Openpilot, focusing on the internal workings of its modules and deep learning models, such as the **Supercombo** model.

In addition to generating adversarial examples, this guide demonstrates how to simulate attacks in real-world scenarios using **CARLA**, a high-fidelity driving simulator. CARLA will be used to test adversarial attacks on autonomous vehicles in controlled environments, enabling users to observe the practical impact of these examples on complex systems.

By the end of this guide, users will have a solid understanding of:
- The nature and creation of adversarial examples.
- How to implement white-box and black-box attacks on various machine learning models.
- Security assessment of systems like Openpilot.
- How to use **CARLA** for testing adversarial examples in simulated autonomous driving scenarios.

The methodology presented in this guide offers a structured approach for both beginners and intermediate users who wish to delve into the field of **Adversarial Machine Learning** and its application in critical real-world systems like image recognition and autonomous driving.

# What is the Openpilot Autonomous Driving System?
**Openpilot** is an open-source autonomous driving system developed by Comma.ai that provides advanced driver-assistance functionalities, such as Adaptive Cruise Control (ACC) and Lane Keeping Assist System (LKAS). It operates by processing data from cameras, radars, and other sensors through deep learning models, such as the **Supercombo model**, which performs end-to-end driving tasks, including lane detection, vehicle following, and road edge identification.

As an open-source project, Openpilot is particularly susceptible to adversarial attacks. Adversarial examples could cause the system to misinterpret its environment, leading to dangerous situations such as improper lane changes or failure to recognize obstacles. In this project, we explore how adversarial examples can trick the perception of machine learning models within Openpilot, demonstrating the practical risks posed by these attacks.

# Table of Contents
- [Introduction](#introduction)
- [Openpilot Version History](#openpilot-version-history)
- [Accidents Related to Openpilot](#accidents-related-to-openpilot)
- [Methodology](#methodology)
- [Openpilot internals](#openpilot-internals)
- [White-Box Attacks](#white-box-attacks)
- [Black-Box Attacks](#black-box-attacks)
- [Installation and Usage](#installation-and-usage)
- [Conclusions](#conclusions)

# Introduction

## What are Machine Learning Models?
**Machine Learning (ML)** models are algorithms that learn patterns from data to make predictions or decisions based on new, unseen data. These models range from simple linear regressions to complex DNNs, which are capable of recognizing intricate patterns in large datasets. DNNs have become the cornerstone of modern artificial intelligence, powering systems in fields like image classification, speech recognition, and autonomous driving.

## What is an Adversarial Example?
An [**Adversarial Example**](https://www.tensorflow.org/tutorials/generative/adversarial_fgsm) **(AE)** is a carefully crafted input designed to deceive deep neural network (DNN)-based models into making incorrect predictions. These examples are typically created by adding small, often imperceptible, perturbations to the original input data, such as images, which causes the model to misclassify or predict erroneously with high confidence. While adversarial perturbations may be subtle and unnoticeable to humans, they can significantly disrupt machine learning models, highlighting vulnerabilities that are especially concerning in safety-critical applications like autonomous driving and facial recognition.

![Adversarial Example Perturbation](images/ae_traffic_sign.png)

### Example:
Imagine an image classification model that correctly identifies a stop sign. By adding a minimal adversarial perturbation to the image, the model might mistakenly classify the stop sign as a speed limit sign, posing serious safety risks in an autonomous driving context. This type of vulnerability exemplifies the critical importance of understanding and mitigating adversarial examples.

## What is Adversarial Machine Learning?
**Adversarial Machine Learning (AML)** is a field that investigates how adversarial examples exploit the weaknesses of machine learning models. AML explores the methods attackers use to generate adversarial examples and focuses on crafting defenses to make models more robust. Attacks in this domain are typically divided into two categories: **white-box** and **black-box** attacks.

AML is of great importance in areas where security and safety are critical, such as autonomous vehicles.

## White-box and Black-box Approaches
This guide will introduce and implement both **white-box** and **black-box** adversarial example attacks:
- **White-box attacks** assume the attacker has complete knowledge of the model, including its structure, parameters, and training data. An example of this is the **Carlini & Wagner (CW) attack**, which is highly effective in finding adversarial examples by minimizing the perturbation needed to mislead the model.
- **Black-box attacks**, on the other hand, assume the attacker has no knowledge of the model. These attacks rely on probing the model through queries and observing the outputs to infer its vulnerabilities. Techniques like **Evolution Strategies (ES)** and **Gaussian Mutation** are commonly employed in black-box scenarios.

Both approaches will be explored in this project, targeting models trained from scratch and pre-trained models like **ResNet-50** for image classification, as well as the **Openpilot** autonomous driving system. Through a series of practical examples and step-by-step instructions, users will gain hands-on experience implementing these adversarial attack strategies.

# Openpilot Version History

This section will provide a brief summary of the key features and improvements in different versions of Openpilot since november 2021.

## [Openpilot 0.8.10](https://blog.comma.ai/0810release/)
- Introduced a new driving model trained on over 1 million minutes of driving, improving localization and cut-in prediction. Updated driver monitoring model with wider FOV for comma three.

## [Openpilot 0.8.11](https://blog.comma.ai/0811release/)
- Introduced smoother acceleration trajectories to improve user comfort and support for CAN FD for more vehicles. Added six new car ports.

## [Openpilot 0.8.12](https://blog.comma.ai/0812release/)
- Improved longitudinal control, redesigned the alert system, and introduced new sounds for alerts. Enhanced stopping behavior and reduced follow distance.

## [Openpilot 0.8.13](https://blog.comma.ai/0813release/)
- Implemented improvements in laneline detection and enhanced longitudinal control, especially for stop-and-go traffic.

## [Openpilot 0.8.14](https://blog.comma.ai/0814release/)
- Further enhancements to longitudinal and lateral control systems, as well as bug fixes for braking disengagements in supported cars.

## [Openpilot 0.8.15](https://blog.comma.ai/0815release/)
- Improved model performance by reducing unplanned disengagements and introduced the ability to export and share video clips of drives. UI updates and stability improvements.

## [Openpilot 0.8.16](https://blog.comma.ai/0816release/)
- Introduced a new stop-and-go longitudinal control and several user interface updates. Enhanced braking accuracy and responsiveness.

## [Openpilot 0.9.0](https://blog.comma.ai/090release/)
- Major improvements to driving models, with updates that included refined steering control and new platforms support.

## [Openpilot 0.9.2](https://blog.comma.ai/092release/)
- New dataset used for the driving model, expanding supported platforms. Improved path visualization for better trajectory understanding.

## [Openpilot 0.9.3](https://blog.comma.ai/093release/)
- Introduced fuzzy fingerprinting for Hyundai, Kia, and Genesis models, improving first-time setup. Driving personality settings were added to control how aggressively Openpilot drives.

## [Openpilot 0.9.4](https://blog.comma.ai/094release/)
- Enhanced driving performance in different environments and fixed several bugs related to steering control and braking.

## [Openpilot 0.9.5](https://blog.comma.ai/095release/)
- Implemented smoother handling of long turns, with more precise lateral control, focusing on user comfort during high-speed driving.

## [Openpilot 0.9.6](https://blog.comma.ai/096release/)
- Introduced a new driving model and improved driver monitoring, along with a neural-based steering control model. Fuzzy fingerprinting was further improved, and support was added for new Toyota models. The update also included a new format for log segment management and bug fixes in the CAN parser.

## [Openpilot 0.9.7](https://blog.comma.ai/097release/)
- New updates focused on long-term stability of the driving models, with further refinements to trajectory control.

# Accidents Related to Openpilot

## [1. Adversarial Perturbation Research (2018)](https://par.nsf.gov/biblio/10128310)
- **Incident**: Research from 2018 demonstrated that small, physical perturbations—such as black-and-white stickers on a stop sign—could trick neural networks into misclassifying the sign as a different traffic sign, such as a speed limit sign. This raised alarms about the vulnerability of autonomous driving systems to adversarial inputs in real-world settings.
- **Impact on Openpilot**: While the research did not directly target Openpilot, it highlighted the potential risks for vision-based ADAS like Openpilot, which rely heavily on camera inputs for road sign interpretation&#8203;:contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}.

## [2. Tesla Adversarial Image Attack (2020)](https://arxiv.org/abs/2003.01265)
- **Incident**: Researchers demonstrated how small, strategically placed stickers on a road could cause Tesla’s Autopilot system to misinterpret lane markings, leading to improper lane changes or unplanned deceleration.
- **Impact on Openpilot**: Similar adversarial attacks could potentially trick Openpilot’s camera-based system into making dangerous driving decisions by misreading lane lines&#8203;:contentReference[oaicite:2]{index=2}.

## [3. Adversarial Attack on Camera-based Perception Systems (2020)](https://keenlab.tencent.com/en/2020/03/30/Exploring-Security-Implications-of-AI-in-Autonomous-Driving-%E2%80%93-Case-Studies-on-Tesla/)
- **Incident**: Tencent’s Keen Security Lab researchers used projected images to fool Tesla’s Autopilot into misinterpreting lane markers and road signs. The manipulated inputs caused the system to take erroneous actions, like steering off-course or failing to stop.
- **Impact on Openpilot**: The attack highlighted the vulnerability of any ADAS relying on camera-based systems, including Openpilot, to adversarial attacks that could manipulate the system's perception of the road environment&#8203;:contentReference[oaicite:3]{index=3}.

## [4. Image Classifier Misinterpretation by Adversarial Attacks (2021)](https://arxiv.org/abs/2101.04232)
- **Incident**: Research showed that imperceptible noise added to images could cause deep neural networks to misinterpret traffic signs and other visual inputs, potentially leading to catastrophic decisions by autonomous driving systems.
- **Impact on Openpilot**: Given that Openpilot relies on convolutional neural networks for its perception model, it could also be susceptible to such adversarial examples, especially in earlier versions that may lack robust adversarial defenses&#8203;:contentReference[oaicite:4]{index=4}&#8203;:contentReference[oaicite:5]{index=5}.

## [5. Adversarial Examples Leading to Over- or Under-Braking (Theoretical Impact)](https://arxiv.org/abs/1807.00459)
- **Incident**: Researchers showed that adversarial perturbations could alter the perception of nearby obstacles in autonomous driving systems, causing them to either brake unnecessarily or fail to brake when needed.
- **Potential Openpilot Impact**: Although theoretical, this kind of attack could mislead Openpilot’s obstacle detection and lead to dangerous over- or under-braking, with severe consequences if exploited in real-world driving&#8203;:contentReference[oaicite:6]{index=6}.

# Methodology

A generical step-by-step methodology for researching and developing attacks, including the information gathering, tools setup, data collection, algorithm structures, and experiment results, can be found in [Methodology](docs/methodology.md).

Then, **white-box** attack and **black-box** strategies against Openpilot are explained more deeply in the following sections.

# Openpilot Internals

```TODO: Explicaciones del source code```

# White-Box Attacks

White-box attacks have full access to the target model, including its architecture, parameters, and weights. In this section, an algorithm is developed to craft an Adversarial Example, exploring how to manage data for the Supercombo model:
- Input data types: [YUV 4:2:0](https://github.com/peter-popov/unhack-openpilot) format
- Model Output: [Array](https://github.com/commaai/openpilot/blob/fa310d9e2542cf497d92f007baec8fd751ffa99c/selfdrive/modeld/models/driving.h#L239) of 6120 floats
- Iterative algorithm: Train an Adversarial Example by making small changes (e.g. [FGSM tutorial](https://www.tensorflow.org/tutorials/generative/adversarial_fgsm))
- Loss function: [Disappearance attack](https://iccv21-adv-workshop.github.io/short_paper/yanan_arow.pdf)
- Gradients: Optimization vector indicating the direction in which the loss function increases the most
- Expectation over Transform: [EoT](https://proceedings.mlr.press/v80/athalye18b/athalye18b.pdf) consists of applying transformations to the Adversarial Example to make more robust the effect under different conditions

Learn more in [White-Box Attacks](docs/white-box.md).

# Black-Box Attacks

In black-box attacks, the attacker only has access to the inputs and outputs of the model. This means that the Supercombo model cannot be used, therefore Evolution Strategies and Gaussian mutations are implemented.

Learn more in [Black-Box Attacks](docs/black-box.md).

# Installation and Usage

## Prerequisites
- Python 3.x
- OpenPilot 0.9.4
- TensorFlow/PyTorch

## Setup
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

# Conclusions

```TODO: a partir de la version 0.9.0 surgen todas las mejoras en las DNNs por lo que desde la 0.8.3 se han solucionado los EA: https://ar5iv.labs.arxiv.org/html/2103.00345```
