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

# Table of Contents
- [Introduction](#introduction)
  - [What are Machine Learning Models?](#what-are-machine-learning-models)
  - [What is an Adversarial Example?](#What-is-an-Adversarial-Example)
  - [What is Adversarial Machine Learning?](#What-is-Adversarial-Machine-Learning)
  - [White-box and Black-box Approaches](#White-box-and-Black-box-Approaches)
- [Openpilot](#openpilot)
  - [What is the Openpilot Autonomous Driving System?](#what-is-the-openpilot-autonomous-driving-system)
  - [Openpilot Version History](#openpilot-version-history)
  - [Accidents Related to Openpilot](#accidents-related-to-openpilot)
  - [Openpilot Internals](#openpilot-internals)
- [Running Openpilot in CARLA simulator](#running-openpilot-in-carla-simulator)
  - [Setup](#setup)
  - [...]()
- [White-Box Attacks](#white-box-attacks)
  - [Carlini & Wagner against image classification](#carlini-&-wagner-against-image-classification)
  - [Openpilot](#openpilot)
- [Black-Box Attacks](#black-box-attacks)
  - [Openpilot](#openpilot)
- [Installation and Usage](#installation-and-usage)
- [Conclusions](#conclusions)

# Introduction

## What are Machine Learning Models?
**Machine Learning (ML)** models are algorithms that learn patterns from data to make predictions or decisions based on new, unseen data. These models range from simple linear regressions to complex DNNs, which are capable of recognizing intricate patterns in large datasets. DNNs have become the cornerstone of modern artificial intelligence, powering systems in fields like image classification, speech recognition, and autonomous driving.

## What is an Adversarial Example?
An [**Adversarial Example**](https://www.tensorflow.org/tutorials/generative/adversarial_fgsm) **(AE)** is a carefully crafted input designed to deceive deep neural network (DNN)-based models into making incorrect predictions. These examples are typically created by adding small, often imperceptible, perturbations to the original input data, such as images, which causes the model to misclassify or predict erroneously with high confidence. While adversarial perturbations may be subtle and unnoticeable to humans, they can significantly disrupt machine learning models, highlighting vulnerabilities that are especially concerning in safety-critical applications like autonomous driving and facial recognition.

![Adversarial Example Perturbation](images/ae_traffic_sign.png)

### [Example](https://arxiv.org/pdf/1801.02780):
Imagine an image classification model that identifies a benign logo. By adding a minimal adversarial perturbation to the image, the model might mistakenly classify it as a stop sign with a confidence of 1.0, posing serious safety risks in an autonomous driving context. This type of vulnerability exemplifies the critical importance of understanding and mitigating adversarial examples.

## What is Adversarial Machine Learning?
**Adversarial Machine Learning (AML)** is a field that investigates how adversarial examples exploit the weaknesses of machine learning models. AML explores the methods attackers use to generate adversarial examples and focuses on crafting defenses to make models more robust. Attacks in this domain are typically divided into two categories: **white-box** and **black-box** attacks.

AML is of great importance in areas where security and safety are critical, such as autonomous vehicles.

## White-box and Black-box Approaches
This guide will introduce and implement both **white-box** and **black-box** adversarial example attacks:
- **White-box attacks** assume the attacker has complete knowledge of the model, including its structure, parameters, and training data. An example of this is the **Carlini & Wagner (CW) attack**, which is highly effective in finding adversarial examples by minimizing the perturbation needed to mislead the model.
- **Black-box attacks**, on the other hand, assume the attacker has no knowledge of the model. These attacks rely on probing the model through queries and observing the outputs to infer its vulnerabilities. Techniques like **Evolution Strategies (ES)** and **Gaussian Mutation** are commonly employed in black-box scenarios.

Both approaches will be explored in this project, targeting models trained from scratch and pre-trained models like **ResNet-50** for image classification, as well as the **Openpilot** autonomous driving system. Through a series of practical examples and step-by-step instructions, users will gain hands-on experience implementing these adversarial attack strategies.

# Openpilot
## What is the Openpilot Autonomous Driving System?
[**Openpilot**](https://comma.ai/openpilot) is an [open-source](https://github.com/commaai/openpilot), advanced driver assistance system (ADAS) developed by Comma.ai. It provides autonomous driving capabilities such as **adaptive cruise control (ACC)**, **lane keeping assistance (LKAS)**, and **forward collision warnings (FCW)**. Openpilot uses a combination of sensors (including cameras, radars, and GNSS) along with deep learning models to interpret the driving environment and make real-time decisions. It has been designed to enhance the driving experience, offering partial autonomy that assists drivers in various tasks.

At the heart of Openpilot is a **deep neural network (DNN)**, specifically the [**Supercombo model**](https://github.com/commaai/openpilot/tree/master/selfdrive/modeld/models), which processes sensor inputs and predicts driving actions such as lane positioning, speed adjustments, and obstacle detection. The system integrates with supported vehicles' onboard sensors and actuators to control acceleration, braking, and steering autonomously, reducing the driver's workload and improving safety.

Openpilot is not a fully autonomous system; it is classified as **Level 2 autonomy**, meaning that while the system can manage some driving tasks, the driver is required to remain attentive and ready to take control at any time.

## Openpilot Version History

Since its initial release, Openpilot has evolved some versions, each improving on the system's robustness, functionality, and ease of use. Below is a brief overview of the key Openpilot versions and their main improvements:

### Version 0.1 (2016-11-29)
- Initial release of Openpilot.
- **Adaptive Cruise Control (ACC)** and **Lane Keep Assist (LKA)** functionalities were introduced.
- Supported vehicles: **Acura ILX 2016** with AcuraWatch Plus and **Honda Civic 2016 Touring Edition**.

### Version 0.2 (2016-12-12)
- Major refactor of the **controls** and **vision** modules.
- Added **car and radar abstraction layers** for broader compatibility.
- Docker container introduced for testing on a PC.
- **Plant model** for testing maneuvers was shipped}.

### Version 0.3 (2017-05-12)
- Added **CarParams** struct to improve abstraction layers.
- New model trained with more crowdsourced data for improved lane tracking.
- Initial **GPS** and **navigation support**.

### Version 0.4 (2018-01-18)
- Significant updates to **UI aesthetics** and **autofocus**.
- Added alpha support for **2017 Toyota Corolla** and **2017 Lexus RX Hybrid**.
- Focus on improving lane tracking when only one lane line is detected.

### Version 0.5 (2018-07-11)
- Introduced **Driver Monitoring (beta)** feature.
- Major efficiency improvements to **vision**, **logger**, and **UI** modules.
- New side-bar with stats, making UI more intuitive.

### Version 0.6 (2019-07-01)
- **New driving model**: increased the temporal context tenfold, making lane-keeping more reliable.
- Openpilot now uses only ~65% of CPU resources, improving system stability and efficiency.

### Version 0.7 (2019-12-13)
- Introduced **Lane Departure Warning (LDW)** for all supported vehicles.
- **Supercombo model**: Combined calibration and driving models for better estimates of lead vehicles.

### Version 0.8 (2020-11-30)
- Fully 3D driving model introduced, significantly improving cut-in detection.
- **UI** now draws 2 road edges, 4 lane lines, and paths in 3D.

### Version 0.9 (2023-11-17)
- New **Vision Transformer Architecture** for improved driving performance.
- Major refinements in lane-keeping and obstacle avoidance, and enhanced compatibility with new Toyota models.

These updates reflect the ongoing improvements in Openpilot's neural network models, safety systems, and vehicle compatibility, ensuring the system remains at the forefront of autonomous driving technology.

## Accidents Related to Openpilot

### [1. Adversarial Perturbation Research (2018)](https://par.nsf.gov/biblio/10128310)
- **Incident**: Research from 2018 demonstrated that small, physical perturbations—such as black-and-white stickers on a stop sign—could trick neural networks into misclassifying the sign as a different traffic sign, such as a speed limit sign. This raised alarms about the vulnerability of autonomous driving systems to adversarial inputs in real-world settings.
- **Impact on Openpilot**: While the research did not directly target Openpilot, it highlighted the potential risks for vision-based ADAS like Openpilot, which rely heavily on camera inputs for road sign interpretation&#8203;:contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}.

### [2. Tesla Adversarial Image Attack (2020)](https://arxiv.org/abs/2003.01265)
- **Incident**: Researchers demonstrated how small, strategically placed stickers on a road could cause Tesla’s Autopilot system to misinterpret lane markings, leading to improper lane changes or unplanned deceleration.
- **Impact on Openpilot**: Similar adversarial attacks could potentially trick Openpilot’s camera-based system into making dangerous driving decisions by misreading lane lines&#8203;:contentReference[oaicite:2]{index=2}.

### [3. Adversarial Attack on Camera-based Perception Systems (2020)](https://keenlab.tencent.com/en/2020/03/30/Exploring-Security-Implications-of-AI-in-Autonomous-Driving-%E2%80%93-Case-Studies-on-Tesla/)
- **Incident**: Tencent’s Keen Security Lab researchers used projected images to fool Tesla’s Autopilot into misinterpreting lane markers and road signs. The manipulated inputs caused the system to take erroneous actions, like steering off-course or failing to stop.
- **Impact on Openpilot**: The attack highlighted the vulnerability of any ADAS relying on camera-based systems, including Openpilot, to adversarial attacks that could manipulate the system's perception of the road environment&#8203;:contentReference[oaicite:3]{index=3}.

### [4. Image Classifier Misinterpretation by Adversarial Attacks (2021)](https://arxiv.org/abs/2101.04232)
- **Incident**: Research showed that imperceptible noise added to images could cause deep neural networks to misinterpret traffic signs and other visual inputs, potentially leading to catastrophic decisions by autonomous driving systems.
- **Impact on Openpilot**: Given that Openpilot relies on convolutional neural networks for its perception model, it could also be susceptible to such adversarial examples, especially in earlier versions that may lack robust adversarial defenses&#8203;:contentReference[oaicite:4]{index=4}&#8203;:contentReference[oaicite:5]{index=5}.

### [5. Adversarial Examples Leading to Over- or Under-Braking (Theoretical Impact)](https://arxiv.org/abs/1807.00459)
- **Incident**: Researchers showed that adversarial perturbations could alter the perception of nearby obstacles in autonomous driving systems, causing them to either brake unnecessarily or fail to brake when needed.
- **Potential Openpilot Impact**: Although theoretical, this kind of attack could mislead Openpilot’s obstacle detection and lead to dangerous over- or under-braking, with severe consequences if exploited in real-world driving&#8203;:contentReference[oaicite:6]{index=6}.

## Openpilot Internals

```TODO: Explicaciones del source code```

# Running Openpilot in CARLA simulator

TODO

# White-Box Attacks

- C&W
- Openpilot

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
