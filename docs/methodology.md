# Methodology for Adversarial Example Attacks

## Introduction
Adversarial example attacks involve introducing carefully crafted perturbations into input data to fool machine learning models, especially deep learning models. These perturbations are often imperceptible to human eyes but can significantly alter the output of the model. In the context of autonomous driving systems like Openpilot, adversarial attacks could trick the system into making erroneous driving decisions by manipulating its perception model.

This methodology outlines the steps for conducting adversarial attacks, focusing on both **whitebox** and **blackbox** approaches, with an emphasis on Openpilot's object detection systems and longitudinal planning.

## General Overview of Adversarial Attacks

Adversarial attacks aim to deceive machine learning models by introducing perturbations to their inputs. In the context of an autonomous driving system, these inputs could be camera images, lidar data, or radar signals. The goal is to manipulate the system into making incorrect predictions, such as not detecting existent obstacles. 

There are two main categories of adversarial attacks:
1. **Whitebox Attacks**: The attacker has full knowledge of the model architecture, parameters, and gradients.
2. **Blackbox Attacks**: The attacker has no direct access to the model’s internals and must rely only on input-output observations.

## Whitebox Attacks

A white-box attack consists of accessing the model's parameters, architecture, and internal workings. This allows the attacker to analyze the entire neural network, including its weights, biases, and the data flow through the network layers. To craft adversarial examples (AEs) against the Openpilot autonomous driving (AD) system, the white-box approach involves the following steps:

### Steps for Whitebox Attacks

1. **Model Analysis**:
   Understanding the architecture and components of the Openpilot system, including the neural network models it uses. The key resources for this analysis are:
   - [Openpilot components and architecture](https://blog.comma.ai/openpilot-in-2021/)
   - [Supercombo model](https://arxiv.org/pdf/2206.08176)
   - [Inputs for Openpilot](https://github.com/commaai/openpilot/tree/fa310d9e2542cf497d92f007baec8fd751ffa99c/selfdrive/modeld/models)
   - [Outputs of the driving model](https://github.com/commaai/openpilot/blob/fa310d9e2542cf497d92f007baec8fd751ffa99c/selfdrive/modeld/models/driving.h#L239)
   - [Qcom cameras](https://github.com/commaai/openpilot/tree/fa310d9e2542cf497d92f007baec8fd751ffa99c/system/camerad/cameras)
   - [Interaction with the Supercombo model](https://github.com/commaai/openpilot/blob/fa310d9e2542cf497d92f007baec8fd751ffa99c/selfdrive/modeld/models/driving.cc)
   - [Model execution](https://github.com/commaai/openpilot/blob/fa310d9e2542cf497d92f007baec8fd751ffa99c/selfdrive/modeld/runners/onnx_runner.py)
   - [Cereal messaging package](https://github.com/commaai/msgq/tree/a9082c826872e5650e8a8e9a6f3e5f95a4d27572)
   - [Visionipc system](https://github.com/commaai/msgq/tree/a9082c826872e5650e8a8e9a6f3e5f95a4d27572/visionipc)
   - [Simulation connection script to gather environmental data](https://github.com/commaai/openpilot/blob/fa310d9e2542cf497d92f007baec8fd751ffa99c/tools/sim/bridge.py)
   - [YUV image formats](https://github.com/peter-popov/unhack-openpilot) and [formats from another source](https://gist.github.com/Jim-Bar/3cbba684a71d1a9d468a6711a6eddbeb)
   - [Existing adversarial attack efforts on Openpilot](https://github.com/noobmasterbala/Adversarial-Attack-and-Defence-On-Openpilot) and [another project](https://github.com/MTammvee/openpilot-supercombo-model/tree/main)

2. **Model Format Conversion**:
   Since the Openpilot Supercombo model uses the ONNX format, it’s necessary to convert the model into a more manageable framework, such as PyTorch, to enable gradient-based attacks. Conversion can be done using libraries like [onnx2torch](https://pypi.org/project/onnx2torch/).

3. **Data Preparation**:
   Collect camera frame images from the car’s sensors. These images will be saved and used later for feeding into the Supercombo model to craft adversarial examples.

4. **Data Parsing**:
   Convert the data formats to ensure compatibility with the neural network's input specifications.

5. **Model Queries**:
   Access the model's parameters, such as weights and biases, to calculate gradients. These gradients are crucial for updating the adversarial example and steering the output toward the desired misclassification.

6. **Loss Function**:
   Define a loss function that measures the impact of the adversarial example. The loss will be used to update the pixels of the adversarial example iteratively.

7. **Iterative Algorithm**:
   Use an iterative process to query the Supercombo model, evaluate the output with the loss function, and update the adversarial example using gradients. Continue until the perturbation reaches a predetermined threshold or number of iterations.

8. **Testing and Validation**:
   Once the adversarial example is crafted, run simulations or real-world tests to validate its effectiveness. This step is crucial to ensure that the adversarial example consistently fools the model across various scenarios.

This method requires significant knowledge of the system's internals but can be highly effective in scenarios where the attacker has access to the model's parameters. The white-box approach is one of the most precise ways to craft adversarial examples as it allows for fine-tuning of perturbations based on complete access to the model.

## Blackbox Attacks

A black-box attack consists of crafting adversarial examples (AEs) without access to the model's architecture, weights, and biases. In this scenario, the attacker interacts with the system only through its inputs and outputs, treating the model as a "black box." The goal is to manipulate the input data, such as camera images, to mislead the system into making incorrect decisions. To attack the Openpilot autonomous driving (AD) system using a black-box strategy, the following steps can be followed (during simulation executions, customizing [bridge.py](https://github.com/commaai/openpilot/blob/fa310d9e2542cf497d92f007baec8fd751ffa99c/tools/sim/bridge.py) file):

### Steps for Blackbox Attacks

1. **Initialize Random Patch**:
   If no adversarial patch has been created yet, generate a random RGB pixel patch. This patch will serve as the basis for subsequent iterations and modifications.

2. **Place Patch**:
   The adversarial patch should be strategically placed on the rear part of the lead vehicle in the driving scenario. There are two approaches to do this:
   - **Approximate Positioning**: Use real-time simulation to approximate the patch’s position in each iteration. This method is faster but may lack precision.
   - **UnrealEditor4 Positioning**: Use the UnrealEditor4 simulation environment to place the patch more accurately during each iteration, although this method is slower due to the overhead of editing.

3. **Data Preparation**:
   Load the simulation data from previous iterations, enabling comparison of the current simulation’s output with previous runs. This comparison helps in assessing whether the adversarial patch is having the intended effect or needs further updates.

4. **Model Interaction**:
   Since you don’t have access to the internal workings of the model, interact with Openpilot’s end-to-end model via messaging packages, replicating how the real Openpilot interacts with the model. Use messaging frameworks such as [VisionIPC](https://github.com/commaai/msgq/tree/a9082c826872e5650e8a8e9a6f3e5f95a4d27572/visionipc) to send inputs and retrieve outputs.

5. **Loss Function**:
   Evaluate the model’s output by comparing it with the results from previous executions. Define a loss function that measures how well the adversarial patch is achieving its goal, such as causing lane deviations or improper braking decisions.

6. **Patch Update**:
   After evaluating the loss, update the adversarial patch’s parameters. These updates can be applied randomly, for example using a Gaussian distribution, or any other update rule that seeks to maximize the model’s misclassification.

7. **Iterative Method**:
   Continuously update the patch through an iterative process. After each iteration, evaluate the patch’s performance using the loss function and continue updating until a predefined threshold is reached, or a set number of iterations have been completed.

This black-box approach is less direct than the white-box method but is more practical in real-world settings where attackers typically do not have access to the internal model parameters. By interacting with the system only through its inputs and outputs, the attacker can iteratively refine adversarial examples to exploit vulnerabilities in Openpilot’s perception model.


## Defenses Against Adversarial Attacks

Adversarial example attacks pose a serious threat to autonomous driving systems. To mitigate these risks, several defense techniques can be implemented:

1. **Adversarial Training**:
    - During training, expose the model to adversarial examples so it learns to recognize and resist them. This approach improves the model’s robustness to adversarial inputs.
  
2. **Input Preprocessing**:
    - Apply preprocessing techniques such as **JPEG compression** or **image denoising** to remove adversarial noise before passing the input to the model.

3. **Gradient Masking**:
    - This defense method obfuscates the gradient of the model, making it difficult for attackers to compute effective perturbations. However, this technique is vulnerable to certain attack methods like **ZOO**.

4. **Model Regularization**:
    - Apply regularization techniques like **weight decay** and **dropout** to the model during training, which can make the model less sensitive to small perturbations in the input.

## Conclusion

Adversarial attacks represent a significant challenge to the safety and reliability of autonomous driving systems like Openpilot. While whitebox attacks allow for precise and targeted perturbations, blackbox attacks can still be highly effective even without full access to the model. Developing robust defenses and continuously testing against adversarial attacks is crucial to ensuring the long-term safety of autonomous vehicles.

---

**References**:
- [Goodfellow et al. (2015) - Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572)
- [Carlini & Wagner (2017) - Towards Evaluating the Robustness of Neural Networks](https://arxiv.org/abs/1608.04644)
- [Kurakin et al. (2016) - Adversarial Machine Learning at Scale](https://arxiv.org/abs/1611.01236)
- [Madry et al. (2017) - Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083)
