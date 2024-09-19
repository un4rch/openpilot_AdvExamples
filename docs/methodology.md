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

In whitebox attacks, the attacker has full access to the model, including its architecture, weights, and gradients. This allows for precise perturbations to be crafted that target specific weaknesses in the model.

### Steps for Whitebox Attacks

1. **Model Access and Analysis**: 
    - Acquire the model architecture and weights of the target system (e.g., Openpilot’s neural networks).
    - Understand the decision-making process, such as how the model perceives lane lines or road signs.

2. **Gradient Calculation**:
    - Compute the gradient of the model's loss function with respect to the input.
    - Use backpropagation to determine how small changes in the input affect the model's output.
  
3. **Crafting Perturbations**:
    - Apply gradient-based attack methods, such as the **Fast Gradient Sign Method (FGSM)**, which alters the input image based on the computed gradient.
    - For more sophisticated attacks, use the **Carlini & Wagner (C&W) Attack**, which minimizes the perturbation size while maximizing the model’s misclassification rate.

4. **Evaluation**:
    - Test the adversarial example against the target system, ensuring that the perturbations are imperceptible to human drivers.
    - Perform robustness checks across multiple inputs to assess the generalizability of the attack.

### Example: Carlini & Wagner Attack on Openpilot
We will perform a Carlini & Wagner attack on Openpilot’s image recognition models to test the system's robustness. We will compare the effectiveness of this attack between Openpilot versions 0.9.4 and 0.8.3 to assess whether newer versions offer better defense mechanisms.

## Blackbox Attacks

In blackbox attacks, the attacker has no direct access to the model’s internal parameters or architecture. Instead, they must rely on input-output pairs and make inferences about the model’s behavior.

### Steps for Blackbox Attacks

1. **Querying the Model**:
    - Send a series of inputs to the system (e.g., camera images of road signs or lanes) and record the outputs (e.g., the classification of the road sign or detected lane lines).
    - Build a surrogate model that mimics the behavior of the target model by training it on the input-output pairs obtained from the target system.

2. **Crafting Perturbations**:
    - Use optimization techniques like **Genetic Algorithms (GA)** or **Differential Evolution** to search for adversarial perturbations that lead to incorrect outputs from the surrogate model.
    - Generate adversarial examples by using methods such as **ZOO (Zeroth Order Optimization)** to iteratively perturb the inputs based on the surrogate model.

3. **Transferability**:
    - Transfer the adversarial examples generated using the surrogate model to the target model, as adversarial examples crafted for one model often transfer to others, especially if they share similar architectures.
  
4. **Evaluation**:
    - Test the adversarial examples on the target system (Openpilot) and measure the success rate of the attack. Evaluate the system's resilience by trying multiple inputs and conditions, such as different lighting or weather scenarios.

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
