# OpenPilot Black-box Adversarial Attack
This guide outlines the implementation of blackbox optimization using the (1+1) Evolution Strategy (ES) with Gaussian mutation for black-box optimization scenarios. The approach focuses on generating Adversarial Examples (AEs) for attacking deep neural networks, specifically, the Openpilot Supercombo model, in environments where model parameters are inaccessible.

## Table of contents
- [Introduction](#introduction)
- [(1+1) Evolution Strategy](#11-evolution-strategy)
- [Gaussian Mutation](#gaussian-mutation)
- [Disappearance Loss Function](#Disappearance-Loss-Function)
- [Black-Box Algorithm](#black-box-algorithm)
  - [Environment Setup](#environment-setup)
  - [Implementation Steps](#implementation-steps)
- [Visualization and Monitoring](#visualization-and-monitoring)

## Introduction
Black-box optimization is a method where the inner workings of a system are unknown, and only input-output relations are accessible. For optimizing such systems, [(1+1) Evolution Strategy (ES)](https://watermark.silverchair.com/evco_a_00248.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAzgwggM0BgkqhkiG9w0BBwagggMlMIIDIQIBADCCAxoGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQM5asW0jnGUK5Ntr-rAgEQgIIC6-Qe-zgkQ4y8cRdKKQRMNmrN7a8RgV6ZXP0cCep4WURMnPE923Cm3O5W3RainU3r5PDqDcpays_D-2oLtRZv6NiVGDXsrB28hWTRauo8jcJWCLiBrthjJwBT9UAueoboNzD4s8_dmIyb9Z9jCaMtNbRpN27SBki3hTnTPsylzGXDJnF6UhrVe8pdSRa6AQjoAycqzjRGEugjQ6vLRf8r1VTxnVMhwCKeClJ4avl3JFsjlxRZtoBpIW3MS1xGSdZdD-vfxn0u3kOysKXXxYlSGuAWMDScgTazsJC_R78h4zHV2LJ06QEVik-4d9YXIwXKu7JATl7erpS1k3hNsm2RzkpXSyXo8CUCFz1ooxF3FXVenThxmI8Qr_b8Ak9hoLGuxH71fKzn5e122KtRIrPDDB1P2_3IvnqZ1X9T6lPAEKrgenNOB-LHQ9tDwQgE2xN-E1sBVkhLD5-BIdJirM5aYdGAtbWjZY4IVBjUMecXeLqzCAgYamPgJpxYd_Y8epnYzqbulTM_JIrTjyJgVrGEd-wdecMeHDvMiYWpSAj7sJ5dZQyUXa41p5UsUP0PUI-27XRkgmNmdifrkFLSVbxaJ2X-zU0u4bmKkq82hFnsDyHFPDjnO3oYgkDr94S0QCP0cgCZagDqpN4oZZzeF8Ww2qksw3C38hlM7hWd6QjFkiLBRZ4zzMaaExQhFfGVIWNzi-NuN3iWEdDsMvb_z0Sqe4Kh3l2n7cAZs0_ju22cFEKatmrLfg15toptzCUTjxkoyqO3dXb6hUXGiDMo0tkGKrT8Ny08BLbMH8TYG9sAY83B0mZFJQ3kNvoAj2jikUfLLp1GZNYeanIYuYrfi90fQR4W18IPT0EJCSYBbw4IdIqTMPbc6i-m9alsGGm99R1rcz-2t05pn2-RTmxH9NJ8wzNPTBYXohjWPIJrl-fIf1U9w1qCo7CvHUdbJPx7N6gX_eZLEd4jDIhvsPAPyXQvUbC75-2ps5fRpFLdRQ) combined with [Gaussian mutation](https://arxiv.org/pdf/2002.03001) is an effective method.

This repository applies this technique to adversarial attacks in a black-box setting where the goal is to find an adversarial example (AE) that perturbs the system's output in a desired direction. The methodology assumes that there is no access to the internal parameters of the model, relying only on observing changes in the model's output during iterations.

## (1+1) Evolution Strategy
The (1+1) ES is a simple optimization strategy that simulates the process of natural selection. It works by generating a new solution (offspring) from an existing solution (parent) in each generation. If the offspring outperforms the parent, it replaces the parent; otherwise, the parent is retained for the next iteration. In this guide, the (1+1) ES is used to find an optimal AE for black-box systems by evaluating the performance of the AE in a simulated environment.

### Algorithm Outline:
1. **Initialize**: Start with a random adversarial example.
2. **Mutation**: Generate offspring by mutating the parent solution using Gaussian mutation.
3. **Evaluation**: Evaluate the offspring by running it through the black-box system and comparing its performance to the parent.
4. **Selection**: If the offspring performs better, it becomes the new parent; otherwise, the parent remains unchanged.
5. **Repeat**: This process continues for a predefined number of iterations or until a desired performance is achieved.

## Gaussian Mutation
Gaussian mutation introduces random perturbations to the offspring, ensuring that the mutation does not follow a fixed pattern. This randomness helps explore the solution space more effectively.

```python
def gaussian_mutation(image, sigma=0.05, blur_sigma=1):
    """
    Apply Gaussian mutation to an image.
    """
    noise = np.random.normal(0, sigma, image.shape)
    smooth_noise = gaussian_filter(noise, sigma=blur_sigma)
    mutated_image = image + smooth_noise
    mutated_image = np.clip(mutated_image, 0, 255).astype(np.uint8)
    return mutated_image
```
This function adds Gaussian noise to the image, making the mutation gradual and smooth, which helps in finding better solutions.

## Disappearance Loss Function
This loss function is designed to penalize high confidence values whilst maximizing the perception of the lead distance with respect to the lead vehicle. Additionally, values a smooth pattern of the patch.

```python
def disappearance_loss(image, conf, dist, real_dist, l1=0.01, l2=0.001):
    l_conf = -math.log(1 - conf)
    l_dis = -abs(dist / real_dist)
    ind = np.arange(0, 50 - 1)
    l_tv = np.sum(abs(image[ind + 1, :, :] - image[ind, :, :])) + np.sum(abs(image[:, ind + 1, :] - image[:, ind, :]))
    loss = l_conf + l1 * l_dis + l2 * l_tv
    return loss
```

## Black-Box Algorithm
![Black-box graphical diagram](/images/blackbox_attack.jpg)
### Environment Setup
This black-box attack is going to be performed during simulation loop in CARLA simulator. To setup the environment, read [this](/README.md#running-openpilot-in-carla-simulator) guide about CARLA simulation installation and establishing a connection with Openpilot.

Additionally, place this [bridge.py](/attacks/bridge.py) file in your `openpilot/tools/sim` directory. This customized bridge file contains the whole black-box attack implementation.

### Implementation Steps
1. **Initialize the Population**: A random adversarial example (AE) is generated. This example is then evaluated by running it through the black-box system.
2. **Mutate the Offspring**: Using Gaussian mutation, the offspring is generated from the parent AE. This offspring is a slightly modified version of the parent based on random noise.
3. **Evaluate the Offspring**: Run the simulation using the generated offspring and collect output data, such as detection confidence and predicted distances in the black-box system.
4. **Select the Best Solution**: The current and previous solutions (parent and offspring) are compared based on their performance in the simulation. The better solution is selected for the next iteration.
```python
def one_plus_one_evolution_strategy_algorithm(data_list, lr=100, sgth=25):
    d_mean = np.mean([elem[2] for elem in data_list])
    c_mean = np.mean([elem[3] for elem in data_list])
    l_mean = np.mean([elem[4] for elem in data_list])
    d_mean_prev = np.mean([elem[2] for elem in patch_prev_info[1]])
    c_mean_prev = np.mean([elem[3] for elem in patch_prev_info[1]])
    l_mean_prev = np.mean([elem[4] for elem in patch_prev_info[1]])

    if (d_mean > d_mean_prev and c_mean < c_mean_prev) or patch_prev is None:
        patch_next = gaussian_mutation(patch_act, lr)
    else:
        patch_next = gaussian_mutation(patch_prev, lr)

    return patch_next
```

## Visualization and Monitoring
During optimization, it is crucial to visualize how the performance evolves across iterations. Charts showing detection confidence, adversarial distances, and other metrics can help determine if the AE is improving. This [python script](/attacks/generate_charts.py) creates charts of the gathered information metrics.

### Monitoring Metrics:
- **Detection confidence**: How confident the system is in detecting an object.
- **Distance predictions**: Comparing predicted distances with real distances to assess the impact of the AE.
