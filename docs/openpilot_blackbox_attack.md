# OpenPilot Black-box Adversarial Attack
This guide outlines the implementation of blackbox optimization using the (1+1) Evolution Strategy (ES) with Gaussian mutation for black-box optimization scenarios. The approach focuses on generating Adversarial Examples (AEs) for attacking deep neural networks, specifically, the Openpilot Supercombo model, in environments where model parameters are inaccessible.

## Table of contents
- [Introduction](#introduction)
- [(1+1) Evolution Strategy](#11-evolution-strategy)
- [Gaussian Mutation](#gaussian-mutation)
- [Disappearance Loss Function](#Disappearance-Loss-Function)
- [Black-Box Algorithm](#black-box-algorithm)

## Introduction
Black-box optimization is a method where the inner workings of a system are unknown, and only input-output relations are accessible. For optimizing such systems, [(1+1) Evolution Strategy (ES)](https://watermark.silverchair.com/evco_a_00248.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAzgwggM0BgkqhkiG9w0BBwagggMlMIIDIQIBADCCAxoGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQM5asW0jnGUK5Ntr-rAgEQgIIC6-Qe-zgkQ4y8cRdKKQRMNmrN7a8RgV6ZXP0cCep4WURMnPE923Cm3O5W3RainU3r5PDqDcpays_D-2oLtRZv6NiVGDXsrB28hWTRauo8jcJWCLiBrthjJwBT9UAueoboNzD4s8_dmIyb9Z9jCaMtNbRpN27SBki3hTnTPsylzGXDJnF6UhrVe8pdSRa6AQjoAycqzjRGEugjQ6vLRf8r1VTxnVMhwCKeClJ4avl3JFsjlxRZtoBpIW3MS1xGSdZdD-vfxn0u3kOysKXXxYlSGuAWMDScgTazsJC_R78h4zHV2LJ06QEVik-4d9YXIwXKu7JATl7erpS1k3hNsm2RzkpXSyXo8CUCFz1ooxF3FXVenThxmI8Qr_b8Ak9hoLGuxH71fKzn5e122KtRIrPDDB1P2_3IvnqZ1X9T6lPAEKrgenNOB-LHQ9tDwQgE2xN-E1sBVkhLD5-BIdJirM5aYdGAtbWjZY4IVBjUMecXeLqzCAgYamPgJpxYd_Y8epnYzqbulTM_JIrTjyJgVrGEd-wdecMeHDvMiYWpSAj7sJ5dZQyUXa41p5UsUP0PUI-27XRkgmNmdifrkFLSVbxaJ2X-zU0u4bmKkq82hFnsDyHFPDjnO3oYgkDr94S0QCP0cgCZagDqpN4oZZzeF8Ww2qksw3C38hlM7hWd6QjFkiLBRZ4zzMaaExQhFfGVIWNzi-NuN3iWEdDsMvb_z0Sqe4Kh3l2n7cAZs0_ju22cFEKatmrLfg15toptzCUTjxkoyqO3dXb6hUXGiDMo0tkGKrT8Ny08BLbMH8TYG9sAY83B0mZFJQ3kNvoAj2jikUfLLp1GZNYeanIYuYrfi90fQR4W18IPT0EJCSYBbw4IdIqTMPbc6i-m9alsGGm99R1rcz-2t05pn2-RTmxH9NJ8wzNPTBYXohjWPIJrl-fIf1U9w1qCo7CvHUdbJPx7N6gX_eZLEd4jDIhvsPAPyXQvUbC75-2ps5fRpFLdRQ) combined with [Gaussian mutation](https://arxiv.org/pdf/2002.03001) is an effective method.

This repository applies this technique to adversarial attacks in a black-box setting where the goal is to find an adversarial example (AE) that perturbs the system's output in a desired direction. The methodology assumes that there is no access to the internal parameters of the model, relying only on observing changes in the model's output during iterations.

![TODO: graphical abstract]()

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
xdxdxd
