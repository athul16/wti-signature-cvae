# WTI Signature CVAE — Project Context

## Project Role
This repository is Project 1 in a two-project architecture.

Its job is to learn a **generative model of WTI crude oil market paths** using:
- Path Signatures
- Conditional Variational Autoencoder (CVAE)
- Lead-Lag transform
- Signature-based loss (Sig-MMD)

The output of this repo feeds directly into:
../wti-rl-policy

This repo is NOT a predictor.
It is a **market simulator / synthetic data generator**.

---

# System Architecture (Two-Project Pipeline)

Project 1: wti-signature-cvae
    ↓ generates synthetic WTI paths
Project 2: wti-rl-policy
    ↓ trains RL trader
Final Output:
    learned trading policy robust across regimes

---

# Core Idea

We use **Path Signature Theory** as a universal feature map for financial paths.

Instead of modeling prices directly:
- transform path → lead-lag
- compute signature / log-signature
- compress into latent space
- decode into synthetic path

This preserves:
- volatility structure
- path geometry
- roughness
- higher order dependencies

---

# Mathematical Foundation

Signature of a path = iterated integrals

Level 1:
- displacement
- total return

Level 2:
- area
- volatility
- lead-lag interaction

Higher levels:
- nonlinear interactions
- regime structure

Key insight:
Any functional of a path can be approximated as linear in signature space.

---

# Lead-Lag Transform

Given price path P:

Lead path:
P_t

Lag path:
P_{t-1}

Creates 2D path:
(P_t , P_{t-1})

This encodes:
- quadratic variation
- volatility
- roughness

This is critical preprocessing.

---

# Model Architecture

Encoder:
- GRU or RNN
- input: signature / log-signature
- output: latent distribution

Latent Space:
- Gaussian
- represents market regimes
    - trending
    - mean reverting
    - high volatility
    - crash

Decoder:
- GRU
- input: latent vector
- output: synthetic price path

---

# Loss Function

Total Loss:

Reconstruction Loss
+ KL Divergence
+ Sig-MMD

Sig-MMD:
Measures distribution similarity in signature space.

This ensures generated paths:
- match stylized facts
- match distributional structure
- not just pointwise similarity

---

# Stylized Facts We Must Preserve

Generated data must exhibit:

1. Volatility clustering
2. Leverage effect
3. Fat tails
4. Non-Gaussian returns
5. Regime switching
6. Serial dependence in volatility

---

# Output Contract for RL Repo

This repo MUST output:

NPZ file:
paths: shape (N, T)

Preferred:
log returns

Alternative:
prices (must document clearly)

Optional metadata:
- regime labels
- volatility stats
- generator config
- seed
- latent samples

Saved to:
../wti-rl-policy/data/sims/

---

# References

Primary Research Foundations:

Financial Signature Theory — Miquel Noguer I Alonso
Market Generators — Horvath, Plenk, Vuletic
MMD Signature Learning — Generative modelling with MMD
SigMA — Signatures + Multihead Attention

---

# Implementation Priorities

1. Correct lead-lag transform
2. Correct signature computation
3. Stable CVAE training
4. Sig-MMD loss
5. Realistic path generation
6. Clean export interface

---

# DO NOT

- train RL here
- predict next step
- overfit to historical sample
- output unrealistic Gaussian noise

---

# This Repo Exists To

Generate **infinite realistic WTI market scenarios**
for downstream reinforcement learning.
