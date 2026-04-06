# AGENTS.md

## Project
WTI Signature CVAE

## Purpose
This repository is Project 1 of a two-project system.

Its job is to learn a generative model of WTI crude oil market paths using a signature-based conditional variational autoencoder (Sig-CVAE / CVAE with path-signature features).

The goal is not just next-step prediction. The goal is to generate realistic synthetic WTI market histories that preserve important statistical and financial properties of real data.

These generated paths will be used downstream by a second repository:
`wti-rl-policy`

## System-level context
Overall pipeline:

1. Real WTI market data
2. Preprocessing / path construction
3. Lead-lag transform
4. Signature or log-signature computation
5. Encoder maps path features into latent space
6. Decoder generates synthetic price/return paths
7. Export generated simulated paths
8. `wti-rl-policy` consumes those paths to train a DQN trading policy

## Core modeling idea
The path signature is the main feature representation.
The model should learn latent market regimes and generate realistic alternative return paths.

Important stylized facts to preserve:
- volatility clustering
- leverage effect
- fat tails
- realistic dependence structure across time

## Expectations for the codebase
When analyzing or editing this repo:

- First explain the full pipeline before changing code
- Identify how raw data becomes model-ready input
- Identify where lead-lag transforms are applied
- Identify where signature/log-signature features are computed
- Identify encoder, latent space, decoder, and loss logic
- Check whether Sig-MMD, reconstruction loss, and KL loss are implemented correctly
- Identify how synthetic paths are exported for the RL repo

## Output contract for downstream RL repo
This repo should ideally produce standardized output files that are easy for `wti-rl-policy` to load.

Preferred format:
- NPZ file
- key: `paths`
- shape: `(N, T)`
- values: preferably log returns, unless explicitly documented otherwise

Optional metadata:
- generator config
- latent regime labels
- volatility statistics
- train/val/test split metadata
- seed used for generation

## What to prioritize
Highest-priority outcomes:
1. Correct path preprocessing
2. Correct signature/log-signature computation
3. Stable training
4. Realistic synthetic path generation
5. Strong evaluation of stylized facts
6. Clean export format for RL consumption

## Evaluation expectations
The repo should support or eventually support:
- distribution comparison between real and synthetic returns
- volatility clustering diagnostics
- leverage effect checks
- fat-tail / kurtosis / tail behavior checks
- pathwise visual comparisons
- regime-level comparisons
- signature-space distance metrics
- MMD-based validation

## Editing guidance
- Prefer small, high-confidence changes
- Do not change data conventions silently
- Document whether arrays are prices or returns
- Keep interfaces simple for downstream RL use
- Flag unclear assumptions instead of guessing
