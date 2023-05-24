# Pre-RMSNorm and Pre-CRMSNorm Transformers: Equivalent and Efficient Pre-LN Transformers

This repo is the official implementation of "Pre-RMSNorm and Pre-CRMSNorm Transformers: Equivalent and Efficient Pre-LN Transformers".

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Introduction
We propose a solution to unify two mainstream Transformer architectures, Pre-LN and Pre-RMSNorm Transformers. By removing the inherent redundant mean information in the main branch of Pre-LN Transformers, we can reduce LayerNorm to RMSNorm, achieving higher efficiency. We also propose the Compressed RMSNorm (CRMSNorm) and Pre-CRMSNorm Transformer based on a lossless compression of the zero-mean vectors. We formally establish the equivalence of Pre-LN, Pre-RMSNorm, and Pre-CRMSNorm Transformer variants in both training and inference. It implies that Pre-LN Transformers can be substituted with Pre-(C)RMSNorm counterparts at almost no cost, offering the same arithmetic functionality along with free efficiency improvement. We can reduce the training and inference time of Pre-LN Transformers by up to 10%.

# Dependencies
* Python >= 3.8
* einops >= 0.0.1.
* PyTorch >= 1.13.
* [apex](https://github.com/NVIDIA/apex) >= 0.1 (optional)
* jax >= 0.4.10 (optional)
* flax >= 0.6.9 (optional)

# Structures
* `example.py`. A self-explained demo of translating a Pre-LN Transformer into an equivalent Pre-RMSNorm and Pre-CRMSNorm Transformers.
Pre-(C)RMSNorm Transformers and equivalence evaluation
* `jax/`. JAX implementation for comparing the inference and training time of different Transformer variants.
* `torch/`. PyTorch implementation for comparing the inference and training time of different Transformer variants.

# Usage
Play with `example.py` to see how we simplify the widely-used Pre-LN Transformers. The expected results are all True to show the equivalence among different variants.
```bash
python example.py
>>>
Pre-LN and Pre-LN-with-Zero-Mean-Main-Branch are close: True
Pre-LN and Pre-RMSNorm are close: True
Pre-LN-with-Zero-Mean-Main-Branch and Pre-RMSNorm are close: True
Pre-LN and Pre-CRMSNorm are close: True
Pre-LN-with-Zero-Mean-Main-Branch and pre_crms_result are close: True
Pre-RMSNorm and Pre-CRMSNorm are close: True
```

You can use the JAX and PyTorch scripts to compare inference and training time for ViT and GPT with different settings.
Please refer to the README in the corresponding directory.
* CPU, GPU, TPU, and other accelerators
* Training on a single accelerator or multiple accelerators with distributed data parallel (DDP) processing
* Different precision
