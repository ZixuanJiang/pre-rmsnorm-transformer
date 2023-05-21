# Pre-RMSNorm and Pre-CRMSNorm Transformers: Equivalent and Efficient Pre-LN Transformers

This repo is the official implementation of "Pre-RMSNorm and Pre-CRMSNorm Transformers: Equivalent and Efficient Pre-LN Transformers"

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Introduction
we propose a solution to unify two mainstream Transformer architectures, Pre-LN and Pre-RMSNorm Transformers. By removing the inherent redundant mean information in the main branch of Pre-LN Transformers, we can reduce LayerNorm to RMSNorm, achieving higher efficiency. We also propose the Compressed RMSNorm (CRMSNorm) and Pre-CRMSNorm Transformer based on a lossless compression of the zero-mean vectors. We formally establish the equivalence of Pre-LN, Pre-RMSNorm, and Pre-CRMSNorm Transformer variants in both training and inference. It implies that Pre-LN Transformers can be substituted with Pre-(C)RMSNorm counterparts at almost no cost, offering the same arithmetic functionality along with free efficiency improvement. We can reduce the training and inference time of Pre-LN Transformers by up to 10%.

# Dependencies
* Python >= 3.8
* einops >= 0.0.1.
* PyTorch >= 1.13.
* apex >= 0.1 (optional) see [apex](https://github.com/NVIDIA/apex) for installation.
* jax >= 0.4.10 (optional)
* flax >= 0.6.9 (optional)

# Structures
* jax/: JAX implementation
    * model.py: Transformer block definition
    * utils.py: utility functions
    * vit_infer.py: ViT inference code
    * vit_train.py: ViT training code   
* torch/: PyTorch implementation
    * model.py: Transformer block definition
    * utils.py: utility functions
    * vit_infer.py: ViT inference code
    * vit_train.py: ViT training code   
* example.py: Example implementation of Pre-(C)RMSNorm Transformer and equivalence evaluation

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

In the directories `jax` and `torch,` we provide scripts to compare the inference and training (on a single accelerator or multiple accelerators with distributed data-parallel (DDP) processing).
