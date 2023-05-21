Our implementation is based on [PyTorch](https://pytorch.org/) and [apex](https://github.com/NVIDIA/apex).
We provide scripts for measuring the inference and training time of ViT and GPT, as listed below.

1. `vit_infer.py`, inference on a single device
2. `vit_train_single_gpu.py`, training on a single GPU
3. `vit_train_ddp.py`, training on multiple GPUs with data parallel training method

We have the following variants. Variants 2, 3, and 4 are equivalent.

1. `no-normalization`. We disable the normalization in Transformers. It is used to investigate the percentage of normalization layers in the whole model.
2. `pre-ln`, `pre-apex-ln`, `pre-customized-ln`. The widely-used Pre-LN Transformers with `torch.nn.LayerNorm` or [`apex.normalization.FusedLayerNorm`](https://nvidia.github.io/apex/layernorm.html), or our customized LayerNorm implementation.
3. `pre-apex-rms` or `pre-rms`. Pre-RMSNorm Transformers with `apex.normalization.FusedRMSNorm` or our customized RMSNorm implementation.
PyTorch does not provide an official API for RMSNorm, as discussed in [this issue](https://github.com/pytorch/pytorch/issues/72643).
4. `pre-crms`. Pre-CRMSNorm Transformers with the main branch vectors in $\mathbb{R}^{d-1}$.
5. `pre-crms-same-dim`. Pre-CRMSNorm Transformers with the main branch vectors in $\mathbb{R}^{d}$.

We refer to the following implementation.
1. [A simplified ViT implementation in PyTorch](https://github.com/lucidrains/vit-pytorch)
2. [GPT2 implementation from OpenAI](https://github.com/openai/gpt-2)
