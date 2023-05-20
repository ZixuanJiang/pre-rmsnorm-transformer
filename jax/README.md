We provide the script for measuring the inference and training time of ViT and GPT.
The implementation is based on [JAX](https://github.com/google/jax) and [Flax](https://github.com/google/flax).
Run `python vit_infer.py` or `python vit_train.py` to compare the execution time of different Transformer variants.

We have the following variants. Variants 2, 3, and 4 are equivalent.

1. `no-normalization`. We disable the normalization in Transformers. It is used to investigate the percentage of normalization layers in the whole model.
2. `pre-ln` or `pre-customized-ln`. The widely-used Pre-LN Transformers with `flax.linen.LayerNorm` or our customized LayerNorm implementation.
3. `pre-rms` or `pre-customized-rms`. Pre-RMSNorm Transformers with `flax.linen.RMSNorm` or our customized RMSNorm implementation.
4. `pre-crms`. Pre-CRMSNorm Transformers with the main branch vectors in $\mathbb{R}^{d-1}$.
5. `pre-crms-same-dim`. Pre-CRMSNorm Transformers with the main branch vectors in $\mathbb{R}^{d}$.

We refer to the following implementation.

1. [Original ViT implementation in JAX from Google](https://github.com/google-research/vision_transformer)
2. [A simplified ViT implementation in JAX](https://github.com/conceptofmind/vit-flax)
