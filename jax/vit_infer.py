import jax
import jax.numpy as jnp
import flax

import time
import functools

from model import PreDefinedVit

warmup_steps = 50
steps = 1000
dtype = jnp.float16
image_size = 224
num_classes = 1000
prng_key = jax.random.PRNGKey(0)
batch_size_list = [1, 4, 16, 64, 256, 1024]

for model_variant in [['Tiny', 16], ['Small', 16], ['Base', 16], ['Large', 16], ['Huge', 14], ['Giant', 14], ['22B', 14]]:
    model_name, patch_size = model_variant
    for method in ['pre-ln', 'pre-customized-ln', 'no-normalization', 'pre-rms', 'pre-customized-rms', 'pre-crms', 'pre-crms-same-dim']:
        raw_model = PreDefinedVit(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            variant=model_name,
            method=method,
        )

        # Make sure initial model parameters (before replication) are on CPU only.
        @functools.partial(jax.jit, backend='cpu')
        def init(): return raw_model.init(prng_key, x=jax.random.normal(prng_key, [1, image_size, image_size, 3], dtype=dtype))
        variables = init()

        params_repl = flax.jax_utils.replicate(variables['params'])
        # pmap replicates the models over all TPUs/GPUs
        model = jax.pmap(functools.partial(raw_model.apply))

        for batch_size in batch_size_list:
            images = jax.random.normal(prng_key, [jax.local_device_count(), batch_size // jax.local_device_count(), image_size, image_size, 3], dtype=dtype)

            for _ in range(warmup_steps):
                logits = model(flax.core.FrozenDict(params=params_repl), images)
                logits.block_until_ready()

            times = []
            for _ in range(steps):
                t0 = time.time()
                logits = model(flax.core.FrozenDict(params=params_repl), images)
                logits.block_until_ready()
                times.append(time.time() - t0)
            times = jnp.array(times)
            q25, q50, q75 = jnp.percentile(times, jnp.array([25, 50, 75]))
            print(f'ViT-{model_name}-{patch_size} {method}, batch size {batch_size}, mean {times.mean()}, std {times.std()}, q25 {q25}, q50 {q50}, q75 {q75}')
