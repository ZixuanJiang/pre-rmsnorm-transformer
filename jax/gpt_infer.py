import jax
import jax.numpy as jnp
import flax

import time
import functools

from model import PreDefinedGPT3

warmup_steps = 50
steps = 1000
vocab_size = 50257
max_seq_length = 2048
prng_key = jax.random.PRNGKey(0)
batch_size_list = [1, 4, 16]
seq_len_list = [128, 512, 2048]

for model_name in ['Small', 'Medium', 'Large', 'XL', '2.7B', '6.7B', '13B', '175B']:
    for method in ['pre-ln', 'pre-customized-ln', 'no-normalization', 'pre-rms', 'pre-customized-rms', 'pre-crms', 'pre-crms-same-dim']:

        raw_model = PreDefinedGPT3(vocab_size=vocab_size, max_seq_length=max_seq_length, variant=model_name, method=method)

        # Make sure initial model parameters (before replication) are on CPU only.
        @functools.partial(jax.jit, backend='cpu')
        def init(): return raw_model.init(prng_key, x=jax.random.randint(prng_key, [1, max_seq_length], minval=0, maxval=vocab_size))
        variables = init()

        params_repl = flax.jax_utils.replicate(variables['params'])
        # pmap replicates the models over all TPUs/GPUs
        model = jax.pmap(functools.partial(raw_model.apply))

        for batch_size in batch_size_list:
            for seq_len in seq_len_list:
                x = jax.random.randint(prng_key, [jax.local_device_count(), batch_size // jax.local_device_count(), seq_len], minval=0, maxval=vocab_size)

                for _ in range(warmup_steps):
                    logits = model(flax.core.FrozenDict(params=params_repl), x)
                    logits.block_until_ready()

                times = []
                for _ in range(steps):
                    t0 = time.time()
                    logits = model(flax.core.FrozenDict(params=params_repl), x)
                    logits.block_until_ready()
                    times.append(time.time() - t0)
                times = jnp.array(times)
                q25, q50, q75 = jnp.percentile(times, jnp.array([25, 50, 75]))
                print(f'GPT3 {model_name} {method}, batch size {batch_size}, seq len {seq_len}, mean {times.mean()}, std {times.std()}, q25 {q25}, q50 {q50}, q75 {q75}')
