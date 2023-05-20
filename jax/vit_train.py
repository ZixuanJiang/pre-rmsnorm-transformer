import jax
import jax.numpy as jnp
import flax
import optax

import time
import functools

from model import PreDefinedVit
import utils

warmup_steps = 50
steps = 100
dtype = jnp.float16
image_size = 224
num_classes = 1000
prng_key = jax.random.PRNGKey(0)
batch_size_list = [2048]

lr_fn = utils.create_learning_rate_schedule(total_steps=10000, base=1e-3, decay_type='cosine', warmup_steps=100)
tx = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=lr_fn, weight_decay=0.0001, mask=None),
)

for model_variant in [['Tiny', 16], ['Small', 16], ['Base', 16], ['Large', 16], ['Huge', 14], ['Giant', 14], ['22B', 14]]:
    model_name, patch_size = model_variant
    for method in ['pre-ln', 'pre-customized-ln', 'no-normalization', 'pre-rms', 'pre-customized-rms', 'pre-crms', 'pre-crms-same-dim']:
        for batch_size in batch_size_list:
            raw_model = PreDefinedVit(
                image_size=image_size,
                patch_size=patch_size,
                num_classes=num_classes,
                variant=model_name,
                method=method,
                pre_rms_training=method in ['pre-rms', 'pre-customized-rms'],
            )

            images = jax.random.normal(prng_key, [jax.local_device_count(), batch_size // jax.local_device_count(), image_size, image_size, 3], dtype=dtype)
            labels = jax.random.randint(prng_key, [jax.local_device_count(), batch_size // jax.local_device_count()], minval=0, maxval=num_classes)
            labels = jax.nn.one_hot(labels, num_classes)

            @functools.partial(jax.jit, backend='cpu')
            def init(): return raw_model.init(prng_key, x=jax.random.normal(prng_key, images.shape[1:], dtype=dtype))
            variables = init()
            params = variables['params']
            params_repl, opt_state_repl = flax.jax_utils.replicate((params, tx.init(params)))
            del params

            update_fn_repl = utils.make_update_fn(apply_fn=raw_model.apply, accum_steps=1, tx=tx)

            for _ in range(warmup_steps):
                with jax.profiler.StepTraceAnnotation('train'):
                    params_repl, opt_state_repl = update_fn_repl(params_repl, opt_state_repl, images, labels)

            times = []
            for _ in range(steps):
                t0 = time.time()
                with jax.profiler.StepTraceAnnotation('train'):
                    params_repl, opt_state_repl = update_fn_repl(params_repl, opt_state_repl, images, labels)
                times.append(time.time() - t0)
            times = jnp.array(times)
            q25, q50, q75 = jnp.percentile(times, jnp.array([25, 50, 75]))
            print(f'ViT-{model_name}-{patch_size} {method}, batch size {batch_size}, mean {times.mean()}, std {times.std()}, q25 {q25}, q50 {q50}, q75 {q75}')
