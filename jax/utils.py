import jax
import jax.numpy as jnp
import optax


def create_learning_rate_schedule(total_steps,
                                  base,
                                  decay_type,
                                  warmup_steps,
                                  linear_end=1e-5):
    """Creates learning rate schedule.

    Currently only warmup + {linear,cosine} but will be a proper mini-language
    like preprocessing one in the future.

    Args:
      total_steps: The total number of steps to run.
      base: The starting learning-rate (without warmup).
      decay_type: 'linear' or 'cosine'.
      warmup_steps: how many steps to warm up for.
      linear_end: Minimum learning rate.

    Returns:
      A function learning_rate(step): float -> {"learning_rate": float}.
    """

    def step_fn(step):
        """Step to learning rate function."""
        lr = base

        progress = (step - warmup_steps) / float(total_steps - warmup_steps)
        progress = jnp.clip(progress, 0.0, 1.0)
        if decay_type == 'linear':
            lr = linear_end + (lr - linear_end) * (1.0 - progress)
        elif decay_type == 'cosine':
            lr = lr * 0.5 * (1. + jnp.cos(jnp.pi * progress))
        else:
            raise ValueError(f'Unknown lr type {decay_type}')

        if warmup_steps:
            lr = lr * jnp.minimum(1., step / warmup_steps)

        return jnp.asarray(lr, dtype=jnp.float32)

    return step_fn


def accumulate_gradient(loss_and_grad_fn, params, images, labels, accum_steps):
    """Accumulate gradient over multiple steps to save on memory."""
    if accum_steps and accum_steps > 1:
        assert images.shape[0] % accum_steps == 0, (
            f'Bad accum_steps {accum_steps} for batch size {images.shape[0]}')
        step_size = images.shape[0] // accum_steps
        l, g = loss_and_grad_fn(params, images[:step_size], labels[:step_size])

        def acc_grad_and_loss(i, l_and_g):
            imgs = jax.lax.dynamic_slice(images, (i * step_size, 0, 0, 0),
                                         (step_size,) + images.shape[1:])
            lbls = jax.lax.dynamic_slice(labels, (i * step_size, 0),
                                         (step_size, labels.shape[1]))
            li, gi = loss_and_grad_fn(params, imgs, lbls)
            l, g = l_and_g
            return (l + li, jax.tree_map(lambda x, y: x + y, g, gi))

        l, g = jax.lax.fori_loop(1, accum_steps, acc_grad_and_loss, (l, g))
        return jax.tree_map(lambda x: x / accum_steps, (l, g))
    else:
        return loss_and_grad_fn(params, images, labels)


def make_update_fn(*, apply_fn, accum_steps, tx):
    """Returns update step for data parallel training."""

    def update_fn(params, opt_state, images, labels):
        def loss_fn(params, images, labels):
            logits = apply_fn(dict(params=params), x=images)
            logp = jax.nn.log_softmax(logits)
            return -jnp.mean(jnp.sum(logp * labels, axis=1))

        l, g = accumulate_gradient(jax.value_and_grad(loss_fn), params, images, labels, accum_steps)
        g = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name='batch'), g)
        updates, opt_state = tx.update(g, opt_state, params)
        params = optax.apply_updates(params, updates)

        return params, opt_state

    return jax.pmap(update_fn, axis_name='batch', donate_argnums=(0,))
