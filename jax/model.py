import jax
import jax.numpy as jnp
from jax.numpy import einsum
import flax.linen as nn

import functools

from einops import rearrange, repeat

# three normalizations without elementwise affine transformation
# normalization is applied along the last dimension


class LayerNorm(nn.Module):
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x):
        dtype = jnp.promote_types(x.dtype, jnp.float32)
        y = jnp.asarray(x, dtype)

        mean = y.mean(axis=-1, keepdims=True)
        mean_square = jax.lax.square(y).mean(axis=-1, keepdims=True)
        var = jnp.maximum(mean_square - jax.lax.square(mean), 0.)
        return (x - mean) * jax.lax.rsqrt(var + self.epsilon)


class RMSNorm(nn.Module):
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x):
        dtype = jnp.promote_types(x.dtype, jnp.float32)
        y = jnp.asarray(x, dtype)

        rms_value = jax.lax.square(y).mean(axis=-1, keepdims=True)
        return x * jax.lax.rsqrt(rms_value + self.epsilon)


class CRMSNorm(nn.Module):
    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x):
        dtype = jnp.promote_types(x.dtype, jnp.float32)
        y = jnp.asarray(x, dtype)

        discarded_value = y.sum(axis=-1, keepdims=True)
        rms_value = (jax.lax.square(y).sum(axis=-1, keepdims=True) + jax.lax.square(discarded_value)) / (x.shape[-1] + 1)
        return x * jax.lax.rsqrt(rms_value + self.epsilon)


class DenseZeroMeanOutput(nn.Dense):
    @nn.compact
    def __call__(self, inputs):
        kernel = self.param('kernel',
                            self.kernel_init,
                            (jnp.shape(inputs)[-1], self.features),
                            self.param_dtype)
        if self.use_bias:
            bias = self.param('bias', self.bias_init, (self.features,),
                              self.param_dtype)
        else:
            bias = None
        inputs, kernel, bias = nn.dtypes.promote_dtype(inputs, kernel, bias, dtype=self.dtype)

        zero_mean_kernel = kernel - jnp.mean(kernel, axis=0, keepdims=True)
        y = self.dot_general(
            inputs,
            zero_mean_kernel,
            (((inputs.ndim - 1,), (0,)), ((), ())),
            precision=self.precision,
        )
        if bias is not None:
            zero_mean_bias = bias - jnp.mean(bias)
            y += jnp.reshape(zero_mean_bias, (1,) * (y.ndim - 1) + (-1,))
        return y


class FeedForward(nn.Module):
    dim: int
    hidden_dim: int
    zero_mean_output: bool = False

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.hidden_dim)(x)
        x = nn.gelu(x)
        final_dense_layer = DenseZeroMeanOutput if self.zero_mean_output else nn.Dense
        x = final_dense_layer(features=self.dim)(x)
        return x


class Attention(nn.Module):
    dim: int
    heads: int
    dim_head: int
    is_causal: bool = False
    zero_mean_output: bool = False

    @nn.compact
    def __call__(self, x):
        to_qkv = nn.Dense(features=self.dim_head * self.heads * 3, use_bias=True)(x)
        qkv = jnp.split(to_qkv, 3, axis=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b n h d', h=self.heads), qkv)
        mask = nn.make_causal_mask(x) if self.is_causal else None
        out = nn.dot_product_attention(q, k, v, mask=mask)
        out = rearrange(out, 'b n h d -> b n (h d)')

        final_dense_layer = DenseZeroMeanOutput if self.zero_mean_output else nn.Dense
        out = final_dense_layer(features=self.dim)(out)
        return out


class Transformer(nn.Module):
    dim: int
    depth: int
    heads: int
    dim_head: int
    mlp_dim: int
    norm_layer: nn.Module = nn.LayerNorm
    is_causal: bool = False
    zero_mean_output: bool = False

    @nn.compact
    def __call__(self, x):
        for _ in range(self.depth):
            residual_input_1 = self.norm_layer()(x) if self.norm_layer is not None else x
            x = x + Attention(self.dim, self.heads, self.dim_head, self.is_causal, self.zero_mean_output)(residual_input_1)
            residual_input_2 = self.norm_layer()(x) if self.norm_layer is not None else x
            x = x + FeedForward(self.dim, self.mlp_dim, self.zero_mean_output)(residual_input_2)
        return x


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class ViT(nn.Module):
    image_size: int
    patch_size: int
    num_classes: int
    dim: int
    depth: int
    heads: int
    mlp_dim: int = None
    dim_head: int = None
    pool: str = 'cls'
    method: str = 'pre-ln'
    pre_rms_training: bool = False

    @nn.compact
    def __call__(self, x):
        if self.dim_head is None:
            dim_head = self.dim // self.heads
            assert dim_head * self.heads == self.dim, 'dimension must be divisible by number of heads'
        else:
            dim_head = self.dim_head
        mlp_dim = self.mlp_dim if self.mlp_dim is not None else self.dim * 4
        dim = self.dim
        pre_rms_training = self.pre_rms_training and self.method in ['pre-rms', 'pre-customized-rms']

        if self.method == 'pre-ln':
            norm_layer = functools.partial(nn.LayerNorm, use_scale=False, use_bias=False)
        elif self.method == 'pre-customized-ln':
            norm_layer = LayerNorm
        elif self.method == 'no-normalization':
            norm_layer = None
        elif self.method == 'pre-rms':
            norm_layer = functools.partial(nn.RMSNorm, use_scale=False)
        elif self.method == 'pre-customized-rms':
            norm_layer = RMSNorm
        elif self.method == 'pre-crms':
            norm_layer = CRMSNorm
            dim -= 1
        elif self.method == 'pre-crms-same-dim':
            norm_layer = CRMSNorm
        else:
            raise NotImplementedError

        image_height, image_width = pair(self.image_size)
        patch_height, patch_width = pair(self.patch_size)

        assert image_height % patch_height == 0
        assert image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        assert self.pool in {'cls', 'mean'}

        x = rearrange(x, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width)
        to_patch_linear = DenseZeroMeanOutput if pre_rms_training else nn.Dense
        x = to_patch_linear(features=dim)(x)

        pos_embedding_param = self.param('pos_embedding', nn.initializers.zeros, [1, num_patches, dim])
        pos_embedding = pos_embedding_param - jnp.mean(pos_embedding_param, axis=-1, keepdims=True) if pre_rms_training else pos_embedding_param
        x += pos_embedding

        cls_token_param = self.param('cls', nn.initializers.zeros, [1, 1, dim])
        cls_token = cls_token_param - jnp.mean(cls_token_param, axis=-1, keepdims=True) if pre_rms_training else cls_token_param
        cls_tokens = repeat(cls_token, '() n d -> b n d', b=x.shape[0])
        x = jnp.concatenate([cls_tokens, x], axis=1)

        x = Transformer(dim, self.depth, self.heads, dim_head, mlp_dim, norm_layer, zero_mean_output=pre_rms_training)(x)
        if norm_layer is not None:
            x = norm_layer()(x)

        x = jnp.mean(x, axis=1) if self.pool == 'mean' else x[:, 0]
        x = nn.Dense(features=self.num_classes)(x)
        return x


vit_variants = {  # (dim, heads, depth, mlp_dim), if mlp_dim is Nont, then mlp_dim = dim * 4
    'Tiny':  (192,  3,  12, None),
    'Small': (384,  6,  12, None),
    'Base':  (768,  12, 12, None),
    'Large': (1024, 16, 24, None),
    'Huge':  (1280, 16, 32, None),
    'Giant': (1664, 16, 48, 8192),
    '22B':   (6144, 48, 48, None),
}


class PreDefinedVit(nn.Module):
    variant: str
    image_size: int
    patch_size: int
    num_classes: int
    pool: str = 'cls'
    method: str = 'pre-ln'
    pre_rms_training: bool = False

    @nn.compact
    def __call__(self, x):
        assert self.variant in vit_variants
        dim, heads, depth, mlp_dim = vit_variants[self.variant]
        return ViT(self.image_size, self.patch_size, self.num_classes, dim, depth, heads, mlp_dim, pool=self.pool, method=self.method, pre_rms_training=self.pre_rms_training)(x)
