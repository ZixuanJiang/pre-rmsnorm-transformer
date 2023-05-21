# %%
# import libraries and define parameters
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

device = torch.device('cuda')
dtype = torch.float64

image_size = 32
batch_size = 64
patch_size = 4
dim = 64
depth = 4
heads = 4
dim_head = dim // heads
mlp_dim = dim * 4
num_classes = 100

# %%
# process the raw images input
x = torch.randn(batch_size, 3, image_size, image_size).to(device).to(dtype)  # raw images
x = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)  # flattened patches
_, num_patches, patch_dim = x.shape

# %%
# define parameters in ViT
patch_linear_weight, patch_linear_bias = torch.randn(dim, patch_dim).to(device).to(dtype), torch.randn(dim).to(device).to(dtype)
pos_embedding = torch.randn(1, num_patches, dim).to(device).to(dtype)
cls_token = torch.randn(1, 1, dim).to(device).to(dtype)

att_in_linear_weight, att_in_linear_bias = torch.randn(depth, dim * 3, dim).to(device).to(dtype), torch.randn(depth, dim * 3).to(device).to(dtype)
att_out_linear_weight, att_out_linear_bias = torch.randn(depth, dim, dim).to(device).to(dtype), torch.randn(depth, dim).to(device).to(dtype)
mlp_in_linear_weight, mlp_in_linear_bias = torch.randn(depth, mlp_dim, dim).to(device).to(dtype), torch.randn(depth, mlp_dim).to(device).to(dtype)
mlp_out_linear_weight, mlp_out_linear_bias = torch.randn(depth, dim, mlp_dim).to(device).to(dtype), torch.randn(depth, dim).to(device).to(dtype)

cls_head_linear_weight, cls_head_linear_bias = torch.randn(num_classes, dim).to(device).to(dtype), torch.randn(num_classes).to(device).to(dtype)

# %%
# define three kinds of normalization


def layer_norm(x, eps: float = 1e-5):
    x_mean = x.mean(dim=-1, keepdim=True)
    x_var = x.var(dim=-1, keepdim=True, correction=0)
    return (x - x_mean) * torch.rsqrt(x_var + eps)


def rms_norm(x, eps: float = 1e-5):
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + eps)


def crms_norm(x, eps: float = 1e-5):
    discarded_element = x.sum(dim=-1, keepdim=True)
    return x * torch.rsqrt((x.square().sum(dim=-1, keepdim=True) + discarded_element.square()) / (x.shape[-1] + 1) + eps)

# %%
# define pre-normalization vit


def attention(x, in_linear_weight, in_linear_bias, out_linear_weight, out_linear_bias, is_causal=False):
    qkv = F.linear(x, in_linear_weight, in_linear_bias).chunk(3, dim=-1)
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=heads), qkv)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal)
    out = rearrange(out, 'b h n d -> b n (h d)')
    return F.linear(out, out_linear_weight, out_linear_bias)


def mlp(x, in_linear_weight, in_linear_bias, out_linear_weight, out_linear_bias):
    x = F.linear(x, in_linear_weight, in_linear_bias)
    x = F.gelu(x)
    return F.linear(x, out_linear_weight, out_linear_bias)


def pre_normalization_vit(
        x,
        patch_linear_weight, patch_linear_bias,
        pos_embedding, cls_token,
        att_in_linear_weight, att_in_linear_bias,
        att_out_linear_weight, att_out_linear_bias,
        mlp_in_linear_weight, mlp_in_linear_bias,
        mlp_out_linear_weight, mlp_out_linear_bias,
        cls_head_linear_weight, cls_head_linear_bias,
        norm_layer):
    # preprocessing
    x = F.linear(x, patch_linear_weight, patch_linear_bias)
    x = x + pos_embedding
    cls_tokens = repeat(cls_token, '1 1 d -> b 1 d', b=batch_size)
    x = torch.cat((cls_tokens, x), dim=1)

    # Transformer blocks
    for i in range(depth):
        x = x + attention(norm_layer(x), att_in_linear_weight[i], att_in_linear_bias[i], att_out_linear_weight[i], att_out_linear_bias[i])
        x = x + mlp(norm_layer(x), mlp_in_linear_weight[i], mlp_in_linear_bias[i], mlp_out_linear_weight[i], mlp_out_linear_bias[i])

    # postprocessing
    x = norm_layer(x)
    x = x[:, 0]
    x = F.linear(x, cls_head_linear_weight, cls_head_linear_bias)
    return x


# %%
# Variant 1. The widely-used Pre-LN Transformer
pre_ln_result = pre_normalization_vit(
    x,
    patch_linear_weight, patch_linear_bias,
    pos_embedding, cls_token,
    att_in_linear_weight, att_in_linear_bias,
    att_out_linear_weight, att_out_linear_bias,
    mlp_in_linear_weight, mlp_in_linear_bias,
    mlp_out_linear_weight, mlp_out_linear_bias,
    cls_head_linear_weight, cls_head_linear_bias,
    norm_layer=layer_norm)

# %%
# Variant 2. Recentering (1) the first input of Transformer blocks, and (2) the output of residual branches has no impact on the functionality
# Recenter the parameters in preprocessing such that the preprocessing generates the corresponding zero-mean vectors.
# zm means zerm-mean
zm_patch_linear_weight = patch_linear_weight - patch_linear_weight.mean(dim=0, keepdim=True)
zm_patch_linear_bias = patch_linear_bias - patch_linear_bias.mean()
zm_pos_embedding = pos_embedding - pos_embedding.mean(dim=-1, keepdim=True)
zm_cls_token = cls_token - cls_token.mean(dim=-1, keepdim=True)

# Recenter the output linear projection such that the residual branches generate the zero-mean parts of the original output.
zm_att_out_linear_weight = att_out_linear_weight - att_out_linear_weight.mean(dim=1, keepdim=True)
zm_att_out_linear_bias = att_out_linear_bias - att_out_linear_bias.mean(dim=1, keepdim=True)
zm_mlp_out_linear_weight = mlp_out_linear_weight - mlp_out_linear_weight.mean(dim=1, keepdim=True)
zm_mlp_out_linear_bias = mlp_out_linear_bias - mlp_out_linear_bias.mean(dim=1, keepdim=True)

pre_ln_result_2 = pre_normalization_vit(
    x,
    zm_patch_linear_weight, zm_patch_linear_bias,
    zm_pos_embedding, zm_cls_token,
    att_in_linear_weight, att_in_linear_bias,
    zm_att_out_linear_weight, zm_att_out_linear_bias,
    mlp_in_linear_weight, mlp_in_linear_bias,
    zm_mlp_out_linear_weight, zm_mlp_out_linear_bias,
    cls_head_linear_weight, cls_head_linear_bias,
    norm_layer=layer_norm)

print('Pre-LN and Pre-LN-With-Zero-Mean-Main-Branch are close:', torch.allclose(pre_ln_result, pre_ln_result_2))

# %%
# Variant 3.
# In Variant 2, we maintain zero-mean on the main branch and still layer norm.
# Given zero-mean input x, LayerNorm(x) = RMSNorm(x).
# Hence, we can replace LayerNorm with RMSNorm directly to simplify the computation.
pre_rms_result = pre_normalization_vit(
    x,
    zm_patch_linear_weight, zm_patch_linear_bias,
    zm_pos_embedding, zm_cls_token,
    att_in_linear_weight, att_in_linear_bias,
    zm_att_out_linear_weight, zm_att_out_linear_bias,
    mlp_in_linear_weight, mlp_in_linear_bias,
    zm_mlp_out_linear_weight, zm_mlp_out_linear_bias,
    cls_head_linear_weight, cls_head_linear_bias,
    norm_layer=rms_norm)

print('Pre-LN and Pre-RMSNorm are close:', torch.allclose(pre_ln_result, pre_rms_result))
print('Pre-LN-With-Zero-Mean-Main-Branch and Pre-RMSNorm are close:', torch.allclose(pre_ln_result_2, pre_rms_result))

# %%
# Variant 4. We apply lossless compression on the zero-mean vectors.
# compress the parameters in the preprocessing such that the preprocessing generates the compressed embedding vectors
compressed_patch_linear_weight = zm_patch_linear_weight[:-1, :]  # shape (dim, patch_dim) -> (dim - 1, patch_dim)
compressed_patch_linear_bias = zm_patch_linear_bias[:-1]  # shape (dim) -> (dim - 1)
compressed_pos_embedding = zm_pos_embedding[:, :, :-1]  # shape (1, num_patches, dim) -> (1, num_patches, dim - 1)
compressed_cls_token = zm_cls_token[:, :, :-1]  # shape (1, 1, dim) -> (1, 1, dim - 1)

# compress the input linear projection
compressed_att_in_linear_weight = att_in_linear_weight[:, :, :-1] - att_in_linear_weight[:, :, -1:]
compressed_mlp_in_linear_weight = mlp_in_linear_weight[:, :, :-1] - mlp_in_linear_weight[:, :, -1:]

# compress the output linear projection
compressed_att_out_linear_weight = zm_att_out_linear_weight[:, :-1, :]
compressed_att_out_linear_bias = zm_att_out_linear_bias[:, :-1]
compressed_mlp_out_linear_weight = zm_mlp_out_linear_weight[:, :-1, :]
compressed_mlp_out_linear_bias = zm_mlp_out_linear_bias[:, :-1]

# compress the final classification linear head
compressed_cls_head_linear_weight = cls_head_linear_weight[:, :-1] - cls_head_linear_weight[:, -1:]

pre_crms_result = pre_normalization_vit(
    x,
    compressed_patch_linear_weight, compressed_patch_linear_bias,
    compressed_pos_embedding, compressed_cls_token,
    compressed_att_in_linear_weight, att_in_linear_bias,
    compressed_att_out_linear_weight, compressed_att_out_linear_bias,
    compressed_mlp_in_linear_weight, mlp_in_linear_bias,
    compressed_mlp_out_linear_weight, compressed_mlp_out_linear_bias,
    compressed_cls_head_linear_weight, cls_head_linear_bias,
    norm_layer=crms_norm)

print('Pre-LN and Pre-CRMSNorm are close:', torch.allclose(pre_ln_result, pre_crms_result))
print('Pre-LN-With-Zero-Mean-Main-Branch and Pre-CRMSNorm are close:', torch.allclose(pre_ln_result_2, pre_crms_result))
print('Pre-RMSNorm and Pre-CRMSNorm are close:', torch.allclose(pre_rms_result, pre_crms_result))
