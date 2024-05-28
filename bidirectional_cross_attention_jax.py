import jax.numpy as jnp
from flax import linen as nn
from einops import rearrange

class BidirectionalCrossAttention(nn.Module):
    dim: int
    heads: int = 8
    dim_head: int = 64
    context_dim: int = None
    dropout: float = 0.
    talking_heads: bool = False
    prenorm: bool = False

    @nn.compact
    def __call__(
        self,
        x,
        context,
        mask = None,
        context_mask = None,
        return_attn = False,
        rel_pos_bias = None
    ):
        context_dim = self.context_dim if self.context_dim is not None else self.dim

        x = nn.LayerNorm(self.dim) if self.prenorm else x
        context = nn.LayerNorm(context_dim) if self.prenorm else context

        inner_dim = self.dim_head * self.heads
        scale = self.dim_head ** -0.5

        b, i, j, h = x.shape[0], x.shape[-2], context.shape[-2], self.heads

        qk = nn.Dense(inner_dim, use_bias = False)(x)
        v = nn.Dense(inner_dim, use_bias = False)(x)
        context_qk = nn.Dense(inner_dim, use_bias = False)(context)
        context_v = nn.Dense(inner_dim, use_bias = False)(context)

        qk, context_qk, v, context_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (qk, context_qk, v, context_v))
        sim = jnp.einsum('b h i d, b h j d -> b h i j', qk, context_qk) * scale

        if rel_pos_bias is not None:
            sim = sim + rel_pos_bias

        if (mask is not None) or (context_mask is not None):
            mask = mask if mask is not None else jnp.ones((b, i), dtype = jnp.bool_)
            context_mask = context_mask if context_mask is not None else jnp.ones((b, j), dtype = jnp.bool_)
            attn_mask = rearrange(mask, 'b i -> b 1 i 1') * rearrange(context_mask, 'b j -> b 1 1 j')
            sim = jnp.where(attn_mask, sim, -jnp.finfo(sim.dtype).max)

        attn = nn.softmax(sim, axis=-1)
        context_attn = nn.softmax(sim, axis=-2)
        attn = nn.Dropout(rate = self.dropout)(attn, deterministic=False)
        context_attn = nn.Dropout(rate = self.dropout)(context_attn, deterministic=False)

        if self.talking_heads:
            attn = nn.Conv(features = self.heads, kernel_size = (1, 1), strides = (1, 1), padding = 'VALID', use_bias = False)(attn)
            context_attn = nn.Conv(features = self.heads, kernel_size = (1, 1), strides = (1, 1), padding = 'VALID', use_bias = False)(context_attn)

        out = jnp.einsum('b h i j, b h j d -> b h i d', attn, context_v)
        context_out = jnp.einsum('b h j i, b h j d -> b h i d', context_attn, v)

        out, context_out = map(lambda t: rearrange(t, 'b h n d -> b n (h d)'), (out, context_out))
        out = nn.Dense(self.dim, use_bias = False)(out)
        context_out = nn.Dense(context_dim, use_bias = False)(context_out)
        
        if return_attn:
            return out, context_out, attn, context_attn
        return out, context_out