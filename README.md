# Bidirectional Cross Attention

JAX implementation of [lucidrains/bidirectional-cross-attention](https://github.com/lucidrains/bidirectional-cross-attention).

### Installation
`pip install bidirectional-cross-attention-jax`

### Usage
```py
import jax
import jax.numpy as jnp
from bidirectional_cross_attention_jax import BidirectionalCrossAttention

key = jax.random.PRNGKey(0)
video = jax.random.normal(key, (1, 4096, 512))
audio = jax.random.normal(key, (1, 8192, 386))

video_mask = jnp.ones((1, 4096), dtype=jnp.bool_)
audio_mask = jnp.ones((1, 8192), dtype=jnp.bool_)

joint_cross_attn = BidirectionalCrossAttention(
    dim = 512,
    heads = 8,
    dim_head = 64,
    context_dim = 386
)

init = joint_cross_attn.init(key, video, audio)
video_out, audio_out = joint_cross_attn.apply(
    init,
    video,
    audio,
    mask = video_mask,
    context_mask = audio_mask
)

# attended output should have the same shape as input

assert video_out.shape == video.shape
assert audio_out.shape == audio.shape
```

### Citations

```bibtex
@article{Hiller2024PerceivingLS,
    title   = {Perceiving Longer Sequences With Bi-Directional Cross-Attention Transformers},
    author  = {Markus Hiller and Krista A. Ehinger and Tom Drummond},
    journal = {ArXiv},
    year    = {2024},
    volume  = {abs/2402.12138},
    url     = {https://api.semanticscholar.org/CorpusID:267751060}
}
```