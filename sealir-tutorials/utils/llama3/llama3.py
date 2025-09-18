import math
import sys
import time

import numpy as np

from .config import ModelArgs
from .tokenizer import Tokenizer
from .utils import load_parameters

np.random.seed(42)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def silu(x):
    result = x * (1 / (1 + np.exp(-x)))
    return result


def compute_cos_sin_cache(head_dim, max_seq_len, base=10000, dtype=np.float32):
    positions = np.arange(0, head_dim, 2, dtype=dtype)  # [0, 2, 4, ...]
    positions = positions[: (head_dim // 2)]  # Take only half
    dim_factors = positions / head_dim  # [0/dim, 2/dim, 4/dim, ...]
    base = dtype(base)
    power_factors = base**dim_factors  # [base^0, base^(2/dim), ...]
    inv_freq = 1.0 / power_factors  # [1/base^0, 1/base^(2/dim), ...]
    timesteps = np.arange(max_seq_len, dtype=dtype)
    freqs = np.outer(timesteps, inv_freq)
    cos_result = np.cos(freqs)
    sin_result = np.sin(freqs)

    return cos_result, sin_result


def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    xqri = xq.reshape(*xq.shape[:-1], -1, 2)
    xkri = xk.reshape(*xk.shape[:-1], -1, 2)
    xq_r = xqri[..., 0]
    xq_i = xqri[..., 1]
    xk_r = xkri[..., 0]
    xk_i = xkri[..., 1]
    freqs_cos = np.expand_dims(freqs_cos, axis=(0, 2))
    freqs_sin = np.expand_dims(freqs_sin, axis=(0, 2))
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # Combine real and imaginary parts
    xq_out = np.stack([xq_out_r, xq_out_i], axis=-1).reshape(
        xq_out_r.shape[:-1] + (-1,)
    )
    xk_out = np.stack([xk_out_r, xk_out_i], axis=-1).reshape(
        xk_out_r.shape[:-1] + (-1,)
    )

    return xq_out, xk_out


def feed_forward(x, up_weight, gate_weight, down_weight):
    swish = silu(x @ gate_weight.T)
    x_v = x @ up_weight.T
    x_ff = swish * x_v
    x_out = x_ff @ down_weight.T
    return x_out


def rmsnorm(x, weight, eps):
    z_float = np.mean(x**2, -1, keepdims=True) + eps
    z = x / np.sqrt(z_float)
    result = z * weight
    return result


def attention(
    x,
    start_pos,
    mask,
    freqs_cos,
    freqs_sin,
    attn_weights,
    args,
    cache_k,
    cache_v,
):
    q_weight, k_weight, v_weight, o_weight = [w.T for w in attn_weights]

    n_local_heads = args.n_heads
    head_dim = args.dim // args.n_heads

    batch_size, seq_len, _ = x.shape

    xq = x @ q_weight
    xk = x @ k_weight
    xv = x @ v_weight

    xq = xq.reshape(batch_size, seq_len, n_local_heads, head_dim)
    xk = xk.reshape(batch_size, seq_len, n_local_heads, head_dim)
    xv = xv.reshape(batch_size, seq_len, n_local_heads, head_dim)

    xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

    cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
    cache_v[:batch_size, start_pos : start_pos + seq_len] = xv
    ks = cache_k[:batch_size, : start_pos + seq_len]
    vs = cache_v[:batch_size, : start_pos + seq_len]

    xq = xq.transpose(0, 2, 1, 3)
    xk = ks.transpose(0, 2, 1, 3)
    xv = vs.transpose(0, 2, 1, 3)

    attention_scores = (xq @ xk.transpose(0, 1, 3, 2)) / math.sqrt(head_dim)
    if mask is not None:
        attention_scores = attention_scores + mask[None, None, :, :]
    attn = softmax(attention_scores)
    output = attn @ xv
    output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
    output = output @ o_weight
    return output, cache_k, cache_v


def transformer_block(
    x,
    start_pos,
    mask,
    freqs_cos,
    freqs_sin,
    block_weights,
    args,
    cache_k,
    cache_v,
):
    attn_weights, ff_weights, in_norm_weight, post_norm_weight = block_weights

    norm_x = rmsnorm(x, in_norm_weight, args.norm_eps)
    h1, cache_k, cache_v = attention(
        norm_x,
        start_pos,
        mask,
        freqs_cos,
        freqs_sin,
        attn_weights,
        args,
        cache_k,
        cache_v,
    )
    z = x + h1
    norm_z = rmsnorm(z, post_norm_weight, args.norm_eps)
    h2 = feed_forward(norm_z, *ff_weights)
    out = z + h2
    return out, cache_k, cache_v


def llama_forward(model, input_ids, start_pos):
    args = model["args"]
    dtype = model["dtype"]

    _, seq_len = input_ids.shape
    h = model["tok_embedding"][input_ids]

    freqs_cos = model["freqs_cos"][start_pos : start_pos + seq_len]
    freqs_sin = model["freqs_sin"][start_pos : start_pos + seq_len]

    mask = None
    if seq_len > 1:
        mask = np.full((seq_len, seq_len), float("-inf"), dtype=dtype)
        mask = np.triu(mask, k=1)
        zeros_shape = (seq_len, start_pos)
        mask = np.concatenate([np.zeros(zeros_shape, dtype=dtype), mask], axis=1)

    caches_k = model["caches_k"]
    caches_v = model["caches_v"]

    for i, block in enumerate(model["layer_blocks"]):
        h, caches_k[i], caches_v[i] = transformer_block(
            h,
            start_pos,
            mask,
            freqs_cos,
            freqs_sin,
            block,
            args,
            caches_k[i],
            caches_v[i],
        )

    h = rmsnorm(h, model["norm_weight"], args.norm_eps)
    logit = h[:, [-1], :] @ model["lm_head_weight"]
    return logit


# -- LLAMA init and generation methods below this point --

def llama_init(model_path, args):
    dtype = getattr(np, args.dtype)

    weights = load_parameters(model_path)
    weights = {k: v.astype(dtype) for k, v in weights.items()}

    tok_embedding = weights["model.embed_tokens.weight"]

    freqs_cos, freqs_sin = compute_cos_sin_cache(
        args.dim // args.n_heads, args.max_seq_len, dtype=dtype
    )
    layer_blocks = []
    for layer_id in range(args.n_layers):
        attn_weights = [
            weights[f"model.layers.{layer_id}.self_attn.q_proj.weight"],
            weights[f"model.layers.{layer_id}.self_attn.k_proj.weight"],
            weights[f"model.layers.{layer_id}.self_attn.v_proj.weight"],
            weights[f"model.layers.{layer_id}.self_attn.o_proj.weight"],
        ]
        ff_weights = [
            weights[f"model.layers.{layer_id}.mlp.up_proj.weight"],
            weights[f"model.layers.{layer_id}.mlp.gate_proj.weight"],
            weights[f"model.layers.{layer_id}.mlp.down_proj.weight"],
        ]
        in_norm = weights[f"model.layers.{layer_id}.input_layernorm.weight"]
        post_norm = weights[f"model.layers.{layer_id}.post_attention_layernorm.weight"]
        layer_blocks.append((attn_weights, ff_weights, in_norm, post_norm))
    norm_weight = weights["model.norm.weight"]
    lm_head_weight = weights["lm_head.weight"].T
    del weights

    # Preallocate caches for all layers (list of np.arrays)
    caches_k = [
        np.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                args.n_heads,
                args.dim // args.n_heads,
            ),
            dtype=dtype,
        )
        for _ in range(args.n_layers)
    ]
    caches_v = [np.zeros_like(caches_k[0]) for _ in range(args.n_layers)]

    return {
        "args": args,
        "dtype": dtype,
        "tok_embedding": tok_embedding,
        "freqs_cos": freqs_cos,
        "freqs_sin": freqs_sin,
        "layer_blocks": layer_blocks,
        "norm_weight": norm_weight,
        "lm_head_weight": lm_head_weight,
        "caches_k": caches_k,
        "caches_v": caches_v,
    }


def llama_generate(model, input_ids, max_new_tokens):
    batch_size, prompt_len = input_ids.shape
    current_len = prompt_len
    next_id = None  # Initialize next_id to avoid undefined variable error
    for i in range(max_new_tokens):
        current_pos = prompt_len + i
        if i == 0:
            current_input_ids = input_ids
            pos = 0
        else:
            current_input_ids = next_id
            pos = current_pos - 1
        logits = llama_forward(model, current_input_ids, pos)
        next_id = np.argmax(logits[:, -1, :], axis=-1, keepdims=True).astype(np.int32)
        yield next_id
        current_len += 1
        if current_len >= model["args"].max_seq_len:
            break


# -- Main script
if __name__ == "__main__":
    args = ModelArgs()
    print(f"Using precision: {args.dtype}")
    tokenizer = Tokenizer("./tokenizer.model.np")
    model = llama_init("./stories15M.model.npz", args)

    if len(sys.argv) == 1:
        prompt = "Once upon a time"
    else:
        prompt = sys.argv[1]
    print(f"\n{prompt}", end="")
    input_ids = np.array([tokenizer.encode(prompt)])
    start = time.time()
    _, seq_len = input_ids.shape
    generated_tokens_count = 0
    for id_val in llama_generate(model, input_ids, args.max_new_tokens):
        seq_len += 1
        generated_tokens_count += 1
        output_id = id_val[0].tolist()
        if output_id[-1] in [tokenizer.eos_id, tokenizer.bos_id]:
            break
        print(tokenizer.decode(output_id), end="", flush=True)
    elapsed = time.time() - start
    print(
        f"\n\nToken count: {seq_len}, elapsed: {elapsed:.2f}s, {round(seq_len / elapsed)} tokens/s"  # noqa: E501
    )
