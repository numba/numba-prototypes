import numpy as np
import math
import inspect
import pytest
import time

from llm_midend import (
    MlirBackend,
    compiler_config,
    TypeFloat64,
    rule,
    ArgFact,
    ruleset,
    Report,
    TodoException,
    TypeInt64,
    py_eqsat_rules,
    module_rules,
    module_rulesets,
    ruleset_extra_builtin_operations,
    ruleset_ufunc_reduce_array_desc,
    ruleset_explain_array_desc,
    ruleset_typevar_annotate,
    ruleset_tuple,
    ruleset_slice,
    ruleset_more_constant_folding,
    ruleset_more_typing,
    global_time_recorder,
    NumPyExecRecord
)

from mlir_backend import jit_compiler, setup_argtypes
from typeinfer_array import array_desc_rules, ruleset_broadcasting
from typeinfer_loops import base_ruleset

##########COMPILER##############


def run_compiler(target_function, args):
    input_shapes = []
    input_types = []
    input_type_rules = []

    for i, a in enumerate(args):
        if isinstance(a, np.ndarray):
            assert a.dtype == np.float64
            assert a.flags.c_contiguous
            input_shapes.append(a.shape)
            desc, eg_facts = array_desc_rules(
                f"array_{i}", shape=a.shape, dtype=TypeFloat64, layout="c"
            )
            input_types.append(desc.toType())
            input_type_rules.extend(eg_facts)

            # HACK
            input_type_rules.append(rule(desc.toType()).then(ArgFact(i, desc.toType())))
        elif isinstance(a, int):
            input_types.append(TypeInt64)
            input_type_rules.append(rule().then(ArgFact(i, TypeInt64)))
        else:
            raise TypeError(type(a))

    ruleset_array_facts = ruleset(*input_type_rules)

    # FIXME: egraph function parameters are sorted.
    #        they don't match the ordering of the actual parameters.
    #        this should be handled elsewhere.
    #        for now we just reorder it here.
    argnames = list(inspect.signature(target_function).parameters.keys())
    arg_ordered = sorted([(v, k) for k, v in enumerate(argnames)])
    input_types = [input_types[i] for k, i in arg_ordered]

    report = Report(default_expanded=True, enable_nested_metadata=True)
    try:
        out = jit_compiler(
            fn=target_function,
            argtypes=tuple(input_types),
            ruleset=(
                base_ruleset
                | py_eqsat_rules()
                | ruleset_broadcasting
                | setup_argtypes(*input_types)
                | ruleset_array_facts
                | module_rules
                | module_rulesets
                | ruleset_extra_builtin_operations
                | ruleset_ufunc_reduce_array_desc
                | ruleset_explain_array_desc
                | ruleset_typevar_annotate
                | ruleset_tuple
                | ruleset_slice
                | ruleset_more_constant_folding
                | ruleset_more_typing
            ),
            # pipeline_report=report,
            # pipeline_debug=True,
            # display_egraph=True,
            **compiler_config,
        )

    finally:
        pass
        # print(report.display())
        # report.display(view_html=True)

    return out


###########TESTING###############


def softmax_max(x):
    return np.max(x, axis=-1, keepdims=True)


def test_softmax_max():
    np.random.seed(0)
    _run_array_unary_test(softmax_max, np.random.random((3, 5)))


def softmax_x_minus_max(x):
    return x - np.max(x, axis=-1, keepdims=True)


def test_softmax_x_minux_max_1d():
    np.random.seed(0)
    _run_array_unary_test(softmax_x_minus_max, np.random.random(100000))


def test_softmax_x_minux_max_2d():
    np.random.seed(0)
    _run_array_unary_test(softmax_x_minus_max, np.random.random((1000, 1000)))


def test_softmax_x_minux_max():
    np.random.seed(0)
    _run_array_unary_test(softmax_x_minus_max, np.random.random((1, 4)))
    _run_array_unary_test(softmax_x_minus_max, np.random.random((1, 2, 6, 4)))


def binop_performance(x, y):
    a = np.asarray(2.0)
    return a * x + y


def test_binop_performance_2d():
    np.random.seed(0)
    _run_array_test(
        binop_performance,
        (np.random.random((1000, 1000)), np.random.random((1000, 1000))),
    )


def softmax_sum(x):
    return np.sum(x, axis=-1, keepdims=True)


def test_softmax_sum():
    np.random.seed(0)
    _run_array_unary_test(softmax_sum, np.random.random((1, 4)))
    _run_array_unary_test(softmax_sum, np.random.random((1, 2, 6, 4)))


def softmax_full(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def test_softmax_full():
    np.random.seed(0)
    _run_array_unary_test(softmax_full, np.random.random((2000, 4000)))


def apply_rotary_emb_reshape(xq):
    xqri = xq.reshape(xq.shape[:-1] + (-1, 2))
    return xqri


def test_apply_rotary_emb_reshape():
    np.random.seed(0)
    _run_array_unary_test(apply_rotary_emb_reshape, np.random.random((1, 2, 3, 4)))


def test_apply_rotary_emb_fancy_index_equiv():
    np.random.seed(0)
    arr = np.random.random((1, 2, 3, 4))
    np.testing.assert_equal(arr[..., 0], np.take(arr, 0, axis=-1))


def test_apply_rotary_emb_broadcast_to_expanddims_equiv():
    seq_len, head_dim = 5, 24
    freqs_cos = np.random.random((seq_len, head_dim))
    desired = np.broadcast_to(np.expand_dims(freqs_cos, axis=(0, 2)), (1, 5, 6, 24))
    got = np.broadcast_to(
        freqs_cos.reshape(1, freqs_cos.shape[0], 1, freqs_cos.shape[1]), (1, 5, 6, 24)
    )
    np.testing.assert_equal(got, desired)


def apply_rotary_emb_fancy_index_0(xqri):
    # xq_r = xqri[..., 0]
    xq_r = np.take(xqri, 0, axis=-1)
    return xq_r


def apply_rotary_emb_fancy_index_1(xqri):
    # xq_r = xqri[..., 1]
    xq_r = np.take(xqri, 1, axis=-1)
    return xq_r


def test_apply_rotary_emb_fancy_index():
    np.random.seed(0)
    _run_array_unary_test(
        apply_rotary_emb_fancy_index_0, np.random.random((1, 2, 3, 4))
    )
    _run_array_unary_test(
        apply_rotary_emb_fancy_index_1, np.random.random((1, 2, 3, 4))
    )


def apply_rotary_emb_expand_dims(freqs_cos):
    # np.expand_dims(freqs_cos, axis=(0, 2))
    # TODO actually support np.expand_dims
    return freqs_cos.reshape(
        (1,) + (freqs_cos.shape[0],) + (1,) + (freqs_cos.shape[1],)
    )


def test_apply_rotary_emb_expand_dims():
    np.random.seed(0)
    seq_len, head_dim = 5, 24
    freqs_cos = np.random.random((seq_len, head_dim))
    _run_array_unary_test(apply_rotary_emb_expand_dims, freqs_cos)


def apply_rotary_emb_broadcast_to(freqs_cos_expanded):
    return np.broadcast_to(freqs_cos_expanded, (1, 5, 6, 24))


def test_apply_rotary_emb_broadcast_to():
    np.random.seed(0)
    seq_len, head_dim = 5, 24
    freqs_cos_expanded = np.random.random((1, seq_len, 1, head_dim))
    # print("???", apply_rotary_emb_broadcast_to(freqs_cos_expanded).shape)
    _run_array_unary_test(apply_rotary_emb_broadcast_to, freqs_cos_expanded)


def apply_rotary_emb_ufuncs(xq_r, xq_i, freqs_cos, freqs_sin):
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin  # adjusted
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    return xq_out_r + xq_out_i


def test_apply_rotary_emb_ufuncs():
    np.random.seed(0)
    shape = 1, 5, 6, 24
    xq_r = np.random.random(shape)
    xq_i = np.random.random(shape)
    freqs_cos = np.random.random(shape)
    freqs_sin = np.random.random(shape)
    _run_array_test(apply_rotary_emb_ufuncs, (xq_r, xq_i, freqs_cos, freqs_sin))


def apply_rotary_emb_stack(xq_out_r, xq_out_i):
    return np.stack((xq_out_r, xq_out_i), axis=-1)


def test_apply_rotary_emb_stack():
    np.random.seed(0)
    shape = 1, 5, 6
    xq_out_r = np.random.random(shape)
    xq_out_i = np.random.random(shape)
    _run_array_test(apply_rotary_emb_stack, (xq_out_r, xq_out_i))


def apply_rotary_emb(xq, xk, freqs_cos, freqs_sin):
    xqri = xq.reshape(xq.shape[:-1] + (-1, 2))
    xkri = xk.reshape(xk.shape[:-1] + (-1, 2))
    # xq_r = xqri[..., 0]
    xq_r = np.take(xqri, 0, axis=-1)
    # xq_i = xqri[..., 1]
    xq_i = np.take(xqri, 1, axis=-1)
    # xk_r = xkri[..., 0]
    xk_r = np.take(xkri, 0, axis=-1)
    # xk_i = xkri[..., 1]
    xk_i = np.take(xkri, 1, axis=-1)

    # freqs_cos = np.broadcast_to(np.expand_dims(freqs_cos, axis=(0, 2)), (1, 5, 6, 24))
    freqs_cos = np.broadcast_to(
        freqs_cos.reshape((1,) + (freqs_cos.shape[0],) + (1,) + (freqs_cos.shape[1],)),
        (1, 5, 6, 24),
    )
    # freqs_sin = np.broadcast_to(np.expand_dims(freqs_sin, axis=(0, 2)), (1, 5, 6, 24))
    freqs_sin = np.broadcast_to(
        freqs_sin.reshape((1,) + (freqs_sin.shape[0],) + (1,) + (freqs_sin.shape[1],)),
        (1, 5, 6, 24),
    )

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # Combine real and imaginary parts
    xq_out = np.stack((xq_out_r, xq_out_i), axis=-1).reshape(
        xq_out_r.shape[:-1] + (-1,)
    )
    xk_out = np.stack((xk_out_r, xk_out_i), axis=-1).reshape(
        xk_out_r.shape[:-1] + (-1,)
    )

    return np.stack((xq_out, xk_out), axis=-1)


def test_apply_rotary_emb_full():
    np.random.seed(0)

    batch_size, seq_len, n_heads, dims = 1, 5, 6, 288
    n_local_heads, head_dim = n_heads, dims // n_heads

    xq = np.random.random((batch_size, seq_len, n_local_heads, head_dim))
    xk = np.random.random((batch_size, seq_len, n_local_heads, head_dim))
    freqs_cos = np.random.random((seq_len, head_dim // 2))
    freqs_sin = np.random.random((seq_len, head_dim // 2))
    _run_array_test(apply_rotary_emb, (xq, xk, freqs_cos, freqs_sin))


#######################################
# Attention


def attention_transpose(q_weight):
    q_weight = np.transpose(q_weight)
    return q_weight


def test_attention_transpose():
    # Testing for:
    #   q_weight, k_weight, v_weight, o_weight = [w.T for w in attn_weights]
    np.random.seed(0)
    q_weight = np.random.random((2, 3, 4))
    # first check equivalent expression
    np.testing.assert_array_equal(q_weight.T, np.transpose(q_weight))

    # test compiler
    _run_array_test(attention_transpose, (q_weight,))

    q_weight = np.random.random((288, 288))
    # test compiler on llm use case
    _run_array_test(attention_transpose, (q_weight,))


def attention_transpose_2(xq):
    return np.transpose(xq, (0, 2, 1, 3))


def test_attention_transpose_2():
    # Usecase
    #   xq = xq.transpose(0, 2, 1, 3)
    np.random.seed(0)
    xq = np.random.random((1, 5, 6, 48))
    # test compiler on llm use case
    _run_array_test(attention_transpose_2, (xq,))


def attention_matmul(x, q_weight):
    # return x @ q_weight
    return np.matmul(x, q_weight)


def test_attention_matmul():
    # Testing for:
    #   x @ q_weight
    np.random.seed(0)
    x = np.random.random((1, 5, 288))
    q_weight = np.random.random((288, 288))

    # test compiler on llm use case
    _run_array_test(attention_matmul, (x, q_weight))

    # test other shapes
    a = np.random.random((1, 2, 3, 4))
    b = np.random.random((1, 2, 4, 5))
    _run_array_test(attention_matmul, (a, b))


def attention_setitem(cache_k, xk):
    batch_size = 1
    seq_len = 5
    start_pos = 0

    cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
    return cache_k


def test_attention_setitem():
    # Usecase:
    #   cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
    np.random.seed(0)

    xk = np.random.random((1, 5, 6, 48))
    cache_k = np.random.random((1, 256, 6, 48))

    # test compiler on llm use case
    _run_array_test(attention_setitem, (cache_k, xk))


def attention_getitem(cache_k):
    batch_size = 1
    seq_len = 5
    start_pos = 0
    ks = cache_k[:batch_size, : start_pos + seq_len]
    return ks


def test_attention_getitem():
    # Usecase:
    #   ks = cache_k[:batch_size, : start_pos + seq_len]
    np.random.seed(0)
    cache_k = np.random.random((1, 256, 6, 48))
    # test compiler on llm use case
    _run_array_test(attention_getitem, (cache_k,))


def attention_getitem_setitem(cache_k, xk, yk):
    batch_size = 1
    seq_len = 5
    start_pos = 0

    cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
    # Copy is a synchronization point;
    # without it, e.g:
    #    ks = cache_k[:batch_size, : start_pos + seq_len]
    # the getitem and + can be moved after the second setitem.
    ks = np.copy(cache_k[:batch_size, : start_pos + seq_len])
    q = ks + ks
    cache_k[:batch_size, start_pos : start_pos + seq_len] = yk
    # the second setitem
    kr = cache_k[:batch_size, : start_pos + seq_len]

    return kr + q


def test_attention_setitem_getitem_effect():
    np.random.seed(0)

    xk = np.random.random((1, 5, 6, 48))
    yk = np.random.random((1, 5, 6, 48))
    cache_k = np.random.random((1, 256, 6, 48))

    # test compiler on llm use case
    _run_array_test(attention_getitem_setitem, (cache_k, xk, yk))


def attention(
    x,  # shape = (1, 5, 288)
    mask,  # shape = (5, 5)
    freqs_cos,  # shape = (5, 24)
    freqs_sin,  # shape = (5, 24)
    attn_weights_q,  # shape = (288, 288)
    attn_weights_k,  # shape = (288, 288)
    attn_weights_v,  # shape = (288, 288)
    attn_weights_o,  # shape = (288, 288)
    cache_k,  # shape = (1, 256, 6, 48)
    cache_v,  # shape = (1, 256, 6, 48)
):
    # HACK
    start_pos = 0

    q_weight = np.transpose(attn_weights_q)
    k_weight = np.transpose(attn_weights_k)
    v_weight = np.transpose(attn_weights_v)
    o_weight = np.transpose(attn_weights_o)

    n_heads = 6
    dims = 288
    n_local_heads = n_heads  # 6
    head_dim = dims // n_heads  # 288/ 6 = 48

    batch_size = x.shape[0]
    seq_len = x.shape[1]

    # xq = x @ q_weight
    # xk = x @ k_weight
    # xv = x @ v_weight
    xq = np.matmul(x, q_weight)
    xk = np.matmul(x, k_weight)
    xv = np.matmul(x, v_weight)

    xq = xq.reshape((batch_size, seq_len, n_local_heads, head_dim))
    xk = xk.reshape((batch_size, seq_len, n_local_heads, head_dim))
    xv = xv.reshape((batch_size, seq_len, n_local_heads, head_dim))

    # BEGIN inlined apply_rotary_emb()
    xqri = xq.reshape(xq.shape[:-1] + (-1, 2))
    xkri = xk.reshape(xk.shape[:-1] + (-1, 2))
    # xq_r = xqri[..., 0]
    xq_r = np.take(xqri, 0, axis=-1)
    # xq_i = xqri[..., 1]
    xq_i = np.take(xqri, 1, axis=-1)
    # xk_r = xkri[..., 0]
    xk_r = np.take(xkri, 0, axis=-1)
    # xk_i = xkri[..., 1]
    xk_i = np.take(xkri, 1, axis=-1)

    # freqs_cos = np.broadcast_to(np.expand_dims(freqs_cos, axis=(0, 2)), (1, 5, 6, 24))
    freqs_cos = np.broadcast_to(
        freqs_cos.reshape((1,) + (freqs_cos.shape[0],) + (1,) + (freqs_cos.shape[1],)),
        (1, 5, 6, 24),
    )
    # freqs_sin = np.broadcast_to(np.expand_dims(freqs_sin, axis=(0, 2)), (1, 5, 6, 24))
    freqs_sin = np.broadcast_to(
        freqs_sin.reshape((1,) + (freqs_sin.shape[0],) + (1,) + (freqs_sin.shape[1],)),
        (1, 5, 6, 24),
    )

    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # Combine real and imaginary parts
    xq = np.stack((xq_out_r, xq_out_i), axis=-1).reshape(xq_out_r.shape[:-1] + (-1,))
    xk = np.stack((xk_out_r, xk_out_i), axis=-1).reshape(xk_out_r.shape[:-1] + (-1,))
    # END inlined apply_rotary_emb()

    cache_k[:batch_size, start_pos : start_pos + seq_len] = xk
    cache_v[:batch_size, start_pos : start_pos + seq_len] = xv
    ks = cache_k[:batch_size, : start_pos + seq_len]
    vs = cache_v[:batch_size, : start_pos + seq_len]

    xq = np.transpose(xq, (0, 2, 1, 3))
    xk = np.transpose(ks, (0, 2, 1, 3))
    xv = np.transpose(vs, (0, 2, 1, 3))

    # FIXME static_broadcast doesn't do dim-expansion.
    _divisor = np.asarray(math.sqrt(head_dim)).reshape((1, 1, 1, 1))
    attention_scores = np.matmul(xq, np.transpose(xk, (0, 1, 3, 2))) / _divisor

    # attention_scores = attention_scores + mask[None, None, :, :]
    attention_scores = attention_scores + mask.reshape((1, 1) + mask.shape)

    # BEGIN inlined softmax
    softmax_x = attention_scores
    exp_x = np.exp(softmax_x - np.max(softmax_x, axis=-1, keepdims=True))
    softmax_out = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    # END
    attn = softmax_out

    output = np.matmul(attn, xv)
    output = np.transpose(output, (0, 2, 1, 3)).reshape((batch_size, seq_len, -1))
    output = np.matmul(output, o_weight)
    return output


def test_attention_full():
    np.random.seed(0)

    x = np.random.random((1, 5, 288))
    mask = np.random.random((5, 5))
    freqs_cos = np.random.random((5, 24))
    freqs_sin = np.random.random((5, 24))
    attn_weights_q = np.random.random((288, 288))
    attn_weights_k = np.random.random((288, 288))
    attn_weights_v = np.random.random((288, 288))
    attn_weights_o = np.random.random((288, 288))
    cache_k = np.random.random((1, 256, 6, 48))
    cache_v = np.random.random((1, 256, 6, 48))

    start_pos = 0
    # test compiler on llm use case
    _run_array_test(
        attention,
        (
            x,  # shape = (1, 5, 288)
            mask,  # shape = (5, 5)
            freqs_cos,  # shape = (5, 24)
            freqs_sin,  # shape = (5, 24)
            attn_weights_q,  # shape = (288, 288)
            attn_weights_k,  # shape = (288, 288)
            attn_weights_v,  # shape = (288, 288)
            attn_weights_o,  # shape = (288, 288)
            cache_k,  # shape = (1, 256, 6, 48)
            cache_v,  # shape = (1, 256, 6, 48)
        ),
    )


def silu(x):
    # TODO: broadcast doesn't do add dimensions correctly yet
    ones = np.asarray(1.0).reshape((1, 1, 1))
    zeros = np.asarray(0.0).reshape((1, 1, 1))
    result = x * (ones / (ones + np.exp(zeros - x)))
    return result


def test_silu_full():
    np.random.seed(0)
    silu_input = np.random.random(1 * 5 * 768).reshape(1, 5, 768)
    _run_array_test(silu, (silu_input,))


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


def transformer_block(
    x, start_pos, mask, freqs_cos, freqs_sin, block_weights, cache_k, cache_v, norm_eps
):
    attn_weights, ff_weights, in_norm_weight, post_norm_weight = block_weights

    norm_x = rmsnorm(x, in_norm_weight, norm_eps)
    h1, cache_k, cache_v = attention(
        norm_x,
        start_pos,
        mask,
        freqs_cos,
        freqs_sin,
        attn_weights,
        cache_k,
        cache_v,
    )
    z = x + h1
    norm_z = rmsnorm(z, post_norm_weight, norm_eps)
    h2 = feed_forward(norm_z, *ff_weights)
    out = z + h2
    return out, cache_k, cache_v


def identity(x):
    return x


def test_identity():
    x = np.random.random(10)
    _run_array_test(identity, (x,))


#######################################

DEBUG = False
PLOT = False
n_repeats = 5
n_runs = 10
func1_name = "NumPy"
func2_name = "MLIRGen"
done_names = set()


def _run_array_unary_test(target_function, inary):
    return _run_array_test(target_function, [inary])


def _run_array_test(target_function, args):
    global_time_recorder.record_func = target_function.__name__
    for _ in range(n_repeats):
        with NumPyExecRecord():
            desired = target_function(*args)
    try:
        cres = run_compiler(target_function, args)
    except TodoException:
        # still try to test the shape output
        be = compiler_config["backend"]
        retty = be.get_last_compiled_return_type()

        assert desired.shape == retty.shape
        assert desired.ndim == retty.ndim

    jit_func = cres.jit_func
    got = jit_func(*args)
    if DEBUG:
        print("GOT".center(80, "-"))
        print(got)
        print("DESIRED".center(80, "-"))
        print(desired)
    np.testing.assert_allclose(got, desired)
    fn_name = cres.fn.__name__

    while fn_name in done_names:
        fn_name = fn_name + "_"

    done_names.add(fn_name)

    if PLOT:
        import timeit
        import matplotlib.pyplot as plt

        # Collect multiple timing measurements for each function
        times1 = []
        times2 = []

        print(
            f"Running {n_repeats} timing measurements, each with {n_runs} executions..."
        )

        for i in range(n_repeats):
            # Time function 1
            t1 = timeit.Timer(lambda: target_function(*args))
            time1 = t1.timeit(number=n_runs) / n_runs  # Average time per execution
            times1.append(time1)

            # Time function 2
            t2 = timeit.Timer(lambda: jit_func(*args))
            time2 = t2.timeit(number=n_runs) / n_runs  # Average time per execution
            times2.append(time2)

            if (i + 1) % 10 == 0:
                print(f"  Completed {i + 1}/{n_repeats} measurements")

        with open("output.txt", "a") as file:
            file.write(fn_name)
            file.write("\n")
            file.write(str(times1))
            file.write("\n")
            file.write(str(times2))
            file.write("\n")
            file.write("\n")

        plt.boxplot([times1, times2], labels=[func1_name, func2_name])
        plt.ylabel("Execution Time (milliseconds)")
        plt.xlabel("Function")
        plt.title("Function Performance Comparison")
        plt.grid(True, alpha=0.3)

        # Calculate and display statistics
        stats_data = [(times1, func1_name), (times2, func2_name)]

        stats_text = []
        for times, name in stats_data:
            mean_time = np.mean(times)
            std_time = np.std(times)
            median_time = np.median(times)
            min_time = np.min(times)
            max_time = np.max(times)
            stats_text.append(
                f"{name}:\n"
                f"  Mean: {mean_time:.4f} ms\n"
                f"  Std: {std_time:.4f} ms\n"
                f"  Median: {median_time:.4f} ms\n"
                f"  Range: [{min_time:.4f}, {max_time:.4f}] ms"
            )

        # Add text box with statistics
        textstr = "\n\n".join(stats_text)
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
        plt.text(
            0.02,
            0.98,
            textstr,
            transform=plt.gca().transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=props,
        )

        # Add performance comparison
        mean1 = np.mean(times1)
        mean2 = np.mean(times2)
        if mean1 < mean2:
            faster = func1_name
            ratio = mean2 / mean1
        else:
            faster = func2_name
            ratio = mean1 / mean2

        comparison_text = f"{faster} is {ratio:.2f}x faster (on average)"
        plt.figtext(
            0.5,
            0.02,
            comparison_text,
            ha="center",
            fontsize=11,
            fontweight="bold",
            color="darkgreen",
        )

        plt.show()


#######################################
# Inlined backend tests


def _run_internal_tests(test_func_str, gen_fn_args, in_shapes, out_shape):
    from mlir.dialects import func
    from mlir import ir
    global_time_recorder.record_func = test_func_str
    test_backend = MlirBackend()
    context = test_backend.context
    loc = ir.Location.name(f"{test_backend}.lower()", context=context)
    module = ir.Module.create(loc=loc)

    # Get the module body pointer so we can insert content into the
    # module.
    module_body = ir.InsertionPoint(module.body)

    with context, loc, module_body:
        # Constuct a function that emits a callable C-interface.
        element_type = ir.F64Type.get()
        input_argtys = [ir.MemRefType.get(x, element_type) for x in in_shapes]
        output_argty = [ir.MemRefType.get(out_shape, element_type)]

        fun = func.FuncOp("func", (input_argtys, output_argty))
        fun.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
        func_block = fun.add_entry_block()

        # Define entry point
        function_entry = ir.InsertionPoint(func_block)

        # Within this function we declare the symbolic representation of
        # input and output arrays of appropriate shapes using memrefs.
        with function_entry:
            test_fn_gen = getattr(test_backend, test_func_str)
            ret = test_fn_gen(
                *fun.arguments, *gen_fn_args, in_shapes=in_shapes, out_shape=out_shape
            )
            func.ReturnOp([ret])

        # Add an empty global init invocation
        func_type = ir.FunctionType.get([], [])

        func_op = func.FuncOp("global_init", func_type)
        func_op.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

        with ir.InsertionPoint(func_op.add_entry_block()):
            func.ReturnOp([])

    module = test_backend.run_passes(module)

    return test_backend.jit_compile_extra(module, input_argtys, output_argty)


def bench_np(np_func):

    def wrapper(*args):
        with NumPyExecRecord():
            res = np_func(*args)

        return res

    return wrapper


def test_unary():
    from mlir.dialects import math as mlir_math

    input_array = np.random.rand(3, 5)

    @bench_np
    def np_func(args):
        return np.exp(args)

    jit_func = _run_internal_tests(
        "_gen_unary_ufunc",
        gen_fn_args=(mlir_math.exp,),
        in_shapes=((3, 5),),
        out_shape=(3, 5),
    )

    np.testing.assert_allclose(jit_func(input_array), np_func(input_array))


def test_binary():
    from mlir.dialects import arith

    input_array_1 = np.random.rand(3, 5)
    input_array_2 = np.random.rand(3, 5)

    @bench_np
    def np_func(a, b):
        return np.add(a, b)

    jit_func = _run_internal_tests(
        "_gen_binop_ufunc",
        gen_fn_args=(arith.addf,),
        in_shapes=((3, 5), (3, 5)),
        out_shape=(3, 5),
    )

    np.testing.assert_allclose(
        jit_func(input_array_1, input_array_2), np_func(input_array_1, input_array_2)
    )


def test_reduce():
    from mlir.dialects import arith

    input_array_1 = np.random.rand(30, 50)

    @bench_np
    def np_func(a):
        return np.sum(a, axis=1)

    jit_func = _run_internal_tests(
        "_gen_reduce_ufunc",
        gen_fn_args=(1, arith.addf),
        in_shapes=((30, 50),),
        out_shape=(30, 1),
    )

    np.testing.assert_allclose(
        jit_func(input_array_1), np_func(input_array_1).reshape(-1, 1)
    )


def test_reshape():
    input_array_1 = np.random.rand(30, 50)

    @bench_np
    def np_func(a):
        return np.reshape(a, (300, 5))

    jit_func = _run_internal_tests(
        "_gen_reshape", gen_fn_args=(), in_shapes=((30, 50),), out_shape=(300, 5)
    )

    np.testing.assert_allclose(jit_func(input_array_1), np_func(input_array_1))


def test_take():
    input_array_1 = np.random.rand(30, 40, 50)

    def np_func(a):
        return np.take(a, indices=1, axis=-1)

    jit_func = _run_internal_tests(
        "_gen_take_shaped",
        gen_fn_args=(1, 3),
        in_shapes=((30, 40, 50),),
        out_shape=(30, 40),
    )

    np.testing.assert_allclose(jit_func(input_array_1), np_func(input_array_1))


def test_broadcast():
    input_array_1 = np.random.rand(1, 50)

    @bench_np
    def np_func(a):
        return np.broadcast_to(a, (30, 50))

    jit_func = _run_internal_tests(
        "_gen_static_broadcast",
        gen_fn_args=(),
        in_shapes=((1, 50),),
        out_shape=(30, 50),
    )

    np.testing.assert_allclose(jit_func(input_array_1), np_func(input_array_1))


def test_expand_dims():
    input_array_1 = np.random.rand(1, 50)

    @bench_np
    def np_func(a):
        return np.expand_dims(a, axis=1)

    jit_func = _run_internal_tests(
        "_gen_array_expand_dims_shaped",
        gen_fn_args=((1,),),
        in_shapes=((1, 50),),
        out_shape=(1, 1, 50),
    )

    np.testing.assert_allclose(jit_func(input_array_1), np_func(input_array_1))


@pytest.mark.skip("Value not an operand")
def test_stack():
    input_array_1 = np.random.rand(3, 5)
    input_array_2 = np.random.rand(3, 5)

    def np_func(a, b):
        return np.stack(a, b)

    jit_func = _run_internal_tests(
        "_gen_array_stack_shaped",
        gen_fn_args=(-1,),
        in_shapes=((3, 5), (3, 5)),
        out_shape=(3, 5, 2),
    )

    np.testing.assert_allclose(
        jit_func(input_array_1, input_array_2), np_func(input_array_1, input_array_2)
    )


def test_transpose():
    input_array_1 = np.random.rand(30, 50)

    @bench_np
    def np_func(a):
        return np.transpose(a)

    jit_func = _run_internal_tests(
        "_gen_inline_array_transpose_shaped",
        gen_fn_args=(),
        in_shapes=((30, 50),),
        out_shape=(50, 30),
    )

    np.testing.assert_allclose(jit_func(input_array_1), np_func(input_array_1))


def test_matmul():
    input_array_1 = np.random.rand(30, 50)
    input_array_2 = np.random.rand(50, 25)

    def np_func(a, b):
        return np.matmul(a, b)

    jit_func = _run_internal_tests(
        "_gen_array_matmul_shaped",
        gen_fn_args=(),
        in_shapes=((30, 50), (50, 25)),
        out_shape=(30, 25),
    )

    np.testing.assert_allclose(
        jit_func(input_array_1, input_array_2), np_func(input_array_1, input_array_2)
    )


@pytest.mark.skip("Value not an operand")
def test_setitem():
    input_array_1 = np.random.rand(30, 40, 50)
    input_array_2 = np.random.rand(30, 5, 50)

    jit_array_1 = input_array_1.copy()
    jit_array_2 = input_array_2.copy()

    def np_func(a, b):
        b[slice(30), slice(5), slice(50)] = a

    jit_func = _run_internal_tests(
        "_gen_array_setitem_shaped",
        gen_fn_args=((slice(30), slice(5), slice(50)),),
        in_shapes=([30, 40, 50], [30, 5, 50]),
        out_shape=(),
    )
    jit_func(jit_array_1, jit_array_2)
    np_func(input_array_1, input_array_2)
    np.testing.assert_allclose(input_array_1, input_array_2)


def test_getitem():
    input_array_1 = np.random.rand(30, 40, 50)

    def np_func(a):
        return a[slice(0, 4), 3]

    jit_func = _run_internal_tests(
        "_gen_array_getitem_shaped",
        gen_fn_args=((slice(0, 4), 3),),
        in_shapes=((30, 40, 50),),
        out_shape=(4, 50),
    )

    np.testing.assert_allclose(jit_func(input_array_1), np_func(input_array_1))
