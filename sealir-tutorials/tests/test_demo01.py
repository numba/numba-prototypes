import demo01_gelu_tanh_approx
from demo01_gelu_tanh_approx import *

from .autotests import autotest_notebook


def test_demo01_autotest():
    autotest_notebook(demo01_gelu_tanh_approx)


def test_demo01_baseline():
    cres = jit_compiler(
        fn=gelu_tanh_forward,
        argtypes=(Float32,),
        rule_schedule=(
            base_ruleset | setup_argtypes(TypeFloat32) | additional_rules
        ).saturate()
        + finalize_ruleset.saturate(),
        **compiler_config
    )
    llvm_module = cres.module
    jt = cres.jit_func
    assert "llvm.call @tanhf" in str(llvm_module)
    run_test(gelu_tanh_forward, jt, (0.234,))


def test_demo01_optimized():
    cres = jit_compiler(
        fn=gelu_tanh_forward,
        argtypes=(Float32,),
        rule_schedule=(
            base_ruleset
            | setup_argtypes(TypeFloat32)
            | additional_rules
            | optimize_rules
        ).saturate()
        + finalize_ruleset.saturate(),
        **compiler_config
    )
    llvm_module = cres.module
    jt = cres.jit_func
    # tanhf not used
    assert "llvm.call @tanhf" not in str(llvm_module)
    # powf not used
    assert "llvm.call @powf" not in str(llvm_module)
    # test correctness
    relclose = lambda x, y: np.allclose(x, y, rtol=1e-6)
    run_test(gelu_tanh_forward, jt, (0.234,), equal=relclose)
