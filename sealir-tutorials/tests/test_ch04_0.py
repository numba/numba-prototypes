from sealir import ase

import ch04_0_typeinfer_prelude
from ch04_0_typeinfer_prelude import *

from .autotests import autotest_notebook


def test_ch04_0_autotest():
    autotest_notebook(ch04_0_typeinfer_prelude)


def check(fn, ruleset):
    cres = pipeline_backend(
        fn=fn,
        rule_schedule=ruleset.saturate(),
        converter_class=ExtendEGraphToRVSDG,
        cost_model=MyCostModel(),
        codegen_extension=codegen_extension,
    )
    return cres.extracted


def test_ch04_0_code_functioning():
    """
    Test the final code that uses everything
    """
    jt = compiler_pipeline(
        fn=chained_additions,
        rule_schedule=optimized_ruleset.saturate(),
        converter_class=ExtendEGraphToRVSDG,
        codegen_extension=codegen_extension,
        cost_model=MyCostModel(),
    ).jit_func
    run_test(chained_additions, jt, (321, 4535))


def test_ch04_0_typeinfer():
    """
    Test type-inference rules
    """
    extracted = check(add_x_y, ruleset=typeinfer_ruleset)
    inferred_ops = [
        cur
        for ps, cur in ase.walk_descendants_depth_first_no_repeat(extracted)
        if isinstance(cur, NbOp_Unboxed_Add_Int64)
    ]
    assert len(inferred_ops) == 1


def test_ch04_0_boxing_optimization():
    """
    Count the number of boxing and unboxing operations before and after
    apply the optimized-ruleset.
    """
    walkfn = ase.walk_descendants_depth_first_no_repeat

    def get_boxing_unboxing_ops(ruleset):
        extracted = check(chained_additions, ruleset=ruleset)
        boxing_ops = [
            cur
            for ps, cur in walkfn(extracted)
            if isinstance(cur, NbOp_Box_Int64)
        ]
        unboxing_ops = [
            cur
            for ps, cur in walkfn(extracted)
            if isinstance(cur, NbOp_Unbox_Int64)
        ]
        return boxing_ops, unboxing_ops

    before_boxing_ops, before_unboxing_ops = get_boxing_unboxing_ops(
        typeinfer_ruleset
    )
    after_boxing_ops, after_unboxing_ops = get_boxing_unboxing_ops(
        optimized_ruleset
    )

    assert len(before_boxing_ops) > len(after_boxing_ops)
    assert len(before_unboxing_ops) > len(after_unboxing_ops)
