from sealir import ase
from sealir.eqsat.rvsdg_convert import egraph_conversion
from sealir.eqsat.rvsdg_eqsat import GraphRoot
from sealir.eqsat.rvsdg_extract import egraph_extraction
from sealir.rvsdg import grammar as rg

import ch03_egraph_program_rewrites
from ch03_egraph_program_rewrites import *

from .autotests import autotest_notebook


def test_ch03_autotest():
    autotest_notebook(ch03_egraph_program_rewrites)


def test_ch03_ifelse_fold_internal():
    def ifelse_fold(a, b):
        c = 0
        if c:
            return a
        else:
            return b

    def check(fn, rule_schedule):
        cres = compiler_pipeline(fn=fn, rule_schedule=rule_schedule)
        return [
            cur
            for ps, cur in ase.walk_descendants_depth_first_no_repeat(
                cres.extracted
            )
            if is_if_else(cur)
        ]

    # Check there's no if-else
    def is_if_else(expr):
        match expr:
            case rg.IfElse():
                return True
        return False

    ifelse_nodes = check(
        ifelse_fold,
        rule_schedule=(
            rvsdg_eqsat.ruleset_rvsdg_basic | ruleset_const_propagate
        ).saturate(),
    )
    # folding shouldn't occur
    assert len(ifelse_nodes) == 1

    ifelse_nodes = check(
        ifelse_fold,
        rule_schedule=(
            rvsdg_eqsat.ruleset_rvsdg_basic
            | ruleset_const_propagate
            | ruleset_const_fold_if_else
        ).saturate(),
    )
    # folding should occur
    assert len(ifelse_nodes) == 0
