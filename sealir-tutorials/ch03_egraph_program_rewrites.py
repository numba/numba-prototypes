# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Chapter 3: EGraph Program Rewrites
#
# In this chapter, we walk through implementing our first program rewrite
# using EGraphs. We show how to define rewrite rules, propagate constants,
# and fold if-else branches at compile time. This chapter demonstrates the
# power of EGraph-based optimizations in the compiler pipeline.
#
# The chapter covers:
# - How to define and use EGraph rewrite rules
# - How to propagate constants and fold branches
# - How to extend the compiler pipeline for EGraph-based optimizations

# ## Imports and Setup
#
# Import all necessary modules for EGraph program rewrites.

from __future__ import annotations

from egglog import (
    EGraph,
    Schedule,
    Unit,
    function,
    i64,
    rewrite,
    rule,
    ruleset,
)
from sealir import rvsdg
from sealir.eqsat import rvsdg_eqsat
from sealir.eqsat.rvsdg_eqsat import GraphRoot, Term, TermList

# We'll be extending from chapter 2.
from ch02_egraph_basic import (
    EGraphOutput,
)
from ch02_egraph_basic import compiler_pipeline as _ch02_compiler_pipeline
from ch02_egraph_basic import (
    run_test,
)
from utils import IN_NOTEBOOK, Report, display

# Next, we'll explore a new compiler pipeline designed with customizable
# rule schedules. To enable this flexibility, we've introduced a `rule_schedule`
# argument, allowing you to tailor the pipeline's behavior to your specific needs.
# A rule schedule defines how and when rules are applied during saturation.


def egraph_saturation(
    egraph: EGraph,
    egraph_root: GraphRoot,
    rule_schedule: Schedule,
    pipeline_report=Report.Sink(),
) -> EGraphOutput:
    # Apply the rule schedule to the egraph for saturation
    egraph.run(rule_schedule)
    pipeline_report.append("EGraph Saturated", egraph)
    return {"egraph": egraph, "egraph_root": egraph_root}


compiler_pipeline = _ch02_compiler_pipeline.replace(
    "egraph_action", egraph_saturation
)


if __name__ == "__main__":
    display(compiler_pipeline.visualize())

# ## Rules for defining constants
#
# Now, let's define a simple rule by specifying what makes a constant boolean.
# We'll use `egglog.function` to annotate properties on terms (`Term`
# instances). Each term directly corresponds to an RVSDG-IR node, which in turn
# maps to a Python AST node. As a result, a term can represent various
# constructs—such as an expression, a literal constant, an operation, or a
# control-flow element.
#
# An `egglog.function` acts as a symbolic entity, meaning it doesn't require a
# function body. In our case, we'll use it to mark specific terms: a term is
# labeled as `IsConstantTrue(Term)` if it represents an expression of a non-zero
# literal int64, indicating a constant `True`. Conversely, we mark a term as
# `IsConstantFalse(Term)` if it's an expression of a literal zero, signifying a
# constant `False`.
#


# +
@function
def IsConstantTrue(t: Term) -> Unit: ...


@function
def IsConstantFalse(t: Term) -> Unit: ...


# -


# Rules can be organized into groups known as `ruleset`. Below, we'll define a
# set of rules for recognizing constants, laying the groundwork for our
# optimization process.


@ruleset
def ruleset_const_propagate(a: Term, ival: i64):
    # a Literal Int64 is constant True if it's non-zero
    yield rule(
        # Given a LiteralI64 where the integer-value is non zero
        a == Term.LiteralI64(ival),
        ival != 0,
    ).then(
        # Setup the following fact
        IsConstantTrue(a)
    )
    # a Literal Int64 is constant False if it's zero
    yield rule(
        # Given a LiteralI64 where the integer-value is zero
        a == Term.LiteralI64(ival),
        ival == 0,
    ).then(
        # Setup the following fact
        IsConstantFalse(a)
    )


# Now, we'll test our newly defined ruleset. This complete ruleset combines a
# few built-in RVSDG rules with our recently crafted simple constant-propagation
# rules.

if __name__ == "__main__":

    def ifelse_fold(a, b):
        c = 0
        if c:
            return a
        else:
            return b

    # Add our const-propagation rule to the basic rvsdg ruleset.
    # We use .saturate() to create a schedule that runs the rules to saturation.
    my_ruleset = rvsdg_eqsat.ruleset_rvsdg_basic | ruleset_const_propagate

    report = Report("Test", default_expanded=True)
    jt = compiler_pipeline(
        fn=ifelse_fold,
        pipeline_report=report,
        rule_schedule=my_ruleset.saturate(),
    ).jit_func
    report.display()
    run_test(ifelse_fold, jt, (12, 34))


# In the example above, observe that a new `IsConstantFalse` node appears on the
# `LiteralI64(0)`. This indicates that our ruleset is successfully identifying
# constants as intended.

# ## Rules for folding if-else
#
# Now, let's create a more complex rule. This time, we'll fold an if-else
# expression when its condition is a constant, simplifying the code by resolving
# the branch at compile time.


@ruleset
def ruleset_const_fold_if_else(a: Term, b: Term, c: Term, operands: TermList):
    yield rewrite(
        # Define the if-else pattern to match
        Term.IfElse(cond=a, then=b, orelse=c, operands=operands),
        subsume=True,  # subsume to disable extracting the original term
    ).to(
        # Define the target expression
        # This apply region `b` (then) using the `operands`.
        Term.Apply(b, operands),
        # Given that the condition is constant True
        IsConstantTrue(a),
    )
    yield rewrite(
        # Define the if-else pattern to match
        Term.IfElse(cond=a, then=b, orelse=c, operands=operands),
        subsume=True,  # subsume to disable extracting the original term
    ).to(
        # Define the target expression.
        # This apply region `c` (orelse) using the `operands`.
        Term.Apply(c, operands),
        # Given that the condition is constant False
        IsConstantFalse(a),
    )


if __name__ == "__main__":
    my_ruleset = (
        rvsdg_eqsat.ruleset_rvsdg_basic
        | ruleset_const_propagate
        | ruleset_const_fold_if_else  # <-- the new rule for if-else
    )

    report = Report("Test", default_expanded=True)
    jt = compiler_pipeline(
        fn=ifelse_fold,
        pipeline_report=report,
        rule_schedule=my_ruleset.saturate(),
    ).jit_func
    report.display()
    run_test(ifelse_fold, jt, (12, 34))

# After applying the rewrite, the RVSDG simplifies dramatically, leaving a
# nearly empty function body. The `!ret` instruction is now hardcoded to
# `$0[2]`, which represents the variable `b`, aligning with the `return b` from
# the `else` branch.
#
# Meanwhile, the EGraph becomes more intriguing, with numerous nodes merged to
# reflect their equivalence. For example, the `Term.Apply` and `Term.IfElse`
# nodes are now combined, showcasing how the rewrite consolidates equivalent
# expressions.
