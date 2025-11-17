# ## Imports and Setup
#
# Import all necessary modules for EGraph program rewrites.

from __future__ import annotations

from egglog import EGraph, Ruleset, Unit, function, i64, rewrite, rule, ruleset
from sealir.eqsat.rvsdg_eqsat import GraphRoot, Term, TermList

from egraph import (
    EGraphOutput,
)
from egraph import compiler_pipeline as _compiler_pipeline
from utils import Report


def egraph_saturation(
    egraph: EGraph,
    egraph_root: GraphRoot,
    ruleset: Ruleset,
    pipeline_report=Report.Sink(),
) -> EGraphOutput:
    # Apply the ruleset to the egraph
    egraph.run(ruleset.saturate())
    pipeline_report.append("EGraph Saturated", egraph)
    return {"egraph": egraph, "egraph_root": egraph_root}


compiler_pipeline = _compiler_pipeline.replace("egraph_action", egraph_saturation)


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


# After applying the rewrite, the RVSDG simplifies dramatically, leaving a
# nearly empty function body. The `!ret` instruction is now hardcoded to
# `$0[2]`, which represents the variable `b`, aligning with the `return b` from
# the `else` branch.
#
# Meanwhile, the EGraph becomes more intriguing, with numerous nodes merged to
# reflect their equivalence. For example, the `Term.Apply` and `Term.IfElse`
# nodes are now combined, showcasing how the rewrite consolidates equivalent
# expressions.
