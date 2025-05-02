from __future__ import annotations

from egglog import (
    Unit,
    Vec,
    function,
    i64,
    i64Like,
    rewrite,
    rule,
    ruleset,
    set_,
    union,
)
from sealir.eqsat.rvsdg_eqsat import (
    Port,
    PortList,
    Region,
    Term,
    TermList,
    wildcard,
)

from ch03_egraph_program_rewrites import (
    IsConstantFalse,
    IsConstantTrue,
    ruleset_const_fold_if_else,
)
from ch04_1_typeinfer_ifelse import *
from utils import IN_NOTEBOOK

_wc = wildcard


# # Dependent Type
#
# A dependent type is a type that depends on a value, allowing the type system
# to encode more precise relationships between values and types.
#
# In the context of constant-folding If-else expression and its type inference,
# dependent type allows us to encode the type choices based on the condition.
# To do so, we implement a ConditionalType below:


@function
def ConditionalType(cond: Term, then_type: Type, else_type: Type) -> Type: ...


# If the condition (`cond`) is a known constant True, the type becomes just
# `then_type`. If the condition is a known cosntant False, the type becomes
# just `else_type`. If the condition is not known, this type can behave like
# a union type in the runtime.


@ruleset
def ruleset_conditional_type(cond: Term, ta: Type, tb: Type):
    yield rewrite(
        # Fold the type to `then` if the cond is True
        ConditionalType(cond, ta, tb),
        subsume=True,
    ).to(
        ta,
        # given
        IsConstantTrue(cond),
    )

    yield rewrite(
        # Fold the type to `else` if the cond is False
        ConditionalType(cond, ta, tb),
        subsume=True,
    ).to(
        tb,
        # given
        IsConstantFalse(cond),
    )

    yield rewrite(
        # Simplify the type if both sides are the same
        ConditionalType(cond, ta, ta),
        subsume=True,
    ).to(
        ta,
    )


# Now we can redefine the type inference to use the `ConditionalType`.


@ruleset
def ruleset_propagate_typeof_ifelse_conditional_type(
    then_region: Region,
    else_region: Region,
    idx: i64,
    stop: i64,
    ifelse: Term,
    then_ports: PortList,
    else_ports: PortList,
    operands: Vec[Term],
    ta: Type,
    tb: Type,
    ty: Type,
    cond: Term,
    vecports: Vec[Port],
):
    # Most of below is the same as in Ch4.1
    yield rule(
        # Propagate operand types into the contained regions
        Term.IfElse(
            cond=_wc(Term),
            then=Term.RegionEnd(region=then_region, ports=_wc(PortList)),
            orelse=Term.RegionEnd(region=else_region, ports=_wc(PortList)),
            operands=TermList(operands),
        ),
        then_region.get(idx),
    ).then(
        union(TypeVar(operands[idx])).with_(TypedIns(then_region).arg(idx)),
        union(TypeVar(operands[idx])).with_(TypedIns(else_region).arg(idx)),
    )

    @function
    def propagate_ifelse_outs(
        idx: i64Like,
        stop: i64Like,
        then_ports: PortList,
        else_ports: PortList,
        ifelse: Term,
    ) -> Unit: ...

    yield rule(
        # Propagate output types from the contained regions
        ifelse
        == Term.IfElse(
            cond=_wc(Term),
            then=Term.RegionEnd(region=_wc(Region), ports=then_ports),
            orelse=Term.RegionEnd(region=_wc(Region), ports=else_ports),
            operands=TermList(operands),
        ),
        then_ports == PortList(vecports),
    ).then(
        propagate_ifelse_outs(
            0, vecports.length(), then_ports, else_ports, ifelse
        )
    )

    yield rule(
        # Step forward
        propagate_ifelse_outs(idx, stop, then_ports, else_ports, ifelse),
        idx < stop,
    ).then(
        propagate_ifelse_outs(idx + 1, stop, then_ports, else_ports, ifelse),
    )

    # This is the only changed rule.
    yield rule(
        propagate_ifelse_outs(idx, stop, then_ports, else_ports, ifelse),
        ta == TypeVar(then_ports.getValue(idx)).getType(),
        tb == TypeVar(else_ports.getValue(idx)).getType(),
        ifelse
        == Term.IfElse(
            cond=cond,
            then=_wc(Term),
            orelse=_wc(Term),
            operands=_wc(TermList),
        ),
    ).then(
        set_(TypeVar(ifelse.getPort(idx)).getType()).to(
            ConditionalType(cond, ta, tb)
        ),
    )


# Rule to propagate type info for `Term.Apply`, which is used in
# constant-folding of if-else.


@ruleset
def ruleset_typeinfer_region_apply(
    region: Region,
    operands: TermList,
    idx: i64,
    typ: Type,
    term: Term,
    oports: PortList,
):
    yield rule(
        Term.Apply(Term.RegionEnd(region, oports), operands),
        operands[idx],
        typ == TypeVar(operands[idx]).getType(),
    ).then(set_(TypedIns(region).arg(idx).getType()).to(typ))

    yield rule(
        term == Term.Apply(Term.RegionEnd(region, oports), operands),
        typ == TypeVar(term.getPort(idx)).getType(),
    ).then(set_(TypedOuts(region).at(idx).getType()).to(typ))


base_ruleset = (
    basic_ruleset
    | ruleset_propagate_typeof_ifelse_conditional_type
    | ruleset_type_basic
    | ruleset_type_infer_literals
    | ruleset_typeinfer_cast
    | ruleset_type_infer_gt
    | ruleset_type_infer_lt
    | ruleset_type_infer_add
    | ruleset_type_infer_sub
    | ruleset_type_infer_div
    | ruleset_region_types
)


def example_1(a, b):
    cond = 0
    if cond:
        z = a - b  # this as int
    else:
        z = float(b) - 1 / a  # this is float
    return z - float(a)


if __name__ == "__main__":
    jt = compiler_pipeline(
        example_1,
        argtypes=(Int64, Int64),
        ruleset=(
            base_ruleset
            | facts_function_types
            | ruleset_type_infer_float
            | ruleset_failed_to_unify
            # For constant folding
            | (ruleset_const_fold_if_else | ruleset_typeinfer_region_apply)
        ),
        verbose=True,
        converter_class=ExtendEGraphToRVSDG,
        cost_model=MyCostModel(),
        backend=Backend(),
    )
    args = 3, 4
    run_test(example_1, jt, args)
