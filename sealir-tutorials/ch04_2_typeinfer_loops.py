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

# # Chapter 4 Part 2: Fully Typing a Scalar Function with Loops
#
# This chapter extends the type inference system to handle loop constructs.
# We show how to implement type inference for while loops, including the
# propagation of types through loop iterations and the handling of loop
# conditions.
#
# The chapter covers:
# - How to implement type inference for loop regions
# - How to handle loop-back type information
# - How to extend the compiler for loop operations

# ## Imports and Setup

from __future__ import annotations

from egglog import (
    Bool,
    String,
    Unit,
    Vec,
    function,
    i64,
    i64Like,
    rule,
    ruleset,
    set_,
    union,
)
from llvmlite import ir
from sealir import ase
from sealir.eqsat.py_eqsat import Py_NotIO
from sealir.eqsat.rvsdg_eqsat import (
    PortList,
    Region,
    Term,
    TermList,
    i64,
)
from sealir.rvsdg import grammar as rg

from ch03_egraph_program_rewrites import (
    run_test,
)
from ch04_1_typeinfer_ifelse import Backend as _ch04_1_Backend
from ch04_1_typeinfer_ifelse import (
    ExtendEGraphToRVSDG as _ch04_1_ExtendEGraphToRVSDG,
)
from ch04_1_typeinfer_ifelse import (
    Grammar,
    Int64,
    MyCostModel,
    NbOp_Base,
    SExpr,
    Type,
    TypeBool,
    TypedIns,
    TypeInt64,
    TypeVar,
    _wc,
)
from ch04_1_typeinfer_ifelse import base_ruleset as _ch4_1_base_ruleset
from ch04_1_typeinfer_ifelse import (
    jit_compiler,
    ruleset_failed_to_unify,
    ruleset_type_infer_failure_report,
    ruleset_type_infer_float,
    setup_argtypes,
)
from utils import Report

# ## Loop Type Inference Rules
#
# Define type inference for loop regions. The logic is similar to the one
# for if-else. The main difference is the loop-back of type info going
# from the loop outputs back to the loop inputs.


@ruleset
def ruleset_propagate_typeof_loops(
    loop: Term,
    body: Term,
    operands: Vec[Term],
    idx: i64,
    ports: PortList,
    region: Region,
    start: i64,
    stop: i64,
    ty: Type,
):
    @function
    def assign_output_loop_typevar(
        start: i64Like,
        stop: i64Like,
        ports: PortList,
        operands: Vec[Term],
        loop: Term,
    ) -> Unit: ...

    yield rule(
        loop == Term.Loop(body=body, operands=TermList(operands)),
        body == Term.RegionEnd(region=region, ports=ports),
        region.get(idx),
    ).then(
        # propagate loop inputs
        union(TypeVar(operands[idx])).with_(TypedIns(region).arg(idx)),
    )

    yield rule(
        loop == Term.Loop(body=body, operands=TermList(operands)),
        body == Term.RegionEnd(region=region, ports=ports),
    ).then(
        # propagate loop outputs
        assign_output_loop_typevar(0, operands.length(), ports, operands, loop)
    )

    yield rule(
        assign_output_loop_typevar(start, stop, ports, operands, loop),
        start + 1 < stop,
    ).then(
        assign_output_loop_typevar(start + 1, stop, ports, operands, loop),
    )

    yield rule(
        assign_output_loop_typevar(start, stop, ports, operands, loop),
        start > 0,
    ).then(
        # TypeVars of loop-region output ports are loop output ports
        union(TypeVar(ports.getValue(start))).with_(
            # minus 1 in because loop output drop the loop condition
            TypeVar(loop.getPort(start - 1))
        ),
        # TypeVars of loop region output ports are the same the operands
        union(TypeVar(ports.getValue(start))).with_(
            # minus 1 in because loop inputs do not have the loop condition
            TypeVar(operands[start - 1])
        ),
    )


# ## Additional Operation Rulesets
#
# Define rulesets for extra operations needed for loop compilation.


@ruleset
def ruleset_type_infer_undef(x: Term, y: Term, io: Term):
    yield rule(
        # Undef operations that are typed to Int64 becomes
        # a literal i64 0
        x == Term.Undef(_wc(String)),
        TypeVar(x).getType() == TypeInt64,  # output is Int64
    ).then(union(x).with_(Term.LiteralI64(0)))
    yield rule(
        # Undef operations that are typed to Bool becomes
        # a literal bool 0
        x == Term.Undef(_wc(String)),
        TypeVar(x).getType() == TypeBool,  # output is Bool
    ).then(union(x).with_(Term.LiteralBool(False)))


@function
def Nb_Not_Int64(operand: Term) -> Term: ...


@ruleset
def ruleset_type_infer_not(x: Term, y: Term, io: Term):
    yield rule(
        # Type-infer unary not that takes a Int64
        y == Py_NotIO(io=io, term=x),
        TypeVar(x).getType() == TypeInt64,
    ).then(
        # Shortcut IO
        union(y.getPort(0)).with_(io),
        # The result becomes Nb_Not_Int64
        union(y.getPort(1)).with_(Nb_Not_Int64(x)),
        # Output is Bool
        set_(TypeVar(Nb_Not_Int64(x)).getType()).to(TypeBool),
    )


# ## E-Graph to RVSDG Extension
#
# Extend EGraphToRVSDG class from Ch4.1 to handle the extra operations


class NbOp_Not_Int64(NbOp_Base):
    operand: SExpr


class ExtendEGraphToRVSDG(_ch04_1_ExtendEGraphToRVSDG):
    def handle_Term(self, op: str, children: dict | list, grm: Grammar):
        match op, children:
            case "Nb_Not_Int64", {"operand": operand}:
                return grm.write(NbOp_Not_Int64(operand=operand))
        return super().handle_Term(op, children, grm)


# ## LLVM Backend Extension
#
# Extend the LLVM Backend from Ch4.1 to handle loop operations


class Backend(_ch04_1_Backend):

    def lower_expr(self, expr, state):
        builder = state.builder
        match expr:
            case rg.Loop(body=rg.RegionEnd() as body, operands=operands):
                # Implement Loop

                # process operands
                ops = []
                for op in operands:
                    ops.append((yield op))

                # Note this is a tail loop.
                begin = body.begin

                with state.push(*ops):
                    loopentry_values = yield begin

                bb_before = builder.basic_block
                bb_loopbody = builder.append_basic_block("loopbody")
                bb_endloop = builder.append_basic_block("endloop")
                builder.branch(bb_loopbody)
                # loop body
                builder.position_at_end(bb_loopbody)
                # setup phi nodes for loopback variables

                phis = []
                for i, var in enumerate(loopentry_values):
                    phi = builder.phi(var.type, name=f"loop_{i}")
                    phi.add_incoming(var, bb_before)
                    phis.append(phi)

                # generate body
                loop_memo = {begin: tuple(phis)}
                memo = ase.traverse(
                    body,
                    self.lower_expr,
                    state=state,
                    init_memo=loop_memo,
                )

                loopout_values = list(memo[body])
                # get loop condition
                loopcond = loopout_values.pop(0)

                # fix up phis
                for i, phi in enumerate(phis):
                    assert phi.type == loopout_values[i].type, (
                        phi.type,
                        loopout_values[i].type,
                    )
                    phi.add_incoming(loopout_values[i], builder.basic_block)
                # back jump
                builder.cbranch(loopcond, bb_loopbody, bb_endloop)
                # end loop
                builder.position_at_end(bb_endloop)
                # Returns the value from the loop body because this is a tail loop
                return loopout_values

            case NbOp_Not_Int64(operand):
                # Implement unary not
                opval = yield operand
                return builder.icmp_unsigned("==", opval, opval.type(0))

        return (yield from super().lower_expr(expr, state))


# ## Base Ruleset
#
# Combine all rulesets for loop type inference

base_ruleset = (
    _ch4_1_base_ruleset
    | ruleset_type_infer_float
    | ruleset_failed_to_unify
    | ruleset_type_infer_failure_report
    | ruleset_type_infer_undef
    | ruleset_type_infer_not
    | ruleset_propagate_typeof_loops
)

# ## Example 1: Simple While Loop
#
# Demonstrate loop compilation with a simple while loop example


def example_1(init, n):
    c = float(init)
    i = 0
    while i < n:
        c = c + float(i)
        i = i + 1
    return c


compiler_config = dict(
    converter_class=ExtendEGraphToRVSDG,
    backend=Backend(),
    cost_model=MyCostModel(),
    verbose=True,
)

if __name__ == "__main__":
    cres = jit_compiler(
        fn=example_1,
        argtypes=(Int64, Int64),
        ruleset=base_ruleset | setup_argtypes(TypeInt64, TypeInt64),
        pipeline_report=Report(),
        **compiler_config,
    )
    jit_func = cres.jit_func
    cres.pipeline_report.display()
    run_test(example_1, jit_func, (10, 7), verbose=True)

# ## Example 2: Nested Loop
#
# Test nested loop compilation with a more complex example


def example_2(init, n):
    c = float(init)
    i = 0
    while i < n:
        j = 0
        while j < i:
            c = c + float(j)
            j = j + 1
        i = i + 1
    return c


if __name__ == "__main__":
    cres = jit_compiler(
        fn=example_2,
        argtypes=(Int64, Int64),
        ruleset=base_ruleset | setup_argtypes(TypeInt64, TypeInt64),
        **compiler_config,
    )
    jit_func = cres.jit_func
    run_test(example_2, jit_func, (10, 7), verbose=True)
