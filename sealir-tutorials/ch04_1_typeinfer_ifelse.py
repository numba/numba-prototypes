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

# # Chapter 4 Part 1: Fully Typing a Scalar Function with If-Else Branch
#
# This chapter extends the type inference system to handle control-flow
# constructs, specifically the if-else branch. We show how to implement
# complete type inference for scalar functions with conditional logic.
#
# The chapter covers:
# - How to implement type inference for if-else constructs
# - How to handle type unification across branches
# - How to extend the compiler pipeline for type inference

# ## Imports and Setup


from __future__ import annotations

import ctypes
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from traceback import print_exception
from typing import Any, Callable, Sequence, TypedDict

from egglog import (
    EGraph,
    Expr,
    Ruleset,
    String,
    StringLike,
    Unit,
    Vec,
    birewrite,
    eq,
    f64,
    function,
    i64,
    i64Like,
    join,
    method,
    ne,
    rewrite,
    rule,
    ruleset,
    set_,
    subsume,
    union,
    vars_,
)
from llvmlite import binding as llvm
from llvmlite import ir
from sealir import ase, grammar, rvsdg
from sealir.eqsat.py_eqsat import (
    Py_AddIO,
    Py_Call,
    Py_DivIO,
    Py_GtIO,
    Py_LoadGlobal,
    Py_LtIO,
    Py_SubIO,
)
from sealir.eqsat.rvsdg_eqsat import (
    GraphRoot,
    InPorts,
    Port,
    PortList,
    Region,
    Term,
    TermList,
    wildcard,
)
from sealir.eqsat.rvsdg_extract import (
    CostModel,
    EGraphToRVSDG,
    ExtractionError,
    egraph_extraction,
)
from sealir.rvsdg import grammar as rg

from ch02_egraph_basic import (
    EGraphExtractionOutput,
)
from ch03_egraph_program_rewrites import (
    EGraphOutput,
)
from ch03_egraph_program_rewrites import (
    compiler_pipeline as _ch03_compiler_pipeline,
)
from ch03_egraph_program_rewrites import (
    egraph_saturation as _ch03_egraph_saturation,
)
from ch03_egraph_program_rewrites import (
    run_test,
)
from ch04_0_typeinfer_prelude import (
    basic_ruleset,
)
from utils import IN_NOTEBOOK, Pipeline, Report, display

_wc = wildcard

# ## Defining Types
# First we'll define the `Type` and `TypeVar` in the EGraph.


# ### `Type`
# `Type` is the actual type. A simple type will just be identified by its name.
# The only operation that it has is `|` for unifying two types.
# For simplicity, we will actually forbid unifying types so that will not be
# any implicit casting.


class Type(Expr):
    @classmethod
    def simple(self, name: StringLike) -> Type:
        "Construct a Type with name"
        ...

    def __or__(self, other: Type) -> Type:
        "Unify with other Type"
        ...


TypeInt64 = Type.simple("Int64")
TypeBool = Type.simple("Bool")
TypeFloat64 = Type.simple("Float64")


if __name__ == "__main__":
    print("Valid types:")
    print(TypeInt64)
    print(TypeBool)
    print(TypeFloat64)
    print("The following will not be allowed:")
    print(TypeInt64 | TypeFloat64)
    print(TypeInt64 | TypeFloat64 | TypeInt64)


# Let's define some rules that will establish what is disallowed:


@ruleset
def ruleset_type_basic(
    ta: Type,
    tb: Type,
    tc: Type,
    name: String,
    ty: Type,
):
    # If ta == tb. then ta
    yield rewrite(ta | tb, subsume=True).to(ta, ta == tb)
    # Simplify
    yield rewrite(ta | tb).to(tb | ta)
    yield birewrite((ta | tb) | tc).to(ta | (tb | tc))


if __name__ == "__main__":
    eg = EGraph()
    eg.register(TypeInt64)
    eg.register(TypeBool)
    eg.register(TypeFloat64)
    eg.register(TypeInt64 | TypeFloat64)
    eg.register(TypeInt64 | TypeFloat64 | TypeInt64)

    print("First run")
    eg.run(ruleset_type_basic)
    if IN_NOTEBOOK:
        eg.display(graphviz=True)
    print("Second run")
    eg.run(ruleset_type_basic)
    if IN_NOTEBOOK:
        eg.display(graphviz=True)
    print("Fully saturated")
    eg.run(ruleset_type_basic.saturate())
    if IN_NOTEBOOK:
        eg.display(graphviz=True)


# ### `TypeVar`
# A type variable for associating program terms and their type.
# Later on, we will merge type variables that are associating with
# different terms. This is needed for the unification of values after a
# conditional branch (if-else).


class TypeVar(Expr):
    def __init__(self, term: Term):
        "Create a TypeVar for a Term"
        ...

    @method(merge=lambda x, y: x | y)
    def getType(self) -> Type:
        """Get the type for this TypeVar.

        Multiple definitions will be merged by Type.__or__, causing a
        unification.
        """
        ...


# Example use of `TypeVar` showing what happens when type-variables with conflicting types are merged:

if __name__ == "__main__":
    eg = EGraph()
    tv_x = TypeVar(Term.LiteralStr("x"))
    tv_y = TypeVar(Term.LiteralStr("y"))
    eg.register(
        set_(tv_x.getType()).to(TypeInt64),
        set_(tv_y.getType()).to(TypeFloat64),
    )
    eg.run(ruleset_type_basic.saturate())
    if IN_NOTEBOOK:
        eg.display(graphviz=True)

# Merging the two type variables will also merge the type that points to:

if __name__ == "__main__":
    eg.register(union(tv_x).with_(tv_y))
    eg.run(ruleset_type_basic.saturate())
    if IN_NOTEBOOK:
        eg.display(graphviz=True)

# ## Handling type inference errors
# Type inference can fail, so we must provide a mechanism for reporting errors.

# ### `ErrorMsg`
# We'll define a `ErrorMsg` class in the egraph to capture all the error
# message. The compilation will always start with `ErrorMsg.root()` in the
# EGraph. When type inference encounters an error, That root node will be
# merged with `ErroMsg.fail()` nodes.


class ErrorMsg(Expr): ...


@function
def ErrorMessage(msg: StringLike) -> ErrorMsg:
    "A node for failure message"
    ...


# ## Typing addition
# Given a function:


def example_0(a, b):
    return a + b


# To implement the addition, we define the operation as an egraph functions:

# Int64 + Int64 -> Int64


@function
def Nb_Add_Int64(lhs: Term, rhs: Term) -> Term: ...


# Float64 + Float64 -> Float64


@function
def Nb_Add_Float64(lhs: Term, rhs: Term) -> Term: ...


# Helper for binary operations


def make_rules_for_binop(binop, lhs_type, rhs_type, typedop, res_type):
    io, lhs, rhs, op = vars_("io lhs rhs op", Term)
    yield rule(
        op == binop(io, lhs, rhs),
        TypeVar(lhs).getType() == lhs_type,
        TypeVar(rhs).getType() == rhs_type,
    ).then(
        # convert to a typed operation
        union(op.getPort(1)).with_(typedop(lhs, rhs)),
        # shortcut io
        union(op.getPort(0)).with_(io),
    )

    yield rule(op == typedop(lhs, rhs)).then(
        # output type
        set_(TypeVar(op).getType()).to(res_type),
    )


# Define addition


@ruleset
def ruleset_type_infer_add():
    # Int64 + Int64 -> Int64
    yield from make_rules_for_binop(
        Py_AddIO, TypeInt64, TypeInt64, Nb_Add_Int64, TypeInt64
    )
    # Float64 + Float64 -> Float64
    yield from make_rules_for_binop(
        Py_AddIO, TypeFloat64, TypeFloat64, Nb_Add_Float64, TypeFloat64
    )


# Define argument types and their propagations:


def setup_argtypes(*argtypes):
    def rule_gen(region):
        return [
            set_(_AttrRegionInputType(region, i).getType()).to(ty)
            for i, ty in enumerate(argtypes, start=1)
        ]

    @ruleset
    def arg_rules(
        outports: Vec[Port],
        func_uid: String,
        reg_uid: String,
        fname: String,
        region: Region,
    ):
        yield rule(
            # This match the function at graph root
            GraphRoot(
                Term.Func(
                    body=Term.RegionEnd(
                        region=region, ports=PortList(outports)
                    ),
                    uid=func_uid,
                    fname=fname,
                )
            ),
            region == Region(uid=reg_uid, inports=_wc(InPorts)),
        ).then(*rule_gen(region))

    return arg_rules


# Associate type variables to region inputs/outputs.


class Attribute(Expr): ...


@function
def _AttrRegionInputType(region: Region, argidx: i64) -> TypeVar: ...


@function
def _AttrRegionOutputType(region: Region, argidx: i64) -> TypeVar: ...


@function
def RegionInputType(region: Region, argidx: i64, typ: Type) -> Attribute: ...


@function
def RegionOutputType(region: Region, argidx: i64, typ: Type) -> Attribute: ...


@ruleset
def ruleset_region_types(
    region: Region,
    idx: i64,
    typ: TypeVar,
    term: Term,
    portlist: PortList,
):
    # Propagate region types
    yield rule(
        # Inputs
        typ == _AttrRegionInputType(region, idx),
        term == region.get(idx),
    ).then(
        union(TypeVar(term)).with_(typ),
    )

    yield rule(
        # Outputs
        term == Term.RegionEnd(region=region, ports=portlist),
        portlist.getValue(idx),
    ).then(
        union(_AttrRegionOutputType(region, idx)).with_(
            TypeVar(portlist.getValue(idx))
        )
    )


@ruleset
def ruleset_annotate_types(ty: Type, region: Region, idx: i64):
    yield rule(
        ty == _AttrRegionInputType(region, idx).getType(),
    ).then(
        subsume(_AttrRegionInputType(region, idx)),
        RegionInputType(region, idx, ty),
    )
    yield rule(
        ty == _AttrRegionOutputType(region, idx).getType(),
    ).then(
        subsume(_AttrRegionOutputType(region, idx)),
        RegionOutputType(region, idx, ty),
    )


if __name__ == "__main__":
    display(_ch03_compiler_pipeline.visualize())
    rules = (
        basic_ruleset
        | ruleset_region_types
        | ruleset_type_basic
        | ruleset_type_infer_add
        | setup_argtypes(TypeInt64, TypeInt64)
    ).saturate() + ruleset_annotate_types
    report = Report("Compiler Pipeline", default_expanded=True)
    try:
        # this should raise a NotImplementedError because we haven't implemented
        # the conversions of `Nb_Add_Int64` back into RVSDG.
        cres = _ch03_compiler_pipeline(
            fn=example_0, rule_schedule=rules, pipeline_report=report
        )
    except NotImplementedError as e:
        # Expect the error to be raised because we haven't implemented the
        # conversions of `Nb_Add_Int64` back into RVSDG.

        # check __notes__ of the exception
        for note in e.__notes__:
            if "Nb_Add_Int64" in str(note):
                print("Error note:")
                print(note)
                break
        else:
            assert False, "should not reach here"
    else:
        # an error should have been raised
        assert False, "should not reach here"
    finally:
        report.display()


# The above compilation have to stop early because we haven't implemented the
# conversions of `Nb_Add_Int64` back into RVSDG.
#
# Observe in the egraph:
# - `Typedouts`, `TypedIns`, `Type.simple("Int64")`

# ## Extend the rest of the compiler


# ### A more extensible compiler pipeline
# We'll need a more extensible compiler pipeline so capability can be added
# later. The new pipeline also gained error checking base on whether there
# is `ErrorMsg` in the egraph.


class CompilationError(Exception):
    def __init__(self, *messages):
        self.messages = messages

    def __str__(self):
        buf = [""]
        for msg in self.messages:
            buf.append("  - " + str(msg))
        return "\n".join(buf)


def egraph_saturation_with_debug(
    egraph: EGraph,
    egraph_root: GraphRoot,
    rule_schedule: Ruleset,
    pipeline_debug: bool = False,
    pipeline_report=Report.Sink(),
    display_egraph=False,
) -> EGraphOutput:
    with pipeline_report.nest("Egraph Saturation") as report:
        # Define graph root that points to the function
        if pipeline_debug:
            report.append("[debug] initial egraph", egraph)

        # Run all the rules until saturation
        runreport = egraph.run(rule_schedule)
        report.append("saturation report", runreport.updated)

        if display_egraph:
            egraph.display()
            # from sealir.model_explorer.core import prepare_egraph, visualize_egraph
            # visualize_egraph(egraph, filepath="debug")
            # prepare_egraph(egraph, filepath="debug")

        if pipeline_debug:
            report.append("[debug] saturated egraph", egraph)
            report.append(
                "[debug] egglog.extract", egraph.extract(egraph_root)
            )

        return dict(egraph=egraph, egraph_root=egraph_root)


class EGraphExtractionOutputMore(EGraphExtractionOutput):
    extracted_roots: list[SExpr]


def pipeline_egraph_extraction(
    egraph,
    rvsdg_expr,
    converter_class,
    cost_model,
    pipeline_report=Report.Sink(),
) -> EGraphExtractionOutputMore:
    with pipeline_report.nest(
        "EGraph Extraction", default_expanded=True
    ) as report:
        try:
            # This is the same as ch4.0
            extraction = egraph_extraction(
                egraph,
                cost_model=cost_model,
            )
        except ExtractionError as e:
            raise CompilationError("extraction failed") from e

        # Look up by types
        grouped_by_type = defaultdict(set)
        for k, v in extraction.node_types.items():
            grouped_by_type[v].add(k)

        memo = {}
        # Process error message
        errors = []
        for k in grouped_by_type["ErrorMsg"]:
            msg = extraction.extract_enode(k).convert(
                rvsdg_expr,
                converter_class=converter_class,
                memo=memo,
            )
            match msg.name, msg.children:
                case "ErrorMessage", [str(msg)]:
                    errors.append(msg)
                case _:
                    raise NotImplementedError(msg)

        if errors:
            raise CompilationError(*errors)

        # Extract to SExpr
        extresult = extraction.extract_graph_root()
        cost = extresult.cost
        extracted = extresult.convert(
            rvsdg_expr,
            converter_class=converter_class,
            memo=memo,
        )
        report.append("Extraction stats", extraction.stats)
        report.append("Extracted cost", cost)
        report.append(f"Extracted Func", format_rvsdg(extracted))

        extracted_roots = defaultdict(list)
        for k in extraction.iter_graph_root():
            node = extraction.extract_enode(k).convert(
                rvsdg_expr, converter_class, memo=memo
            )
            extracted_roots[extraction.node_types[k]].append(node)

    return dict(
        cost=cost, extracted=extracted, extracted_roots=extracted_roots
    )


pipeline_middle_end = (
    _ch03_compiler_pipeline.trunc("egraph_saturation")
    .extend(egraph_saturation_with_debug)
    .extend(pipeline_egraph_extraction)
)

if __name__ == "__main__":
    display(pipeline_middle_end.visualize())


class BackendOutput(TypedDict):
    module: Any


@pipeline_middle_end.extend
def pipeline_backend(
    extracted,
    argtypes,
    extracted_roots,
    backend,
    pipeline_report=Report.Sink(),
) -> BackendOutput:
    with pipeline_report.nest("Backend") as report:
        module = backend.lower(
            extracted, argtypes, extracted_roots["Attribute"]
        )
        report.append("Lowered module", module)
        return dict(module=module)


class JitCompilerOutput(TypedDict):
    jit_func: Any


@pipeline_backend.extend
def jit_compiler(
    module, extracted, backend, export_func_name="func"
) -> JitCompilerOutput:
    jit_func = backend.jit_compile(module, extracted, export_func_name)
    return dict(jit_func=jit_func)


if __name__ == "__main__":
    display(pipeline_backend.visualize())

# ### Define EGraph functions for new operations:


@function
def Nb_Gt_Int64(lhs: Term, rhs: Term) -> Term: ...
@function
def Nb_Lt_Int64(lhs: Term, rhs: Term) -> Term: ...
@function
def Nb_Sub_Int64(lhs: Term, rhs: Term) -> Term: ...
@function
def Nb_Sub_Float64(lhs: Term, rhs: Term) -> Term: ...
@function
def Nb_Div_Int64(lhs: Term, rhs: Term) -> Term: ...
@function
def Nb_CastI64ToF64(operand: Term) -> Term: ...
@function
def Nb_CastToFloat(arg: Term) -> Term: ...


# ### Define rules for the operations:


@ruleset
def ruleset_type_infer_gt(io: Term, x: Term, y: Term, op: Term):
    yield from make_rules_for_binop(
        Py_GtIO, TypeInt64, TypeInt64, Nb_Gt_Int64, TypeBool
    )


@ruleset
def ruleset_type_infer_lt(io: Term, x: Term, y: Term, op: Term):
    yield from make_rules_for_binop(
        Py_LtIO, TypeInt64, TypeInt64, Nb_Lt_Int64, TypeBool
    )


@ruleset
def ruleset_type_infer_sub(io: Term, x: Term, y: Term, op: Term):
    yield from make_rules_for_binop(
        Py_SubIO, TypeInt64, TypeInt64, Nb_Sub_Int64, TypeInt64
    )

    yield from make_rules_for_binop(
        Py_SubIO, TypeFloat64, TypeFloat64, Nb_Sub_Float64, TypeFloat64
    )


## This works but not needed
# @ruleset
# def ruleset_type_infer_sub_promote(io: Term, x: Term, y: Term, op: Term):
#     # # Promote to float if one side is float
#     yield rule(
#         op == Py_SubIO(io, x, y),
#         TypeVar(x).getType() == TypeInt64,
#         TypeVar(y).getType() == TypeFloat64,
#     ).then(
#         union(op).with_(Py_SubIO(io, Nb_CastI64ToF64(x), y)),
#     )
#
#     yield rule(
#         op == Py_SubIO(io, x, y),
#         TypeVar(x).getType() == TypeFloat64,
#         TypeVar(y).getType() == TypeInt64,
#     ).then(
#         union(op).with_(Py_SubIO(io, x, Nb_CastI64ToF64(y))),
#         subsume(Py_SubIO(io, x, y)),
#     )
@ruleset
def ruleset_type_infer_literals(op: Term, ival: i64, fval: f64):
    yield rule(op == Term.LiteralI64(ival)).then(
        set_(TypeVar(op).getType()).to(TypeInt64)
    )
    yield rule(op == Term.LiteralF64(fval)).then(
        set_(TypeVar(op).getType()).to(TypeFloat64)
    )


@ruleset
def ruleset_typeinfer_cast(op: Term, val: Term):
    yield rule(
        op == Nb_CastI64ToF64(val),
        TypeVar(val).getType() == TypeInt64,
    ).then(
        set_(TypeVar(op).getType()).to(TypeFloat64),
    )


@ruleset
def ruleset_type_infer_div(io: Term, x: Term, y: Term, op: Term):
    yield from make_rules_for_binop(
        Py_DivIO, TypeInt64, TypeInt64, Nb_Div_Int64, TypeFloat64
    )


# ### Rules for type-inference on if-else

# Most of the logic is just propagation. The key is merging the type-variables
# of all outputs.


@ruleset
def ruleset_propagate_typeof_ifelse(
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
    vecports: Vec[Port],
):

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
        union(TypeVar(operands[idx])).with_(
            _AttrRegionInputType(then_region, idx)
        ),
        union(TypeVar(operands[idx])).with_(
            _AttrRegionInputType(else_region, idx)
        ),
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

    yield rule(
        propagate_ifelse_outs(idx, stop, then_ports, else_ports, ifelse),
    ).then(
        # TypeVars of then ports are else ports
        union(TypeVar(then_ports.getValue(idx))).with_(
            TypeVar(else_ports.getValue(idx))
        ),
        # TypeVars of ifelse outputs are else ports
        union(TypeVar(ifelse.getPort(idx))).with_(
            TypeVar(else_ports.getValue(idx))
        ),
    )


SExpr = rvsdg.grammar.SExpr


# ### Extend RVSDG Grammar for the new operations


# +
class NbOp_Base(grammar.Rule):
    pass


class NbOp_Type(NbOp_Base):
    name: str


class NbOp_InTypeAttr(NbOp_Base):
    region: SExpr
    idx: int
    type: NbOp_Type


class NbOp_OutTypeAttr(NbOp_Base):
    region: SExpr
    idx: int
    type: NbOp_Type


class NbOp_Gt_Int64(NbOp_Base):
    lhs: SExpr
    rhs: SExpr


class NbOp_Lt_Int64(NbOp_Base):
    lhs: SExpr
    rhs: SExpr


class NbOp_Add_Int64(NbOp_Base):
    lhs: SExpr
    rhs: SExpr


class NbOp_Add_Float64(NbOp_Base):
    lhs: SExpr
    rhs: SExpr


class NbOp_Sub_Int64(NbOp_Base):
    lhs: SExpr
    rhs: SExpr


class NbOp_Sub_Float64(NbOp_Base):
    lhs: SExpr
    rhs: SExpr


class NbOp_Div_Int64(NbOp_Base):
    lhs: SExpr
    rhs: SExpr


class NbOp_CastI64ToF64(NbOp_Base):
    operand: SExpr


class Grammar(grammar.Grammar):
    start = rvsdg.Grammar.start | NbOp_Base


# -

# Define attribute formating


format_rvsdg = rvsdg.format_rvsdg


# ### Extend EGraph to RVSDG


class ExtendEGraphToRVSDG(EGraphToRVSDG):
    grammar = Grammar
    unknown_use_generic = True

    def is_type_from_egraph(self, node) -> bool:
        op = node["op"]
        if op.startswith("Type."):
            return True
        return False

    def handle_Type(
        self, key: str, op: str, children: dict | list, grm: Grammar
    ):
        match op, children:
            case "Type.simple", {"name": name}:
                return grm.write(NbOp_Type(name))
            case "· | ·", {"self": lhs, "other": rhs}:
                raise NotImplementedError(f"cannot unify: {lhs} and {rhs}")
            case _:
                return NotImplemented

    def handle_Term(self, op: str, children: dict | list, grm: Grammar):
        match op, children:
            case "Nb_Gt_Int64", {"lhs": lhs, "rhs": rhs}:
                return grm.write(NbOp_Gt_Int64(lhs=lhs, rhs=rhs))
            case "Nb_Lt_Int64", {"lhs": lhs, "rhs": rhs}:
                return grm.write(NbOp_Lt_Int64(lhs=lhs, rhs=rhs))
            case "Nb_Add_Int64", {"lhs": lhs, "rhs": rhs}:
                return grm.write(NbOp_Add_Int64(lhs=lhs, rhs=rhs))
            case "Nb_Add_Float64", {"lhs": lhs, "rhs": rhs}:
                return grm.write(NbOp_Add_Float64(lhs=lhs, rhs=rhs))
            case "Nb_Sub_Int64", {"lhs": lhs, "rhs": rhs}:
                return grm.write(NbOp_Sub_Int64(lhs=lhs, rhs=rhs))
            case "Nb_Sub_Float64", {"lhs": lhs, "rhs": rhs}:
                return grm.write(NbOp_Sub_Float64(lhs=lhs, rhs=rhs))
            case "Nb_Div_Int64", {"lhs": lhs, "rhs": rhs}:
                return grm.write(NbOp_Div_Int64(lhs=lhs, rhs=rhs))
            case "Nb_CastI64ToF64", {"operand": operand}:
                return grm.write(NbOp_CastI64ToF64(operand=operand))
            case _:
                # Use parent's implementation for other terms.
                return super().handle_Term(op, children, grm)

    def handle_Attribute(
        self, key: str, op: str, children: dict | list, grm: Grammar
    ):
        match op, children:
            case "RegionInputType", {
                "region": region,
                "argidx": int(idx),
                "typ": typ,
            }:
                return grm.write(
                    NbOp_InTypeAttr(region=region, idx=idx, type=typ)
                )
            case "RegionOutputType", {
                "region": region,
                "argidx": int(idx),
                "typ": typ,
            }:
                return grm.write(
                    NbOp_OutTypeAttr(region=region, idx=idx, type=typ)
                )
            case _:
                return NotImplemented


# ### Define cost model
# penalize Python operations (`Py_` prefix)


class MyCostModel(CostModel):
    def get_cost_function(self, nodename, op, ty, cost, children):
        if "Term.Literal" in op:
            # Literals has very low cost
            return self.get_simple(1)
        elif op.startswith("Py_"):
            # Penalize Python operations
            return self.get_simple(float(1e20))
        elif op.startswith("Nb_"):
            return self.get_simple(cost)
        elif op.endswith("getType") and ty == "Type":
            return self.get_simple(1e20)
        # Fallthrough to parent's cost function
        return super().get_cost_function(nodename, op, ty, cost, children)


# ### Define Attributes


# +
def get_port_by_name(ports: Sequence[rg.Port], name: str):
    for i, p in enumerate(ports):
        if p.name == name:
            return i, p
    raise ValueError(f"{name!r} not found")


class AttributeParser:

    def __init__(self, roots: Sequence[SExpr]):
        region_inputs = {}
        region_outputs = {}
        for node in roots:
            match node:
                case NbOp_InTypeAttr(region=rb, idx=int(idx), type=typ):
                    record = region_inputs.setdefault(rb, {})
                    record[idx] = typ
                case NbOp_OutTypeAttr(region=rb, idx=int(idx), type=typ):
                    record = region_outputs.setdefault(rb, {})
                    record[idx] = typ
                case _:
                    raise ValueError(node)

        self._typedins = region_inputs
        self._typedouts = region_outputs

    def get_region_attr(self, rb: rg.RegionBegin) -> RegionAttribute:
        return RegionAttribute(self._typedins[rb], self._typedouts[rb])


class RegionAttribute:
    def __init__(self, typedins, typedouts):
        self._typedins = typedins
        self._typedouts = typedouts

    def get_return_type(self, func: rg.Func):
        i, _ = get_port_by_name(func.body.ports, rvsdg.internal_prefix("ret"))
        try:
            out = self._typedouts[i]
        except KeyError:
            raise CompilationError("Missing return type") from None
        return out

    def num_input_types(self):
        return len(self._typedins)

    def num_output_types(self):
        return len(self._typedouts)

    def input_types(self):
        for idx in range(1, self.num_input_types() + 1):
            yield self._typedins[idx]

    def get_output_type(self, i):
        return self._typedouts.get(i)


# -


# ### Extend LLVM Backend for the new operations


# +
@dataclass(frozen=True)
class LowerStates(ase.TraverseState):
    builder: ir.IRBuilder
    push: Callable
    tos: Callable


class Backend:
    def __init__(self):
        self.initialize_llvm()

    def initialize_llvm(self):
        # llvm.initialize()
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()

    def lower_type(self, ty: NbOp_Type):
        match ty:
            case NbOp_Type("Int64"):
                return ir.IntType(64)
            case NbOp_Type("Float64"):
                return ir.DoubleType()
            case NbOp_Type("Bool"):
                return ir.IntType(1)
        raise NotImplementedError(f"unknown type: {ty}")

    def lower_io_type(self):
        # IO type is an empty struct
        return ir.LiteralStructType(())

    def get_result_type(self, ta: NbOp_Type, tb: NbOp_Type) -> NbOp_Type:
        match (ta, tb):
            case (NbOp_Type("Int64"), NbOp_Type("Float64")) | (
                NbOp_Type("Float64"),
                NbOp_Type("Int64"),
            ):
                return NbOp_Type("Float64")
            case _:
                raise NotImplementedError(f"unsupported cast: {ta} {tb}")

    def lower_cast(self, builder, value, fromty, toty):
        match fromty, toty:
            case (NbOp_Type("Int64"), NbOp_Type("Float64")):
                return builder.sitofp(value, self.lower_type(toty))
            case _:
                raise NotImplementedError(
                    f"unsupported lower_cast: {fromty} -> {toty}"
                )

    def lower(self, root: rg.Func, argtypes, other_roots):
        mod = ir.Module()
        llargtypes = [*map(self.lower_type, argtypes)]

        fname = root.fname
        self.attributes = AttributeParser(other_roots)
        retty = self.attributes.get_region_attr(
            root.body.begin
        ).get_return_type(root)
        llrettype = self.lower_type(retty)

        fnty = ir.FunctionType(llrettype, llargtypes)
        fn = ir.Function(mod, fnty, name=fname)
        # init entry block and builder
        builder = ir.IRBuilder(fn.append_basic_block("entry"))
        iostate = ir.LiteralStructType(())(ir.Undefined)

        # Emit the function body
        reg_args_stack = []

        @contextmanager
        def push(*regionargs):
            reg_args_stack.append(regionargs)
            yield
            reg_args_stack.pop()

        def tos():
            return reg_args_stack[-1]

        try:
            with push(iostate, *fn.args):
                memo = ase.traverse(
                    root.body,
                    self.lower_expr,
                    state=LowerStates(builder=builder, push=push, tos=tos),
                )
        except:
            print(mod)
            raise

        func_region_outs = memo[root.body]

        i, p = get_port_by_name(root.body.ports, rvsdg.internal_prefix("ret"))
        builder.ret(func_region_outs[i])

        return mod

    def lower_expr(self, expr: SExpr, state: LowerStates):
        builder = state.builder
        match expr:
            case rg.RegionBegin(inports=inports):
                values = state.tos()
                assert len(values) == len(inports)
                return values
            case rg.RegionEnd(begin=begin, ports=ports):
                yield begin
                portvalues = []
                for p in ports:
                    pv = yield p.value
                    portvalues.append(pv)
                return portvalues

            case rg.IfElse(
                cond=cond, body=body, orelse=orelse, operands=operands
            ):
                condval = yield cond

                # process operands
                ops = []
                for op in operands:
                    ops.append((yield op))

                # unpack pybool
                match condval.type:
                    case ir.IntType() if condval.type.width == 1:
                        condbit = condval
                    case _:
                        raise NotImplementedError(
                            f"unhandled if-cond type: {condval.type}"
                        )

                bb_then = builder.append_basic_block("then")
                bb_else = builder.append_basic_block("else")
                bb_endif = builder.append_basic_block("endif")

                builder.cbranch(condbit, bb_then, bb_else)

                # Then
                with builder.goto_block(bb_then):
                    with state.push(*ops):
                        value_then = yield body

                    builder.branch(bb_endif)
                    bb_then_end = builder.basic_block
                # Else
                with builder.goto_block(bb_else):
                    with state.push(*ops):
                        value_else = yield orelse

                    builder.branch(bb_endif)
                    bb_else_end = builder.basic_block
                # EndIf
                builder.position_at_end(bb_endif)
                phis = []
                paired = zip(value_then, value_else, strict=True)
                for i, (left, right) in enumerate(paired):
                    assert (
                        left.type == right.type
                    ), f"{left.type} != {right.type}"
                    phi = builder.phi(left.type, name=f"ifelse_{i}")
                    phi.add_incoming(left, bb_then_end)
                    phi.add_incoming(right, bb_else_end)
                    phis.append(phi)
                return phis

            case rg.Unpack(val=source, idx=int(idx)):
                return (yield source)[idx]

            case NbOp_Gt_Int64(lhs, rhs):
                lhs = yield lhs
                rhs = yield rhs
                return builder.icmp_signed(">", lhs, rhs)

            case NbOp_Lt_Int64(lhs, rhs):
                lhs = yield lhs
                rhs = yield rhs
                return builder.icmp_signed("<", lhs, rhs)

            case NbOp_Add_Int64(lhs, rhs):
                lhs = yield lhs
                rhs = yield rhs
                return builder.add(lhs, rhs)

            case NbOp_Add_Float64(lhs, rhs):
                lhs = yield lhs
                rhs = yield rhs
                return builder.fadd(lhs, rhs)

            case NbOp_Sub_Int64(lhs, rhs):
                lhs = yield lhs
                rhs = yield rhs
                return builder.sub(lhs, rhs)

            case NbOp_Sub_Float64(lhs, rhs):
                lhs = yield lhs
                rhs = yield rhs
                return builder.fsub(lhs, rhs)

            case NbOp_Div_Int64(lhs, rhs):
                lhs = yield lhs
                rhs = yield rhs
                x = builder.sitofp(lhs, ir.DoubleType())
                y = builder.sitofp(rhs, ir.DoubleType())
                return builder.fdiv(x, y)

            case NbOp_CastI64ToF64(operand):
                val = yield operand
                return builder.sitofp(val, ir.DoubleType())

            ##### more

            case rg.PyBool(val):
                return ir.IntType(1)(val)

            case rg.PyInt(val):
                return ir.IntType(64)(val)

        raise NotImplementedError(expr)

    def jit_compile(self, llmod: ir.Module, func_node: rg.Func, func_name):
        sym = func_node.fname
        # Create JIT
        lljit = llvm.create_lljit_compiler()
        rt = (
            llvm.JITLibraryBuilder()
            .add_ir(str(llmod))
            .export_symbol(sym)
            .add_current_process()
            .link(lljit, sym)
        )
        ptr = rt[sym]

        fnty = llmod.get_global(sym).type.pointee
        ct_args = list(map(self.get_ctype, fnty.args))
        ct_ret = self.get_ctype(fnty.return_type)

        return JitCallable.from_pointer(rt, ptr, ct_args, ct_ret)

    def get_ctype(self, lltype: ir.Type):
        match lltype:
            case ir.IntType():
                match lltype.width:
                    case 64:
                        return ctypes.c_int64
            case ir.DoubleType():
                return ctypes.c_double
        raise NotImplementedError(lltype)

    def run_passes(self, module, passes):
        pass


# -


# Define a new `JitCallable` with control of the argument


@dataclass(frozen=True)
class JitCallable:
    rt: llvm.ResourceTracker
    pyfunc: Callable

    @classmethod
    def from_pointer(cls, rt: llvm.ResourceTracker, ptr: int, argtys, retty):
        pyfunc = ctypes.PYFUNCTYPE(retty, *argtys)(ptr)
        return cls(rt=rt, pyfunc=pyfunc)

    def __call__(self, *args: Any) -> Any:
        return self.pyfunc(*args)


Int64 = NbOp_Type("Int64")

base_ruleset = (
    basic_ruleset
    | ruleset_propagate_typeof_ifelse
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


# ## Example 1: simple if-else


def example_1(a, b):
    if a > b:
        z = a - b
    else:
        z = b - a
    return z + a


if __name__ == "__main__":
    pipeline_report = Report(
        "Pipeline execution report", enable_nested_metadata=True
    )
    jit_func = jit_compiler(
        fn=example_1,
        argtypes=(Int64, Int64),
        rule_schedule=(
            base_ruleset | setup_argtypes(TypeInt64, TypeInt64)
        ).saturate()
        + ruleset_annotate_types.saturate(),
        converter_class=ExtendEGraphToRVSDG,
        backend=Backend(),
        cost_model=MyCostModel(),
        pipeline_debug=True,
        pipeline_report=pipeline_report,
    ).jit_func
    pipeline_report.display()

    args = (10, 33)
    run_test(example_1, jit_func, args, verbose=True)
    args = (7, 3)
    run_test(example_1, jit_func, args, verbose=True)


# ## Example 2: add `float()`


def example_2(a, b):
    if a > b:
        z = float(a - b)
    else:
        z = float(b) - 1 / a
    return z - float(a)


# Add rules for `float()`


@ruleset
def ruleset_type_infer_float(
    io: Term, loadglb: Term, callstmt: Term, args: Vec[Term], arg: Term
):
    yield rule(
        # Convert Python float(arg)
        loadglb == Py_LoadGlobal(io=_wc(Term), name="float"),
        callstmt == Py_Call(io=io, func=loadglb, args=TermList(args)),
        eq(args.length()).to(i64(1)),
    ).then(
        union(callstmt.getPort(1)).with_(Nb_CastToFloat(args[0])),
        union(callstmt.getPort(0)).with_(io),
    )
    # Type check and specialize
    yield rewrite(
        Nb_CastToFloat(arg),
        subsume=True,
    ).to(
        Nb_CastI64ToF64(arg),
        # given
        TypeVar(arg).getType() == TypeInt64,
    )


if __name__ == "__main__":
    report = Report("Pipeline execution report")
    cres = jit_compiler(
        fn=example_2,
        argtypes=(Int64, Int64),
        rule_schedule=(
            base_ruleset
            | setup_argtypes(TypeInt64, TypeInt64)
            | ruleset_type_infer_float  # < --- added for float()
        ).saturate()
        + ruleset_annotate_types.saturate(),
        converter_class=ExtendEGraphToRVSDG,
        backend=Backend(),
        cost_model=MyCostModel(),
        pipeline_debug=True,
        pipeline_report=report,
    )
    jit_func = cres.jit_func
    args = (10, 33)
    run_test(example_2, jit_func, args, verbose=True)
    args = (7, 3)
    run_test(example_2, jit_func, args, verbose=True)

# ## Example 3: unify mismatching type
#
# What if type of `z` does not match across the branch?


def example_3(a, b):
    if a > b:
        z = a - b  # this as int
    else:
        z = float(b) - 1 / a  # this is float
    return z - float(a)


# Add rules to signal error


@function
def failed_to_unify(ty: Type) -> Unit: ...


@ruleset
def ruleset_prune_type_unify(ty: Type):
    yield rewrite(ty | ty, subsume=True).to(ty)


@ruleset
def ruleset_failed_to_unify(ty: Type, tz: Type):
    yield rule(
        ty | tz,
    ).then(
        failed_to_unify(ty | tz),
        ErrorMessage("fail to unify"),
    )


if __name__ == "__main__":
    report = Report("Pipeline execution report")
    try:
        jit_compiler(
            fn=example_3,
            argtypes=(Int64, Int64),
            rule_schedule=(
                base_ruleset
                | setup_argtypes(TypeInt64, TypeInt64)
                | ruleset_type_infer_float
            ).saturate()
            + (ruleset_prune_type_unify.saturate())
            + (ruleset_failed_to_unify | ruleset_annotate_types).saturate(),
            converter_class=ExtendEGraphToRVSDG,
            backend=Backend(),
            cost_model=MyCostModel(),
            pipeline_debug=True,
            pipeline_report=report,
        )
    except CompilationError as e:
        # Compilation failed because the return type cannot be determined.
        # This indicates that the type inference is incomplete
        print_exception(e)
        assert "fail to unify" in str(e)
    finally:
        report.display()


# ## Example 4: Improve error reporting
#
# Add logics to report error early


@ruleset
def ruleset_type_infer_failure_report(
    ifelse: Term,
    ty: Type,
    idx: i64,
    name: String,
    then_region: Region,
    else_region: Region,
    then_ports: PortList,
    else_ports: PortList,
):
    yield rule(
        ifelse
        == Term.IfElse(
            cond=_wc(Term),
            then=Term.RegionEnd(then_region, ports=then_ports),
            orelse=Term.RegionEnd(else_region, ports=else_ports),
            operands=_wc(TermList),
        ),
        ty == TypeVar(ifelse.getPort(idx)).getType(),
        failed_to_unify(ty),
        name == then_ports[idx].name,
    ).then(
        ErrorMessage(
            join("Failed to unify if-else outgoing variables: ", name)
        ),
    )


if __name__ == "__main__":
    report = Report("Pipeline execution report")
    try:
        jit_compiler(
            fn=example_3,
            argtypes=(Int64, Int64),
            rule_schedule=(
                base_ruleset
                | setup_argtypes(TypeInt64, TypeInt64)
                | ruleset_type_infer_float
            ).saturate()
            + (ruleset_prune_type_unify.saturate())
            + (
                ruleset_failed_to_unify
                | ruleset_type_infer_failure_report
                | ruleset_annotate_types
            ).saturate(),
            converter_class=ExtendEGraphToRVSDG,
            backend=Backend(),
            cost_model=MyCostModel(),
            pipeline_debug=True,
            pipeline_report=report,
        )

    except CompilationError as e:
        print_exception(e)
        assert "Failed to unify if-else outgoing variables: z" in str(e)
    finally:
        report.display()
