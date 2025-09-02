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


@function
def failed_to_unify(ty: Type) -> Unit: ...


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

    # Identify errors
    yield rule(
        # If both sides are valid types and not equal, then fail
        ty == ta | tb,
        ne(ta).to(tb),  # ta != tb
    ).then(failed_to_unify(ty))


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


class ErrorMsg(Expr):
    @classmethod
    def root(cls) -> ErrorMsg:
        "The empty root"
        ...

    @classmethod
    def fail(cls, msg: String) -> ErrorMsg:
        "A node for failure message"
        ...

    @method(preserve=True)
    def eval(self) -> tuple[str, tuple]:
        """
        This is for converting the information in the EGraph back to
        Python. This will parse the EGraph node to extract the message string.
        """
        from egglog.builtins import ClassMethodRef, _extract_call

        call = _extract_call(self)
        if isinstance(call.callable, ClassMethodRef):
            assert call.callable.class_name == "ErrorMsg"
            args = [self.__with_expr__(x).eval() for x in call.args]
            return call.callable.method_name, tuple(args)
        raise TypeError


# Helpers to process the error message


def get_error_message(err_info: tuple[str, tuple]) -> str:
    "Helper to process the result of ErrorMsg.eval()"
    match err_info:
        case "fail", (msg,):
            return msg
        case _:
            raise NotImplementedError


# For example

if __name__ == "__main__":
    root = ErrorMsg.root()
    eg = EGraph()
    eg.register(
        union(root).with_(ErrorMsg.fail("I failed")),
        union(root).with_(ErrorMsg.fail("Failed again")),
    )
    if IN_NOTEBOOK:
        eg.display(graphviz=True)
    msgs = eg.extract_multiple(root, n=3)
    print(msgs)
    for msg in msgs:
        print(msg.eval())
        try:
            print(get_error_message(msg.eval()))
        except NotImplementedError:
            print("no msg")


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
            set_(TypedIns(region).arg(i).getType()).to(ty)
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


class TypedIns(Expr):
    def __init__(self, region: Region): ...

    def arg(self, idx: i64Like) -> TypeVar: ...


class TypedOuts(Expr):
    def __init__(self, region: Region): ...

    def at(self, idx: i64Like) -> TypeVar: ...


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
        typ == TypedIns(region).arg(idx),
        term == region.get(idx),
    ).then(
        union(TypeVar(term)).with_(typ),
    )

    # yield rule(
    #     # Outputs
    #     term == Term.RegionEnd(region=region, ports=portlist),
    #     pv := portlist.getValue(idx),
    # ).then(
    #     union(TypedOuts(region).at(idx)).with_(TypeVar(pv)),
    # )


if __name__ == "__main__":
    display(_ch03_compiler_pipeline.visualize())
    rules = (
        basic_ruleset
        | ruleset_region_types
        | ruleset_type_basic
        | ruleset_type_infer_add
        | setup_argtypes(TypeInt64, TypeInt64)
    )
    report = Report("Compiler Pipeline", default_expanded=True)
    try:
        # this should raise a NotImplementedError because we haven't implemented
        # the conversions of `Nb_Add_Int64` back into RVSDG.
        cres = _ch03_compiler_pipeline(
            fn=example_0, ruleset=rules, pipeline_report=report
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
    pass


def egraph_saturation_with_error_checking(
    egraph: EGraph,
    egraph_root: GraphRoot,
    ruleset: Ruleset,
    pipeline_debug: bool = False,
    pipeline_report=Report.Sink(),
) -> EGraphOutput:
    with pipeline_report.nest("Egraph Saturation") as report:
        # Define graph root that points to the function

        # Define the empty root node for the error messages
        errors = ErrorMsg.root()
        egraph.let("errors", errors)
        if pipeline_debug:
            report.append("[debug] initial egraph", egraph)

        # Run all the rules until saturation
        runreport = egraph.run(ruleset.saturate())
        report.append("saturation report", runreport.updated)

        if pipeline_debug:
            report.append("[debug] saturated egraph", egraph)
            report.append(
                "[debug] egglog.extract", egraph.extract(egraph_root)
            )

        # Use egglog's default extractor to get the error messages
        errmsgs = map(
            lambda x: x.eval(), egraph.extract_multiple(errors, n=10)
        )
        errmsgs_filtered = [
            get_error_message((meth, args))
            for meth, args in errmsgs
            if meth != "root"
        ]
        if errmsgs_filtered:
            # Raise CompilationError if there are compiler errors
            raise CompilationError("\n".join(errmsgs_filtered))

        return dict(egraph=egraph, egraph_root=egraph_root)


def pipeline_egraph_extraction(
    egraph,
    rvsdg_expr,
    converter_class,
    cost_model,
    pipeline_report=Report.Sink(),
) -> EGraphExtractionOutput:
    with pipeline_report.nest(
        "EGraph Extraction", default_expanded=True
    ) as report:
        stats = {}
        try:
            # This is the same as ch4.1
            cost, extracted = egraph_extraction(
                egraph,
                rvsdg_expr,
                converter_class=converter_class,
                cost_model=cost_model,
                stats = stats,
            )
        except ExtractionError as e:
            raise CompilationError("extraction failed") from e

        report.append("Extraction stats", stats)
        report.append("Extracted cost", cost)
        for i, child in enumerate(extracted._args):
            if isinstance(child, rg.Func):
                report.append(f"Extracted[{i}] Func", format_rvsdg(child))
            else:
                report.append(f"Extracted[{i}] {child._head}", ase.pretty_str(child))

    return dict(cost=cost, extracted=extracted)


pipeline_middle_end = (
    _ch03_compiler_pipeline.trunc("egraph_saturation")
    .extend(egraph_saturation_with_error_checking)
    .extend(pipeline_egraph_extraction)
)

if __name__ == "__main__":
    display(pipeline_middle_end.visualize())


class BackendOutput(TypedDict):
    module: Any


@pipeline_middle_end.extend
def pipeline_backend(
    extracted, argtypes, backend, pipeline_report=Report.Sink()
) -> BackendOutput:
    with pipeline_report.nest("Backend") as report:
        module = backend.lower(extracted, argtypes)
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
    idx: int
    type: NbOp_Type


class NbOp_OutTypeAttr(NbOp_Base):
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


def my_attr_format(attrs: rg.Attrs) -> str:
    ins = {}
    outs = {}
    others = []
    for attr in attrs.attrs:
        match attr:
            case NbOp_InTypeAttr(idx=int(idx), type=NbOp_Type(name=str(name))):
                ins[idx] = name
            case NbOp_OutTypeAttr(
                idx=int(idx), type=NbOp_Type(name=str(name))
            ):
                outs[idx] = name
            case _:
                others.append(attr)

    def format(dct):
        if len(dct):
            hi = max(dct.keys())
            out = ", ".join(dct.get(i, "_") for i in range(hi + 1))
            return f"({out})"
        else:
            return "()"

    outbuf = []
    if ins or outs:
        outbuf.append(format(ins) + "->" + format(outs))
    for other in others:
        outbuf.append(ase.pretty_str(other))
    return ", ".join(outbuf)


format_rvsdg = partial(rvsdg.format_rvsdg, format_attrs=my_attr_format)


# ### Extend EGraph to RVSDG


class ExtendEGraphToRVSDG(EGraphToRVSDG):
    grammar = Grammar

    def is_type_from_egraph(self, node) -> bool:
        op = node["op"]
        if op.startswith("Type."):
            return True
        return False

    # def handle_region_attributes(self, key: str, grm: Grammar):
    #     def search_equiv_calls(self_key: str):
    #         nodes = self.gdct["nodes"]
    #         ecl = nodes[self_key]["eclass"]
    #         for k, v in nodes.items():
    #             children = v["children"]
    #             if children and nodes[children[0]]["eclass"] == ecl:
    #                 yield k, v

    #     def get_types(key_arg):
    #         typs = []
    #         for k, v in search_equiv_calls(key_arg):
    #             for j in self.search_eclass_siblings(k):
    #                 if self.is_type_from_egraph(self.gdct["nodes"][j]):
    #                     typ = self.dispatch(j, grm)
    #                     typs.append(typ)
    #         return typs

    #     attrs = []
    #     typedargs = list(self.search_calls(key, "TypedIns"))
    #     if typedargs:
    #         [typedarg] = typedargs
    #         for key_arg in self.search_method_calls(typedarg, "arg"):
    #             _k_self, k_idx = self.get_children(key_arg)
    #             # get the idx in `.arg(idx)`
    #             idx = self.dispatch(k_idx, grm)
    #             typs = get_types(key_arg)

    #             if len(typs) == 1:
    #                 typ = typs[0]
    #                 attrs.append(grm.write(NbOp_InTypeAttr(idx=idx, type=typ)))
    #             else:
    #                 resolved = list(map(ase.pretty_str, typs))
    #                 assert len(typs) == 0, f"multiple types: {resolved}"

    #     typedouts = list(self.search_calls(key, "TypedOuts"))
    #     if typedouts:
    #         [typedout] = typedouts
    #         for key_at in self.search_method_calls(typedout, "at"):
    #             _k_self, k_idx = self.get_children(key_at)
    #             idx = self.dispatch(k_idx, grm)

    #             typs = get_types(key_at)
    #             if len(typs) == 1:
    #                 typ = typs[0]
    #                 attrs.append(
    #                     grm.write(NbOp_OutTypeAttr(idx=idx, type=typ))
    #                 )
    #             else:
    #                 assert len(typs) == 0, "multiple types"

    #     return grm.write(rg.Attrs(tuple(attrs)))

    def handle_Type(
        self, key: str, op: str, children: dict | list, grm: Grammar
    ):
        match children:
            case {"name": name}:
                return grm.write(NbOp_Type(name))
        raise NotImplementedError

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


# ### Define cost model
# penalize Python operations (`Py_` prefix)


class MyCostModel(CostModel):
    def get_cost_function(self, nodename, op, ty, cost, children):
        if "Term.Literal" in op:
            # Literals has very low cost
            return self.get_simple(1)
        elif op.startswith("Py_"):
            # Penalize Python operations
            return self.get_simple(float("1e9"))
        elif op.startswith("Nb_"):
            return self.get_simple(cost)
        elif op == "ArrayDescSpec":
            return self.get_simple(1)
        # Fallthrough to parent's cost function
        return super().get_cost_function(nodename, op, ty, cost, children)


# ### Define Attributes


# +
def get_port_by_name(ports: Sequence[rg.Port], name: str):
    for i, p in enumerate(ports):
        if p.name == name:
            return i, p
    raise ValueError(f"{name!r} not found")


class Attributes:
    _typedins: dict[int, NbOp_InTypeAttr]
    _typedouts: dict[int, NbOp_OutTypeAttr]

    def __init__(self, attrs: rg.Attrs):

        ins = {}
        outs = {}
        for attr in attrs.attrs:
            match attr:
                case NbOp_InTypeAttr(idx=idx):
                    ins[idx] = attr
                case NbOp_OutTypeAttr(idx=idx):
                    outs[idx] = attr
                case _:
                    raise ValueError(attr)

        self._typedins = ins
        self._typedouts = outs

    def get_output_attribute(self, idx: int) -> NbOp_OutTypeAttr | None:
        return self._typedouts.get(idx)

    def get_output_type(self, idx: int) -> NbOp_Type | None:
        at = self._typedouts.get(idx)
        if at is not None:
            return at.type
        return None

    def get_return_type(self, regionend: rg.RegionEnd):
        i, p = get_port_by_name(regionend.ports, rvsdg.internal_prefix("ret"))
        if attr := self.get_output_attribute(i):
            return attr.type
        raise CompilationError("Missing return type")

    def num_input_types(self):
        return len(self._typedins)

    def num_output_types(self):
        return len(self._typedouts)

    def input_types(self):
        for idx in range(1, self.num_input_types() + 1):
            yield self._typedins[idx].type


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
        llvm.initialize()
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

    def lower(self, root: rg.Func, argtypes):
        mod = ir.Module()
        llargtypes = [*map(self.lower_type, argtypes)]

        fname = root.fname
        retty = Attributes(root.body.begin.attrs).get_return_type(root.body)
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
        ruleset=(base_ruleset | setup_argtypes(TypeInt64, TypeInt64)),
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
        ruleset=(
            base_ruleset
            | setup_argtypes(TypeInt64, TypeInt64)
            | ruleset_type_infer_float  # < --- added for float()
        ),
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


@ruleset
def ruleset_failed_to_unify(ty: Type):
    yield rule(
        failed_to_unify(ty),
    ).then(
        union(ErrorMsg.root()).with_(ErrorMsg.fail("fail to unify")),
    )


if __name__ == "__main__":
    report = Report("Pipeline execution report")
    try:
        jit_compiler(
            fn=example_3,
            argtypes=(Int64, Int64),
            ruleset=(
                base_ruleset
                | setup_argtypes(TypeInt64, TypeInt64)
                | ruleset_type_infer_float
                | ruleset_failed_to_unify
            ),
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
        union(ErrorMsg.root()).with_(
            ErrorMsg.fail(
                join("Failed to unify if-else outgoing variables: ", name)
            )
        ),
    )


if __name__ == "__main__":
    report = Report("Pipeline execution report")
    try:
        jit_compiler(
            fn=example_3,
            argtypes=(Int64, Int64),
            ruleset=(
                base_ruleset
                | setup_argtypes(TypeInt64, TypeInt64)
                | ruleset_type_infer_float
                | ruleset_failed_to_unify
                | ruleset_type_infer_failure_report
            ),
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
