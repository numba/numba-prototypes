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

# # Chapter 4 Part 0: Type Inference Prelude
#
# This chapter introduces the basics of type inference in the compiler
# pipeline. We show how to add type inference logic for scalar operations
# and how to extend the compiler pipeline to support type-aware rewrites.
#
# The chapter covers:
# - How to represent types in the e-graph
# - How to add type inference rules for basic operations
# - How to extend the compiler pipeline for type inference

# ## Imports and Setup

from egglog import (
    EGraph,
    Expr,
    String,
    StringLike,
    function,
    rewrite,
    rule,
    ruleset,
    union,
)
from sealir import grammar, rvsdg
from sealir.eqsat import rvsdg_eqsat
from sealir.eqsat.py_eqsat import Py_AddIO
from sealir.eqsat.rvsdg_convert import egraph_conversion
from sealir.eqsat.rvsdg_eqsat import GraphRoot, PortList, Region, Term
from sealir.eqsat.rvsdg_extract import (
    CostModel,
    EGraphToRVSDG,
    Extraction,
    egraph_extraction,
)
from sealir.llvm_pyapi_backend import SSAValue
from sealir.rvsdg import grammar as rg

from ch02_egraph_basic import (
    BackendOutput,
    EGraphExtractionOutput,
    backend,
    jit_compile,
    pipeline_egraph_extraction,
)
from ch03_egraph_program_rewrites import (
    compiler_pipeline as _ch03_compiler_pipeline,
)
from ch03_egraph_program_rewrites import (
    ruleset_const_propagate,
    run_test,
)
from utils import IN_NOTEBOOK, Report, display

# First, we need some modifications to the compiler-pipeline.
# The middle-end is augmented with the following:
#
# - `converter_class` is for customizing EGraph-to-RVSDG conversion as we will be
#   introducing new RVSDG operations for typed operations.
# - `cost_model` is for customizing the cost of the new operations.


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
        extraction = egraph_extraction(
            egraph,
            cost_model=cost_model,  # <-------------- new
        )
        extresult = extraction.extract_graph_root()
        cost = extresult.cost
        extracted = extresult.convert(
            rvsdg_expr,
            converter_class=converter_class,  # <---- new
        )
        report.append("Cost", cost)
        report.append("Extracted", rvsdg.format_rvsdg(extracted))
        return {"cost": cost, "extracted": extracted}


pipeline_new_extract = _ch03_compiler_pipeline.replace(
    "pipeline_egraph_extraction", pipeline_egraph_extraction
)

# The compiler_pipeline will have a `codegen_extension` for defining LLVM
# code-generation for the new operations.


def extended_backend(
    extracted, codegen_extension, pipeline_report=Report.Sink()
) -> BackendOutput:
    with pipeline_report.nest("Backend", default_expanded=True) as report:
        llmod = backend(extracted, codegen_extension=codegen_extension)
        report.append("LLVM", llmod)
        jt = jit_compile(llmod, extracted)
        return {"jit_func": jt, "llmod": llmod}


# extend the pipeline with the new backend
pipeline_backend = pipeline_new_extract.replace(
    "pipeline_backend", extended_backend
)
compiler_pipeline = pipeline_backend

# visualize the pipeline
if __name__ == "__main__":
    display(compiler_pipeline.visualize())

# ## A Simple Type Inference Example

# First, we will start with a simple binary add operation.


def add_x_y(x, y):
    return x + y


# We will start with the same ruleset as in chapter 3.

basic_ruleset = rvsdg_eqsat.ruleset_rvsdg_basic | ruleset_const_propagate


# We will test our base compiler (ch 3 compiler behavior) on our function to set
# the baseline. At this stage, no type inference is happening.

if __name__ == "__main__":
    # start with previous compiler pipeline
    report = Report("Compiler Pipeline", default_expanded=True)
    jt = compiler_pipeline(
        fn=add_x_y,
        rule_schedule=basic_ruleset.saturate(),
        converter_class=EGraphToRVSDG,
        codegen_extension=None,
        cost_model=None,
        pipeline_report=report,
    ).jit_func
    report.display()
    run_test(add_x_y, jt, (123, 321), verbose=True)


# ### Adding type inference
#
# A new EGraph expression class (`Expr`) is added to represent type:


class Type(Expr):
    def __init__(self, name: StringLike): ...


# Then, we add a EGraph function to determine the type-of a `Term`:


@function
def TypeOf(x: Term) -> Type: ...


# Next, we define functions for the new operations:
#
# - `Nb_Unbox_Int64` unboxes a PyObject into a Int64.
# - `Nb_Box_Int64` boxes a Int64 into a PyObject.
# - `Nb_Unboxed_Add_Int64` performs a Int64 addition on unboxed operands.


@function
def Nb_Unbox_Int64(val: Term) -> Term: ...
@function
def Nb_Box_Int64(val: Term) -> Term: ...
@function
def Nb_Unboxed_Add_Int64(lhs: Term, rhs: Term) -> Term: ...


# Now, we define the first type-inference rule:
#
# If a `Py_AddIO()` (a Python binary add operation) is applied to operands
# that are known Int64, convert it into the unboxed add. The output type will
# be Int64. The IO state into the `Py_AddIO()` will be unchanged.

# +
TypeInt64 = Type("Int64")


@ruleset
def ruleset_type_infer_add(io: Term, x: Term, y: Term, add: Term):
    yield rule(
        add == Py_AddIO(io, x, y),
        TypeOf(x) == TypeInt64,
        TypeOf(y) == TypeInt64,
    ).then(
        # convert to a typed operation
        union(add.getPort(1)).with_(
            Nb_Box_Int64(
                Nb_Unboxed_Add_Int64(Nb_Unbox_Int64(x), Nb_Unbox_Int64(y))
            )
        ),
        # shortcut io
        union(add.getPort(0)).with_(io),
        # output type
        union(TypeOf(add.getPort(1))).with_(TypeInt64),
    )


# -

# The following rule defines some fact about the function being compiled.
# It declares that the two arguments are Int64.


@ruleset
def facts_argument_types(
    outports: PortList,
    func_uid: String,
    fname: String,
    region: Region,
    arg_x: Term,
    arg_y: Term,
):
    yield rule(
        GraphRoot(
            Term.Func(
                body=Term.RegionEnd(region=region, ports=outports),
                uid=func_uid,
                fname=fname,
            )
        ),
        arg_x == region.get(1),
        arg_y == region.get(2),
    ).then(
        union(TypeOf(arg_x)).with_(TypeInt64),
        union(TypeOf(arg_y)).with_(TypeInt64),
    )


# ### Defining conversion into RVSDG

# We will expand the RVSDG grammar with the typed operations.
#
# Each of the new typed operations will require a corresponding grammar rule.

# +
SExpr = rvsdg.grammar.SExpr


class NbOp_Base(grammar.Rule):
    pass


class NbOp_Unboxed_Add_Int64(NbOp_Base):
    lhs: SExpr
    rhs: SExpr


class NbOp_Unbox_Int64(NbOp_Base):
    val: SExpr


class NbOp_Box_Int64(NbOp_Base):
    val: SExpr


# -

# The new grammar for our IR is a combination of the new typed-operation grammar
# and the base RVSDG grammar.


class Grammar(grammar.Grammar):
    start = rvsdg.Grammar.start | NbOp_Base


# Now, we define a EGraph-to-RVSDG conversion class that is expanded to handle
# the new grammar.


class ExtendEGraphToRVSDG(EGraphToRVSDG):
    grammar = Grammar

    def handle_Term(self, op: str, children: dict | list, grm: Grammar):
        match op, children:
            case "Nb_Unboxed_Add_Int64", {"lhs": lhs, "rhs": rhs}:
                return grm.write(NbOp_Unboxed_Add_Int64(lhs=lhs, rhs=rhs))
            case "Nb_Unbox_Int64", {"val": val}:
                return grm.write(NbOp_Unbox_Int64(val=val))
            case "Nb_Box_Int64", {"val": val}:
                return grm.write(NbOp_Box_Int64(val=val))
            case _:
                # Use parent's implementation for other terms.
                return super().handle_Term(op, children, grm)

    def handle_Type(
        self, key: str, op: str, children: dict | list, grm: Grammar
    ):

        match op, children:
            case "Type", {"name": str(typename)}:
                if typename == "Int64":
                    return grm.write(
                        rg.Generic(name="Type", children=tuple([typename]))
                    )

        raise NotImplementedError("handle_Type", op, children)


# The LLVM code-generation also needs an extension:


def codegen_extension(expr, args, builder, pyapi):
    match expr._head, args:
        case "NbOp_Unboxed_Add_Int64", (lhs, rhs):
            return SSAValue(builder.add(lhs.value, rhs.value))
        case "NbOp_Unbox_Int64", (val,):
            return SSAValue(pyapi.long_as_longlong(val.value))
        case "NbOp_Box_Int64", (val,):
            return SSAValue(pyapi.long_from_longlong(val.value))
    return NotImplemented


# A new cost model to prioritize the typed operations:


class MyCostModel(CostModel):
    def get_cost_function(self, nodename, op, ty, cost, children):
        self_cost = None
        match op:
            case "Nb_Unboxed_Add_Int64":
                self_cost = 0.1

            case "Nb_Unbox_Int64":
                self_cost = 0.1

            case "Nb_Box_Int64":
                self_cost = 0.1

        if self_cost is not None:
            return self.get_simple(self_cost)

        # Fallthrough to parent's cost function
        return super().get_cost_function(nodename, op, ty, cost, children)


# The new ruleset with the type inference logic and facts about the compiled
# function:

typeinfer_ruleset = (
    basic_ruleset | ruleset_type_infer_add | facts_argument_types
)

# We are now ready to run the compiler:

if __name__ == "__main__":
    report = Report("Compiler Pipeline", default_expanded=True)
    jt = compiler_pipeline(
        fn=add_x_y,
        rule_schedule=typeinfer_ruleset.saturate(),
        converter_class=ExtendEGraphToRVSDG,
        codegen_extension=codegen_extension,
        cost_model=MyCostModel(),
        pipeline_report=report,
    ).jit_func
    report.display()
    run_test(add_x_y, jt, (123, 321), verbose=True)


# Observations:
#
# - In the egraph, observe how the new operations are represented.
# - In the RVSDG, notice the lack of `Py_AddIO()`
# - In the LLVM, notice the addition is now done in native `i64`.

# ## Optimize boxing logic


# A key benefit of EGraph is that there is no need to specify ordering to
# "compiler-passes". To demonstrate this, we will insert optimization rules
# on the boxing and unboxing operation. `unbox(box(x))` is equivalent
# to an no-op. We can remove redundant boxing and unboxing.

# We will need more than one addition to showcase the optimization:


def chained_additions(x, y):
    return x + y + y


if __name__ == "__main__":
    report = Report("Compiler Pipeline", default_expanded=True)
    jt = compiler_pipeline(
        fn=chained_additions,
        rule_schedule=typeinfer_ruleset.saturate(),
        converter_class=ExtendEGraphToRVSDG,
        codegen_extension=codegen_extension,
        cost_model=MyCostModel(),
        pipeline_report=report,
    ).jit_func
    report.display()
    run_test(chained_additions, jt, (123, 321), verbose=True)

# Observations:
#
# ```
#   $4 = NbOp_Box_Int64 $3
#   $5 = NbOp_Unbox_Int64 $4
# ```
#
# The box and unbox chain is redundant (i.e. `$3 = $5`).

# ### Box/Unbox optimization rules


# The needed optimization rule is very simple. Any chained box-unbox; or unbox-box
# are redundant.
#
# (We use `subsume=True` to delete the original EGraph node (enode) to shrink
# the graph early.)


@ruleset
def ruleset_optimize_boxing(x: Term):
    yield rewrite(Nb_Box_Int64(Nb_Unbox_Int64(x)), subsume=True).to(x)
    yield rewrite(Nb_Unbox_Int64(Nb_Box_Int64(x)), subsume=True).to(x)


optimized_ruleset = typeinfer_ruleset | ruleset_optimize_boxing

if __name__ == "__main__":
    report = Report("Compiler Pipeline", default_expanded=True)
    jt = compiler_pipeline(
        fn=chained_additions,
        rule_schedule=optimized_ruleset.saturate(),
        converter_class=ExtendEGraphToRVSDG,
        codegen_extension=codegen_extension,
        cost_model=MyCostModel(),
        pipeline_report=report,
    ).jit_func
    report.display()
    run_test(chained_additions, jt, (123, 321), verbose=True)

# Observations:
#
# ```
#   $1 = NbOp_Unbox_Int64 $0[1]
#   $2 = NbOp_Unbox_Int64 $0[2]
#   $3 = NbOp_Unboxed_Add_Int64 $1 $2
#   $4 = NbOp_Unboxed_Add_Int64 $3 $2
#   $5 = NbOp_Box_Int64 $4
# ```
#
# There is no redundant box-unbox between the two unboxed add anymore.
