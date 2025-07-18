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

# # Demo 3.0: Matrix Multiplication Order: Performance is Not Associative
#
# This demo notebook shows how the order of matrix multiplications can
# dramatically affect performance, even though the mathematical result is the
# same. We use cost-based extraction and e-graphs to find the optimal
# parenthesization for a chain of matrix multiplications, inspired by
# [this blog post](https://www.johndcook.com/blog/2017/12/12/efficiency-is-not-associative-for-matrix-multiplication/).
#
# The notebook demonstrates:
# - How to use e-graphs and cost models to optimize the order
# - How to extract and visualize the optimized computation

# ## Imports and Setup

# +
from typing import Any, TypedDict

import numpy as np
from egglog import (
    EGraph,
    Ruleset,
    function,
    i64,
    i64Like,
    rewrite,
    rule,
    ruleset,
    set_,
    subsume,
    union,
)
from sealir import ase
from sealir.eqsat.py_eqsat import (
    Py_MatMultIO,
)
from sealir.eqsat.rvsdg_convert import egraph_conversion
from sealir.eqsat.rvsdg_eqsat import (
    GraphRoot,
    Term,
)
from sealir.eqsat.rvsdg_extract import (
    egraph_extraction,
)
from sealir.rvsdg import format_rvsdg
from sealir.rvsdg import grammar as rg

from ch01_basic_compiler import pipeline_frontend
from ch04_0_typeinfer_prelude import (
    pipeline_new_extract as _ch04_0_pipeline_new_extract,
)
from ch04_1_typeinfer_ifelse import TypeFloat64
from ch05_typeinfer_array import (
    ArrayDesc,
    Dim,
    ExtendEGraphToRVSDG,
    Grammar,
)
from ch05_typeinfer_array import MyCostModel as _ch05_CostModel
from ch05_typeinfer_array import (
    NbOp_Base,
    TypeVar,
    array_desc_rules,
    base_ruleset,
    setup_argtypes,
)
from utils import Pipeline, Report, display, visualize_expr_tree

# -


SExpr = ase.SExpr

# ## NumPy Example: Two Matrix Multiplications
#
# We start by showing that the result of matrix multiplication is associative,
# but the performance can differ depending on the order of operations.

M = 200
N = 100
K = 200

# (M x N) (N x K) (K x N)

if __name__ == "__main__":
    arr0 = np.random.random((M, N))
    arr1 = np.random.random((N, K))
    arr2 = np.random.random((K, N))
    res0 = (arr0 @ arr1) @ arr2
    res1 = arr0 @ (arr1 @ arr2)
    print(res0.shape)
    print(res1.shape)
    np.testing.assert_allclose(res0, res1)
    # unoptimized
    # %timeit arr0 @ arr1 @ arr2
    # optimized
    # %timeit arr0 @ (arr1 @ arr2)

# ## NumPy Example: Five Matrix Multiplications
#
# Here we define two different ways to multiply five matrices and compare their
# results and performance.

f0 = lambda arr0, arr1, arr2, arr3: arr0 @ arr1 @ arr2 @ arr1 @ arr2 @ arr3


def f1(arr0, arr1, arr2, arr3):
    x = arr1 @ arr2
    return arr0 @ (x @ (x @ arr3))


if __name__ == "__main__":
    arr0 = np.random.random((M, N))
    arr1 = np.random.random((N, K))
    arr2 = np.random.random((K, N))
    arr3 = np.random.random((N, 1))
    res0 = f0(arr0, arr1, arr2, arr3)
    res1 = f1(arr0, arr1, arr2, arr3)
    np.testing.assert_allclose(res0, res1)

    report = Report(
        "Expression tree of different versions",
        default_expanded=True,
    )
    report.append("original version", visualize_expr_tree(f0))
    report.append("hand-optimized version", visualize_expr_tree(f1))
    report.display()

    # unoptimized
    # %timeit f0(arr0, arr1, arr2, arr3)
    # optimized
    # %timeit f1(arr0, arr1, arr2, arr3)

# ## E-Graph Construction for Matrix Multiplication
#
# We now build an e-graph to represent all possible ways to parenthesize a chain
# of matrix multiplications, and encode the associativity rule.

array_0_desc, array_0_infos = array_desc_rules(
    "array_0", shape=(M, N), dtype=TypeFloat64, layout="c"
)
array_1_desc, array_1_infos = array_desc_rules(
    "array_1", shape=(N, K), dtype=TypeFloat64, layout="c"
)
array_2_desc, array_2_infos = array_desc_rules(
    "array_2", shape=(K, N), dtype=TypeFloat64, layout="c"
)
array_3_desc, array_3_infos = array_desc_rules(
    "array_3", shape=(N, 1), dtype=TypeFloat64, layout="c"
)
ruleset_array_facts = ruleset(
    *array_0_infos, *array_1_infos, *array_2_infos, *array_3_infos
)


@function
def MatMul(lhs: Term, rhs: Term) -> Term: ...


@function
def MatMul_KnownShape(
    lhs: Term, rhs: Term, m: i64Like, n: i64Like, k: i64Like
) -> Term: ...


@function
def ArrayDescOp(term: Term) -> ArrayDesc: ...


@ruleset
def ruleset_optimize_matmul(
    ary0: Term,
    ary1: Term,
    ary2: Term,
    ad0: ArrayDesc,
    ad1: ArrayDesc,
    shapeM: i64,
    shapeN: i64,
    shapeK: i64,
):
    # associative
    yield rewrite(MatMul(MatMul(ary0, ary1), ary2)).to(
        MatMul(ary0, MatMul(ary1, ary2))
    )

    yield rule(
        # array desc propagation
        ary2 == MatMul(ary0, ary1),
        ad0.toType() == TypeVar(ary0).getType(),
        ad1.toType() == TypeVar(ary1).getType(),
        ad0.ndim == i64(2),
        ad1.ndim == i64(2),
        m := ad0.dim(0),
        q := ad1.dim(1),
        ad0.dim(1) == ad1.dim(0),
        ad0.dtype == ad1.dtype,
    ).then(
        # set output dims
        ad2 := ArrayDescOp(ary2),
        set_(TypeVar(ary2).getType()).to(ad2.toType()),
        set_(ad2.ndim).to(2),
        set_(ad2.dim(0)).to(m),
        set_(ad2.dim(1)).to(q),
        set_(ad2.dtype).to(ad0.dtype),
    )

    yield rule(
        ary2 == (stmt := MatMul(ary0, ary1)),
        ad0.toType() == TypeVar(ary0).getType(),
        ad1.toType() == TypeVar(ary1).getType(),
        ad0.ndim == i64(2),
        ad1.ndim == i64(2),
        Dim.fixed(shapeM) == ad0.dim(0),
        Dim.fixed(shapeN) == ad0.dim(1),
        Dim.fixed(shapeN) == ad1.dim(0),
        Dim.fixed(shapeK) == ad1.dim(1),
        ad0.dtype == ad1.dtype,
    ).then(
        # Convert to MatMul_KnownShape
        union(ary2).with_(
            MatMul_KnownShape(ary0, ary1, shapeM, shapeN, shapeK)
        ),
    )


@ruleset
def ruleset_matmul_op(io: Term, lhs: Term, rhs: Term, res: Term):
    yield rule(res == Py_MatMultIO(io, lhs, rhs)).then(
        union(res.getPort(1)).with_(MatMul(lhs, rhs)),
        union(res.getPort(0)).with_(io),
    )


# ## Compile E-Graph to Extractable Representation
#
# This section defines the pipeline to convert the e-graph into a form suitable
# for extraction and optimization.


def code_input(ary0, ary1, ary2, ary3):
    return ary0 @ ary1 @ ary2 @ ary1 @ ary2 @ ary3


class EGraphConversionOutput(TypedDict):
    egraph: EGraph
    saturation_steps: list[str] | None


def egraph_saturation(
    egraph: EGraph,
    egraph_root: GraphRoot,
    extra_ruleset: Ruleset,
    argtypes: tuple,
    arg_facts: Ruleset,
    pipeline_report=Report.Sink(),
    pipeline_debug=False,
    animate=False,
) -> EGraphConversionOutput:
    with pipeline_report.nest("Egraph saturation") as report:
        all_rules = (
            setup_argtypes(*argtypes)
            | arg_facts
            | extra_ruleset
            | base_ruleset
        )
        saturation_steps: list[str] | None
        if animate:
            # Manual saturation loop to extract serialized graph
            saturation_steps = []
            while egraph.run(all_rules).updated:
                jsdata = egraph._serialize(
                    n_inline_leaves=0, split_primitive_outputs=False
                ).to_json()
                saturation_steps.append(jsdata)
        else:
            runreport = egraph.run(all_rules.saturate())
            saturation_steps = None
        if pipeline_debug:
            report.append("[debug] Run report", runreport)
            report.append("[debug] Saturated egraph", egraph)
        return dict(egraph=egraph, saturation_steps=saturation_steps)


pipeline_middle_end = _ch04_0_pipeline_new_extract.trunc(
    "pipeline_backend"
).replace("egraph_saturation", egraph_saturation)
pipeline_to_egraph = pipeline_middle_end.trunc("pipeline_egraph_extraction")

extra_ruleset = ruleset_matmul_op | ruleset_optimize_matmul


if __name__ == "__main__":
    display(pipeline_to_egraph.visualize())
    report = Report("Pipeline execution report", enable_nested_metadata=True)
    argtypes = (
        array_0_desc.toType(),
        array_1_desc.toType(),
        array_2_desc.toType(),
        array_3_desc.toType(),
    )
    pipeline_to_egraph(
        fn=code_input,
        argtypes=argtypes,
        arg_facts=ruleset_array_facts,
        extra_ruleset=extra_ruleset,
        pipeline_debug=True,
        pipeline_report=report,
    )
    report.display()


# ## Cost Model for Matrix Multiplication
#
# We define a cost model that estimates the computational cost of each matrix
# multiplication node, based on the classic formula `2 * m * n * k`.


class MyCostModel(_ch05_CostModel):
    def get_cost_function(self, nodename, op, ty, cost, children):
        match op, tuple(children):
            case "MatMul_KnownShape", (_lhs, _rhs, m, n, k):
                return self.get_equation(
                    lambda lhs, rhs, *_, m, n, k: 2 * m * n * k,
                    constants=dict(m=m, n=n, k=k),
                )
            case "MatMul", _:
                return self.get_simple(float("inf"))
        return super().get_cost_function(nodename, op, ty, cost, children)


class NbOp_MatMulKnownShape(NbOp_Base):
    lhs: SExpr
    rhs: SExpr

    m: int
    n: int
    k: int


# Custom converter to handle the MatMul_KnownShape node when converting from
# EGraph to RVSDG. This ensures that matrix multiplication nodes with known
# shapes are represented as NbOp_MatMulKnownShape in the RVSDG, preserving
# shape information for cost modeling and further analysis.


class MyEGraphToRVSDG(ExtendEGraphToRVSDG):

    def handle_Term(self, op: str, children: dict | list, grm: Grammar):
        match op, children:
            case "MatMul_KnownShape", {
                "lhs": lhs,
                "rhs": rhs,
                "m": m,
                "n": n,
                "k": k,
            }:
                return grm.write(
                    NbOp_MatMulKnownShape(lhs=lhs, rhs=rhs, m=m, n=n, k=k)
                )
        return super().handle_Term(op, children, grm)


if __name__ == "__main__":
    display(pipeline_middle_end.visualize())
    report = Report("Pipeline execution report", enable_nested_metadata=True)
    pipeline_middle_end(
        fn=code_input,
        argtypes=argtypes,
        arg_facts=ruleset_array_facts,
        extra_ruleset=extra_ruleset,
        cost_model=MyCostModel(),
        converter_class=MyEGraphToRVSDG,
        pipeline_report=report,
        pipeline_debug=True,
    )
    report.display()

# ## Extract Python Source from Optimized Expression
#
# This section shows how to turn the optimized computation into executable
# Python code, and compare it to the original and hand-optimized versions.


class SourceMaker(ase.TreeVisitor):
    def __init__(self):
        super().__init__()
        self.memo = {}
        self.out = []

    def visit(self, expr: SExpr):
        memo = self.memo
        buf = self.out
        match expr:
            case rg.Attrs():
                return None
            case rg.RegionBegin(attrs=attrs, inports=inports):
                memo[expr] = [f"arg{i}" for i, v in enumerate(inports)]
                self.nargs = len(inports) - 1
            case rg.Unpack(val=val, idx=idx):
                val = memo[val]
                memo[expr] = val[idx]
            case NbOp_MatMulKnownShape(lhs=lhs, rhs=rhs, m=m, n=n, k=k):
                lhs, rhs = memo[lhs], memo[rhs]
                memo[expr] = ref = f"v{len(memo)}"
                buf.append(f"{ref} = ({lhs} @ {rhs})")
            case _:
                raise ValueError(expr)

    def get_source(self):
        source = "; ".join(self.out)
        args = [f"arg{i}" for i in range(1, self.nargs + 1)]
        last = f"v{len(self.memo) - 1}"
        template = f"""
def func({', '.join(args)}):
    {source}
    return {last}
"""
        return template

    def get_function(self, global_ns={}):
        source_code = self.get_source()
        exec(source_code, global_ns)
        return global_ns["func"]


def extract_py_source(extracted, sourcemaker_class=SourceMaker, global_ns={}):
    visitor = sourcemaker_class()
    outports = extracted.body.ports
    [out_equ] = [p.value for p in outports if p.name == "!ret"]
    ase.apply_bottomup(out_equ, visitor)
    return visitor.get_function(global_ns=global_ns), visitor.get_source()


class ExtractedOutput(TypedDict):
    extracted: Any
    source: str


@pipeline_middle_end.extend
def compiler(
    extracted,
    sourcemaker_class,
    global_ns=None,
    pipeline_report=Report.Sink(),
    pipeline_debug=False,
) -> ExtractedOutput:
    extracted, source = extract_py_source(
        extracted,
        sourcemaker_class=sourcemaker_class,
        global_ns=global_ns or {},
    )
    pipeline_report.append("Extracted source", source)
    return dict(extracted=extracted, source=source)


if __name__ == "__main__":
    display(compiler.visualize())

if __name__ == "__main__":
    report = Report("Pipeline execution report", enable_nested_metadata=True)
    cres = compiler(
        fn=code_input,
        argtypes=argtypes,
        extra_ruleset=extra_ruleset,
        arg_facts=ruleset_array_facts,
        cost_model=MyCostModel(),
        converter_class=MyEGraphToRVSDG,
        sourcemaker_class=SourceMaker,
        pipeline_report=report,
        pipeline_debug=True,
    )
    report.display()
    extracted = cres.extracted
    res1 = extracted(arr0, arr1, arr2, arr3)
    res0 = f1(arr0, arr1, arr2, arr3)
    np.testing.assert_allclose(res0, res1)

    report = Report(
        "Expression tree of different versions",
        default_expanded=True,
    )
    report.append("original version", visualize_expr_tree(f0))
    report.append("hand-optimized version", visualize_expr_tree(f1))
    report.append(
        "superoptimized version", visualize_expr_tree(cres.extracted)
    )
    report.display()

    # original
    # %timeit f0(arr0, arr1, arr2, arr3)

    # manual optimized
    # %timeit f1(arr0, arr1, arr2, arr3)

    # cost-based extraction
    # %timeit extracted(arr0, arr1, arr2, arr3)
