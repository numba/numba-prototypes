# Traditional compiler design depends on a series of compiler passes. However,
# this approach suffers from the phase-ordering problem—where the specific order
# of optimization steps can influence the resulting code, often leading to
# missed optimizations or less efficient output due to the inflexible sequence
# of transformations. The use of EGraphs and Equality Saturation overcomes this
# issue. By representing a program as a graph of equivalent expressions and
# evaluating multiple optimization possibilities at once, these techniques
# enable the compiler to identify the most effective code version without being
# limited by a predetermined order of passes.
#
# We will define the middle end—--the stage of a compiler that
# optimizes and transforms an intermediate representation (IR) of a program,
# connecting the front end (which con,verts source code into IR) to the back end
# (which produces target machine code). The egglog library powers our use of
# EGraphs. Before exploring how to write rules for EGraphs, we'll first
# establish a conversion from RVSDG-IR to EGraph. RVSDG-IR must be encoded into
# the EGraph, enabling us to apply rewrite rulesets for program optimization or
# analysis. Afterward, an extraction step selects the most efficient variant
# from the EGraph.

# ## Imports and Setup

from typing import Any, TypedDict

from egglog import EGraph
from sealir import rvsdg
from sealir.eqsat.rvsdg_convert import egraph_conversion
from sealir.eqsat.rvsdg_eqsat import GraphRoot
from sealir.eqsat.rvsdg_extract import egraph_extraction

from compiler import (
    backend,
)
from compiler import compiler_pipeline as pipeline_jit_compile
from compiler import (
    jit_compile,
    pipeline_backend,
    pipeline_frontend,
)
from utils import IN_NOTEBOOK, Report, display

# ## Simple EGraph Roundtripping
#
# Our initial middle-end is a simple roundtripping from RVSDG-IR to EGraph and
# back to RVSDG-IR. `SealIR` provides `egraph_conversion()` for RVSDG-IR to
# EGraph, and `egraph_extraction()` for EGraph to RVSDG-IR.

# ### Convert RVSDG to EGraph
#
# The following code shows the RVSDG-IR and egraph for the `max_if_else()`
# function. The two are almost a direct mapping.


class EGraphOutput(TypedDict):
    egraph: EGraph
    egraph_root: GraphRoot


@pipeline_frontend.extend
def pipeline_egraph_conversion(
    rvsdg_expr, pipeline_report=Report.Sink()
) -> EGraphOutput:
    with pipeline_report.nest("EGraph Conversion", default_expanded=True) as report:
        memo = egraph_conversion(rvsdg_expr)
        egraph = EGraph()
        root = GraphRoot(memo[rvsdg_expr])
        egraph.let("root", root)
        report.append("EGraph", egraph)
        return {"egraph": egraph, "egraph_root": root}


# Here, we will use the default cost model, which is based on the node count.


class EGraphExtractionOutput(TypedDict):
    cost: float
    extracted: Any


@pipeline_egraph_conversion.extend
def pipeline_egraph_extraction(
    egraph, rvsdg_expr, pipeline_report=Report.Sink()
) -> EGraphExtractionOutput:
    with pipeline_report.nest("EGraph Extraction", default_expanded=True) as report:
        cost, extracted = egraph_extraction(egraph, rvsdg_expr)
        report.append("Cost", cost)
        report.append("Extracted", rvsdg.format_rvsdg(extracted))
        return {"cost": cost, "extracted": extracted}


# ## Extended Compiler Pipeline
#
# Redefine the compiler pipeline to include the middle-end with EGraph
# optimization capabilities.


def egraph_action(
    egraph: EGraph,
    egraph_root: GraphRoot,
    pipeline_report=Report.Sink(),
) -> EGraphOutput:
    # For now, the middle end is just an identity function that exercise
    # the encoding into and out of egraph.
    with pipeline_report.nest("EGraph Action") as report:
        report.append("EGraph", egraph)
    return {"egraph": egraph, "egraph_root": egraph_root}


pipeline_middle_end = pipeline_egraph_extraction.insert(-1, egraph_action)


class BackendOutput(TypedDict):
    jit_func: Any
    llmod: Any


@pipeline_middle_end.extend
def pipeline_backend(extracted, pipeline_report=Report.Sink()) -> BackendOutput:
    with pipeline_report.nest("Backend", default_expanded=True) as report:
        llmod = backend(extracted)
        report.append("LLVM", llmod)
        jt = jit_compile(llmod, extracted)
        return {"jit_func": jt, "llmod": llmod}


compiler_pipeline = pipeline_backend
