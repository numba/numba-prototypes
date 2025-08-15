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

# # Chapter 1: Basic Compiler
#
# ## AST frontend and LLVM backend
#
#
# This chapter introduces the fundamental components of our compiler: the
# [Abstract Syntax Tree
# (AST)](https://en.wikipedia.org/wiki/Abstract_syntax_tree) frontend and
# [LLVM](https://en.wikipedia.org/wiki/LLVM) backend. We show how to parse
# Python functions into a [Regionalized Value-State Dependence Graph
# intermediate representation (RVSDG-IR)](https://arxiv.org/abs/1912.05036) and
# then compile them to executable code using LLVM.
#
# The chapter covers:
# - How to implement a frontend that converts Python AST to RVSDG-IR
# - How to implement a backend that generates LLVM IR
# - How to [just-in_time compile
#   (JIT)](https://en.wikipedia.org/wiki/Just-in-time_compilation) and execute
#   the generated code

# ## Imports and Setup
#
# Import all necessary modules for the basic compiler implementation.

from __future__ import annotations

import builtins
from collections import ChainMap

from llvmlite import binding as llvm
from llvmlite import ir
from sealir import ase, rvsdg
from sealir.llvm_pyapi_backend import (
    CodegenCtx,
    CodegenState,
    JitCallable,
    PythonAPI,
    _codegen_loop,
)
from sealir.rvsdg import grammar as rg
from typing_extensions import TypedDict

from utils import Pipeline, Report

# ## Frontend Implementation
#
# The frontend accepts a Python function object, reads its source code, and
# parses its AST. It then
# transforms the AST into a RVSDG, a representation that simplifies
# further intermediate representation (IR) processing. The RVSDG uses a
# data-flow centric encoding in which control-flow constructs are mapped into
# regions. These regions function much like ordinary operations, with clearly
# defined sets of input and output ports. Additionally, state is explicitly
# encoded as I/O, so that every operation maintains a pure operational
# appearance.
#
# The `pipeline_frontend()` transforms Python functions into RVSDG intermediate
# representation for compilation. The pipeline extracts source code, parses it
# into an AST, and converts the AST into a RVSDG—a data-flow representation
# that encapsulates control structures (loops, conditionals) as regions with
# explicit input/output ports. This design eliminates control-flow analysis
# complexity, makes data dependencies explicit, and enables aggressive
# optimizations and parallelization. The pipeline outputs the RVSDG expression
# and debugging information for traceability to source code.

class FrontendOutput(TypedDict):
    """
    Output structure from the frontend pipeline containing the transformed
    representation and debugging information.

    Attributes:
        rvsdg_expr:
        dbginfo:
    """
    rvsdg_expr: object
    """The RVSDG expression representing the transformed function in data-flow
    form
    """
    dbginfo: object
    """Debug information object containing source mappings and metadata
    for tracing transformations back to original source code
    """


@Pipeline
def pipeline_frontend(fn, pipeline_report=Report.Sink()) -> FrontendOutput:
    """
    Transform a Python function into an RVSDG representation for compilation.

    Args:
        fn: Python function object to be compiled
        pipeline_report: Reporting sink for debugging and visualization
                         (optional)

    Returns:
        FrontendOutput: Dictionary containing the RVSDG expression and debug
        info

    Example:
        >>> def my_func(x, y):
        ...     return x + y * 2
        >>> result = pipeline_frontend(my_func)
        >>> # result['rvsdg_expr'] contains the data-flow representation
        >>> # result['dbginfo'] contains source mapping information
    """
    with pipeline_report.nest("Frontend", default_expanded=True) as report:
        rvsdg_expr, dbginfo = rvsdg.restructure_source(fn)
        report.append("Debug Info on RVSDG", dbginfo.show_sources())
        report.append("RVSDG", rvsdg.format_rvsdg(rvsdg_expr))
    return {"rvsdg_expr": rvsdg_expr, "dbginfo": dbginfo}


# ### Simple Frontend Example
#
# Here's a simple function that adds two numbers together and prints the
# RVSDG-IR to illustrate the frontend:

if __name__ == "__main__":

    def exercise_frontend_simple(x, y):
        return x + y

    cres = pipeline_frontend(fn=exercise_frontend_simple)
    # Results are returned as a SimpleNamespace
    print(rvsdg.format_rvsdg(cres.rvsdg_expr))


# Alternatively, use Report to display the results:

if __name__ == "__main__":
    report = Report("Frontend", default_expanded=True)
    cres = pipeline_frontend(
        fn=exercise_frontend_simple, pipeline_report=report
    )
    report.display()

# The function's RVSDG-IR includes a region:
#
# ```
# $0 = Region[236] <- !io x y
# {
#     ...
# } [302] -> !io=$1[0] !ret=$1[1] x=$0[1] y=$0[2]
# ```
#
# The region has three input ports: `!io`, `x`, and `y`.
#
# Its output ports are `!io`, `!ret`, `x`, and `y`.
#
# Here, `!io` represents the state. Because Python functions are imperative,
# each function carries an implicit state that must be updated for any
# side-effect.
#
# The `!ret` port holds the function's return value.
#
# The `x` and `y` output ports convey the region's internal value state, but the
# function node ignores these values.
#
# The single operation in the function is:
#
# ```
#  $1 = PyBinOp + $0[0] $0[1], $0[2]
# ```
#
# This is a Python binary operation. It uses input operands from `$0`, which is
# the header of the function's region. In this notation, `$0[n]` refers to a
# specific port: `$0[0]` corresponds to `!io`, `$0[1]` to `x`, and `$0[2]` to
# `y`.

# ### Complex Frontend Example
#
# Below is a more intricate example that requires restructuring of the control
# flow. This is due to the use of the `break` statement to exit a for-loop.
#
# RVSDG enforces structured control flow. Only three types of control-flow
# regions are permitted:
#
# - Linear region, as `Region {...}`;
# - If-Else region, as `If {...} Else {...} EndIf`;
# - Loop region, as `Loop {...} EndLoop`.

if __name__ == "__main__":

    def exercise_frontend_loop_if_break(x, y):
        """
        Iterates from 0 to x-1, accumulates the sum in c, and breaks the loop
        if i > y. Returns the accumulated sum.
        """
        c = 0
        for i in range(x):
            if i > y:
                break
            c += i
        return c

    report = Report("Frontend", default_expanded=True)
    cres = pipeline_frontend(
        fn=exercise_frontend_loop_if_break, pipeline_report=report
    )
    report.display()

# This example has the same input and output regions as the simple example:
#
# ```
# $0 = Region[1350] <- !io x y
# {
#     ...
# } [3602] -> !io=$50[0] !ret=$50[6]
# ```
#
# But has many more Python binary operations within the function. Here are some
# observations from the RVSDG-IR above:
#
# * The for-loop is restructured into a `Loop` composed of `If-Else` regions.
# * The `Loop` region is tail-controlled; its first output port acts as the
#   loop condition. See `!_loopcond_0002`.
# * An extra `If-Else` follows the loop to adjust the value of `i`.
#
# The beauty of the structured control-flow is that everything can be
# encapsulated as a plain data-flow operation, including any region. Everything
# is just an operation with some input and output ports. This simplifies the
# rest of the compiler.

# ## Backend Implementation
#
# SealIR includes a lightweight LLVM backend that emits the Python C-API, which
# executes the RVSDG-IR. After creating our LLVM backend, we can JIT compile
# and test.
#
# The example below demonstrates how to use this backend:


def _determine_arity(root: ase.SExpr) -> int:
    """Helper function to get the arity of the function node"""
    match root:
        case rg.Func(args=rg.Args() as args):
            return len(args.arguments)
        case _:
            raise TypeError(root._head)


def backend(root, ns=builtins.__dict__, codegen_extension=None):
    """
    Emit LLVM using Python C-API.

    root: the RVSDG expression for the function
    ns: is the dictionary of global names. A JIT is assumed. Object pointer for
        each key is used.
    codegen_extension: Optional. If defined, it is the function to call when an
        unknown operation is encountered by the code generator. This argument
        is used in the later chapters.

    Warning:

    - This is for testing only.
    - Does NOT do proper memory management.
    - Does NOT do proper error handling.
    """

    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()

    mod = ir.Module()

    ll_byte = ir.IntType(8)
    ll_pyobject_ptr = ll_byte.as_pointer()
    # Make LLVM function
    arity = _determine_arity(root)

    assert arity >= 1
    actual_num_args = arity
    fnty = ir.FunctionType(
        ll_pyobject_ptr, [ll_pyobject_ptr] * actual_num_args
    )
    fn = ir.Function(mod, fnty, name="foo")

    # init entry block and builder
    builder = ir.IRBuilder(fn.append_basic_block())
    retval_slot = builder.alloca(ll_pyobject_ptr)
    builder.store(ll_pyobject_ptr(None), retval_slot)  # init retval to NULL

    bb_main = builder.append_basic_block()
    builder.branch(bb_main)
    builder.position_at_end(bb_main)

    ctx = CodegenCtx(
        llvm_module=mod,
        llvm_func=fn,
        builder=builder,
        pyapi=PythonAPI(builder),
        global_ns=ChainMap(ns, __builtins__),
        codegen_extension=codegen_extension,
    )

    # Emit the function body
    ase.traverse(root, _codegen_loop, CodegenState(context=ctx))
    return mod


# ### JIT Compilation
#
# The following function takes a LLVM module and JIT compiles it for execution:


def jit_compile(mod, rvsdg_expr):
    """JIT compile LLVM module into an executable function for this process."""
    llvm_ir = str(mod)

    # Create JIT
    lljit = llvm.create_lljit_compiler()
    rt = (
        llvm.JITLibraryBuilder()
        .add_ir(llvm_ir)
        .export_symbol("foo")
        .add_current_process()
        .link(lljit, "foo")
    )
    ptr = rt["foo"]
    arity = _determine_arity(rvsdg_expr)
    return JitCallable.from_pointer(rt, ptr, arity)


# ### Compiler Pipeline
#
# The following is a simple compiler pipeline:


class BackendOutput(TypedDict):
    llmod: object


@pipeline_frontend.extend
def pipeline_backend(
    rvsdg_expr, pipeline_report=Report.Sink()
) -> BackendOutput:
    with pipeline_report.nest("Backend", default_expanded=True) as report:
        llmod = backend(rvsdg_expr)
        report.append("LLVM", llmod)
    return {"llmod": llmod}


class JITOutput(TypedDict):
    jit_func: object


@pipeline_backend.extend
def compiler_pipeline(llmod, rvsdg_expr) -> JITOutput:
    jit_func = jit_compile(llmod, rvsdg_expr)
    return {"jit_func": jit_func}


# ### Testing Framework
#
# Define a testing framework to verify compiler correctness.


def run_test(fn, jt, args, *, verbose=False, equal=lambda x, y: x == y):
    res = jt(*args)
    got = fn(*args)

    if verbose:
        report = Report("Testing report", default_expanded=False)
        report.append("Args", args)
        report.append("JIT output", res)
        report.append("Expected output", got)
        report.display()

    assert equal(res, got), (res, got)
    return res


# ## Complete Example
#
# The following puts everything from this chapter together. Running the
# frontend to generate RVSDG-IR. Then, emitting LLVM using the backend.
# Finally, JIT'ing it into executable code and verifying it.
#
# Let's use a modified version of our complex example function:

if __name__ == "__main__":

    def sum_ints(n):
        """
        Iterates to n and returns the accumulated sum.
        """
        c = 0
        for i in range(n):
            c += i
        return c

    report = Report("Compiler Pipeline", default_expanded=True)
    jt = compiler_pipeline(fn=sum_ints, pipeline_report=report).jit_func
    report.display()
    run_test(sum_ints, jt, (12,), verbose=True)

# The JIT output matches the expected output, confirming correctness. The
# compilation pipeline generates structured reports containing frontend RVSDG-IR
# and backend LLVM-IR.
#
# Pipeline analysis:
#
# - Each compilation stage outputs structured reports with tabbed organization
#   for separation of concerns and debugging accessibility.
# - LLVM generates control-flow graphs using basic blocks and branch
#   instructions; RVSDG represents computation as structured dataflow regions
#   with explicit I/O ports.
# - Generated LLVM code implements all operations via Python C-API function
#   calls, maintaining compatibility with Python's object model and garbage
#   collector.
