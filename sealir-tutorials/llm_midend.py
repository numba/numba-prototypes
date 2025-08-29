import operator
from functools import reduce
from pprint import pprint
from types import FunctionType

from ch04_1_typeinfer_ifelse import Type, TypeFloat64, TypeVar
from ch05_typeinfer_array import ArrayDesc, Broadcast, Dim
from ch05_typeinfer_array import (
    ExtendEGraphToRVSDG as _ch05_ExtendEGraphToRVSDG,
)
from ch05_typeinfer_array import (
    Grammar,
    Int64,
    NbOp_Base,
    TypeInt64,
    array_desc_rules,
    base_ruleset,
)
from ch05_typeinfer_array import compiler_config as _compiler_config
from ch05_typeinfer_array import jit_compiler, setup_argtypes
from ch09_whole_program_compiler_driver import CallGraphVisitor
from egglog import (
    Bool,
    BoolLike,
    Expr,
    String,
    StringLike,
    Unit,
    Vec,
    function,
    i64,
    i64Like,
    rewrite,
    rule,
    ruleset,
    set_,
    subsume,
    union,
    var,
)
from sealir.eqsat.py_eqsat import (
    Py_AttrIO,
    Py_Call,
    Py_CallKwargs,
    Py_DivIO,
    Py_LoadGlobal,
    Py_NegIO,
    Py_SubIO,
)
from sealir.eqsat.py_eqsat import make_rules as py_eqsat_rules
from sealir.eqsat.rvsdg_eqsat import Term, TermDict, TermList, termlist
from sealir.rvsdg.grammar import SExpr
from utils import Report

source_filename = "llm.py"
with open(source_filename, "r") as fin:
    source_code = fin.read()

cgv = CallGraphVisitor(source_code, source_filename)
cgv.visit_all()

print("########## Symbol Table ##########")
pprint(cgv.functions)
print("########## ------------ ##########")
print("########## Global Calls ##########")
pprint(cgv.global_calls)
print("########## ------------ ##########")
print("########## Call Graph   ##########")
pprint(cgv.get_call_graph())
print("########## ------------ ##########")
print("########## Module Imported   ##########")
pprint(cgv.imported)
print("########## ------------ ##########")


#######################################
class Module(Expr):
    def __init__(self, name: StringLike): ...

    def toType(self) -> Type: ...


@function
def ModuleGetAttr(mod: Module, attrname: StringLike) -> Term: ...


# Install module rules


def make_module_rule(gv_name: str, mod_name):
    def ruleset_module_getattr(
        mod: Module,
        attrname: String,
        io: Term,
        name: String,
        op: Term,
        args: Vec[Term],
        obj: Term,
    ):
        # When loading a global variable that matches the module name,
        # set its type to be the corresponding module type
        yield rule(
            op == Py_LoadGlobal(io, name),
            name == String(gv_name),
        ).then(set_(TypeVar(op).getType()).to(Module(mod_name).toType()))

        # Module getattr
        yield rule(
            op == Py_AttrIO(io, obj, attrname),
            TypeVar(obj).getType() == Module(name).toType(),
        ).then(
            # Shortcut io
            union(op.getPort(0)).with_(io),
            # Setup getattr
            union(op.getPort(1)).with_(ModuleGetAttr(Module(name), attrname)),
        )

    return ruleset_module_getattr


def make_module_fact_ruleset(module_mapping: dict[str, str]):
    facts = []
    for gv_name, mod_name in module_mapping.items():
        facts.append(make_module_rule(gv_name, mod_name))
    return facts


module_rules = ruleset(*make_module_fact_ruleset(cgv.imported))

#######################################

# Install numpy function rules


@function
def NpyOp_Sum(operand: Term, axis: i64Like, keepdims: BoolLike) -> Term: ...


@function
def NpyOp_Max(operand: Term, axis: i64Like, keepdims: BoolLike) -> Term: ...


@function
def NpyOp_Exp(operand: Term) -> Term: ...


@function
def NpyOp_Subtract(lhs: Term, rhs: Term) -> Term: ...


@function
def NpyOp_Divide(lhs: Term, rhs: Term) -> Term: ...


@function
def get_ufunc_reduce_array_desc(
    in_array: ArrayDesc, axis: i64Like, keepdims: BoolLike
) -> ArrayDesc: ...


@function
def _array_dims_reduce_axis_keepdims(
    in_array: ArrayDesc,
    out_array: ArrayDesc,
    axis: i64Like,
    ndim: i64Like,
    i: i64Like,
) -> Unit: ...


@ruleset
def ruleset_ufunc_reduce_array_desc(
    in_array: ArrayDesc,
    out_array: ArrayDesc,
    axis: i64,
    keepdims: Bool,
    ndim: i64,
    idx: i64,
    dim: Dim,
):
    # get_ufunc_reduce_array_desc
    yield rule(
        out_array == get_ufunc_reduce_array_desc(in_array, axis, keepdims),
        ndim == in_array.ndim,
        keepdims == Bool(True),
    ).then(
        set_(out_array.dataLayout).to(in_array.dataLayout),
        set_(out_array.ndim).to(ndim),
        set_(out_array.dtype).to(in_array.dtype),
        _array_dims_reduce_axis_keepdims(in_array, out_array, axis, ndim, 0),
    )

    # _array_dims_reduce_axis_keepdims
    #   normalize axis
    yield rule(
        target := _array_dims_reduce_axis_keepdims(
            in_array, out_array, axis, ndim, idx
        ),
        axis < i64(0),
    ).then(
        _array_dims_reduce_axis_keepdims(
            in_array, out_array, ndim + axis, ndim, idx
        ),
        subsume(target),
    )
    #   out_dim[idx]=in_dim[idx] if idx != axis
    yield rule(
        _array_dims_reduce_axis_keepdims(in_array, out_array, axis, ndim, idx),
        0 <= idx,
        idx < ndim,
        idx != axis,
    ).then(
        union(out_array.dim(idx)).with_(in_array.dim(idx)),
    )
    #   out_dim[idx]=1 if idx == axis
    yield rule(
        _array_dims_reduce_axis_keepdims(in_array, out_array, axis, ndim, idx),
        0 <= idx,
        idx < ndim,
        idx != axis,
    ).then(
        union(out_array.dim(idx)).with_(Dim.fixed(1)),
    )
    #  idx+=1
    yield rule(
        _array_dims_reduce_axis_keepdims(in_array, out_array, axis, ndim, idx),
        0 <= idx,
        idx < ndim - 1,
    ).then(
        _array_dims_reduce_axis_keepdims(
            in_array, out_array, axis, ndim, idx + 1
        ),
    )


@function
def DEBUG(operand: Term) -> Term: ...


class NumPyRules:
    module_name = "numpy"

    @staticmethod
    def _make_reduce_op_rules(op_name: str, op_constructor):
        """
        Common logic for numpy reduce operations (max, sum, etc.) that handle
        keepdims and axis parameters.
        """
        io = var("io", Term)
        obj = var("obj", Term)
        res = var("res", Term)
        args = var("args", TermList)
        kwargs = var("kwargs", TermDict)

        intype = var("intype", Type)
        arrdesc = var("arrdesc", ArrayDesc)

        callee = Py_CallKwargs(
            func=npy_reduce(op_name), io=io, args=args, kwargs=kwargs
        )
        keepdims_val = var("keepdims_val", Bool)
        axis_val = var("axis_val", i64)
        yield rule(callee).then(
            kwargs.lookup("keepdims"), kwargs.lookup("axis")
        )
        yield rewrite(callee.getPort(1)).to(
            op_constructor(args[0], axis=axis_val, keepdims=keepdims_val),
            # conditions
            Term.LiteralI64(axis_val) == kwargs.get("axis"),
            Term.LiteralBool(keepdims_val) == kwargs.get("keepdims"),
        )
        # make it pure
        yield rewrite(callee.getPort(0)).to(io)
        yield rule(
            res == op_constructor(obj, axis=axis_val, keepdims=keepdims_val),
            intype == TypeVar(obj).getType(),
            intype == arrdesc.toType(),
        ).then(
            set_(TypeVar(res).getType()).to(
                get_ufunc_reduce_array_desc(
                    arrdesc, axis_val, keepdims_val
                ).toType()
            )
        )

    @staticmethod
    def _make_binary_rules(op_name: str, op_constructor):
        io = var("io", Term)
        lhs = var("lhs", Term)
        rhs = var("rhs", Term)
        res = var("res", Term)
        lhs_arraydesc = var("lhs_arraydesc", ArrayDesc)
        rhs_arraydesc = var("rhs_arraydesc", ArrayDesc)
        arg_vector = var("arg_vector", Vec[Term])

        the_call = Py_Call(
            func=npy_binary_ufunc(op_name),
            io=io,
            args=TermList(arg_vector),
        )
        yield rule(
            the_call,
            arg_vector.length() == i64(2),
            lhs == arg_vector[0],
            rhs == arg_vector[1],
        ).then(
            union(the_call.getPort(0)).with_(io),
            union(the_call.getPort(1)).with_(op_constructor(lhs, rhs)),
        )
        # Typing and broadcasting
        yield rule(
            res == op_constructor(lhs, rhs),
            lhs_arraydesc.toType() == TypeVar(lhs).getType(),
            rhs_arraydesc.toType() == TypeVar(rhs).getType(),
        ).then(
            set_(TypeVar(res).getType()).to(
                Broadcast(lhs_arraydesc, rhs_arraydesc).toType()
            )
        )

    @staticmethod
    def _make_unary_rules(op_name: str, op_constructor):
        io = var("io", Term)
        operand = var("operand", Term)
        res = var("res", Term)
        operand_arraydesc = var("operand_arraydesc", ArrayDesc)
        arg_vector = var("arg_vector", Vec[Term])

        the_call = Py_Call(
            func=npy_unary_ufunc(op_name),
            io=io,
            args=TermList(arg_vector),
        )
        yield rule(
            the_call,
            arg_vector.length() == i64(1),
            operand == arg_vector[0],
        ).then(
            union(the_call.getPort(0)).with_(io),
            union(the_call.getPort(1)).with_(op_constructor(operand)),
        )
        # Typing and broadcasting
        yield rule(
            res == op_constructor(operand),
            operand_arraydesc.toType() == TypeVar(operand).getType(),
        ).then(set_(TypeVar(res).getType()).to(operand_arraydesc.toType()))

    @staticmethod
    def exp(orig: Term):
        yield rewrite(orig, subsume=True).to(npy_unary_ufunc("exp"))
        yield from NumPyRules._make_unary_rules("exp", NpyOp_Exp)

    @staticmethod
    def max(orig: Term):
        yield rewrite(orig, subsume=True).to(npy_reduce("max"))
        yield from NumPyRules._make_reduce_op_rules("max", NpyOp_Max)

    @staticmethod
    def sum(orig: Term):
        yield rewrite(orig, subsume=True).to(npy_reduce("sum"))
        yield from NumPyRules._make_reduce_op_rules("sum", NpyOp_Sum)

    @staticmethod
    def subtract(orig: Term):
        yield rewrite(orig, subsume=True).to(npy_binary_ufunc("subtract"))
        yield from NumPyRules._make_binary_rules("subtract", NpyOp_Subtract)

    @staticmethod
    def divide(orig: Term):
        yield rewrite(orig, subsume=True).to(npy_binary_ufunc("divide"))
        yield from NumPyRules._make_binary_rules("divide", NpyOp_Divide)


@ruleset
def ruleset_numpy_promote_binop(
    op: Term, lhs: Term, rhs: Term, io: Term, arraydesc: ArrayDesc
):
    def promote_subtract(operand, opname, py_op):
        return rewrite(py_op(io, lhs, rhs)).to(
            Py_Call(
                ModuleGetAttr(Module("numpy"), opname), io, termlist(lhs, rhs)
            ),
            # when
            TypeVar(operand).getType() == arraydesc.toType(),
        )

    for operand in [lhs, rhs]:
        yield promote_subtract(operand, "subtract", Py_SubIO)
        yield promote_subtract(operand, "divide", Py_DivIO)


@function
def npy_unary_ufunc(name: StringLike) -> Term: ...


@function
def npy_binary_ufunc(name: StringLike) -> Term: ...


@function
def npy_reduce(name: StringLike) -> Term: ...


loaded_module = {
    "numpy": NumPyRules,
}


def module_function_rule_lookup(modname: str):
    modrule = loaded_module[modname]
    for fname in dir(modrule):
        if not fname.startswith("_"):
            fn = getattr(modrule, fname)
            if isinstance(fn, FunctionType):
                base_node = ModuleGetAttr(Module(modname), fname)
                yield from fn(base_node)


def make_function_rule(module_mapping: dict[str, str]):
    for modname in module_mapping.values():
        rules = module_function_rule_lookup(modname)
        yield ruleset(*rules, name=f"ruleset_module_{modname}")


module_rulesets = (
    reduce(operator.or_, make_function_rule(cgv.imported))
    | ruleset_numpy_promote_binop
)


#######################################
# Extra operator rules


@function
def Nb_Neg_Int64(operand: Term) -> Term: ...


@ruleset
def ruleset_type_infer_negate(
    io: Term,
    x: Term,
    op: Term,
):
    yield rule(
        op == Py_NegIO(io, x),
        TypeVar(x).getType() == TypeInt64,
    ).then(
        union(op.getPort(1)).with_(Nb_Neg_Int64(x)),
        union(op.getPort(0)).with_(io),
    )

    yield rule(op == Nb_Neg_Int64(x)).then(
        set_(TypeVar(op).getType()).to(TypeInt64)
    )


ruleset_extra_builtin_operations = ruleset_type_infer_negate

#######################################
target_function = cgv.functions["softmax"]
# target_function = cgv.functions["smaller"]
pprint(target_function)


array_x_desc, array_x_infos = array_desc_rules(
    "array_x", shape=("N",), dtype=TypeFloat64, layout="c"
)
ruleset_array_facts = ruleset(
    *array_x_infos,
)


class LLM_generic(NbOp_Base):
    desc: str
    operands: tuple[SExpr, ...]


class LLM_Type(NbOp_Base):
    name: str
    children: tuple[SExpr, ...]


class ExtendEGraphToRVSDG(_ch05_ExtendEGraphToRVSDG):
    def handle_Term(self, op: str, children: dict | list, grm: Grammar):
        parent_output = super().handle_Term(op, children, grm)
        if parent_output is NotImplemented:
            assert isinstance(children, dict)
            return grm.write(
                LLM_generic(
                    desc=op + "{" + f"{', '.join(children)}" + "}",
                    operands=tuple(children.values()),
                )
            )
        return parent_output

    def is_type_from_egraph(self, node) -> bool:
        return node["op"] == "·.toType"

    def handle_ArrayDesc(
        self, key: str, op: str, children: dict | list, grm: Grammar
    ):
        assert isinstance(children, dict)
        return grm.write(
            LLM_Type(name=str(op), children=tuple(children.values()))
        )

    def handle_Type(
        self, key: str, op: str, children: dict | list, grm: Grammar
    ):
        try:
            return super().handle_Type(key, op, children, grm)
        except NotImplementedError:
            assert isinstance(children, dict)
            return grm.write(
                LLM_Type(name=op, children=tuple(children.values()))
            )


compiler_config = _compiler_config.copy()
compiler_config["converter_class"] = ExtendEGraphToRVSDG


class StubBackend:
    def lower(self, root, argtypes):
        from ch04_1_typeinfer_ifelse import Attributes
        from sealir.rvsdg import format_rvsdg

        fname = root.fname
        attrs = Attributes(root.body.begin.attrs)
        retty = attrs.get_return_type(root.body)
        print("RETURN TYPE", argtypes, "->", retty)
        return format_rvsdg(root)

    def jit_compile(self, module, extracted, export_name):
        return module  # TODO


compiler_config["backend"] = StubBackend()

report = Report(default_expanded=True, enable_nested_metadata=True)
try:
    out = jit_compiler(
        fn=target_function.ast,
        argtypes=(array_x_desc.toType(),),
        ruleset=(
            base_ruleset
            | py_eqsat_rules()
            | setup_argtypes(array_x_desc.toType())
            | ruleset_array_facts
            | module_rules
            | module_rulesets
            | ruleset_extra_builtin_operations
            | ruleset_ufunc_reduce_array_desc
        ),
        pipeline_report=report,
        pipeline_debug=True,
        **compiler_config,
    )
finally:
    pass
    # report.display(view_html=True)

print("OUTPUT".center(80, "-"))
print(out.jit_func)
