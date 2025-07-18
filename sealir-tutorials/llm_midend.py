import operator
from functools import reduce
from pprint import pprint
from types import FunctionType

from egglog import (
    Bool,
    BoolLike,
    Expr,
    String,
    StringLike,
    Vec,
    function,
    i64,
    i64Like,
    rewrite,
    rule,
    ruleset,
    set_,
    union,
    var,
)
from sealir.eqsat.py_eqsat import (
    Py_AttrIO,
    Py_CallKwargs,
    Py_LoadGlobal,
)
from sealir.eqsat.py_eqsat import make_rules as py_eqsat_rules
from sealir.eqsat.rvsdg_eqsat import (
    Term,
    TermDict,
    TermList,
)

from ch04_1_typeinfer_ifelse import Type, TypeVar
from ch05_typeinfer_array import (
    Int64,
    TypeInt64,
    base_ruleset,
    compiler_config,
    jit_compiler,
    setup_argtypes,
)
from ch09_whole_program_compiler_driver import CallGraphVisitor
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


class NumPyRules:
    module_name = "numpy"

    @staticmethod
    def exp(orig: Term):
        yield rewrite(orig, subsume=True).to(npy_unary_ufunc("exp"))

    @staticmethod
    def max(orig: Term):
        yield rewrite(orig, subsume=True).to(npy_reduce("max"))

    @staticmethod
    def sum(orig: Term):
        yield rewrite(orig, subsume=True).to(npy_reduce("sum"))

        io = var("io", Term)
        args = var("args", TermList)
        kwargs = var("kwargs", TermDict)
        callee = Py_CallKwargs(
            func=npy_reduce("sum"), io=io, args=args, kwargs=kwargs
        )
        keepdims_val = var("keepdims_val", Bool)
        axis_val = var("axis_val", i64)
        yield rule(callee).then(
            kwargs.lookup("keepdims"), kwargs.lookup("axis")
        )
        yield rewrite(callee.getPort(1)).to(
            NpyOp_Sum(args[0], axis=axis_val, keepdims=keepdims_val),
            # conditions
            Term.LiteralI64(axis_val) == kwargs.get("axis"),
            Term.LiteralBool(keepdims_val) == kwargs.get("keepdims"),
        )


@function
def npy_unary_ufunc(name: StringLike) -> Term: ...


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


module_rulesets = reduce(operator.or_, make_function_rule(cgv.imported))

#######################################
softmax = cgv.functions["softmax"]
pprint(softmax)


report = Report(default_expanded=True, enable_nested_metadata=True)
try:
    jit_compiler(
        fn=softmax.ast,
        argtypes=(Int64, Int64),
        ruleset=(
            base_ruleset
            | py_eqsat_rules()
            | setup_argtypes(TypeInt64, TypeInt64)
            | module_rules
            | module_rulesets
        ),
        pipeline_report=report,
        pipeline_debug=True,
        **compiler_config,
    )
finally:
    report.display(view_html=True)
