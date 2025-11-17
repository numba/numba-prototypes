# # Call Graph Construction
# There are two datastructures which will hold the necessary
# information to schedule compilation. The first is a [Symbol
# Table](https://en.wikipedia.org/wiki/Symbol_table) and the second is a [Call
# Graph](https://en.wikipedia.org/wiki/Call_graph).
#
# The "Symbol Table" is a mapping structure that maps symbol names to symbol
# information. Concretely for the case of a Python function compiler, this will
# map functions to various pieces of information about these functions. In our
# case we consider classes to be syntactic sugar and we consider methods to be
# simply functions where the first argument (`self`) is an instance to a simple
# datastructre that has fields. In this case, the symbol information will
# consist of three parts:
#
#  * The fully qualified name of the function
#  * the complete `ast.AST` node for the function
#  * any calls that can be statically determined
#
# The "Call Graph" represents the relationships between functions. It is a
# directed graph where each node maps to a function and the children are the
# calls within the function. It represents the ordering of calls for a given
# function and thus can be used to schedule the compilation of functions. The
# call graph will be established from the third part of the symbol information.
#
# We develop these capabilities on the Python Abstract Syntax Tree (AST)
# representation of Python. The module `ast` provides a set of utilities to
# work the AST. Specifically we will develop a visitor class by subclassing
# from the `NodeVisitor` class. This visitor will traverse the AST and collect
# the various pieces of information.


# ### Imports

import ast
import symtable
from collections import defaultdict
from dataclasses import dataclass

# ### Symbol Information class


@dataclass
class SymbolInfo:
    name: str
    ast: ast.AST
    calls: list


# ### Call Graph Visitor class
#
# As mentioned above, `the CallGraphVisitor` class is a subclass of the
# `ast.NodeVisitor`. It is used to traverse the AST and collect information
# about the functions and their calls. Only a subset of the AST nodes are
# supported by `visit_*` methods. The most important ones are:
#
#   * `visit_FunctionDef`: Visit a function definition
#   * `visit_ClassDef`: Visit a class definition
#
# Additionally the function `update_calls` is used to rewrite the names such
# that they become qualified. The class itself has various housekeeping
# datastructures such as stack to keep track of the current namespace, which
# class is being visited and so on.
#
# Lastly, the function `get_call_graph` returns the call graph.


class CallGraphVisitor(ast.NodeVisitor):

    def __init__(self, source_code, file_name):
        # Stash the arguments
        self.source_code = source_code
        self.file_name = file_name
        # Get the AST once
        self.tree = ast.parse(source_code)
        # Initialize the cpython symtable
        self.symt = symtable.symtable(source_code, file_name, "exec")
        # Filter out all class definitions from the AST
        self.classes = set(
            (
                node.name
                for node in ast.walk(self.tree)
                if isinstance(node, ast.ClassDef)
            )
        )
        # Setup the namespace and class stacks
        self.namespace_stack = []
        self.class_stack = []
        # Nested dictionary to record class types
        self.class_types = defaultdict(dict)
        # Dictionary to record all functions, this is effectively the symbol
        # table.
        self.functions = {}
        # List of all global ast.Call nodes
        self.global_calls = []
        # Mapping of imported Python modules in the global namespace
        # Key is global name of the imported Module.
        # Value is the fully-qualified import path.
        self.imported = {}

    def get_call_graph(self) -> dict[str : tuple[str]]:
        """Obtain a call graph suitable for processing with networkx.

        Returns a dictionary mapping function names as strings to lists of
        function names as strings.
        """

        return {k: tuple(c[1] for c in v.calls) for k, v in self.functions.items()}

    def update_calls(self, node):
        """Update the calls for a function or register a global call."""
        # Flatten the name of the call from ast.Attribute or ast.Name.
        call_qname = attribute_to_qualified_name(node)
        if call_qname is None:
            return
        # Get the current class name, if we are visiting a class and have a
        # method.
        class_name, method_name = (
            (None, None)
            if not (self.class_stack and self.namespace_stack)
            else (self.class_stack[-1], self.namespace_stack[-1])
        )
        # Get the name of the first paramater (usually 'self') of the method
        # call using symtable module. If we are in a class, we assume this is a
        # method call indeed.  TODO: account for @staticmethod and
        # @classmethod.
        first_param_name = (
            self.symt.lookup(class_name)
            .get_namespace()
            .lookup(method_name)
            .get_namespace()
            .get_parameters()[0]
            if class_name and method_name
            else None
        )
        if first_param_name and call_qname.split(".")[0] == first_param_name:
            # If the call starts with "self" or it's equivalent as determiend
            # above, we replace # the "self" with the current class name to
            # qualify it.
            call_qname = class_name + call_qname[len(first_param_name) :]
        if class_name and call_qname.startswith(class_name):
            # Replace calls from class attributes with their qualified name.
            # First split the qualified name by the dot separator.
            split_qname = call_qname.split(".")
            # Get the types of the current classes attributes.
            current_class_types = self.class_types[class_name]
            # If the second element in the qualified name matches the name of
            # the class attribute, replace the reference to the class.attribute
            # string with the correct type.
            if split_qname[1] in current_class_types:
                call_qname = ".".join(
                    [current_class_types[split_qname[1]]] + split_qname[2:]
                )
        if call_qname in self.classes:
            # If the call ends with the current class name, we replace it with
            # the constructor call, since this is the Python semantics.
            call_qname = call_qname + ".__init__"

        if self.namespace_stack:
            name = ".".join(self.namespace_stack)
            assert name in self.functions, f"Function {name} not found"
            self.functions[name].calls.append((node, call_qname))
        else:
            self.global_calls.append((node, call_qname))

    def visit_all(self):
        """Visit all nodes in the AST."""
        self.visit(self.tree)

    def visit_Import(self, node):
        # Add globals that are imported into self.imported
        for alias in node.names:
            # The name used in the global namespace
            global_name = alias.asname if alias.asname else alias.name
            # The fully-qualified import path
            import_path = alias.name
            self.imported[global_name] = import_path

    def visit_ImportFrom(self, node):
        # Add globals that are imported into self.imported
        module_name = node.module if node.module else ""
        for alias in node.names:
            # The name used in the global namespace
            global_name = alias.asname if alias.asname else alias.name
            # The fully-qualified import path
            if module_name:
                import_path = f"{module_name}.{alias.name}"
            else:
                # Handle relative imports (from . import name)
                import_path = alias.name
            self.imported[global_name] = import_path

    def visit_FunctionDef(self, node):
        """Visit a function definition."""
        # Create a new namespace for the function
        self.namespace_stack.append(node.name)
        name = ".".join(self.namespace_stack)
        self.functions[name] = SymbolInfo(name, node, [])

        # Visit the function body
        self.generic_visit(node)

        # Pop the namespace after visiting the function
        self.namespace_stack.pop()

    def visit_ClassDef(self, node):
        """Visit a class definition."""
        # Create a new namespace for the class
        self.namespace_stack.append(node.name)
        # Push the name of the class onto the class_stack
        self.class_stack.append(node.name)

        # Visit the class body
        self.generic_visit(node)

        # Pop the namespace after visiting the class
        self.namespace_stack.pop()
        # Pop the class name from the class_stack
        self.class_stack.pop()

    def visit_Call(self, node):
        """Visit a function call."""
        # Update the namespace where this call occurs
        self.update_calls(node.func)

        # Visit the arguments of the function call
        for n in node.args + node.keywords:
            self.generic_visit(n)

    def visit_AnnAssign(self, node):
        """Visit an annotated assignment."""
        if self.class_stack[-1] == self.namespace_stack[-1]:
            # Class and namespace stack have the identical last value. This
            # means we are in a class definition.
            class_name = self.class_stack[-1]
            assert isinstance(node.target, ast.Name)
            attribute_name = node.target.id
            assert isinstance(node.annotation, ast.Name)
            attribute_type = node.annotation.id
            # Populate the class_type datastructure
            self.class_types[class_name][attribute_name] = attribute_type


# ### Utilities


def attribute_to_qualified_name(node):
    """
    Converts an ast.Attribute node into a fully qualified name string.

    For example, if the AST represents "module.submodule.function", this
    function will return the string "module.submodule.function". Operates
    recursively to handle nested attributes.

    Args:
        node: An ast.Attribute node or ast.Name node

    Returns:
        str: The fully qualified name as a string
    """
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        return f"{attribute_to_qualified_name(node.value)}.{node.attr}"
    elif isinstance(node, ast.Call):
        return attribute_to_qualified_name(node.func)
    else:
        # Node type not supported for qualified name conversion
        return None


def to_graphviz(cgv):
    # Convert the call graph in a CallGraphVisitor to a graphviz style graph
    # that Jupyter can render natively.
    #

    import networkx as nx
    from graphviz import Source

    # We use the interface "adjacency list" to create a networkx DiGraph
    # (directed graph).  Then convert that to a graphviz style graph for
    # visualization using various APIs.

    return Source(nx.nx_agraph.to_agraph(nx.DiGraph(cgv.get_call_graph())).string())
