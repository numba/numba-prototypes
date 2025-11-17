#!/usr/bin/env python3
"""
Script to convert Python functions into Graphviz expression trees using symbolic evaluation.
Uses operator overloading to automatically build the computation graph.
Enhanced with NumPy-style operations support and unused node filtering.
"""

import argparse
import ast
import inspect
from pathlib import Path
from typing import Any, Dict, Set

import graphviz


class SymbolicNode:
    """A symbolic node that tracks operations and builds the computation graph."""

    _node_counter = 0
    _graph = None

    def __init__(self, name: str, node_type: str = "var", value: Any = None):
        SymbolicNode._node_counter += 1
        self.id = f"n{SymbolicNode._node_counter}"
        self.name = name
        self.node_type = node_type  # 'var', 'op', 'literal'
        self.value = value
        self.operands = []
        self.used_in_computation = (
            False  # Track if this node is part of the computation
        )

    def _mark_as_used(self):
        """Mark this node and all its operands as used in computation."""
        if self.used_in_computation:
            return  # Already processed

        self.used_in_computation = True
        for operand in self.operands:
            if isinstance(operand, SymbolicNode):
                operand._mark_as_used()

    def _add_to_graph(self):
        """Add this node to the current graph."""
        if self.node_type == "var":
            color = "lightgreen" if self.name.startswith("arg") else "lightyellow"
            shape = "ellipse"
        elif self.node_type == "op":
            color = "lightsteelblue"
            shape = "diamond"
        else:  # literal
            color = "lightcoral"
            shape = "ellipse"

        SymbolicNode._graph.node(
            self.id,
            label=self.name,
            shape=shape,
            fillcolor=color,
            style="filled",
        )

    def _binary_op(self, other, op_symbol: str, op_name: str):
        """Create a new node representing a binary operation."""
        if not isinstance(other, SymbolicNode):
            other = SymbolicNode(str(other), "literal", other)

        result = SymbolicNode(op_symbol, "op")
        result.operands = [self, other]
        return result

    def _unary_op(self, op_symbol: str, op_name: str):
        """Create a new node representing a unary operation."""
        result = SymbolicNode(op_symbol, "op")
        result.operands = [self]
        return result

    # Arithmetic operations
    def __add__(self, other):
        return self._binary_op(other, "+", "add")

    def __radd__(self, other):
        if not isinstance(other, SymbolicNode):
            other = SymbolicNode(str(other), "literal", other)
        return other._binary_op(self, "+", "add")

    def __sub__(self, other):
        return self._binary_op(other, "-", "sub")

    def __rsub__(self, other):
        if not isinstance(other, SymbolicNode):
            other = SymbolicNode(str(other), "literal", other)
        return other._binary_op(self, "-", "sub")

    def __mul__(self, other):
        return self._binary_op(other, "*", "mul")

    def __rmul__(self, other):
        if not isinstance(other, SymbolicNode):
            other = SymbolicNode(str(other), "literal", other)
        return other._binary_op(self, "*", "mul")

    def __truediv__(self, other):
        return self._binary_op(other, "/", "div")

    def __rtruediv__(self, other):
        if not isinstance(other, SymbolicNode):
            other = SymbolicNode(str(other), "literal", other)
        return other._binary_op(self, "/", "div")

    def __floordiv__(self, other):
        return self._binary_op(other, "//", "floordiv")

    def __rfloordiv__(self, other):
        if not isinstance(other, SymbolicNode):
            other = SymbolicNode(str(other), "literal", other)
        return other._binary_op(self, "//", "floordiv")

    def __mod__(self, other):
        return self._binary_op(other, "%", "mod")

    def __rmod__(self, other):
        if not isinstance(other, SymbolicNode):
            other = SymbolicNode(str(other), "literal", other)
        return other._binary_op(self, "%", "mod")

    def __pow__(self, other):
        return self._binary_op(other, "**", "pow")

    def __rpow__(self, other):
        if not isinstance(other, SymbolicNode):
            other = SymbolicNode(str(other), "literal", other)
        return other._binary_op(self, "**", "pow")

    # Matrix multiplication
    def __matmul__(self, other):
        return self._binary_op(other, "@", "matmul")

    def __rmatmul__(self, other):
        if not isinstance(other, SymbolicNode):
            other = SymbolicNode(str(other), "literal", other)
        return other._binary_op(self, "@", "matmul")

    # Bitwise operations
    def __and__(self, other):
        return self._binary_op(other, "&", "and")

    def __rand__(self, other):
        if not isinstance(other, SymbolicNode):
            other = SymbolicNode(str(other), "literal", other)
        return other._binary_op(self, "&", "and")

    def __or__(self, other):
        return self._binary_op(other, "|", "or")

    def __ror__(self, other):
        if not isinstance(other, SymbolicNode):
            other = SymbolicNode(str(other), "literal", other)
        return other._binary_op(self, "|", "or")

    def __xor__(self, other):
        return self._binary_op(other, "^", "xor")

    def __rxor__(self, other):
        if not isinstance(other, SymbolicNode):
            other = SymbolicNode(str(other), "literal", other)
        return other._binary_op(self, "^", "xor")

    def __lshift__(self, other):
        return self._binary_op(other, "<<", "lshift")

    def __rlshift__(self, other):
        if not isinstance(other, SymbolicNode):
            other = SymbolicNode(str(other), "literal", other)
        return other._binary_op(self, "<<", "lshift")

    def __rshift__(self, other):
        return self._binary_op(other, ">>", "rshift")

    def __rrshift__(self, other):
        if not isinstance(other, SymbolicNode):
            other = SymbolicNode(str(other), "literal", other)
        return other._binary_op(self, ">>", "rshift")

    # Unary operations
    def __neg__(self):
        return self._unary_op("-", "neg")

    def __pos__(self):
        return self._unary_op("+", "pos")

    def __invert__(self):
        return self._unary_op("~", "invert")

    # Comparison operations (return symbolic nodes for graph building)
    def __eq__(self, other):
        return self._binary_op(other, "==", "eq")

    def __ne__(self, other):
        return self._binary_op(other, "!=", "ne")

    def __lt__(self, other):
        return self._binary_op(other, "<", "lt")

    def __le__(self, other):
        return self._binary_op(other, "<=", "le")

    def __gt__(self, other):
        return self._binary_op(other, ">", "gt")

    def __ge__(self, other):
        return self._binary_op(other, ">=", "ge")

    # Indexing and slicing
    def __getitem__(self, key):
        if not isinstance(key, SymbolicNode):
            key = SymbolicNode(str(key), "literal", key)
        return self._binary_op(key, "[]", "getitem")

    def __call__(self, *args, **kwargs):
        """Handle function calls."""
        # Convert arguments to symbolic nodes, handling tuples/lists
        symbolic_args = []
        for arg in args:
            if isinstance(arg, (tuple, list)):
                # Handle tuple/list arguments (like for hstack, vstack)
                for item in arg:
                    if not isinstance(item, SymbolicNode):
                        symbolic_args.append(SymbolicNode(str(item), "literal", item))
                    else:
                        symbolic_args.append(item)
            elif not isinstance(arg, SymbolicNode):
                symbolic_args.append(SymbolicNode(str(arg), "literal", arg))
            else:
                symbolic_args.append(arg)

        # Create the operation node
        result = SymbolicNode(f"{self.name}()", "op")
        result.operands = symbolic_args
        return result

    def __getattr__(self, name):
        """Handle attribute access - create symbolic nodes for any attribute."""
        full_name = f"{self.name}.{name}"
        return SymbolicNode(full_name, "var")

    def __repr__(self):
        return f"SymbolicNode({self.name})"


class ExpressionTreeBuilder:
    """Builds expression trees from Python functions using symbolic evaluation."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the builder state."""
        SymbolicNode._node_counter = 0
        SymbolicNode._graph = None
        self.all_nodes = []

    def build_tree(self, func, func_name: str = None) -> graphviz.Digraph:
        """Build an expression tree from a function using symbolic evaluation."""
        self.reset()

        # Get function signature
        sig = inspect.signature(func)
        arg_names = list(sig.parameters.keys())

        if func_name is None:
            func_name = func.__name__

        # Create Graphviz graph
        graph = graphviz.Digraph(
            name=func_name, comment=f"Expression tree for {func_name}"
        )
        graph.attr(rankdir="BT")
        graph.attr("node", fontname="Arial")
        graph.attr("edge", fontname="Arial")

        # Set the global graph reference
        SymbolicNode._graph = graph

        # Create symbolic arguments
        symbolic_args = []
        for arg_name in arg_names:
            arg_node = SymbolicNode(arg_name, "var")
            symbolic_args.append(arg_node)
            self.all_nodes.append(arg_node)

        # Create symbolic numpy module
        symbolic_np = SymbolicNode("np", "var")
        self.all_nodes.append(symbolic_np)

        # Hook to collect all created nodes
        original_init = SymbolicNode.__init__

        def tracking_init(self_node, *args, **kwargs):
            original_init(self_node, *args, **kwargs)
            self.all_nodes.append(self_node)

        SymbolicNode.__init__ = tracking_init

        try:
            # Create a namespace with the symbolic numpy module
            func_globals = func.__globals__.copy()
            func_globals["np"] = symbolic_np

            # Create a new function with the modified globals
            import types

            symbolic_func = types.FunctionType(
                func.__code__,
                func_globals,
                func.__name__,
                func.__defaults__,
                func.__closure__,
            )

            result = symbolic_func(*symbolic_args)

            # If the result is a symbolic node, mark it and its dependencies as used
            if isinstance(result, SymbolicNode):
                result._mark_as_used()

                # Add only used nodes to the graph
                self._add_used_nodes_to_graph()

                # Add edges for used nodes
                self._add_edges_for_used_nodes()

                # Add a return node
                return_node = SymbolicNode("return", "op")
                return_node.used_in_computation = True
                return_node._add_to_graph()
                graph.edge(result.id, return_node.id)

        except Exception as e:
            print(f"Error during symbolic execution: {e}")
            print("This might happen if the function uses unsupported operations.")
        finally:
            # Restore original __init__
            SymbolicNode.__init__ = original_init

        return graph

    def _add_used_nodes_to_graph(self):
        """Add only the used nodes to the graph."""
        for node in self.all_nodes:
            if node.used_in_computation:
                node._add_to_graph()

    def _add_edges_for_used_nodes(self):
        """Add edges for all used nodes."""
        for node in self.all_nodes:
            if not node.used_in_computation:
                continue

            if len(node.operands) == 2:  # Binary operation
                left, right = node.operands
                if left.used_in_computation and right.used_in_computation:
                    SymbolicNode._graph.edge(left.id, node.id, label="left")
                    SymbolicNode._graph.edge(right.id, node.id, label="right")
            elif len(node.operands) == 1:  # Unary operation
                operand = node.operands[0]
                if operand.used_in_computation:
                    SymbolicNode._graph.edge(operand.id, node.id)
            elif len(node.operands) > 2:  # Function call with multiple args
                for i, arg in enumerate(node.operands):
                    if arg.used_in_computation:
                        SymbolicNode._graph.edge(arg.id, node.id, label=f"arg{i}")

    def visualize_function(
        self,
        func,
        output_file: str = None,
        format: str = "png",
        view: bool = False,
        func_name: str = None,
    ):
        """Visualize a function and optionally save/display the result."""
        graph = self.build_tree(func, func_name)

        if output_file:
            # Remove extension if provided (graphviz adds it)
            output_file = str(Path(output_file).with_suffix(""))
            graph.render(output_file, format=format, cleanup=True)
            print(f"Graph saved as {output_file}.{format}")

        if view:
            graph.view()

        return graph


def visualize_expr_tree(
    func,
    output_file: str = None,
    format: str = "png",
    view: bool = False,
    func_name: str = None,
):
    return ExpressionTreeBuilder().visualize_function(
        func,
        output_file=output_file,
        format=format,
        view=view,
        func_name=func_name,
    )


def load_function_from_file(filepath: str) -> tuple:
    """Load a function from a Python file."""
    with open(filepath, "r") as f:
        code = f.read()

    # Parse the AST to find function definitions
    tree = ast.parse(code)

    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            functions.append(node.name)

    if not functions:
        raise ValueError("No function definitions found in the file")

    # Execute the code to make functions available
    namespace = {}
    exec(code, namespace)

    # Return the first function found
    func_name = functions[0]
    return namespace[func_name], func_name


def main():
    parser = argparse.ArgumentParser(
        description="Convert Python functions to Graphviz expression trees"
    )
    parser.add_argument(
        "input", nargs="?", help="Input Python file containing function"
    )
    parser.add_argument(
        "-f",
        "--function",
        help="Function name to visualize (if file has multiple)",
    )
    parser.add_argument("-o", "--output", help="Output file name (without extension)")
    parser.add_argument(
        "--format",
        default="png",
        choices=["png", "svg", "pdf", "dot"],
        help="Output format (default: png)",
    )
    parser.add_argument("--view", action="store_true", help="Open the generated graph")
    parser.add_argument(
        "--example", action="store_true", help="Run with example function"
    )
    parser.add_argument(
        "--numpy-example",
        action="store_true",
        help="Run with NumPy example function",
    )

    args = parser.parse_args()

    builder = ExpressionTreeBuilder()

    if args.example:
        # Define example function
        def example_func(arg1, arg2, arg3, arg4):
            v4 = arg2 @ arg3
            v5 = v4 @ v4
            v7 = v5 @ arg4
            v8 = arg1 @ v7
            return v8

        print("Visualizing example function:")
        print(inspect.getsource(example_func))

        graph = builder.visualize_function(
            example_func,
            output_file=args.output or "example_func",
            format=args.format,
            view=args.view,
        )

        if not args.output and not args.view:
            print("\nGraphviz DOT source:")
            print(graph.source)

    elif args.numpy_example:
        # Define NumPy example function
        def numpy_func(arg1, arg2, arg3, arg4):
            import numpy as np

            v3 = np.hstack((arg1, arg2))
            v6 = np.vstack((arg3, arg4))
            v7 = v3 @ v6
            return v7

        print("Visualizing NumPy example function:")
        print(inspect.getsource(numpy_func))

        graph = builder.visualize_function(
            numpy_func,
            output_file=args.output or "numpy_func",
            format=args.format,
            view=args.view,
        )

        if not args.output and not args.view:
            print("\nGraphviz DOT source:")
            print(graph.source)

    elif args.input:
        try:
            func, func_name = load_function_from_file(args.input)

            if args.function:
                # Load specific function if specified
                namespace = {}
                with open(args.input, "r") as f:
                    exec(f.read(), namespace)
                if args.function not in namespace:
                    raise ValueError(f"Function '{args.function}' not found in file")
                func = namespace[args.function]
                func_name = args.function

            print(f"Visualizing function: {func_name}")
            print(inspect.getsource(func))

            output_name = args.output or func_name
            graph = builder.visualize_function(
                func,
                output_file=output_name,
                format=args.format,
                view=args.view,
                func_name=func_name,
            )

            if not args.output and not args.view:
                print(f"\nGraphviz DOT source for {func_name}:")
                print(graph.source)

        except Exception as e:
            print(f"Error: {e}")

    else:
        print("Please provide an input file or use --example or --numpy-example")
        print("Example usage:")
        print("  python script.py --example")
        print("  python script.py --numpy-example")
        print("  python script.py my_function.py -o output --view")


if __name__ == "__main__":
    main()
