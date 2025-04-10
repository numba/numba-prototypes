import ast
from dataclasses import dataclass
from typing import Dict, Optional, Set, List
import sys


@dataclass
class SymbolInfo:
    """Placeholder for symbol information. Will be expanded later."""
    pass


class SymbolTableBuilder(ast.NodeVisitor):
    """AST visitor that builds a symbol table while traversing the parse tree."""

    def __init__(self):
        self.symbols = {}  # Global symbol table
        self.scopes = [{}]  # Stack of scopes, with global scope at index 0
        self.current_scope = self.scopes[0]

    def visit_Name(self, node):
        """Process name nodes to track variable references and definitions."""
        if isinstance(node.ctx, ast.Store):
            # This is a definition or assignment
            self.current_scope[node.id] = SymbolInfo()
            # Also add to global symbol table with fully qualified name
            self.symbols[node.id] = SymbolInfo()
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Process function definitions."""
        # Register the function name in the current scope
        self.current_scope[node.name] = SymbolInfo()
        self.symbols[node.name] = SymbolInfo()

        # Create a new scope for this function
        function_scope = {}
        self.scopes.append(function_scope)
        old_scope = self.current_scope
        self.current_scope = function_scope

        # Add parameters to the function's scope
        for arg in node.args.args:
            self.current_scope[arg.arg] = SymbolInfo()
            # Use qualified name for global table
            qualified_name = f"{node.name}.{arg.arg}"
            self.symbols[qualified_name] = SymbolInfo()

        # Visit the function body
        for statement in node.body:
            self.visit(statement)

        # Restore the previous scope
        self.scopes.pop()
        self.current_scope = old_scope

    def visit_ClassDef(self, node):
        """Process class definitions."""
        # Register the class in the current scope
        self.current_scope[node.name] = SymbolInfo()
        self.symbols[node.name] = SymbolInfo()

        # Create a scope for this class
        class_scope = {}
        self.scopes.append(class_scope)
        old_scope = self.current_scope
        self.current_scope = class_scope

        # Visit the class body
        for statement in node.body:
            self.visit(statement)

        # Process class members for the symbol table
        for name, _ in class_scope.items():
            qualified_name = f"{node.name}.{name}"
            self.symbols[qualified_name] = SymbolInfo()

        # Restore the previous scope
        self.scopes.pop()
        self.current_scope = old_scope


class CompilerDriver:
    """Main driver for the compiler process."""

    def __init__(self):
        self.symbol_table = {}

    def compile(self, source_code):
        """Compile the given source code and return the symbol table."""
        # Parse the source into an AST
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            print(f"Syntax error: {e}")
            return None

        # Build the symbol table
        symbol_builder = SymbolTableBuilder()
        symbol_builder.visit(tree)

        # Return the resulting symbol table
        return symbol_builder.symbols


def main():
    """Entry point for the compiler driver."""
    if len(sys.argv) < 2:
        print("Usage: python compiler_driver.py <python_source_file>")
        sys.exit(1)

    source_file = sys.argv[1]

    try:
        with open(source_file, 'r') as f:
            source_code = f.read()
    except FileNotFoundError:
        print(f"File not found: {source_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    compiler = CompilerDriver()
    symbol_table = compiler.compile(source_code)

    if symbol_table:
        print("Symbol Table:")
        for symbol, info in symbol_table.items():
            print(f"  {symbol}: {info}")


if __name__ == "__main__":
    main()
