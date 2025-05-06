import ast
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, Set, List
import sys


class EntityType:
    pass

class Variable(EntityType):
    pass

class LocalVariable(Variable):
    pass

class Class(EntityType):
    pass

class Member(Variable):
    pass

class Function(EntityType):
    pass

class Closure(EntityType):
    pass

class Constructor(Function):
    pass

class Method(Function):
    pass

class ClassMethod(Function):
    pass

class StaticMethod(Function):
    pass


@dataclass
class SymbolInfo:
    """Symbol Information"""

    entity_type: EntityType 


class Scope(dict):
    """Current Scope"""

    def __init__(self, namespace="", entity_type=EntityType):
        self.namespace = namespace
        self.entity_type = entity_type


class SymbolTableBuilder(ast.NodeVisitor):
    """AST visitor that builds a symbol table while traversing the parse tree."""

    def __init__(self):
        self.symbols = {}  # Global symbol table
        self.scopes = [Scope()]  # Stack of scopes, with global scope at index 0

    @property
    def current_scope(self):
        return self.scopes[-1]

    def in_global_scope(self):
        return len(self.scopes) == 1 and self.scopes[0].namespace == ""

    def create_namespace(self, name):
        prefix = "" if self.current_scope.namespace == "" else f"{self.current_scope.namespace}."
        new_namespace = f"{prefix}{name}"
        return new_namespace

    def new_scope(self, name, entity_type):
        return Scope(self.create_namespace(name), entity_type)

    def visit_Name(self, node):
        """Process name nodes to track variable references and definitions."""
        if isinstance(node.ctx, ast.Store):
            # This is a definition or assignment
            if issubclass(self.current_scope.entity_type, Class):
                symbol_info = SymbolInfo(Member)
            elif issubclass(self.current_scope.entity_type, Function):
                symbol_info = SymbolInfo(LocalVariable)
            else:
                assert self.in_global_scope()
                symbol_info = SymbolInfo(Variable)
            namespace = self.create_namespace(node.id)
            self.current_scope[node.id] = symbol_info
            self.symbols[namespace] = symbol_info
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Process function definitions."""
        # Based on the current scope, determine the EntityType
        if issubclass(self.current_scope.entity_type, Function):
            # this is a function in a function, must be a closure
            symbol_info = SymbolInfo(Closure)
        elif issubclass(self.current_scope.entity_type, Class):
            # must be some kind of method
            # check for decorator
            dl = [n.id for n in node.decorator_list]
            if "classmethod" in dl and "staticmethod" in dl:
                raise NotImplementedError(
                    "can't decorate a method with both classmethod and staticmethod")
            elif "classmethod" in dl:
                symbol_info = SymbolInfo(ClassMethod)
            elif "staticmethod" in dl:
                symbol_info = SymbolInfo(StaticMethod)
            else:
                if node.name == "__init__":
                    symbol_info = SymbolInfo(Constructor)
                else:
                    symbol_info = SymbolInfo(Method)
        else:
            # must be in global scope
            assert self.in_global_scope()
            symbol_info = SymbolInfo(Function)

        # Create a new scope for this function
        scope = self.new_scope(node.name, Function)
        # Add the symbol info to the current scope
        self.current_scope[node.name] = symbol_info
        # Add a fully qualified name to the global symbol table
        self.symbols[scope.namespace] = symbol_info

        ## Add parameters to the function's scope
        #for arg in node.args.args:
        #    self.current_scope[arg.arg] = SymbolInfo()
        #    # Use qualified name for global table
        #    qualified_name = f"{node.name}.{arg.arg}"
        #    self.symbols[qualified_name] = SymbolInfo()

        # Place the functions scope on the scope stack
        self.scopes.append(scope)
        # Visit the function body
        for statement in node.body:
            self.visit(statement)

        # Pop the function's scope from the scope stack
        self.scopes.pop()

    def visit_ClassDef(self, node):
        """Process class definitions."""

        # Based on the current scope, determine the EntityType
        if issubclass(self.current_scope.entity_type, (Function, Class)):
            # this is a function in a function, must be a closure
            symbol_info = SymbolInfo(Closure)
        else:
            # must be in global scope
            assert self.in_global_scope()
            symbol_info = SymbolInfo(Class)

        # Create a new scope for this Class
        scope = self.new_scope(node.name, Class)
        # Add the symbol info to the current scope
        self.current_scope[node.name] = symbol_info
        # Add a fully qualified name to the global symbol table
        self.symbols[scope.namespace] = symbol_info


        # Place the functions scope on the scope stack
        self.scopes.append(scope)
        # Visit the class body
        for statement in node.body:
            self.visit(statement)

        # Pop the classes scope from the scope stack
        self.scopes.pop()


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
