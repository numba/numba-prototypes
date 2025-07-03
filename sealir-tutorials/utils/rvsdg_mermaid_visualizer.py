#!/usr/bin/env python3
"""
RVSDG to Mermaid Diagram Converter

Converts sealir RVSDG expressions (from restructure_source) into clean Mermaid flowchart diagrams.
Provides simple, readable visualization of control flow and data dependencies.
"""

import base64
import uuid
from typing import Dict, List, Set, Tuple

from IPython.display import HTML
from sealir import ase, rvsdg


def quote(s):
    """Return the string s wrapped in double quotes."""
    return f'"{s}"'


class RVSDGMermaidConverter:
    """Convert RVSDG AST expressions to Mermaid flowchart syntax"""

    def __init__(self):
        self.node_counter = 0
        self.node_map: Dict[ase.SExpr, str] = {}  # expr -> mermaid_id
        self.edges: List[Tuple[str, str, str]] = []  # (from, to, label)
        self.visited: Set[ase.SExpr] = set()
        self.region_counter = 0

    def get_node_id(self, expr: ase.SExpr) -> str:
        """Get or create a unique node ID for an expression"""
        if expr not in self.node_map:
            self.node_counter += 1
            self.node_map[expr] = f"n{self.node_counter}"
        return self.node_map[expr]

    def get_node_label(self, expr: ase.SExpr) -> str:
        """Generate a clean label for a node"""
        head = expr._head

        # Special formatting for common RVSDG constructs
        if head == "Func":
            fname = getattr(expr, "fname", "func")
            return f"Function({fname})"
        elif head == "RegionBegin":
            inports = getattr(expr, "inports", ())
            return f"Region Start\\n({', '.join(inports)})"
        elif head == "RegionEnd":
            return "Region End"
        elif head == "IfElse":
            return "If-Else"
        elif head == "Loop":
            return "Loop"
        elif head == "PyForLoop":
            return "For Loop"
        elif head == "IO":
            return "IO State"
        elif head == "Undef":
            name = (
                getattr(expr, "name", "?")
                if hasattr(expr, "name")
                else expr._args[0] if expr._args else "?"
            )
            return f"Undef({name})"
        elif head == "ArgRef":
            idx = getattr(expr, "idx", "?") if hasattr(expr, "idx") else "?"
            name = getattr(expr, "name", "?") if hasattr(expr, "name") else "?"
            return f"Arg({idx}:{name})"
        elif head == "Unpack":
            idx = getattr(expr, "idx", "?") if hasattr(expr, "idx") else "?"
            return f"Unpack[{idx}]"
        elif head == "Port":
            name = getattr(expr, "name", "?") if hasattr(expr, "name") else "?"
            return f"Port({name})"
        elif head == "DbgValue":
            name = getattr(expr, "name", "?") if hasattr(expr, "name") else "?"
            return f"Var({name})"
        # Python operations
        elif head == "PyBinOp":
            op = getattr(expr, "op", "?") if hasattr(expr, "op") else "?"
            return f"BinOp({op})"
        elif head == "PyBinOpPure":
            op = getattr(expr, "op", "?") if hasattr(expr, "op") else "?"
            return f"BinOp({op})"
        elif head == "PyUnaryOp":
            op = getattr(expr, "op", "?") if hasattr(expr, "op") else "?"
            return f"UnaryOp({op})"
        elif head == "PyCall":
            return "Call"
        elif head == "PyCallPure":
            return "Call"
        elif head == "PyLoadGlobal":
            name = getattr(expr, "name", "?") if hasattr(expr, "name") else "?"
            return f"Global({name})"
        elif head == "PyAttr":
            attr = (
                getattr(expr, "attrname", "?")
                if hasattr(expr, "attrname")
                else "?"
            )
            return f"Attr(.{attr})"
        elif head == "PySubscript":
            return "Subscript"
        # Literals
        elif head == "PyInt":
            value = (
                getattr(expr, "value", "?")
                if hasattr(expr, "value")
                else expr._args[0] if expr._args else "?"
            )
            return f"Int({value})"
        elif head == "PyFloat":
            value = (
                getattr(expr, "value", "?")
                if hasattr(expr, "value")
                else expr._args[0] if expr._args else "?"
            )
            return f"Float({value})"
        elif head == "PyBool":
            value = (
                getattr(expr, "value", "?")
                if hasattr(expr, "value")
                else expr._args[0] if expr._args else "?"
            )
            return f"Bool({value})"
        elif head == "PyStr":
            value = (
                getattr(expr, "value", "?")
                if hasattr(expr, "value")
                else expr._args[0] if expr._args else "?"
            )
            return f"Str('{value}')"
        elif head == "PyNone":
            return "None"
        elif head == "PyTuple":
            return "Tuple"
        elif head == "PyList":
            return "List"
        else:
            # Generic formatting
            return head.replace("Py", "")

    def get_node_shape(self, expr: ase.SExpr) -> str:
        """Determine the Mermaid shape for a node based on its type"""
        head = expr._head
        label = quote(self.get_node_label(expr))

        if head == "Func":
            return f"[{label}]"  # Rectangle for functions
        elif head in ["RegionBegin", "RegionEnd"]:
            return f"[{label}]"  # Rectangle for regions
        elif head in ["IfElse", "Loop", "PyForLoop"]:
            return f"{{{label}}}"  # Rhombus for control flow
        elif head in ["PyInt", "PyFloat", "PyBool", "PyStr", "PyNone"]:
            return f"([{label}])"  # Stadium for literals
        elif head in ["ArgRef", "Undef"]:
            return f">{label}]"  # Flag for parameters/undefined
        elif head == "IO":
            return f"({label})"  # Circle for IO state
        elif head in [
            "PyBinOp",
            "PyBinOpPure",
            "PyUnaryOp",
            "PyCall",
            "PyCallPure",
        ]:
            return f"[{label}]"  # Rectangle for operations
        elif head == "DbgValue":
            return f"[{label}]"  # Rectangle for variables
        else:
            return f"[{label}]"  # Default rectangle

    def convert_to_mermaid(self, root_expr: ase.SExpr) -> str:
        """Convert an RVSDG expression to Mermaid flowchart"""
        self.reset()

        # Walk the expression tree and build the graph
        self._build_graph(root_expr)

        # Generate Mermaid syntax
        mermaid_lines = ["flowchart TD"]

        # Add nodes with shapes
        for expr, node_id in self.node_map.items():
            shape = self.get_node_shape(expr)
            mermaid_lines.append(f"    {node_id}{shape}")

        # Add edges
        for from_id, to_id, label in self.edges:
            if label:
                mermaid_lines.append(f"    {from_id} -->|{label}| {to_id}")
            else:
                mermaid_lines.append(f"    {from_id} --> {to_id}")

        # Add styling
        mermaid_lines.extend(
            [
                "",
                "    %% Styling",
                "    classDef function fill:#e3f2fd,stroke:#1976d2,stroke-width:2px",
                "    classDef control fill:#e1f5fe,stroke:#01579b,stroke-width:2px",
                "    classDef operation fill:#f3e5f5,stroke:#4a148c,stroke-width:2px",
                "    classDef literal fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px",
                "    classDef param fill:#fff3e0,stroke:#e65100,stroke-width:2px",
                "    classDef io fill:#fce4ec,stroke:#ad1457,stroke-width:2px",
                "    classDef variable fill:#f1f8e9,stroke:#33691e,stroke-width:2px",
            ]
        )

        return "\n".join(mermaid_lines)

    def _build_graph(
        self, expr: ase.SExpr, parent_id: str = None, edge_label: str = ""
    ):
        """Recursively build the graph structure"""
        if expr in self.visited:
            # Handle cycles by just creating an edge
            if parent_id:
                node_id = self.get_node_id(expr)
                self.edges.append((parent_id, node_id, edge_label))
            return

        self.visited.add(expr)
        node_id = self.get_node_id(expr)

        # Add edge from parent if exists
        if parent_id:
            self.edges.append((parent_id, node_id, edge_label))

        # Process children based on RVSDG structure
        self._process_children(expr, node_id)

    def _process_children(self, expr: ase.SExpr, node_id: str):
        """Process children of an RVSDG node"""
        head = expr._head

        # Handle specific RVSDG constructs
        if head == "Func":
            # Function has args and body
            if hasattr(expr, "args") and isinstance(expr.args, ase.SExpr):
                self._build_graph(expr.args, node_id, "args")
            if hasattr(expr, "body") and isinstance(expr.body, ase.SExpr):
                self._build_graph(expr.body, node_id, "body")

        elif head == "RegionEnd":
            # RegionEnd has begin and ports
            if hasattr(expr, "begin") and isinstance(expr.begin, ase.SExpr):
                self._build_graph(expr.begin, node_id, "begin")
            if hasattr(expr, "ports"):
                for i, port in enumerate(expr.ports):
                    if isinstance(port, ase.SExpr):
                        self._build_graph(port, node_id, f"port{i}")

        elif head == "IfElse":
            # IfElse has cond, body, orelse, operands
            if hasattr(expr, "cond") and isinstance(expr.cond, ase.SExpr):
                self._build_graph(expr.cond, node_id, "cond")
            if hasattr(expr, "body") and isinstance(expr.body, ase.SExpr):
                self._build_graph(expr.body, node_id, "then")
            if hasattr(expr, "orelse") and isinstance(expr.orelse, ase.SExpr):
                self._build_graph(expr.orelse, node_id, "else")
            if hasattr(expr, "operands"):
                for i, op in enumerate(expr.operands):
                    if isinstance(op, ase.SExpr):
                        self._build_graph(op, node_id, f"op{i}")

        elif head in ["Loop", "PyForLoop"]:
            # Loop has body and operands
            if hasattr(expr, "body") and isinstance(expr.body, ase.SExpr):
                self._build_graph(expr.body, node_id, "body")
            if hasattr(expr, "operands"):
                for i, op in enumerate(expr.operands):
                    if isinstance(op, ase.SExpr):
                        self._build_graph(op, node_id, f"op{i}")

        elif head == "Port":
            # Port has value
            if hasattr(expr, "value") and isinstance(expr.value, ase.SExpr):
                self._build_graph(expr.value, node_id, "value")

        elif head == "DbgValue":
            # DbgValue has value
            if hasattr(expr, "value") and isinstance(expr.value, ase.SExpr):
                self._build_graph(expr.value, node_id, "value")

        elif head == "Unpack":
            # Unpack has val
            if hasattr(expr, "val") and isinstance(expr.val, ase.SExpr):
                self._build_graph(expr.val, node_id, "val")

        elif head in ["PyBinOp", "PyBinOpPure"]:
            # Binary operations have io, lhs, rhs
            if hasattr(expr, "io") and isinstance(expr.io, ase.SExpr):
                self._build_graph(expr.io, node_id, "io")
            if hasattr(expr, "lhs") and isinstance(expr.lhs, ase.SExpr):
                self._build_graph(expr.lhs, node_id, "lhs")
            if hasattr(expr, "rhs") and isinstance(expr.rhs, ase.SExpr):
                self._build_graph(expr.rhs, node_id, "rhs")

        elif head == "PyUnaryOp":
            # Unary operations have io, operand
            if hasattr(expr, "io") and isinstance(expr.io, ase.SExpr):
                self._build_graph(expr.io, node_id, "io")
            if hasattr(expr, "operand") and isinstance(
                expr.operand, ase.SExpr
            ):
                self._build_graph(expr.operand, node_id, "operand")

        elif head in ["PyCall", "PyCallPure"]:
            # Calls have func, io, args
            if hasattr(expr, "func") and isinstance(expr.func, ase.SExpr):
                self._build_graph(expr.func, node_id, "func")
            if hasattr(expr, "io") and isinstance(expr.io, ase.SExpr):
                self._build_graph(expr.io, node_id, "io")
            if hasattr(expr, "args"):
                for i, arg in enumerate(expr.args):
                    if isinstance(arg, ase.SExpr):
                        self._build_graph(arg, node_id, f"arg{i}")

        elif head == "PyLoadGlobal":
            # LoadGlobal has io
            if hasattr(expr, "io") and isinstance(expr.io, ase.SExpr):
                self._build_graph(expr.io, node_id, "io")

        elif head in ["PyTuple", "PyList"]:
            # Collections have elems
            if hasattr(expr, "elems"):
                for i, elem in enumerate(expr.elems):
                    if isinstance(elem, ase.SExpr):
                        self._build_graph(elem, node_id, f"elem{i}")

        else:
            # Generic fallback - process all SExpr args
            for i, arg in enumerate(expr._args):
                if isinstance(arg, ase.SExpr):
                    label = f"arg{i}"
                    self._build_graph(arg, node_id, label)

    def reset(self):
        """Reset converter state for new conversion"""
        self.node_counter = 0
        self.node_map.clear()
        self.edges.clear()
        self.visited.clear()
        self.region_counter = 0


def visualize_rvsdg_mermaid(rvsdg_expr: ase.SExpr) -> str:
    """Convert an RVSDG expression to Mermaid diagram and return an HTML button to view it on kroki.io using base64-encoded URL."""
    converter = RVSDGMermaidConverter()
    mermaid_string = converter.convert_to_mermaid(rvsdg_expr)
    return render_mermaid_popup(mermaid_string, button_text="Open Flowchart")


def render_mermaid_popup(
    mermaid_code, theme="default", button_text="View Diagram"
):
    """Create a button that opens mermaid diagram in a new window, with the diagram taking up 90% of the screen."""
    button_id = f"btn-{uuid.uuid4().hex[:8]}"
    diagram_id = f"diagram-{uuid.uuid4().hex[:8]}"

    # Escape any backticks and quotes in the mermaid code
    escaped_code = (
        mermaid_code.replace("`", "\\`")
        .replace('"', '\\"')
        .replace("'", "\\'")
    )

    # Create the HTML content for the new window
    new_window_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mermaid Diagram</title>
        <style>
            html, body {{
                height: 100%;
                margin: 0;
                padding: 0;
            }}
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                background: #f5f5f5;
                display: flex;
                flex-direction: column;
                align-items: center;
                height: 100vh;
                width: 100vw;
            }}
            .container {{
                background: white;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                width: 90vw;
                height: 90vh;
                max-width: 90vw;
                max-height: 90vh;
                display: flex;
                flex-direction: column;
                overflow: hidden;
            }}
            .controls {{
                margin-bottom: 10px;
                text-align: center;
            }}
            .diagram {{
                flex: 1 1 auto;
                width: 100%;
                height: 100%;
                overflow: auto;
                display: flex;
                align-items: center;
                justify-content: center;
            }}
            button {{
                background: #007cba;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                cursor: pointer;
                margin: 0 5px;
            }}
            button:hover {{
                background: #005a8a;
            }}
            svg {{
                max-width: 100%;
                max-height: 100%;
                height: auto;
                width: auto;
                display: block;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="controls">
                <button onclick="downloadSVG()">Download SVG</button>
                <button onclick="window.print()">Print</button>
                <button onclick="window.close()">Close</button>
            </div>
            <div id="{diagram_id}" class="diagram">
                <div>Loading diagram...</div>
            </div>
        </div>

        <script type="module">
            async function renderDiagram() {{
                try {{
                    const mermaid = await import('https://cdn.skypack.dev/mermaid@10.6.1');

                    mermaid.default.initialize({{
                        startOnLoad: false,
                        theme: '{theme}',
                        securityLevel: 'loose',
                        fontFamily: 'Arial, sans-serif'
                    }});

                    const container = document.getElementById('{diagram_id}');
                    const diagramCode = `{escaped_code}`;

                    const {{svg}} = await mermaid.default.render('diagram-render-{diagram_id}', diagramCode);
                    container.innerHTML = svg;

                    // Store SVG for download
                    window.currentSVG = svg;

                }} catch (error) {{
                    console.error('Mermaid rendering error:', error);
                    const container = document.getElementById('{diagram_id}');
                    container.innerHTML = `<p style="color: red;">Error: ${{error.message}}</p>`;
                }}
            }}

            window.downloadSVG = function() {{
                if (window.currentSVG) {{
                    const blob = new Blob([window.currentSVG], {{type: 'image/svg+xml'}});
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = 'mermaid-diagram.svg';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                }}
            }}

            renderDiagram();
        </script>
    </body>
    </html>
    """

    # Encode the HTML content
    encoded_html = base64.b64encode(new_window_html.encode("utf-8")).decode(
        "utf-8"
    )

    # Create the button HTML
    button_html = f"""
    <div style="margin: 10px 0;">
        <button id="{button_id}"
                style="background: #007cba; color: white; border: none; padding: 10px 20px;
                       border-radius: 4px; cursor: pointer; font-size: 14px;">
            {button_text}
        </button>
    </div>

    <script>
        document.getElementById('{button_id}').onclick = function() {{
            const htmlContent = atob('{encoded_html}');
            const newWindow = window.open('', '_blank', 'width=1200,height=800,scrollbars=yes,resizable=yes');
            newWindow.document.write(htmlContent);
            newWindow.document.close();
        }}
    </script>
    """

    return HTML(button_html)
