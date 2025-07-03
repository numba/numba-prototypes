from __future__ import annotations

import base64
import html
import re
import uuid
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pprint import pformat
from timeit import default_timer as timer

from egglog import EGraph
from IPython.display import HTML, SVG, display

from .notebookutils import IN_NOTEBOOK


def remove_svg_constraints_xml(svg_str):
    # Parse the SVG
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    root = ET.fromstring(svg_str)

    # Remove the attributes
    if "width" in root.attrib:
        del root.attrib["width"]
    if "height" in root.attrib:
        del root.attrib["height"]
    # Convert back to string
    return ET.tostring(root, encoding="unicode")


def egraph_to_svg(egraph: EGraph) -> HTML:
    content = egraph._graphviz()
    svg_raw = content.pipe(format="svg", quiet=True)
    svg_str = (
        svg_raw.decode("utf-8") if isinstance(svg_raw, bytes) else svg_str
    )

    svg_data = svg_str

    # Escape the SVG data properly for JavaScript
    svg_escaped = (
        svg_data.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    )

    return HTML(
        f"""
    <div style="margin: 10px 0;">
        <button onclick="openSVGInNewTab()" style="
            margin-bottom: 10px;
            padding: 8px 16px;
            background: #007cba;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        ">Open Full Size in New Tab</button>

        <div style="
            overflow: auto;
            border: 1px solid #ccc;
            resize: both;
            min-width: 10em;
            min-height: 3em;
        ">
            {svg_data}
        </div>
    </div>

    <script>
    function openSVGInNewTab() {{
        const svgData = `{svg_escaped}`;
        const blob = new Blob([svgData], {{type: 'image/svg+xml;charset=utf-8'}});
        const url = URL.createObjectURL(blob);
        const newWindow = window.open();
        newWindow.location.href = url;

        // Clean up after a delay
        setTimeout(() => URL.revokeObjectURL(url), 2000);
    }}
    </script>
    """
    )


class ReportInterface(ABC):
    """
    Abstract interface for report classes.
    """

    @abstractmethod
    def __init__(self, title="Report", default_expanded=False, **kwargs):
        pass

    @abstractmethod
    def append(self, title, content):
        pass

    @abstractmethod
    def display(self):
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def nest(self, *args, **kwargs):
        pass


class DummyReport(ReportInterface):
    """
    A dummy version of Report that provides the same interface but does nothing.
    Useful for disabling reporting in non-notebook or headless environments.
    """

    def __init__(self, title="", default_expanded=False, **kwargs):
        self.title = title
        self.default_expanded = default_expanded
        self.panes = []
        self.report_id = None

    @contextmanager
    def nest(self, *args, **kwargs):
        yield DummyReport()

    def append(self, title, content):
        pass

    def display(self):
        pass

    def clear(self):
        self.panes = []

    def __len__(self):
        return 0

    def __repr__(self):
        return f"DummyReport()"


class Report(ReportInterface):
    """
    A utility class for creating collapsible panes in Jupyter notebooks.

    Supports both text content and IPython display-able objects (images, plots, etc.).
    """

    Sink = DummyReport

    def __init__(
        self,
        title="Report",
        default_expanded=False,
        enable_nested_metadata=False,
    ):
        """
        Initialize a new Report.

        Args:
            title (str): The main title for the report
            default_expanded (bool): Whether panes should be expanded by default
        """
        self.title = title
        self.default_expanded = default_expanded
        self.enable_nested_metadata = enable_nested_metadata
        self.panes = []
        self.report_id = f"report_{uuid.uuid4().hex[:8]}"
        self._start_time = timer()

    def _compute_metadata(self):
        """Compute metadata as a dict for timing breakdown."""
        last_time = self._start_time
        timing = []
        for pane_info in self.panes:
            title = pane_info["title"]
            end_time = pane_info["end_time"]
            timing.append(
                {
                    "title": title,
                    "elapsed_ms": (end_time - last_time) * 1000,
                }
            )
            last_time = end_time
        total_elapsed = (last_time - self._start_time) * 1000
        return {
            "total_elapsed_ms": total_elapsed,
            "timing": timing,
        }

    def _format_metadata(self, metadata_dict):
        """Format the metadata dict as a string."""
        buf = [
            f"time elapsed {metadata_dict['total_elapsed_ms']:.2f}ms",
            "timing breakdown:",
        ]
        for entry in metadata_dict["timing"]:
            buf.append(f"  {entry['elapsed_ms']:.2f}ms: {entry['title']:20}")
        return "\n".join(buf)

    @contextmanager
    def nest(self, *args, metadata=False, **kwargs) -> "Report":
        report = Report(*args, **kwargs)
        try:
            yield report
        finally:
            if self.enable_nested_metadata or metadata:
                meta = report._compute_metadata()
                elapsed = meta["total_elapsed_ms"]
                report.append("[metadata]", report._format_metadata(meta))
                title = f"{report.title} ({elapsed:.2f}ms)"
            else:
                title = report.title
            self.append(title, report)

    def append(self, title, content):
        pane_id = f"pane_{uuid.uuid4().hex[:8]}"

        # Store original content for terminal display
        if not hasattr(self, "_original_contents"):
            self._original_contents = {}
        self._original_contents[pane_id] = content

        if isinstance(content, EGraph):
            content = egraph_to_svg(content)

        # Process different content types (same as original)
        if isinstance(content, Report):
            # Nested Report support
            html_content = self._render_nested_report(content)
            is_nested_report = True
        elif hasattr(content, "_repr_html_"):
            # IPython display-able object with HTML representation
            html_content = content._repr_html_()
            is_nested_report = False
        elif hasattr(content, "_repr_png_"):
            # Image content
            png_data = content._repr_png_()
            if isinstance(png_data, bytes):
                b64_data = base64.b64encode(png_data).decode("utf-8")
                html_content = f'<img src="data:image/png;base64,{b64_data}" style="max-width: 100%; height: auto;">'
            else:
                html_content = f"<pre>{str(content)}</pre>"
            is_nested_report = False
        elif hasattr(content, "_repr_svg_"):
            # SVG content
            svg_content = content._repr_svg_()
            html_content = svg_content
            is_nested_report = False
        elif hasattr(content, "_repr_image_svg_xml"):  # support graphviz
            # SVG content
            svg_content = content._repr_image_svg_xml()
            html_content = svg_content
            is_nested_report = False
        elif isinstance(content, str):
            # Plain text content - check if it looks like HTML
            if content.strip().startswith("<") and content.strip().endswith(
                ">"
            ):
                html_content = content
            else:
                # Escape HTML and preserve whitespace

                escaped_content = html.escape(content)
                html_content = f'<pre style="white-space: pre-wrap; font-family: monospace; background-color: #2a2a2a; color: #e0e0e0; padding: 10px; border-radius: 4px; overflow-x: auto; border: 1px solid #404040;">{escaped_content}</pre>'
            is_nested_report = False
        else:
            # Fallback to string representation
            escaped_content = html.escape(str(content))
            html_content = f'<pre style="white-space: pre-wrap; font-family: monospace; background-color: #2a2a2a; color: #e0e0e0; padding: 10px; border-radius: 4px; overflow-x: auto; border: 1px solid #404040;">{escaped_content}</pre>'
            is_nested_report = False

        self.panes.append(
            {
                "id": pane_id,
                "title": title,
                "content": html_content,
                "is_nested_report": is_nested_report,
                "fallback_content": str(content),
                "end_time": timer(),
            }
        )

    def _render_nested_report(self, nested_report):
        """Render a nested Report as HTML content."""
        # Generate HTML for the nested report with modified styling
        nested_html = nested_report._generate_nested_html(
            parent_id=self.report_id
        )

        return nested_html

    def _generate_nested_html(self, parent_id):
        """Generate HTML for a nested report with modified styling."""
        # Use a different CSS class prefix to avoid conflicts
        nested_id = f"nested_{self.report_id}"

        # CSS styles for nested report
        css = f"""
        <style>
        .nested-report-container-{nested_id} {{
            font-family: inherit;
            border: 1px solid #555;
            border-radius: 6px;
            margin: 8px 0;
            background-color: #2a2a2a;
            box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);
        }}

        .nested-report-title-{nested_id} {{
            background-color: #3a3a3a;
            padding: 10px 15px;
            margin: 0;
            font-size: 1em;
            font-weight: 600;
            color: #e0e0e0;
            border-bottom: 1px solid #555;
            border-radius: 6px 6px 0 0;
        }}

        .nested-pane-{nested_id} {{
            border-bottom: 1px solid #555;
        }}

        .nested-pane-{nested_id}:last-child {{
            border-bottom: none;
        }}

        .nested-pane-header-{nested_id} {{
            background-color: #2a2a2a;
            padding: 8px 15px;
            cursor: pointer;
            user-select: none;
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: background-color 0.2s ease;
            font-weight: 400;
            color: #c0c0c0;
            font-size: 1em;
        }}

        .nested-pane-header-{nested_id}:hover {{
            background-color: #333;
        }}

        .nested-pane-toggle-{nested_id} {{
            font-size: 1.2em;
            color: #888;
            transition: transform 0.2s ease;
        }}

        .nested-pane-content-{nested_id} {{
            padding: 12px 15px;
            background-color: #252525;
            display: none;
            border-top: 1px solid #555;
            color: #d0d0d0;
            font-size: 0.9em;
        }}

        .nested-pane-content-{nested_id}.expanded {{
            display: block;
        }}

        .nested-pane-toggle-{nested_id}.expanded {{
            transform: rotate(90deg);
        }}
        </style>
        """

        # JavaScript for nested toggle functionality
        js = f"""
        <script>
        function toggleNestedPane_{nested_id}(paneId) {{
            const content = document.getElementById(paneId + '_content');
            const toggle = document.getElementById(paneId + '_toggle');

            if (content.classList.contains('expanded')) {{
                content.classList.remove('expanded');
                toggle.classList.remove('expanded');
            }} else {{
                content.classList.add('expanded');
                toggle.classList.add('expanded');
            }}
        }}
        </script>
        """

        # Generate HTML for each pane in nested report
        panes_html = ""
        for pane in self.panes:
            expanded_class = "expanded" if self.default_expanded else ""
            panes_html += f"""
            <div class="nested-pane-{nested_id}">
                <div class="nested-pane-header-{nested_id}" onclick="toggleNestedPane_{nested_id}('{pane['id']}')">
                    <span>{pane['title']}</span>
                    <span id="{pane['id']}_toggle" class="nested-pane-toggle-{nested_id} {expanded_class}">▶</span>
                </div>
                <div id="{pane['id']}_content" class="nested-pane-content-{nested_id} {expanded_class}">
                    {pane['content']}
                </div>
            </div>
            """

        # Complete nested HTML
        html = f"""
        {css}
        <div class="nested-report-container-{nested_id}">
            <div class="nested-report-title-{nested_id}">{self.title}</div>
            {panes_html}
        </div>
        {js}
        """

        return html

    def _generate_html(self):
        """Generate the HTML for the complete report."""

        # CSS styles
        css = f"""
        <style>
        .report-container-{self.report_id} {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            border: 1px solid #404040;
            border-radius: 8px;
            margin: 10px 0;
            background-color: #1e1e1e;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
            color: #e0e0e0;
            display: flex;
            flex-wrap: wrap;
            gap: 16px;
            padding: 16px;
            width: auto;
            max-width: 100%;
            position: relative;
        }}

        .report-title-{self.report_id} {{
            background-color: #2d2d2d;
            padding: 15px 20px;
            margin: 0 0 10px 0;
            font-size: 1.2em;
            font-weight: 600;
            color: #f0f0f0;
            border-bottom: 1px solid #404040;
            border-radius: 8px 8px 0 0;
            width: 100%;
            flex-basis: 100%;
        }}

        .pane-{self.report_id} {{
            border: 1px solid #404040;
            border-radius: 8px;
            background-color: #232323;
            min-width: 300px;
            max-width: 600px;
            flex: 1 1 300px;
            display: flex;
            flex-direction: column;
            margin-bottom: 0;
            transition: max-width 0.3s, width 0.3s, flex 0.3s, box-shadow 0.3s;
            position: relative;
            z-index: 1;
        }}

        .pane-{self.report_id}.expanded-fullwidth {{
            max-width: 100% !important;
            width: 100% !important;
            flex: 1 1 100%;
            z-index: 2;
            box-shadow: 0 0 0 3px #007cba, 0 2px 8px rgba(0,0,0,0.4);
        }}

        .pane-header-{self.report_id} {{
            background-color: #252525;
            padding: 12px 20px;
            cursor: pointer;
            user-select: none;
            display: flex;
            align-items: center;
            justify-content: space-between;
            transition: background-color 0.2s ease;
            font-weight: 500;
            color: #d0d0d0;
            border-radius: 8px 8px 0 0;
        }}

        .pane-header-{self.report_id}:hover {{
            background-color: #2d2d2d;
        }}

        .pane-toggle-{self.report_id} {{
            font-size: 1.2em;
            color: #a0a0a0;
            transition: transform 0.2s ease;
        }}

        .pane-content-{self.report_id} {{
            padding: 20px;
            background-color: #1a1a1a;
            display: none;
            border-top: 1px solid #404040;
            color: #e0e0e0;
            border-radius: 0 0 8px 8px;
        }}

        .pane-content-{self.report_id}.expanded {{
            display: block;
        }}

        .pane-toggle-{self.report_id}.expanded {{
            transform: rotate(90deg);
        }}

        .expand-btn-{self.report_id} {{
            margin-left: 10px;
            padding: 2px 10px;
            background: #007cba;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 0.95em;
            transition: background 0.2s;
        }}
        .expand-btn-{self.report_id}:hover {{
            background: #005f8a;
        }}
        </style>
        """

        # JavaScript for toggle functionality
        js = f"""
        <script>
        function togglePane_{self.report_id}(paneId) {{
            const content = document.getElementById(paneId + '_content');
            const toggle = document.getElementById(paneId + '_toggle');

            if (content.classList.contains('expanded')) {{
                content.classList.remove('expanded');
                toggle.classList.remove('expanded');
            }} else {{
                content.classList.add('expanded');
                toggle.classList.add('expanded');
            }}
        }}

        function expandPaneFullWidth_{self.report_id}(paneId) {{
            const allPanes = document.querySelectorAll('.pane-{self.report_id}');
            const thisPane = document.getElementById(paneId);
            const expandBtn = document.getElementById(paneId + '_expandbtn');
            const isExpanded = thisPane.classList.contains('expanded-fullwidth');
            if (!isExpanded) {{
                allPanes.forEach(pane => {{
                    pane.classList.remove('expanded-fullwidth');
                }});
                thisPane.classList.add('expanded-fullwidth');
                expandBtn.textContent = 'Collapse';
            }} else {{
                thisPane.classList.remove('expanded-fullwidth');
                expandBtn.textContent = 'Expand';
            }}
        }}
        </script>
        """

        # Generate HTML for each pane
        panes_html = ""
        for idx, pane in enumerate(self.panes, 1):
            expanded_class = "expanded" if self.default_expanded else ""
            pane_number = f"{idx}. "
            pane_id = pane["id"]
            panes_html += f"""
            <div class="pane-{self.report_id}" id="{pane_id}">
                <div class="pane-header-{self.report_id}" onclick="togglePane_{self.report_id}('{pane_id}')">
                    <span>{pane_number}{pane['title']}</span>
                    <span>
                        <button type=\"button\" class=\"expand-btn-{self.report_id}\" id=\"{pane_id}_expandbtn\" onclick=\"event.stopPropagation();expandPaneFullWidth_{self.report_id}('{pane_id}')\">Expand</button>
                        <span id="{pane_id}_toggle" class="pane-toggle-{self.report_id} {expanded_class}">▶</span>
                    </span>
                </div>
                <div id="{pane_id}_content" class="pane-content-{self.report_id} {expanded_class}">
                    {pane['content']}
                </div>
            </div>
            """

        # Complete HTML
        html = f"""
        {css}
        <div class="report-container-{self.report_id}" id="report_container_{self.report_id}">
            <p class="report-title-{self.report_id}">{self.title}</p>
            {panes_html}
        </div>
        {js}
        """

        return html

    def display(self):
        """Display the report in the Jupyter notebook."""
        if IN_NOTEBOOK:
            html_content = self._generate_html()
            return display(HTML(html_content))
        else:
            self._display_terminal()

    def _display_terminal(self, indent_level=0, _stored_contents=None):
        """
        Enhanced terminal display that properly handles nested reports.
        This version requires modifications to the append method to store original objects.
        """
        # Terminal color codes
        BOLD = "\033[1m"
        DIM = "\033[2m"
        RESET = "\033[0m"
        BLUE = "\033[94m"
        GREEN = "\033[92m"
        YELLOW = "\033[93m"
        CYAN = "\033[96m"
        MAGENTA = "\033[95m"

        # Create indentation for nested reports
        base_indent = "  " * indent_level
        content_indent = "  " * (indent_level + 1)

        # Report title with appropriate styling based on nesting level
        if indent_level == 0:
            # Top-level report
            print(f"\n{base_indent}{BOLD}{BLUE}{'=' * 60}{RESET}")
            print(f"{base_indent}{BOLD}{BLUE} {self.title.center(56)} {RESET}")
            print(f"{base_indent}{BOLD}{BLUE}{'=' * 60}{RESET}\n")
        else:
            # Nested report
            separator_length = max(len(self.title) + 4, 20)
            print(
                f"\n{base_indent}{BOLD}{MAGENTA}{'┌' + '─' * separator_length + '┐'}{RESET}"
            )
            print(
                f"{base_indent}{BOLD}{MAGENTA}│ {self.title.ljust(separator_length - 2)} │{RESET}"
            )
            print(
                f"{base_indent}{BOLD}{MAGENTA}{'└' + '─' * separator_length + '┘'}{RESET}"
            )

        # Use stored contents if available, otherwise fall back to current method
        contents_to_use = _stored_contents or getattr(
            self, "_original_contents", {}
        )

        for i, pane in enumerate(self.panes, 1):
            # Pane header with nesting-appropriate styling
            expanded_indicator = "▼" if self.default_expanded else "▶"
            if indent_level == 0:
                print(
                    f"{content_indent}{BOLD}{GREEN}[{i}] {expanded_indicator} {pane['title']}{RESET}"
                )
                print(
                    f"{content_indent}{DIM}{'─' * (len(pane['title']) + 10)}{RESET}"
                )
            else:
                print(
                    f"{content_indent}{BOLD}{YELLOW}[{i}] {expanded_indicator} {pane['title']}{RESET}"
                )
                print(
                    f"{content_indent}{DIM}{'─' * (len(pane['title']) + 8)}{RESET}"
                )

            # Get original content if stored
            original_content = contents_to_use.get(pane["id"])

            if isinstance(original_content, Report):
                # Recursively display nested report
                original_content._display_terminal(
                    indent_level + 1, contents_to_use
                )
            else:
                # Regular content processing
                if original_content is not None:
                    content = str(original_content)
                else:
                    content = pane.get(
                        "fallback_content", str(pane["content"])
                    )

                # Strip HTML tags for better terminal display

                clean_content = re.sub(r"<[^>]+>", "", content)

                # Add indentation for readability
                indented_content = "\n".join(
                    f"{content_indent}  {line}"
                    for line in clean_content.split("\n")
                    if line.strip()
                )
                print(f"{CYAN}{indented_content}{RESET}")

            print()  # Add spacing between panes

    def clear(self):
        """Clear all panes from the report."""
        self.panes = []

    def __len__(self):
        """Return the number of panes in the report."""
        return len(self.panes)

    def __repr__(self):
        return f"Report(title='{self.title}', panes={len(self.panes)})"


def create_nested_demo():
    """Create a demo with nested reports."""
    # Create sub-reports
    config_report = Report("Configuration Details")
    config_report.append(
        "Model Settings",
        """
    Model: GPT-4
    Temperature: 0.7
    Max Tokens: 2048
    """,
    )
    config_report.append(
        "Training Data",
        """
    Dataset: Custom Training Data
    Size: 10GB
    Format: JSON Lines
    """,
    )

    analysis_report = Report("Analysis Results")
    analysis_report.append(
        "Performance Metrics",
        """
    <div style="background: linear-gradient(45deg, #2a2a2a, #1e1e1e); padding: 15px; border-radius: 5px; border: 1px solid #404040; color: #e0e0e0;">
        <h4 style="color: #f0f0f0; margin-top: 0;">Key Metrics</h4>
        <ul style="color: #d0d0d0;">
            <li><strong style="color: #ffffff;">Accuracy:</strong> 94.2%</li>
            <li><strong style="color: #ffffff;">Precision:</strong> 91.8%</li>
            <li><strong style="color: #ffffff;">Recall:</strong> 89.3%</li>
        </ul>
    </div>
    """,
    )
    analysis_report.append(
        "Error Analysis", "Detailed error breakdown and recommendations..."
    )

    # Create main report with nested reports
    main_report = Report("Machine Learning Pipeline Report")
    main_report.append(
        "Overview",
        "This report contains detailed analysis of our ML pipeline.",
    )
    main_report.append("Configuration", config_report)  # Nested report
    main_report.append(
        "LLVM-IR Code",
        """
    define i32 @main() {
    entry:
      %retval = alloca i32, align 4
      store i32 0, i32* %retval, align 4
      ret i32 0
    }
    """,
    )
    main_report.append("Analysis", analysis_report)  # Another nested report
    main_report.append(
        "Conclusions",
        "The pipeline shows excellent performance with room for optimization.",
    )

    return main_report


# Example usage and helper functions
def create_sample_report():
    """Create a sample report for demonstration purposes."""
    report = Report("Sample Analysis Report")

    # Add some sample content
    report.append(
        "Configuration",
        """
    Model: GPT-4
    Temperature: 0.7
    Max Tokens: 2048
    Dataset: Custom Training Data
    """,
    )

    report.append(
        "LLVM-IR Code",
        """
    define i32 @main() {
    entry:
      %retval = alloca i32, align 4
      store i32 0, i32* %retval, align 4
      %call = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([13 x i8], [13 x i8]* @.str, i32 0, i32 0))
      ret i32 0
    }
    """,
    )

    report.append(
        "Analysis Results",
        """
    <div style="background: linear-gradient(45deg, #2a2a2a, #1e1e1e); padding: 15px; border-radius: 5px; border: 1px solid #404040; color: #e0e0e0;">
        <h4 style="color: #f0f0f0; margin-top: 0;">Performance Metrics</h4>
        <ul style="color: #d0d0d0;">
            <li><strong style="color: #ffffff;">Execution Time:</strong> 1.23ms</li>
            <li><strong style="color: #ffffff;">Memory Usage:</strong> 45.2MB</li>
            <li><strong style="color: #ffffff;">Cache Hits:</strong> 98.5%</li>
        </ul>
    </div>
    """,
    )

    return report
