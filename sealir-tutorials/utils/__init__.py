from .expr_tree_builder import visualize_expr_tree
from .notebookutils import *
from .pipeline import Pipeline
from .report import Report


from egglog import EGraph
from egglog.bindings import RunReport
from IPython.display import HTML



# Report extensions

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

def _egraph_renderer(report, content: EGraph):
    return report._render_content(egraph_to_svg(content))

Report.renderer[EGraph] = _egraph_renderer



def _egraph_runreport_renderer(report, content: RunReport):
    """
    class RunReport:
        updated: bool
        search_time_per_rule: dict[str, timedelta]
        apply_time_per_rule: dict[str, timedelta]
        search_time_per_ruleset: dict[str, timedelta]
        apply_time_per_ruleset: dict[str, timedelta]
        rebuild_time_per_ruleset: dict[str, timedelta]
        num_matches_per_rule: dict[str, int]
    """

    def format_timedelta(td):
        """Format timedelta to a readable string"""
        if td is None:
            return "N/A"
        total_seconds = td.total_seconds()
        if total_seconds < 0.001:
            return f"{total_seconds * 1000000:.1f}μs"
        elif total_seconds < 1:
            return f"{total_seconds * 1000:.1f}ms"
        else:
            return f"{total_seconds:.3f}s"

    def create_table(title, data_dict, value_formatter=None):
        """Create an HTML table for a dictionary"""
        if not data_dict:
            return f"""
            <div style="margin: 15px 0;">
                <h3 style="color: #e1e1e1; margin-bottom: 10px;">{title}</h3>
                <p style="color: #999; font-style: italic;">No data available</p>
            </div>
            """

        if value_formatter is None:
            value_formatter = str

        rows = []
        for key, value in sorted(data_dict.items()):
            formatted_value = value_formatter(value)
            rows.append(f"""
                <tr>
                    <td style="padding: 8px 12px; border-bottom: 1px solid #444; font-family: 'SF Mono', 'Monaco', 'Menlo', 'Consolas', monospace; color: #e1e1e1;">{key}</td>
                    <td style="padding: 8px 12px; border-bottom: 1px solid #444; text-align: right; color: #e1e1e1;">{formatted_value}</td>
                </tr>
            """)

        return f"""
        <div style="margin: 15px 0;">
            <h3 style="color: #e1e1e1; margin-bottom: 10px; font-weight: 600;">{title}</h3>
            <table style="border-collapse: collapse; width: 100%; max-width: 600px; border: 1px solid #555; background-color: #1e1e1e;">
                <thead>
                    <tr style="background-color: #2a2a2a;">
                        <th style="padding: 10px 12px; text-align: left; border-bottom: 2px solid #555; color: #e1e1e1; font-weight: 600;">Item</th>
                        <th style="padding: 10px 12px; text-align: right; border-bottom: 2px solid #555; color: #e1e1e1; font-weight: 600;">Value</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """

    # Build HTML content
    html_parts = []# Add overall status
    html_parts.append(f"""
    <div style="margin: 15px 0; padding: 12px 16px;
                background-color: {'rgba(40, 167, 69, 0.15)' if content.updated else 'rgba(220, 53, 69, 0.15)'};
                border: 1px solid {'#28a745' if content.updated else '#dc3545'};
                border-radius: 6px; color: #e1e1e1;">
        <strong style="color: {'#4ade80' if content.updated else '#f87171'};">Run Status:</strong>
        <span style="color: #e1e1e1;">{'Updated' if content.updated else 'No updates'}</span>
    </div>
    """)

    # Create tables for each dictionary
    html_parts.append(create_table("Search Time Per Rule", content.search_time_per_rule, format_timedelta))
    html_parts.append(create_table("Apply Time Per Rule", content.apply_time_per_rule, format_timedelta))
    html_parts.append(create_table("Search Time Per Ruleset", content.search_time_per_ruleset, format_timedelta))
    html_parts.append(create_table("Apply Time Per Ruleset", content.apply_time_per_ruleset, format_timedelta))
    html_parts.append(create_table("Rebuild Time Per Ruleset", content.rebuild_time_per_ruleset, format_timedelta))
    html_parts.append(create_table("Number of Matches Per Rule", content.num_matches_per_rule))

    # Wrap everything in a container
    full_html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                line-height: 1.4; background-color: #0d1117; padding: 20px; border-radius: 8px;
                border: 1px solid #30363d; color: #e1e1e1;">
        <h2 style="color: #f0f6fc; margin-bottom: 20px; border-bottom: 2px solid #30363d;
                   padding-bottom: 10px; font-weight: 600;">
            EGraph Run Report
        </h2>
        {''.join(html_parts)}
    </div>
    """

    return report._render_content(HTML(full_html))


def _egraph_runreport_renderer_nested(report, content: RunReport):
    """
    Alternative implementation using nested Report objects instead of HTML tables.

    class RunReport:
        updated: bool
        search_time_per_rule: dict[str, timedelta]
        apply_time_per_rule: dict[str, timedelta]
        search_time_per_ruleset: dict[str, timedelta]
        apply_time_per_ruleset: dict[str, timedelta]
        rebuild_time_per_ruleset: dict[str, timedelta]
        num_matches_per_rule: dict[str, int]
    """

    def format_timedelta(td):
        """Format timedelta to a readable string"""
        if td is None:
            return "N/A"
        total_seconds = td.total_seconds()

        if total_seconds < 0.000001:
            return f"{total_seconds * 1000000000:.1f}ns"
        elif total_seconds < 0.001:
            return f"{total_seconds * 1000000:.1f}μs"
        elif total_seconds < 1:
            return f"{total_seconds * 1000:.1f}ms"
        else:
            return f"{total_seconds:.3f}s"

    def add_dict_to_report(parent_report, title, data_dict, value_formatter=None):
        """Add a dictionary as a nested report"""
        if value_formatter is None:
            value_formatter = str

        section_report = Report(title, default_expanded=True)

        for key, value in sorted(data_dict.items(), key=lambda x: x[1], reverse=True):
            formatted_value = value_formatter(value)
            section_report.append(key, formatted_value)

        parent_report.append(title, section_report)

    # Import Report class

    # Create main report
    main_report = Report("EGraph Run Report")

    # Add overall status
    main_report.append("Updated", content.updated)

    # Add each section as a nested report
    add_dict_to_report(main_report, "Search Time Per Rule", content.search_time_per_rule, format_timedelta)
    add_dict_to_report(main_report, "Apply Time Per Rule", content.apply_time_per_rule, format_timedelta)
    add_dict_to_report(main_report, "Search Time Per Ruleset", content.search_time_per_ruleset, format_timedelta)
    add_dict_to_report(main_report, "Apply Time Per Ruleset", content.apply_time_per_ruleset, format_timedelta)
    add_dict_to_report(main_report, "Rebuild Time Per Ruleset", content.rebuild_time_per_ruleset, format_timedelta)
    add_dict_to_report(main_report, "Number of Matches Per Rule", content.num_matches_per_rule)

    return report._render_content(main_report)

Report.renderer[RunReport] = _egraph_runreport_renderer_nested