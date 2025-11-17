from __future__ import annotations

import inspect
from types import SimpleNamespace
from typing import (
    Any,
    Callable,
    Dict,
    Union,
    get_args,
    get_origin,
    get_type_hints,
)

from typing_extensions import TypedDict, is_typeddict

from .report import Report

try:
    import graphviz

    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False


class Pipeline:
    """
    An immutable pipeline that chains functions together.
    Each function takes named arguments and returns a TypedDict.
    The pipeline automatically passes only the arguments each function needs.
    Functions must be annotated with return type that is a TypedDict subclass.
    """

    def __init__(
        self,
        *functions: Callable[..., TypedDict],
        report_key: str = "pipeline_report",
    ):
        """
        Initialize pipeline with a sequence of functions.

        Args:
            *functions: Variable number of callable functions that return
            TypedDict subclasses

        Raises:
            TypeError: If any function doesn't have proper TypedDict return
            type annotation
        """
        self._validate_functions(functions)
        self._functions = tuple(functions)
        self._report_key = report_key

    @property
    def last(self):
        return self._functions[-1]

    def replace(self, name, fn) -> "Pipeline":
        """
        Create a new pipeline by replacing a function with a new function by
        name.
        """
        # search for the function with the given name
        for i, func in enumerate(self._functions):
            if getattr(func, "__name__", str(func)) == name:
                return Pipeline(
                    *(self._functions[:i] + (fn,) + self._functions[i + 1 :])
                )
        raise ValueError(f"Function {name} not found in pipeline")

    def trunc(self, name: str) -> Pipeline:
        """
        Create a new pipeline by truncating the pipeline at the given function.
        """
        for i, func in enumerate(self._functions):
            if getattr(func, "__name__", str(func)) == name:
                return Pipeline(*self._functions[:i])
        raise ValueError(f"Function {name} not found in pipeline")

    def _validate_functions(self, functions):
        """
        Validate that all functions have proper TypedDict return type
        annotations.

        Args:
            functions: Tuple of functions to validate

        Raises:
            TypeError: If any function lacks proper TypedDict return type
            annotation
        """
        for func in functions:
            try:
                # Get type hints for the function
                type_hints = get_type_hints(func)
                return_type = type_hints.get("return")

                if return_type is None:
                    raise TypeError(
                        f"Function '{getattr(func, '__name__', str(func))}' "
                        "must have a return type annotation"
                    )

                # Check if return type is a TypedDict
                if not self._is_typeddict_type(return_type):
                    raise TypeError(
                        f"Function '{getattr(func, '__name__', str(func))}' "
                        f"must return a TypedDict subclass, got {return_type}"
                    )

            except AttributeError:
                # Function doesn't support type hints (e.g., built-in functions)
                raise TypeError(
                    f"Function '{getattr(func, '__name__', str(func))}' must "
                    "have return type annotation that is a TypedDict subclass"
                )

    def _is_typeddict_type(self, type_hint) -> bool:
        """
        Check if a type hint represents a TypedDict.

        Args:
            type_hint: The type hint to check

        Returns:
            bool: True if the type hint is a TypedDict, False otherwise
        """
        # Direct TypedDict check
        if is_typeddict(type_hint):
            return True

        # Handle Union types (e.g., Optional[SomeTypedDict])
        origin = get_origin(type_hint)
        if origin is Union:
            args = get_args(type_hint)
            # Check if any of the union members is a TypedDict (excluding None for Optional)
            return any(is_typeddict(arg) for arg in args if arg is not type(None))

        return False

    def _validate_runtime_result(self, func, result, expected_type):
        """
        Validate that the runtime result matches the expected TypedDict
        structure.

        Args:
            func: The function that produced the result result: The actual
            result from the function expected_type: The expected TypedDict type
        """
        func_name = getattr(func, "__name__", str(func))

        if not isinstance(result, dict):
            raise TypeError(
                f"Function {func_name} must return a dict, got {type(result)}"
            )

        # Get the TypedDict type from Union if needed
        actual_typed_dict = expected_type
        if get_origin(expected_type) is Union:
            args = get_args(expected_type)
            typed_dict_types = [arg for arg in args if is_typeddict(arg)]
            if typed_dict_types:
                actual_typed_dict = typed_dict_types[0]  # Use first TypedDict in Union

        if is_typeddict(actual_typed_dict):
            # Get required and optional keys from TypedDict
            annotations = getattr(actual_typed_dict, "__annotations__", {})
            required_keys = getattr(
                actual_typed_dict, "__required_keys__", set(annotations.keys())
            )
            optional_keys = getattr(actual_typed_dict, "__optional_keys__", set())
            all_allowed_keys = set(annotations.keys())

            # Check for missing required keys
            missing_keys = required_keys - set(result.keys())
            if missing_keys:
                raise TypeError(
                    f"Function {func_name} missing required keys: "
                    f"{missing_keys}. Expected TypedDict "
                    f"{actual_typed_dict.__name__} with keys: {annotations}"
                )

            # Check for unexpected keys
            unexpected_keys = set(result.keys()) - all_allowed_keys
            if unexpected_keys:
                raise TypeError(
                    f"Function {func_name} returned unexpected keys: "
                    f"{unexpected_keys}. Expected TypedDict "
                    f"{actual_typed_dict.__name__} with keys: {annotations}"
                )

    def __call__(self, **initial_kwargs) -> SimpleNamespace:
        """
        Execute the pipeline with initial keyword arguments.

        Args:
            **initial_kwargs: Initial keyword arguments to pass to the first
            function

        Returns:
            SimpleNamespace: The final result after all functions have been
            applied
        """
        # Keep track of all arguments available throughout the pipeline
        all_kwargs = initial_kwargs.copy()
        result = initial_kwargs

        for func in self._functions:
            if not isinstance(result, dict):
                raise TypeError(
                    f"Function {getattr(func, '__name__', str(func))} must "
                    "return a dict, got {type(result)}"
                )

            # Merge current result with all accumulated kwargs
            merged_kwargs = {**all_kwargs, **result}

            # Get the function signature to determine what arguments it needs
            sig = inspect.signature(func)
            func_params = set(sig.parameters.keys())

            # Filter kwargs to only include what this function expects
            filtered_kwargs = {
                k: v for k, v in merged_kwargs.items() if k in func_params
            }

            # Call function with only the arguments it needs
            result = func(**filtered_kwargs)

            # Validate runtime return type against TypedDict annotation
            type_hints = get_type_hints(func)
            expected_return_type = type_hints.get("return")
            if expected_return_type:
                self._validate_runtime_result(func, result, expected_return_type)

            # Update the accumulated kwargs with new values from this function
            if isinstance(result, dict):
                all_kwargs.update(result)

        return SimpleNamespace(**all_kwargs)

    def extend(self, *functions: Callable[..., TypedDict]) -> "Pipeline":
        """
        Create a new pipeline by adding functions to the end.
        This method is immutable - it returns a new Pipeline instance.

        Args:
            *functions: Functions to add to the pipeline

        Returns:
            A new Pipeline instance with the additional functions
        """
        return Pipeline(*self._functions, *functions)

    def insert(self, index, fn) -> "Pipeline":
        fns = list(self._functions)
        fns.insert(index, fn)
        return Pipeline(*fns)

    def compose(self, other_pipeline: "Pipeline") -> "Pipeline":
        """
        Create a new pipeline by composing with another pipeline.
        The other pipeline's functions are added to the end.

        Args:
            other_pipeline: Another Pipeline instance to compose with

        Returns:
            A new Pipeline instance with combined functions
        """
        if not isinstance(other_pipeline, Pipeline):
            raise TypeError("Can only compose with another Pipeline instance")
        return Pipeline(*(self._functions + other_pipeline._functions))

    def __len__(self) -> int:
        """Return the number of functions in the pipeline."""
        return len(self._functions)

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        func_names = [getattr(f, "__name__", str(f)) for f in self._functions]
        return f"Pipeline({' -> '.join(func_names)})"

    @property
    def functions(self) -> tuple:
        """Get a tuple of all functions in the pipeline (read-only)."""
        return self._functions

    def describe(self) -> str:
        """
        Generate a static visualization of the pipeline structure and data flow.
        Shows required inputs and outputs based purely on type annotations.

        Returns:
            str: A formatted string showing the pipeline structure
        """
        lines = []
        lines.append("Pipeline Structure Visualization")
        lines.append("=" * 50)

        # Track all fields produced by the pipeline stages
        produced_fields = set()

        # First pass: collect all outputs to determine what's produced
        # internally
        for func in self._functions:
            type_hints = get_type_hints(func)
            return_type = type_hints.get("return", "Any")

            if self._is_typeddict_type(return_type):
                actual_typed_dict = self._extract_typeddict_from_type(return_type)
                if actual_typed_dict:
                    annotations = getattr(actual_typed_dict, "__annotations__", {})
                    produced_fields.update(annotations.keys())

        # Determine pipeline inputs (parameters needed but not produced
        # internally)
        pipeline_required_inputs = set()
        pipeline_optional_inputs = set()

        for func in self._functions:
            sig = inspect.signature(func)
            type_hints = get_type_hints(func)

            for param_name, param in sig.parameters.items():
                if param_name not in produced_fields:
                    # This parameter is not produced by any stage, so it's a
                    # pipeline input
                    param_type = type_hints.get(param_name, "Any")
                    if param.default is inspect.Parameter.empty:  # Required parameter
                        pipeline_required_inputs.add((param_name, param_type))
                    else:  # Optional parameter
                        pipeline_optional_inputs.add(
                            (param_name, param_type, param.default)
                        )

        # Show pipeline inputs
        lines.append(f"\nPipeline Inputs (Required):")
        if pipeline_required_inputs:
            for param_name, param_type in sorted(pipeline_required_inputs):
                param_type_str = self._format_type(param_type)
                lines.append(f"  {param_name}: {param_type_str}")
        else:
            lines.append("  <none>")

        lines.append(f"\nPipeline Inputs (Optional):")
        if pipeline_optional_inputs:
            for param_name, param_type, default_value in sorted(
                pipeline_optional_inputs,
                key=lambda x: x[0],
            ):
                param_type_str = self._format_type(param_type)
                lines.append(f"  {param_name}: {param_type_str} = {default_value}")
        else:
            lines.append("  <none>")

        lines.append(f"\nPipeline Stages:")
        lines.append("-" * 30)

        # Show each stage
        for i, func in enumerate(self._functions, 1):
            func_name = getattr(func, "__name__", str(func))
            lines.append(f"\nStage {i}: {func_name}")

            # Get function signature and type hints
            sig = inspect.signature(func)
            type_hints = get_type_hints(func)

            # Show input parameters
            lines.append(f"  Inputs:")
            for param_name, param in sig.parameters.items():
                param_type = type_hints.get(param_name, "Any")
                param_type_str = self._format_type(param_type)

                # Determine if this is a pipeline input or from previous stage
                if param_name in {name for name, _ in pipeline_required_inputs}:
                    source = "pipeline input"
                elif param_name in {name for name, _, _ in pipeline_optional_inputs}:
                    source = "pipeline input (optional)"
                elif param_name in produced_fields:
                    source = "from previous stage"
                else:
                    source = (
                        "optional parameter"
                        if param.default is not inspect.Parameter.empty
                        else "unknown source"
                    )

                param_info = f"    {param_name}: {param_type_str}"
                if param.default is not inspect.Parameter.empty:
                    param_info += f" = {param.default}"
                param_info += f" ({source})"
                lines.append(param_info)

            # Show return type and output fields
            return_type = type_hints.get("return", "Any")
            return_type_str = self._format_type(return_type)
            lines.append(f"  Returns: {return_type_str}")

            # Show expected output fields if it's a TypedDict
            if self._is_typeddict_type(return_type):
                actual_typed_dict = self._extract_typeddict_from_type(return_type)
                if actual_typed_dict:
                    annotations = getattr(actual_typed_dict, "__annotations__", {})
                    required_keys = getattr(
                        actual_typed_dict,
                        "__required_keys__",
                        set(annotations.keys()),
                    )

                    lines.append(f"  Output Fields:")
                    for field_name, field_type in annotations.items():
                        type_str = self._format_type(field_type)
                        required_str = (
                            "required" if field_name in required_keys else "optional"
                        )
                        lines.append(f"    {field_name}: {type_str} ({required_str})")

        # Show final pipeline outputs
        lines.append(f"\nPipeline Outputs:")
        lines.append("-" * 20)

        final_outputs = set()
        for func in self._functions:
            type_hints = get_type_hints(func)
            return_type = type_hints.get("return", "Any")

            if self._is_typeddict_type(return_type):
                actual_typed_dict = self._extract_typeddict_from_type(return_type)
                if actual_typed_dict:
                    annotations = getattr(actual_typed_dict, "__annotations__", {})
                    for field_name, field_type in annotations.items():
                        field_type_str = self._format_type(field_type)
                        final_outputs.add((field_name, field_type_str))

        if final_outputs:
            for field_name, field_type_str in sorted(final_outputs):
                lines.append(f"  {field_name}: {field_type_str}")
        else:
            lines.append("  <no typed outputs>")

        # Show pipeline summary
        lines.append(f"\nPipeline Summary:")
        lines.append("-" * 20)
        lines.append(f"  Stages: {len(self._functions)}")
        lines.append(f"  Required inputs: {len(pipeline_required_inputs)}")
        lines.append(f"  Optional inputs: {len(pipeline_optional_inputs)}")
        lines.append(f"  Total outputs: {len(final_outputs)}")

        # Data flow analysis
        return "\n".join(lines)

    def visualize(self, title: str = ""):
        """
        Generate a Graphviz visualization of the pipeline structure.

        Args:
            title: Title for the graph

        Returns:
            An object with _repr_svg_() and _repr_html_() for Jupyter display

        Raises:
            ImportError: If graphviz package is not installed
        """
        if not GRAPHVIZ_AVAILABLE:
            raise ImportError(
                "graphviz package is required for visualization. "
                "Install it with: pip install graphviz"
            )

        if not title:
            title = "Pipeline-" + self._functions[-1].__name__

        # Create the graph
        dot = graphviz.Digraph(name=title)

        # Set graph attributes for top-down layout
        dot.attr(rankdir="TB")
        dot.attr("node", shape="box", style="filled")
        dot.attr("edge", fontsize="10")

        # First pass: collect all outputs to determine what's produced internally
        produced_fields = set()
        stage_outputs = {}  # Track which stage produces which fields

        for i, func in enumerate(self._functions, 1):
            type_hints = get_type_hints(func)
            return_type = type_hints.get("return", "Any")
            stage_outputs[i] = set()

            if self._is_typeddict_type(return_type):
                actual_typed_dict = self._extract_typeddict_from_type(return_type)
                if actual_typed_dict:
                    annotations = getattr(actual_typed_dict, "__annotations__", {})
                    produced_fields.update(annotations.keys())
                    stage_outputs[i].update(annotations.keys())

        # Determine pipeline inputs
        pipeline_required_inputs = set()
        pipeline_optional_inputs = set()

        # Functions for each stage of the pipeline
        for func in self._functions:
            sig = inspect.signature(func)
            type_hints = get_type_hints(func)

            for param_name, param in sig.parameters.items():
                if param_name not in produced_fields:
                    param_type = type_hints.get(param_name, "Any")
                    if param.default is inspect.Parameter.empty:
                        pipeline_required_inputs.add((param_name, param_type))
                    else:
                        pipeline_optional_inputs.add(
                            (param_name, param_type, param.default)
                        )

        # Create input nodes and group them at the top
        with dot.subgraph() as inputs:
            inputs.attr(rank="min")  # Force inputs to stay at the top

            for param_name, param_type in sorted(pipeline_required_inputs):
                param_type_str = self._format_type(param_type)
                inputs.node(
                    f"input_{param_name}",  # Use prefixed ID to avoid conflicts
                    label=f"{param_name}\\n{param_type_str}",
                    fillcolor="lightblue",
                )

            for param_name, param_type, default_value in sorted(
                pipeline_optional_inputs, key=lambda x: x[:2]
            ):
                param_type_str = self._format_type(param_type)
                inputs.node(
                    f"input_{param_name}",  # Use prefixed ID to avoid conflicts
                    label=(f"{param_name}\\n{param_type_str}\\n= " f"{default_value}"),
                    fillcolor="lightcyan",
                )

        # Create stage nodes in the middle
        with dot.subgraph() as stages:
            for i, func in enumerate(self._functions, 1):
                func_name = getattr(func, "__name__", str(func))
                type_hints = get_type_hints(func)
                return_type = type_hints.get("return", "Any")
                return_type_str = self._format_type(return_type)

                stage_id = f"stage_{i}"
                label = f"Stage {i}\\n{func_name}\\n→ {return_type_str}"
                stages.node(stage_id, label=label, fillcolor="lightyellow")

        # Create output nodes for final results
        final_outputs = {}  # Use dict to avoid duplicates: field_name -> field_type_str

        # Collect all unique output fields with their types
        for func in self._functions:
            type_hints = get_type_hints(func)
            return_type = type_hints.get("return", "Any")

            if self._is_typeddict_type(return_type):
                actual_typed_dict = self._extract_typeddict_from_type(return_type)
                if actual_typed_dict:
                    annotations = getattr(actual_typed_dict, "__annotations__", {})
                    for field_name, field_type in annotations.items():
                        field_type_str = self._format_type(field_type)
                        final_outputs[field_name] = (
                            field_type_str  # This will overwrite duplicates
                        )

        # Create output nodes and group them at the bottom
        with dot.subgraph() as outputs:
            outputs.attr(rank="max")  # Force outputs to stay at the bottom

            for field_name, field_type_str in sorted(final_outputs.items()):
                outputs.node(
                    f"out_{field_name}",
                    label=f"{field_name}\\n{field_type_str}",
                    fillcolor="lightgreen",
                )

        # Create edges with proper tracking of field versions
        # Track the most recent producer of each field
        field_producers = {}  # field_name -> node_id of most recent producer

        # Initialize with pipeline inputs
        for param_name, _ in pipeline_required_inputs:
            field_producers[param_name] = f"input_{param_name}"
        for param_name, _, _ in pipeline_optional_inputs:
            field_producers[param_name] = f"input_{param_name}"

        for i, func in enumerate(self._functions, 1):
            sig = inspect.signature(func)
            stage_id = f"stage_{i}"

            # Input edges to this stage
            for param_name, param in sig.parameters.items():
                if param_name in field_producers:
                    # Connect from the most recent producer of this field
                    source_node = field_producers[param_name]
                    dot.edge(source_node, stage_id, label=param_name)

            # Update field producers with outputs from this stage
            type_hints = get_type_hints(func)
            return_type = type_hints.get("return", "Any")
            stage_produced_fields = (
                set()
            )  # Track what this stage produces to avoid duplicates

            if self._is_typeddict_type(return_type):
                actual_typed_dict = self._extract_typeddict_from_type(return_type)
                if actual_typed_dict:
                    annotations = getattr(actual_typed_dict, "__annotations__", {})
                    for field_name in annotations.keys():
                        if field_name not in stage_produced_fields:
                            # This stage now becomes the producer of this field
                            field_producers[field_name] = stage_id
                            stage_produced_fields.add(field_name)

        # Output edges from final producers to output nodes
        # Only connect the most recent (final) producer of each field to avoid duplicates
        connected_edges = set()  # Track (source, target, label) to avoid duplicates

        for field_name in final_outputs.keys():
            if field_name in field_producers:
                producer_node = field_producers[field_name]
                target_node = f"out_{field_name}"
                edge_key = (producer_node, target_node, field_name)

                # Only connect if the producer is a stage and we haven't created this edge yet
                if (
                    producer_node.startswith("stage_")
                    and edge_key not in connected_edges
                ):
                    dot.edge(producer_node, target_node, label=field_name)
                    connected_edges.add(edge_key)

        # Add legend at the bottom right
        with dot.subgraph(name="cluster_legend") as legend:
            legend.attr(label="Legend", style="filled", fillcolor="white", rankdir="TB")
            legend.node("legend_input", label="Required Input", fillcolor="lightblue")
            legend.node(
                "legend_optional",
                label="Optional Input",
                fillcolor="lightcyan",
            )
            legend.node(
                "legend_stage",
                label="Processing Stage",
                fillcolor="lightyellow",
            )
            legend.node("legend_output", label="Output", fillcolor="lightgreen")

            # Keep legend nodes in vertical arrangement
            legend.edge("legend_input", "legend_optional", style="invisible")
            legend.edge("legend_optional", "legend_stage", style="invisible")
            legend.edge("legend_stage", "legend_output", style="invisible")

        class _IPythonWrapper:
            def __init__(self, dot):
                self._dot = dot

            def _repr_html_(self):
                svg_str = self._repr_svg_()
                return f'<div style="width: 100%;">{svg_str}</div>'

            def _repr_svg_(self):
                return self._dot._repr_image_svg_xml()

        return _IPythonWrapper(dot)

    def _find_producing_stage(self, field_name: str, max_stage: int) -> int:
        """
        Find which stage produces a given field.

        Args:
            field_name: Name of the field to find
            max_stage: Maximum stage number to search (exclusive)

        Returns:
            Stage number that produces the field, or 0 if not found
        """
        for i in range(1, max_stage + 1):
            func = self._functions[i - 1]  # Convert to 0-based index
            type_hints = get_type_hints(func)
            return_type = type_hints.get("return", "Any")

            if self._is_typeddict_type(return_type):
                actual_typed_dict = self._extract_typeddict_from_type(return_type)
                if actual_typed_dict:
                    annotations = getattr(actual_typed_dict, "__annotations__", {})
                    if field_name in annotations:
                        return i
        return 0

    def _format_type(self, type_hint) -> str:
        """
        Format a type hint into a readable string.

        Args:
            type_hint: The type hint to format

        Returns:
            str: Formatted type string
        """
        if hasattr(type_hint, "__forward_arg__"):
            return type_hint.__forward_arg__
        elif hasattr(type_hint, "__name__"):
            return type_hint.__name__
        elif hasattr(type_hint, "_name"):
            return type_hint._name
        elif is_typeddict(type_hint):
            return getattr(type_hint, "__name__", str(type_hint))
        else:
            return str(type_hint).replace("typing.", "")

    def _extract_typeddict_from_type(self, type_hint):
        """
        Extract the actual TypedDict class from a type hint (handling Union
        types).

        Args:
            type_hint: The type hint that may contain a TypedDict

        Returns:
            The TypedDict class or None
        """
        if is_typeddict(type_hint):
            return type_hint

        # Handle Union types (e.g., Optional[SomeTypedDict])
        origin = get_origin(type_hint)
        if origin is Union:
            args = get_args(type_hint)
            for arg in args:
                if is_typeddict(arg):
                    return arg

        return None
