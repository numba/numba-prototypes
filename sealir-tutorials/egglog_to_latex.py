from typing import List, Union
from egglog import EGraph


def tokenize(egglog_str: str) -> List[str]:
    """
    Splits an Egglog S-expression string into a flat list of tokens.
    Tokens are either "(" or ")", or atoms (any sequence of non-whitespace, non-parenthesis chars).
    """
    tokens = []
    i = 0
    while i < len(egglog_str):
        c = egglog_str[i]
        if c.isspace():
            i += 1
            continue
        if c in ("(", ")"):
            tokens.append(c)
            i += 1
        else:
            j = i
            while j < len(egglog_str) and not egglog_str[j].isspace() and egglog_str[j] not in ("(", ")"):
                j += 1
            tokens.append(egglog_str[i:j])
            i = j
    return tokens


def parse_sexps(tokens: List[str]) -> List[Union[str, list]]:
    """
    Parses a flat list of tokens into a nested list of S-expression forms.
    Each form is either an atom (string) or a list whose first element is the head.
    Returns a flat list of top-level S-expressions (each itself a nested list).
    """
    stack: List[List] = []
    current: List[Union[str, list]] = []
    for tok in tokens:
        if tok == "(":
            stack.append(current)
            current = []
        elif tok == ")":
            completed = current
            current = stack.pop()
            current.append(completed)
        else:
            current.append(tok)
    # If the entire parse wrapped everything in a single list, unwrap it:
    if len(current) == 1 and isinstance(current[0], list):
        return current[0]
    return current


def sexp_to_string(sexp):
    """Convert parsed S-expression back to original string format"""
    if isinstance(sexp, str):
        return sexp
    elif isinstance(sexp, list):
        inner = ' '.join(sexp_to_string(item) for item in sexp)
        return f"({inner})"
    else:
        return str(sexp)


LATEX_ESCAPE = str.maketrans({
    "_": r"\_",
    "#": r"\#",
    "$": r"\$",
    "%": r"\%",
    "&": r"\&",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\^{}",
    "\\": r"\\",
})


def _atom_tex(a: str) -> str:
    try:
        float(a)        # leave numerics bare
        return a
    except ValueError:
        return r"\text{" + a.translate(LATEX_ESCAPE) + "}"


INFIX_OPS = {"=", "!=", "<", "<=", ">", ">=", "+", "-", "*", "/", "%", "**"}


def _sexp_tex(x) -> str:
    if isinstance(x, str):
        return _atom_tex(x)

    head, *args = x

    # infix pretty-printing for common binary ops
    if head in INFIX_OPS and len(args) == 2:
        return f"{_sexp_tex(args[0])} {head} {_sexp_tex(args[1])}"

    return (
        r"\text{" + head.translate(LATEX_ESCAPE) + "}"
        + "(" + ", ".join(_sexp_tex(a) for a in args) + ")"
    )


def _is_set_expr(x):
    # Detects if x is a set-like S-expression: ['set', lhs, rhs]
    return isinstance(x, list) and len(x) == 3 and x[0] == "set"


def _set_tex(x):
    # Renders set(lhs, rhs) as lhs \to rhs
    return f"{_sexp_tex(x[1])} \\to {_sexp_tex(x[2])}"


def to_latex(sexp):
    """
    Render (rewrite …) or (rule …) as KaTeX-safe LaTeX.
    """
    if not (isinstance(sexp, list) and sexp):
        return None

    tag = sexp[0]

    # ───────────────  REWRITE  ────────────────
    if tag == "rewrite" and len(sexp) >= 3:
        lhs, rhs = sexp[1], sexp[2]

        # harvest optional :when clause (list of conditions)
        when_conds = []
        i = 3
        while i < len(sexp):
            if sexp[i] == ":when" and i + 1 < len(sexp):
                when_conds = sexp[i + 1]          # list of cond S-exps
                break
            i += 1                                # <- step only ONE token

        lhs_tex = _sexp_tex(lhs)
        rhs_tex = _sexp_tex(rhs)

        cond_tex = ""
        if when_conds:
            joined = r",\; ".join(_sexp_tex(c) for c in when_conds)
            cond_tex = rf",\; {joined}"

        num = rf"\text{{expr}} = {lhs_tex}{cond_tex}"
        den = rf"\text{{expr}} \to {rhs_tex}"

        return rf"\frac{{{num}}}{{{den}}}"

    # ────────────────  RULE  ─────────────────
    if tag == "rule" and len(sexp) >= 3:
        premises, conclusions = sexp[1], sexp[2]

        def render_stack(exprs):
            lines = []
            for e in exprs:
                if _is_set_expr(e):
                    lines.append(_set_tex(e))
                else:
                    lines.append(_sexp_tex(e))
            return r"\\ ".join(lines)

        prem_tex  = render_stack(premises)
        concl_tex = render_stack(conclusions)

        num = rf"\begin{{array}}{{c}}{prem_tex}\end{{array}}"
        den = rf"\begin{{array}}{{c}}{concl_tex}\end{{array}}"

        return rf"\frac{{{num}}}{{{den}}}"

    return None


def visualize_ruleset_latex(ruleset, verbose=True):
    """
    Visualize an egglog ruleset by converting it to LaTeX representation.
    Only works in notebook environments.

    Args:
        ruleset: The egglog ruleset to visualize
        verbose: If True, prints the original S-expression before LaTeX display

    Returns:
        None, but displays LaTeX representation if in notebook environment
    """
    try:
        shell = get_ipython().__class__.__name__
        is_notebook = shell == "ZMQInteractiveShell"
    except NameError:
        is_notebook = False

    if not is_notebook:
        return

    # Create demo egraph and run ruleset
    demo_egraph = EGraph(save_egglog_string=True)
    demo_egraph.run(ruleset)
    egglog_str = demo_egraph.as_egglog_string

    # Parse into S-expressions
    tokens = tokenize(egglog_str)
    sexps = parse_sexps(tokens)

    from IPython.display import display, Math

    for sexp in sexps:
        tex = to_latex(sexp)
        if tex:
            if verbose:
                print(sexp_to_string(sexp))
            display(Math(tex))
            if verbose:
                print()
