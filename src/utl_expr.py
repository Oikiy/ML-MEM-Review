"""Restricted arithmetic evaluation for config equations."""

from __future__ import annotations

import ast
import operator
from typing import Any, Mapping

import pandas as pd

_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
}
_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_ALLOWED_NODES = (
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Name,
    ast.Load,
    ast.Constant,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.UAdd,
    ast.USub,
)


def _assert_safe(tree: ast.AST, equation: str) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(
                f"Equation is not restricted arithmetic "
                f"(unsupported {type(node).__name__}): {equation}"
            )
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError(f"Equation constants must be numeric: {equation}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError(f"Equation cannot reference dunder names: {equation}")


def _eval_node(node: ast.AST, names: Mapping[str, Any], equation: str):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in names:
            raise ValueError(f"Unknown name '{node.id}' in equation: {equation}")
        return names[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARYOPS:
        return _UNARYOPS[type(node.op)](_eval_node(node.operand, names, equation))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left = _eval_node(node.left, names, equation)
        right = _eval_node(node.right, names, equation)
        return _BINOPS[type(node.op)](left, right)
    raise ValueError(f"Equation is not restricted arithmetic: {equation}")


def evaluate_arithmetic(equation: str, names: Mapping[str, Any]) -> Any:
    """Evaluate a numeric expression using only + - * / ** and names from ``names``."""
    if not isinstance(equation, str) or not equation.strip():
        raise ValueError("Equation must be a non-empty string.")
    try:
        tree = ast.parse(equation, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid equation syntax: {equation}") from exc
    _assert_safe(tree, equation)
    return _eval_node(tree.body, names, equation)


def evaluate_arithmetic_frame(df: pd.DataFrame, equation: str) -> pd.Series:
    """Evaluate an arithmetic equation against DataFrame columns."""
    names = {str(column): df[column] for column in df.columns}
    result = evaluate_arithmetic(equation, names)
    if isinstance(result, pd.Series):
        return result.reindex(df.index)
    return pd.Series(result, index=df.index)
