"""Black-Scholes Monte Carlo pricing utilities with safe payoff evaluation."""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping

import numpy as np


class UnsafeExpressionError(ValueError):
    """Raised when a payoff expression contains unsupported constructs."""


class _SafeExpressionValidator(ast.NodeVisitor):
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Num,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.Call,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.IfExp,
    )

    def __init__(self, allowed_names: Iterable[str], allowed_funcs: Mapping[str, object]):
        self.allowed_names = set(allowed_names)
        self.allowed_funcs = set(allowed_funcs)

    def generic_visit(self, node):
        if not isinstance(node, self.allowed_nodes):
            raise UnsafeExpressionError(f"Unsupported expression element: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Name(self, node: ast.Name):
        if node.id not in self.allowed_names and node.id not in self.allowed_funcs:
            raise UnsafeExpressionError(f"Unknown name '{node.id}' in payoff expression")
        return self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if not isinstance(node.func, ast.Name):
            raise UnsafeExpressionError("Only simple function calls are allowed")
        func_name = node.func.id
        if func_name not in self.allowed_funcs:
            raise UnsafeExpressionError(f"Call to disallowed function '{func_name}'")
        return self.generic_visit(node)


ALLOWED_FUNCTIONS: Dict[str, object] = {
    name: getattr(math, name)
    for name in (
        "exp",
        "log",
        "sqrt",
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "fabs",
        "floor",
        "ceil",
        "pow",
    )
}
ALLOWED_FUNCTIONS.update({"max": max, "min": min, "abs": abs})


def _safe_eval(expression: str, variables: Mapping[str, float]) -> float:
    tree = ast.parse(expression, mode="eval")
    validator = _SafeExpressionValidator(variables.keys(), ALLOWED_FUNCTIONS)
    validator.visit(tree)
    compiled = compile(tree, filename="<payoff>", mode="eval")
    env = {**ALLOWED_FUNCTIONS, **variables}
    return float(eval(compiled, {"__builtins__": {}}, env))


@dataclass
class OptionInput:
    spot: float
    strike: float
    rate: float
    volatility: float
    maturity: float
    paths: int = 50_000

    def to_variables(self, terminal_price: float) -> Dict[str, float]:
        return {
            "S0": self.spot,
            "K": self.strike,
            "r": self.rate,
            "sigma": self.volatility,
            "T": self.maturity,
            "ST": terminal_price,
        }


def simulate_terminal_prices(params: OptionInput, rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    drift = (params.rate - 0.5 * params.volatility**2) * params.maturity
    diffusion = params.volatility * math.sqrt(params.maturity) * rng.standard_normal(params.paths)
    return params.spot * np.exp(drift + diffusion)


def price_option(params: OptionInput, payoff_expression: str) -> float:
    terminal_prices = simulate_terminal_prices(params)
    payoffs = np.array(
        [
            _safe_eval(payoff_expression, params.to_variables(float(st)))
            for st in terminal_prices
        ],
        dtype=float,
    )
    discounted = math.exp(-params.rate * params.maturity) * payoffs
    return float(discounted.mean())


def summarize_pricing(params: OptionInput, payoff_expression: str) -> Dict[str, float]:
    price = price_option(params, payoff_expression)
    return {"price": price}
