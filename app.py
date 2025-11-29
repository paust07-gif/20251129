from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any, Dict

from flask import Flask, jsonify, render_template, request

from bs_model import OptionInput, UnsafeExpressionError, price_option

app = Flask(__name__)


def _parse_option_input(data: Dict[str, Any]) -> OptionInput:
    return OptionInput(
        spot=float(data.get("spot", 0.0)),
        strike=float(data.get("strike", 0.0)),
        rate=float(data.get("rate", 0.0)),
        volatility=float(data.get("volatility", 0.0)),
        maturity=float(data.get("maturity", 0.0)),
        paths=int(data.get("paths", 50000)),
    )


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None, error=None)


@app.route("/calculate", methods=["POST"])
def calculate():
    form = request.form
    payoff_expression = form.get("payoff", "max(ST - K, 0)")
    try:
        params = _parse_option_input(form)
        price = price_option(params, payoff_expression)
        result = {
            "input": asdict(params),
            "payoff": payoff_expression,
            "price": price,
        }
        return render_template("index.html", result=result, error=None)
    except (ValueError, UnsafeExpressionError) as exc:
        return render_template("index.html", result=None, error=str(exc)), 400


@app.route("/api/calculate", methods=["POST"])
def api_calculate():
    try:
        payload: Dict[str, Any] = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON body"}), 400

    payoff_expression = payload.get("payoff")
    if not payoff_expression:
        return jsonify({"error": "Missing 'payoff' expression"}), 400

    try:
        params = _parse_option_input(payload)
        price = price_option(params, payoff_expression)
        return jsonify({"price": price, "payoff": payoff_expression, "input": asdict(params)})
    except (ValueError, UnsafeExpressionError) as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
