from typing import Dict, Tuple
import numpy as np
import pandas as pd


def compute_wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calcule le WAPE (Weighted Absolute Percentage Error) en %."""
    total_sales = np.sum(y_true)
    if total_sales == 0:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / total_sales * 100)


def compute_inventory_costs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    unit_cost_overstock: float = 2.0,
    unit_cost_understock: float = 10.0,
) -> Dict[str, float]:
    """Calcule l'impact financier de l'erreur de prévision.

    - Overstock : prévision > réel (coût d'immobilisation / stockage)
    - Understock : prévision < réel (marge perdue / insatisfaction client)
    """
    errors = y_pred - y_true

    overstock_units = np.maximum(0, errors)
    understock_units = np.maximum(0, -errors)

    cost_overstock = float(np.sum(overstock_units * unit_cost_overstock))
    cost_understock = float(np.sum(understock_units * unit_cost_understock))
    total_cost = cost_overstock + cost_understock

    return {
        "total_cost": total_cost,
        "cost_overstock": cost_overstock,
        "cost_understock": cost_understock,
        "units_overstocked": float(np.sum(overstock_units)),
        "units_understocked": float(np.sum(understock_units)),
    }


def optimize_safety_stock_buffer(
    y_true: np.ndarray,
    y_pred_base: np.ndarray,
    cost_overstock: float = 2.0,
    cost_understock: float = 10.0,
    buffer_range: Tuple[float, float, float] = (-0.20, 0.30, 0.01),
) -> Tuple[float, pd.DataFrame]:
    """Trouve le taux de stock de sécurité (buffer) qui minimise le coût financier global.

    Simule des ajustements de -20 % à +30 % sur la prévision brute.
    """
    results = []
    best_buffer = 0.0
    min_cost = float("inf")

    buffers = np.arange(
        buffer_range[0], buffer_range[1] + buffer_range[2], buffer_range[2]
    )

    for buf in buffers:
        y_adjusted = np.maximum(0, y_pred_base * (1 + buf))
        costs = compute_inventory_costs(
            y_true,
            y_adjusted,
            unit_cost_overstock=cost_overstock,
            unit_cost_understock=cost_understock,
        )

        total_cost = costs["total_cost"]
        wape = compute_wape(y_true, y_adjusted)

        results.append(
            {
                "buffer_pct": round(buf * 100, 1),
                "total_cost": total_cost,
                "cost_overstock": costs["cost_overstock"],
                "cost_understock": costs["cost_understock"],
                "wape": round(wape, 2),
            }
        )

        if total_cost < min_cost:
            min_cost = total_cost
            best_buffer = buf

    df_results = pd.DataFrame(results)
    return best_buffer, df_results