"""Riesgo de rotura de stock priorizado por margen. Responsable: Persona 3.

Idea: cobertura = (on_hand + on_order) / demanda semanal esperada, comparada con el lead time.
Prioridad = probabilidad de rotura x margen unitario x demanda -> margen en riesgo (EUR).
Decisión táctica: qué combinaciones tienda-producto reponer esta semana.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _normal_sf(z: np.ndarray) -> np.ndarray:
    """Función de supervivencia 1 - Phi(z) de la normal estándar."""
    vec_erfc = np.vectorize(math.erfc)
    return 0.5 * vec_erfc(z / math.sqrt(2.0))


def stockout_risk(inventory: pd.DataFrame, sales: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Calcula cobertura en semanas, probabilidad de rotura y margen en riesgo (EUR) por tienda y producto."""
    inv = inventory.copy()
    inv["week"] = pd.to_datetime(inv["week"])

    latest_week = inv["week"].max()
    snap = inv[inv["week"] == latest_week].copy()

    sl = sales.copy()
    sl["sale_ts"] = pd.to_datetime(sl["sale_ts"])
    if "channel" in sl.columns:
        sl = sl[sl["channel"] == "tienda"].copy()

    sl["week"] = sl["sale_ts"].dt.to_period("W-SUN").dt.start_time
    max_sale_week = sl["week"].max()
    cutoff_week = max_sale_week - pd.Timedelta(weeks=11)
    recent = sl[sl["week"] >= cutoff_week]

    all_weeks = pd.date_range(cutoff_week, max_sale_week, freq="W-MON")

    weekly_sales = (
        recent.groupby(["store_id", "product_id", "week"])["units"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=all_weeks, fill_value=0)
    )

    # Factor total de tráfico de tienda: consumo total observado en inventario vs ventas identificadas en `sales`
    inv_sorted = inv.sort_values(["store_id", "product_id", "week"])
    inv_diff = -inv_sorted.groupby(["store_id", "product_id"])["on_hand_units"].diff()
    total_inv_drop = float(inv_diff[inv_diff > 0].sum())

    sl_in_inv = sl[(sl["week"] >= inv["week"].min()) & (sl["week"] <= latest_week)]
    total_recorded_units = float(sl_in_inv["units"].sum())
    traffic_factor = float(np.clip(total_inv_drop / max(total_recorded_units, 1.0), 1.0, 30.0))

    scaled_weekly = weekly_sales * traffic_factor
    local_mean = scaled_weekly.mean(axis=1)
    local_std = scaled_weekly.std(axis=1, ddof=0)

    # Suavizado empírico con la velocidad media del producto para evitar ceros por baja muestra
    prod_global_mean = (
        recent.groupby("product_id")["units"].sum() / max(1, len(all_weeks)) * traffic_factor
        / max(1, inv["store_id"].nunique())
    )

    df = snap.merge(local_mean.rename("local_dem").reset_index(), on=["store_id", "product_id"], how="left")
    df = df.merge(local_std.rename("local_std").reset_index(), on=["store_id", "product_id"], how="left")
    df["prod_dem"] = df["product_id"].map(prod_global_mean).fillna(0.5)

    df["expected_weekly_demand"] = (
        0.75 * df["local_dem"].fillna(df["prod_dem"]) + 0.25 * df["prod_dem"]
    ).clip(lower=0.5)
    df["demand_std"] = np.maximum(
        df["local_std"].fillna(0.0), np.sqrt(df["expected_weekly_demand"])
    ).clip(lower=0.5)

    df["available_units"] = df["on_hand_units"] + df["on_order_units"]
    df["coverage_weeks"] = (df["available_units"] / df["expected_weekly_demand"]).round(2)
    df["lead_time_weeks"] = (df["lead_time_days"] / 7.0).round(2)

    mu_lt = df["expected_weekly_demand"] * df["lead_time_weeks"]
    sigma_lt = np.maximum(df["demand_std"] * np.sqrt(df["lead_time_weeks"]), 0.5)
    z = (df["available_units"] - mu_lt) / sigma_lt
    prob = _normal_sf(z.to_numpy())
    prob = np.where(df["available_units"] <= 0, 1.0, prob)
    df["stockout_prob"] = np.round(np.clip(prob, 0.0, 1.0), 4)

    prod_cols = ["product_id", "product_name", "category", "list_price_eur", "unit_cost_eur"]
    df = df.merge(products[prod_cols], on="product_id", how="left")
    df["unit_margin_eur"] = (df["list_price_eur"] - df["unit_cost_eur"]).round(2)
    df["demand_lead_time_units"] = mu_lt.round(2)
    df["margin_at_risk_eur"] = (
        df["stockout_prob"] * df["unit_margin_eur"] * df["demand_lead_time_units"]
    ).round(2)

    target_stock = np.ceil(df["expected_weekly_demand"] * (df["lead_time_weeks"] + 2.0))
    df["reorder_now"] = (df["coverage_weeks"] < df["lead_time_weeks"]) | (df["stockout_prob"] >= 0.50)
    df["recommended_order_units"] = np.where(
        df["reorder_now"], np.maximum(1, (target_stock - df["available_units"]).astype(int)), 0
    )

    cols = [
        "store_id", "product_id", "product_name", "category", "week",
        "on_hand_units", "on_order_units", "lead_time_days", "lead_time_weeks",
        "expected_weekly_demand", "coverage_weeks", "stockout_prob",
        "unit_margin_eur", "demand_lead_time_units", "margin_at_risk_eur",
        "reorder_now", "recommended_order_units",
    ]
    return df[cols].sort_values("margin_at_risk_eur", ascending=False).reset_index(drop=True)
