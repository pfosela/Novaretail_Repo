"""Análisis de cohortes de clientes. Responsable: Persona 3.

Cohorte = mes de alta (signup_date). Métrica = % de clientes que compran en el mes N tras el alta,
segmentado por canal de adquisición. Base para el clasificador de inactividad (entrega 3).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def retention_matrix(orders: pd.DataFrame, customers: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Matriz de retención (%) por mes de alta y meses transcurridos, opcionalmente agrupada por `by`."""
    ord_df = orders.copy()
    if "status" in ord_df.columns:
        ord_df = ord_df[ord_df["status"] != "cancelled"].copy()
    ord_df["created_ts"] = pd.to_datetime(ord_df["created_ts"])
    ord_df["order_month"] = ord_df["created_ts"].dt.to_period("M")

    min_order_month = ord_df["order_month"].min()
    max_order_month = ord_df["order_month"].max()

    cust = customers.copy()
    cust["signup_date"] = pd.to_datetime(cust["signup_date"])
    cust["cohort_month"] = cust["signup_date"].dt.to_period("M")

    cust = cust[(cust["cohort_month"] >= min_order_month) & (cust["cohort_month"] <= max_order_month)].copy()

    group_keys = [by, "cohort_month"] if by else ["cohort_month"]
    cohort_sizes = cust.groupby(group_keys)["customer_id"].nunique().rename("cohort_size")

    merged = ord_df[["customer_id", "order_month"]].merge(
        cust[["customer_id", *group_keys]], on="customer_id", how="inner"
    )
    merged["period_number"] = (merged["order_month"] - merged["cohort_month"]).apply(lambda p: p.n)
    merged = merged[merged["period_number"] >= 0]

    active = (
        merged.groupby([*group_keys, "period_number"])["customer_id"]
        .nunique()
        .rename("active_customers")
        .reset_index()
    )
    active = active.merge(cohort_sizes.reset_index(), on=group_keys, how="left")
    active["retention_pct"] = (active["active_customers"] / active["cohort_size"] * 100.0).round(2)

    max_period = int(active["period_number"].max()) if len(active) else 0
    all_periods = list(range(max_period + 1))

    mat = (
        active.pivot(index=group_keys, columns="period_number", values="retention_pct")
        .reindex(index=cohort_sizes.index, columns=all_periods)
    )

    if by:
        c_months = mat.index.get_level_values("cohort_month")
    else:
        c_months = mat.index

    max_observable = np.array([(max_order_month - cm).n for cm in c_months])
    for p in all_periods:
        observable_mask = p <= max_observable
        col_vals = mat[p].to_numpy(dtype=float)
        col_vals[observable_mask & np.isnan(col_vals)] = 0.0
        col_vals[~observable_mask] = np.nan
        mat[p] = col_vals

    if by:
        mat.index = pd.MultiIndex.from_arrays(
            [mat.index.get_level_values(0), mat.index.get_level_values(1).astype(str)],
            names=[by, "cohort_month"],
        )
    else:
        mat.index = mat.index.astype(str)
        mat.index.name = "cohort_month"

    mat.columns.name = "month_offset"
    return mat
