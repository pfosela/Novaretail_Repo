"""Métricas de flujo a partir del event log. Responsable: Persona 3.

Lee los eventos ya validados (data/clean) y calcula:
- tiempo de cada etapa (created->picked->packed->shipped->delivered), P50 y P90
  por almacén, transportista y mes
- WIP diario (pedidos empezados y no entregados)
- cuello de botella (etapa con mayor P90 / mayor WIP)
- pedidos que requieren intervención hoy (decisión operativa)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STAGES = [("created", "picked"), ("picked", "packed"), ("packed", "shipped"), ("shipped", "delivered")]
STAGE_COLS = [f"{a}_to_{b}" for a, b in STAGES]


def stage_durations(events: pd.DataFrame) -> pd.DataFrame:
    """Una fila por pedido con las horas de cada etapa."""
    df = events.copy()
    df["event_ts"] = pd.to_datetime(df["event_ts"])

    piv = df.pivot_table(index="order_id", columns="activity", values="event_ts", aggfunc="min")

    meta = (
        df.sort_values("event_ts")
        .groupby("order_id", as_index=True)
        .agg(warehouse=("warehouse", "first"), carrier=("carrier", "first"))
    )

    out = meta.reindex(piv.index).copy()
    if "created" in piv.columns:
        out["created_ts"] = piv["created"]
        out["month"] = piv["created"].dt.to_period("M").astype(str)
    else:
        out["created_ts"] = pd.NaT
        out["month"] = pd.NA

    for src, dst in STAGES:
        col = f"{src}_to_{dst}"
        if src in piv.columns and dst in piv.columns:
            out[col] = (piv[dst] - piv[src]).dt.total_seconds() / 3600.0
        else:
            out[col] = np.nan

    if "created" in piv.columns and "delivered" in piv.columns:
        out["total_hours"] = (piv["delivered"] - piv["created"]).dt.total_seconds() / 3600.0
    else:
        out["total_hours"] = np.nan

    return out.reset_index()


def percentiles(durations: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """P50 y P90 de cada etapa agrupado por `by`."""
    cols = [c for c in [*STAGE_COLS, "total_hours"] if c in durations.columns]
    grouped = durations.groupby(by, dropna=False)

    res = grouped[cols[0]].count().rename("n_orders").to_frame()
    for c in cols:
        res[f"{c}_p50"] = grouped[c].quantile(0.50).round(2)
        res[f"{c}_p90"] = grouped[c].quantile(0.90).round(2)

    return res.reset_index()


def daily_wip(events: pd.DataFrame) -> pd.DataFrame:
    """Pedidos en curso (empezados con 'created' y aún no 'delivered') al cierre de cada día."""
    df = events.copy()
    df["event_ts"] = pd.to_datetime(df["event_ts"])

    piv = df.pivot_table(index="order_id", columns="activity", values="event_ts", aggfunc="min")
    if "created" not in piv.columns:
        return pd.DataFrame(columns=["date", "wip_total"])

    wh_map = df.groupby("order_id")["warehouse"].first()
    start_day = piv["created"].dt.normalize()
    max_day = df["event_ts"].max().normalize()

    if "delivered" in piv.columns:
        end_day = piv["delivered"].dt.normalize().fillna(max_day + pd.Timedelta(days=1))
    else:
        end_day = pd.Series(max_day + pd.Timedelta(days=1), index=piv.index)

    days = pd.date_range(start_day.min(), max_day, freq="D")
    day_to_idx = pd.Series(np.arange(len(days)), index=days)

    s_idx = day_to_idx.reindex(start_day).to_numpy()
    e_idx = day_to_idx.reindex(end_day).fillna(len(days)).astype(int).to_numpy()

    def _sweep(mask: np.ndarray) -> np.ndarray:
        diff = np.zeros(len(days) + 1, dtype=int)
        np.add.at(diff, s_idx[mask], 1)
        np.add.at(diff, np.clip(e_idx[mask], 0, len(days)), -1)
        return np.cumsum(diff[:-1])

    out = pd.DataFrame({"date": days, "wip_total": _sweep(np.ones(len(piv), dtype=bool))})
    warehouses = sorted(wh_map.dropna().unique())
    wh_arr = wh_map.reindex(piv.index).to_numpy()
    for wh in warehouses:
        out[wh] = _sweep(wh_arr == wh)

    return out


def orders_needing_action(events: pd.DataFrame, orders: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    """Pedidos activos que han incumplido o van a incumplir la promesa si nadie actúa hoy."""
    today = pd.Timestamp(today)

    ord_df = orders.copy()
    for c in ["created_ts", "promised_ts", "delivered_ts"]:
        ord_df[c] = pd.to_datetime(ord_df[c])

    active = ord_df[
        (ord_df["channel"] != "tienda")
        & (ord_df["created_ts"] <= today)
        & (ord_df["status"].isin(["pending", "processing", "shipped"]))
        & (ord_df["delivered_ts"].isna() | (ord_df["delivered_ts"] > today))
    ].copy()

    if active.empty:
        return pd.DataFrame(
            columns=[
                "order_id", "channel", "warehouse", "carrier", "current_stage",
                "created_ts", "promised_ts", "hours_to_promise", "risk_level", "recommended_action",
            ]
        )

    ev = events.copy()
    ev["event_ts"] = pd.to_datetime(ev["event_ts"])
    ev_until = ev[ev["event_ts"] <= today].sort_values("event_ts")

    last_ev = (
        ev_until.groupby("order_id", as_index=False)
        .agg(
            current_stage=("activity", "last"),
            last_event_ts=("event_ts", "last"),
            warehouse=("warehouse", "first"),
            carrier=("carrier", "first"),
        )
    )

    merged = active.merge(last_ev, on="order_id", how="left")
    merged["current_stage"] = merged["current_stage"].fillna("created")
    merged["hours_to_promise"] = ((merged["promised_ts"] - today).dt.total_seconds() / 3600.0).round(1)

    min_hours_needed = merged["current_stage"].map(
        {"created": 42.0, "picked": 36.0, "packed": 30.0, "shipped": 18.0}
    ).fillna(36.0)

    breached = merged["hours_to_promise"] < 0
    at_risk = (~breached) & (merged["hours_to_promise"] < min_hours_needed)
    flagged = merged[breached | at_risk].copy()

    flagged["risk_level"] = np.where(flagged["hours_to_promise"] < 0, "Vencido", "Riesgo alto")

    def _action(row) -> str:
        st = row["current_stage"]
        if st in ("created", "picked"):
            return f"Priorizar picking/packing urgente en {row['warehouse']} (cola fast-track)"
        if st == "packed":
            return f"Forzar salida en próxima recogida de {row['carrier']} en {row['warehouse']}"
        return f"Escalar incidencia con transportista {row['carrier']} y avisar al cliente"

    flagged["recommended_action"] = flagged.apply(_action, axis=1)

    cols = [
        "order_id", "channel", "warehouse", "carrier", "current_stage",
        "created_ts", "promised_ts", "hours_to_promise", "risk_level", "recommended_action",
    ]
    return flagged[cols].sort_values("hours_to_promise", ascending=True).reset_index(drop=True)
