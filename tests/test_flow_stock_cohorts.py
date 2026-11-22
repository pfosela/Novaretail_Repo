import pandas as pd
from novaretail import cohorts, flow, stock


def test_flow_metrics_and_action():
    events = pd.DataFrame([
        {"order_id": "NR0000001", "activity": "created", "event_ts": "2026-10-28 10:00:00", "resource": "web", "warehouse": "WH-BCN", "carrier": "RapidLog"},
        {"order_id": "NR0000001", "activity": "picked", "event_ts": "2026-10-28 14:00:00", "resource": "PK01", "warehouse": "WH-BCN", "carrier": "RapidLog"},
        {"order_id": "NR0000001", "activity": "packed", "event_ts": "2026-10-29 10:00:00", "resource": "EM01", "warehouse": "WH-BCN", "carrier": "RapidLog"},
        {"order_id": "NR0000001", "activity": "shipped", "event_ts": "2026-10-29 12:00:00", "resource": "RapidLog", "warehouse": "WH-BCN", "carrier": "RapidLog"},
        {"order_id": "NR0000001", "activity": "delivered", "event_ts": "2026-10-30 12:00:00", "resource": "RapidLog", "warehouse": "WH-BCN", "carrier": "RapidLog"},
        {"order_id": "NR0000002", "activity": "created", "event_ts": "2026-10-29 09:00:00", "resource": "web", "warehouse": "WH-MAD", "carrier": "EnviaYa"},
        {"order_id": "NR0000002", "activity": "picked", "event_ts": "2026-10-29 12:00:00", "resource": "PK02", "warehouse": "WH-MAD", "carrier": "EnviaYa"},
    ])
    dur = flow.stage_durations(events)
    assert len(dur) == 2
    assert dur.loc[dur["order_id"] == "NR0000001", "created_to_picked"].iloc[0] == 4.0
    assert dur.loc[dur["order_id"] == "NR0000001", "picked_to_packed"].iloc[0] == 20.0

    perc = flow.percentiles(dur, by=["warehouse"])
    assert {"WH-BCN", "WH-MAD"} == set(perc["warehouse"])

    wip = flow.daily_wip(events)
    assert wip["wip_total"].max() >= 1

    orders = pd.DataFrame([
        {"order_id": "NR0000001", "channel": "web", "created_ts": "2026-10-28 10:00:00", "promised_ts": "2026-10-31 10:00:00", "delivered_ts": "2026-10-30 12:00:00", "status": "delivered"},
        {"order_id": "NR0000002", "channel": "web", "created_ts": "2026-10-29 09:00:00", "promised_ts": "2026-11-01 09:00:00", "delivered_ts": None, "status": "processing"},
    ])
    flagged = flow.orders_needing_action(events, orders, today=pd.Timestamp("2026-10-31 23:59:59"))
    assert list(flagged["order_id"]) == ["NR0000002"]


def test_stockout_risk():
    inv = pd.DataFrame([
        {"store_id": "S001", "product_id": "P0001", "week": "2026-10-19", "on_hand_units": 20, "on_order_units": 0, "lead_time_days": 14},
        {"store_id": "S001", "product_id": "P0001", "week": "2026-10-26", "on_hand_units": 2, "on_order_units": 0, "lead_time_days": 14},
    ])
    sales = pd.DataFrame([
        {"sale_id": "SL1", "order_id": "NR1", "sale_ts": "2026-10-20 12:00:00", "product_id": "P0001", "store_id": "S001", "customer_id": "C1", "channel": "tienda", "units": 10, "unit_price_eur": 50.0, "discount_pct": 0},
    ])
    products = pd.DataFrame([
        {"product_id": "P0001", "product_name": "Producto Test", "category": "Moda", "list_price_eur": 50.0, "unit_cost_eur": 20.0},
    ])
    res = stock.stockout_risk(inv, sales, products)
    assert len(res) == 1
    assert res.loc[0, "margin_at_risk_eur"] > 0
    assert bool(res.loc[0, "reorder_now"]) is True


def test_retention_matrix():
    customers = pd.DataFrame([
        {"customer_id": "C1", "signup_date": "2026-01-10", "acquisition_channel": "tienda"},
        {"customer_id": "C2", "signup_date": "2026-01-15", "acquisition_channel": "web"},
    ])
    orders = pd.DataFrame([
        {"order_id": "NR1", "customer_id": "C1", "created_ts": "2026-01-12", "status": "delivered"},
        {"order_id": "NR2", "customer_id": "C1", "created_ts": "2026-02-14", "status": "delivered"},
        {"order_id": "NR3", "customer_id": "C2", "created_ts": "2026-01-20", "status": "delivered"},
    ])
    m_global = cohorts.retention_matrix(orders, customers)
    assert m_global.loc["2026-01", 0] == 100.0
    assert m_global.loc["2026-01", 1] == 50.0

    m_ch = cohorts.retention_matrix(orders, customers, by="acquisition_channel")
    assert m_ch.loc[("tienda", "2026-01"), 1] == 100.0
    assert m_ch.loc[("web", "2026-01"), 1] == 0.0
