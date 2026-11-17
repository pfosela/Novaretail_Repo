import pandas as pd

from novaretail.contracts import clean_orders, parse_ts


def _row(**kw):
    base = {
        "order_id": "NR0000001", "customer_id": "C000001", "store": "S001", "channel": "web",
        "created_ts": "2025-07-12 10:00:00", "promised_ts": "2025-07-15 10:00:00",
        "delivered_ts": "2025-07-14 10:00:00", "status": "delivered",
        "revenue_eur": "50.0", "variable_cost_eur": "30.0",
    }
    base.update(kw)
    return base


def test_parse_ts_iso_no_se_confunde_con_dia_mes():
    s = parse_ts(pd.Series(["2025-07-12 15:37:56", "12/07/2025 15:37"]))
    assert (s.dt.month == 7).all() and (s.dt.day == 12).all()


def test_pedido_correcto_pasa():
    valid, q = clean_orders(pd.DataFrame([_row()]))
    assert len(valid) == 1 and len(q) == 0


def test_errores_van_a_cuarentena():
    raw = pd.DataFrame([
        _row(),
        _row(),                                                   # duplicado exacto -> se elimina
        _row(order_id="NR0000002", variable_cost_eur=None),      # coste ausente
        _row(order_id="NR0000003", revenue_eur="-10"),           # importe negativo
        _row(order_id="NR0000004", delivered_ts="2025-07-01 10:00:00"),  # entrega antes de creación
        _row(order_id="NR0000005", channel=" WEB "),             # se corrige, no va a cuarentena
    ])
    valid, q = clean_orders(raw)
    assert set(valid["order_id"]) == {"NR0000001", "NR0000005"}
    assert len(q) == 3
