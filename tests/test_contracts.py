import pandas as pd

from novaretail.contracts import (MAX_QUARANTINE_RATE, clean_events, clean_inventory, clean_orders, clean_sales,
                                  parse_ts)


# ---------------------------------------------------------------- utilidades
def test_parse_ts_iso_no_se_confunde_con_dia_mes():
    s = parse_ts(pd.Series(["2025-07-12 15:37:56", "12/07/2025 15:37", "2025-07-12 15:37:56.123456789"]))
    assert (s.dt.month == 7).all() and (s.dt.day == 12).all()


# ---------------------------------------------------------------- orders
def _order(**kw):
    base = {
        "order_id": "NR0000001", "customer_id": "C000001", "store": "S001", "channel": "web",
        "created_ts": "2025-07-12 10:00:00", "promised_ts": "2025-07-15 10:00:00",
        "delivered_ts": "2025-07-14 10:00:00", "status": "delivered",
        "revenue_eur": "50.0", "variable_cost_eur": "30.0",
    }
    base.update(kw)
    return base


def test_pedido_correcto_pasa():
    valid, q, rep = clean_orders(pd.DataFrame([_order()]))
    assert len(valid) == 1 and len(q) == 0 and not rep.critical


def test_correcciones_seguras_no_van_a_cuarentena():
    raw = pd.DataFrame([_order(), _order(),                                  # duplicado exacto
                        _order(order_id="NR0000002", channel=" WEB ", store="S001  "),
                        _order(order_id="NR0000003", created_ts="12/07/2025 10:00")])
    valid, q, rep = clean_orders(raw)
    assert len(valid) == 3 and len(q) == 0
    assert rep.corrections["duplicados_exactos_eliminados"] == 1
    assert rep.corrections["fecha_formato_europeo"] == 1


def test_errores_de_pedido_van_a_cuarentena_con_motivo():
    raw = pd.DataFrame([
        _order(),
        _order(order_id="NR0000001", created_ts="2025-07-12 10:05:00"),       # reenvío
        _order(order_id="NR0000002", variable_cost_eur=None),
        _order(order_id="NR0000003", revenue_eur="-10"),
        _order(order_id="NR0000004", delivered_ts="2025-07-01 10:00:00"),
        _order(order_id="NR0000005", store=None),
        _order(order_id="NR0000006", status=""),
        _order(order_id="NR0000007", created_ts="2027-07-12 10:00:00", promised_ts="2027-07-15 10:00:00"),
    ])
    valid, q, rep = clean_orders(raw)
    assert list(valid["order_id"]) == ["NR0000001"]
    assert set(rep.quarantine_by_reason) == {
        "order_id_reenviado", "coste_ausente", "revenue_eur:importe_positivo", "fila:entrega_tras_creacion",
        "store:not_nullable", "status:not_nullable", "fecha_futura"}


def test_conciliacion_detecta_importe_inflado():
    raw = pd.DataFrame([_order(), _order(order_id="NR0000002", revenue_eur="5000.0")])
    lines = pd.DataFrame({"order_id": ["NR0000001", "NR0000002"], "unit_price_eur": [50.0, 50.0],
                          "units": [1, 1], "discount_pct": [0, 0]})
    valid, q, rep = clean_orders(raw, valid_lines=lines)
    assert list(valid["order_id"]) == ["NR0000001"]
    assert rep.quarantine_by_reason == {"importe_no_cuadra_con_lineas": 1}


def test_regla_critica_si_hay_demasiada_cuarentena():
    raw = pd.DataFrame([_order(order_id=f"NR000000{i}", variable_cost_eur=None) for i in range(5)])
    _, _, rep = clean_orders(raw)
    assert rep.quarantine_rate > MAX_QUARANTINE_RATE and rep.critical


def test_regla_critica_si_faltan_columnas():
    _, _, rep = clean_orders(pd.DataFrame([_order()]).drop(columns=["revenue_eur"]))
    assert any("faltan columnas" in c for c in rep.critical)


# ---------------------------------------------------------------- sales
def _sale(**kw):
    base = {"sale_id": "SL1", "order_id": "NR0000001", "sale_ts": pd.Timestamp("2025-07-12"), "product_id": "P0001",
            "store_id": "S001", "customer_id": "C000001", "channel": "web", "units": 1, "unit_price_eur": 10.0,
            "discount_pct": 0}
    base.update(kw)
    return base


def test_sales_reglas():
    products = pd.DataFrame({"product_id": ["P0001"]})
    raw = pd.DataFrame([_sale(), _sale(sale_id="SL2", product_id="P9999"), _sale(sale_id="SL3", discount_pct=150),
                        _sale(sale_id="SL4", units=0)])
    valid, q, rep = clean_sales(raw, products)
    assert list(valid["sale_id"]) == ["SL1"]
    assert set(rep.quarantine_by_reason) == {"producto_inexistente", "discount_pct:descuento_0_100",
                                             "units:unidades_positivas"}


# ---------------------------------------------------------------- events
def _events(order, times, carrier="RapidLog"):
    acts = ["created", "picked", "packed", "shipped", "delivered"]
    return [{"order_id": order, "activity": a, "event_ts": t, "resource": "r", "warehouse": "WH-MAD", "carrier": carrier}
            for a, t in zip(acts, times)]


def test_events_corrige_y_detecta_secuencia_incoherente():
    ok = _events("NR1", ["2025-01-01 10:00:00", "2025-01-01 11:00:00", "2025-01-01 12:00:00",
                         "2025-01-01 18:00:00", "2025-01-02 10:00:00"])
    ok[1]["activity"] = "Picked"          # mayúscula -> se corrige
    ok[2]["carrier"] = None               # transportista -> se deduce del pedido
    bad = _events("NR2", ["2025-01-01 10:00:00", "2025-01-01 11:00:00", "2025-01-01 12:00:00",
                          "2025-01-01 18:00:00", "2025-01-01 09:00:00"])   # entregado antes de crearse
    raw = pd.DataFrame(ok + [ok[0]] + bad)                                  # + un evento duplicado
    valid, q, rep = clean_events(raw)
    assert set(valid["order_id"]) == {"NR1"} and len(valid) == 5
    assert rep.corrections == {"actividad_en_minusculas": 1, "eventos_duplicados_eliminados": 1,
                               "transportista_deducido": 1}
    assert rep.quarantine_by_reason == {"secuencia_incoherente": 5}


# ---------------------------------------------------------------- inventory
def test_inventory_stock_negativo_y_semanas_faltantes():
    raw = pd.DataFrame({"store_id": ["S001"] * 3 + ["S002"] * 2, "product_id": ["P1"] * 5,
                        "week": ["2025-01-06", "2025-01-13", "2025-01-20", "2025-01-06", "2025-01-20"],
                        "on_hand_units": [5, -1, 3, 2, 2], "on_order_units": [0] * 5, "lead_time_days": [7] * 5})
    valid, q, rep = clean_inventory(raw)
    assert len(q) == 1 and rep.warnings["semanas_faltantes"] == 1
