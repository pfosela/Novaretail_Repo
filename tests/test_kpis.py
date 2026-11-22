import pandas as pd
import pytest

from novaretail import kpis


@pytest.fixture
def orders():
    return pd.DataFrame({
        "order_id": ["A", "B", "C", "D", "D"],
        "store": ["S001", "S001", "S002", "S002", "S002"],
        "channel": ["web", "web", "tienda", "app", "app"],
        "created_ts": ["2025-01-01 10:00:00", "2025-01-02 10:00:00", "02/01/2025 10:00", "2025-02-01 10:00:00", "2025-02-01 10:00:00"],
        "promised_ts": ["2025-01-04 10:00:00", "2025-01-05 10:00:00", "02/01/2025 10:00", "2025-02-04 10:00:00", "2025-02-04 10:00:00"],
        "delivered_ts": ["2025-01-03 10:00:00", "2025-01-06 10:00:00", "02/01/2025 10:00", None, None],
        "status": ["delivered", "returned", "delivered", "cancelled", "cancelled"],
        "revenue_eur": ["100", "50", "20", "30", "30"],
        "variable_cost_eur": ["70", "40", "10", "3", "3"],
    })


def test_order_kpis_basic(orders):
    k = kpis.order_kpis(orders)
    assert k["pedidos"] == 4 and k["filas"] == 5          # D duplicado: pedidos únicos vs filas
    assert k["tasa_devolucion"] == pytest.approx(1 / 3)   # 1 devuelto de 3 entregados o devueltos
    assert k["tasa_cancelacion"] == pytest.approx(2 / 5)
    assert k["entrega_a_tiempo"] == pytest.approx(2 / 3)  # B llega un día tarde
    assert k["ingresos_netos_eur"] == pytest.approx(120)  # sin devueltos ni cancelados


def test_dates_in_european_format_are_parsed(orders):
    df = kpis.prepare_orders(orders)
    assert df.loc[2, "created_ts"] == pd.Timestamp("2025-01-02 10:00")
    assert df["month"].tolist()[:3] == ["2025-01"] * 3


def test_kpis_by_channel(orders):
    by = kpis.kpis_by(orders, "channel")
    assert set(by.index) == {"web", "tienda", "app"}
    assert by.loc["tienda", "entrega_a_tiempo"] == 1.0


def test_region_unknown_store_is_not_dropped(orders):
    stores = pd.DataFrame({"store_id": ["S001"], "region": ["Norte"]})
    by = kpis.kpis_by(orders, "region", stores)
    assert "(sin región)" in by.index                      # S002 no está en stores


def test_margin_by_discount():
    sales = pd.DataFrame({"channel": ["web", "web"], "discount_pct": [0, 50], "product_id": ["P1", "P1"],
                          "units": [1, 1], "unit_price_eur": [100.0, 100.0]})
    products = pd.DataFrame({"product_id": ["P1"], "unit_cost_eur": [70.0]})
    g = kpis.margin_by_discount(sales, products).set_index("discount_pct")["margen_bruto_pct"]
    assert g[0] == pytest.approx(0.30) and g[50] == pytest.approx(-0.40)
