"""KPIs de negocio y árbol de valor. Responsable: Marcos de Andrés Torrejón.

Las mismas funciones se aplican a datos en bruto (KPI "ingenuo", el dashboard actual) y a datos
validados (KPI "validado") para la comparación M2.

Definiciones (idénticas en bruto y validado, para que la diferencia sea solo calidad del dato):

* ``pedidos``: nº de order_id distintos. ``filas``: filas del fichero (las duplicadas inflan la cifra).
* ``ingresos_eur`` y ``margen_contribucion_eur``: suma de revenue_eur y de revenue_eur - variable_cost_eur
  sobre todas las filas, tal como los suma el dashboard actual (incluye devueltos y cancelados).
* ``ingresos_netos_eur``: igual, pero sin pedidos devueltos ni cancelados.
* ``tasa_devolucion``: devueltos / (entregados + devueltos).
* ``tasa_cancelacion``: cancelados / filas con estado informado.
* ``entrega_a_tiempo``: entregas con delivered_ts <= promised_ts / entregas con fecha de entrega.
"""
from __future__ import annotations

import pandas as pd

# Preparación común

def _as_ts(s: pd.Series) -> pd.Series:
    """Convierte a fecha si hace falta (acepta ISO y dd/mm/aaaa hh:mm); si ya es fecha, no toca nada.

    No se usa ``format="mixed", dayfirst=True``: interpretaría 2025-07-12 como 7 de diciembre.
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    iso = pd.to_datetime(s, format="ISO8601", errors="coerce")
    eu = pd.to_datetime(s, format="%d/%m/%Y %H:%M", errors="coerce")
    return iso.fillna(eu)


def prepare_orders(orders: pd.DataFrame, stores: pd.DataFrame | None = None) -> pd.DataFrame:
    """Añade columnas de análisis sin limpiar nada: importes numéricos, fechas, mes, región y banderas.

    No corrige canales ni tiendas: así, aplicada al fichero en bruto, reproduce lo que ve la dirección.
    """
    df = orders.copy()
    df["revenue_eur"] = pd.to_numeric(df["revenue_eur"], errors="coerce")
    df["variable_cost_eur"] = pd.to_numeric(df["variable_cost_eur"], errors="coerce")
    for c in ["created_ts", "promised_ts", "delivered_ts"]:
        df[c] = _as_ts(df[c])
    df["status"] = df["status"].astype("string").str.strip().str.lower()
    df["month"] = df["created_ts"].dt.to_period("M").astype("string")
    df["margin_eur"] = df["revenue_eur"] - df["variable_cost_eur"]
    if stores is not None and "region" not in df.columns:
        reg = stores.set_index("store_id")["region"]
        df["region"] = df["store"].map(reg).fillna("(sin región)")
    return df

# KPIs de pedidos

def order_kpis(orders: pd.DataFrame) -> pd.Series:
    """KPIs de pedidos: volumen, ingresos, margen, ticket medio, devolución, cancelación y puntualidad."""
    df = prepare_orders(orders)
    revenue, margin = df["revenue_eur"], df["margin_eur"]
    delivered = df[df["delivered_ts"].notna()]
    returned = (df["status"] == "returned").sum()
    shipped_ok = (df["status"] == "delivered").sum() + returned
    with_status = df["status"].notna() & (df["status"] != "")
    net = ~df["status"].isin(["returned", "cancelled"])
    return pd.Series({
        "pedidos": df["order_id"].nunique(),
        "filas": len(df),
        "ingresos_eur": revenue.sum(),
        "ingresos_netos_eur": revenue[net].sum(),
        "margen_contribucion_eur": margin.sum(),
        "margen_pct": margin.sum() / revenue.sum() if revenue.sum() else float("nan"),
        "ticket_medio_eur": revenue.mean(),
        "tasa_devolucion": returned / shipped_ok if shipped_ok else float("nan"),
        "tasa_cancelacion": (df["status"] == "cancelled").sum() / with_status.sum() if with_status.sum() else float("nan"),
        "entrega_a_tiempo": (delivered["delivered_ts"] <= delivered["promised_ts"]).mean() if len(delivered) else float("nan"),
    })


def kpis_by(orders: pd.DataFrame, by: str | list[str], stores: pd.DataFrame | None = None) -> pd.DataFrame:
    """KPIs agrupados por canal, región, mes... (``by`` puede ser 'channel', 'region' o 'month')."""
    df = prepare_orders(orders, stores)
    return df.groupby(by).apply(order_kpis, include_groups=False)


def current_dashboard(orders_raw: pd.DataFrame, stores: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """El "dashboard actual": KPIs sobre datos en bruto por canal, región y mes (sin ninguna limpieza)."""
    return {
        "total": order_kpis(orders_raw).to_frame("valor"),
        "canal": kpis_by(orders_raw, "channel", stores),
        "region": kpis_by(orders_raw, "region", stores),
        "mes": kpis_by(orders_raw, "month", stores),
    }

# Margen por canal y nivel de descuento (a nivel de línea de venta)

def margin_by_discount(sales: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Margen bruto (precio con descuento - coste de producto) por canal y nivel de descuento.

    Es margen *antes* de logística: con descuentos >= 30 % ya es nulo o negativo.
    """
    s = sales.merge(products[["product_id", "unit_cost_eur"]], on="product_id", how="left")
    s["ingreso"] = s["unit_price_eur"] * s["units"] * (1 - s["discount_pct"] / 100)
    s["coste"] = s["unit_cost_eur"] * s["units"]
    g = s.groupby(["channel", "discount_pct"]).agg(lineas=("ingreso", "size"), ingreso=("ingreso", "sum"),
                                                     coste=("coste", "sum")).reset_index()
    g["margen_bruto_pct"] = (g["ingreso"] - g["coste"]) / g["ingreso"]
    return g

# Árbol de valor

# Supuestos del grupo, a validar con las áreas indicadas. Cada uno tiene escenario bajo / base / alto.
# "recuperable" = parte del importe que una decisión basada en datos fiables podría evitar.
ASSUMPTIONS = {
    "descuentos_recuperable": {"bajo": 0.25, "base": 0.50, "alto": 0.75},   # Comercial / Finanzas
    "roturas_recuperable": {"bajo": 0.30, "base": 0.50, "alto": 0.70},      # Operaciones / Compras
    "cancelaciones_recuperable": {"bajo": 0.10, "base": 0.25, "alto": 0.40},  # Operaciones
    "tickets_incidentes_evitables": {"bajo": 0.20, "base": 0.30, "alto": 0.40},  # Soporte (puntos de % mal atendido)
    "tickets_retraso_evitables": {"bajo": 0.30, "base": 0.50, "alto": 0.70},  # Operaciones
    "coste_gestion_ticket_eur": 6.5,        # ~12 min de agente a ~32 EUR/h de coste empresa (Soporte)
    # coste esperado de un ticket de prioridad alta mal atendido, por categoría (Soporte / Finanzas)
    "coste_incidente_eur": {"cobro_duplicado": 25.0,       # contracargo bancario + gestión
                            "cambio_direccion": 12.0,      # reenvío del paquete
                            "retraso_entrega": 16.0,       # gestión + riesgo de cancelación
                            "proteccion_datos": 0.0},      # riesgo regulatorio: no se monetiza (apartado 5)
}


def stockout_lost_margin(inventory: pd.DataFrame, sales: pd.DataFrame, products: pd.DataFrame,
                         stores: pd.DataFrame, since: str) -> pd.DataFrame:
    """Margen perdido observado por semanas con stock cero (desde ``since``), por tienda-producto.

    demanda semanal = unidades vendidas / semanas con stock; margen unitario = margen bruto realizado
    del producto (con sus descuentos), con suelo en 0. Es una estimación del margen *ya perdido*; el margen
    *en riesgo* hacia delante lo calcula ``stock.stockout_risk`` (Persona 3: Daniel).
    """
    inv = inventory.copy()
    inv["week"] = pd.to_datetime(inv["week"])
    units = sales.groupby(["store_id", "product_id"])["units"].sum().rename("unidades")
    in_stock_weeks = inv[inv["on_hand_units"] > 0].groupby(["store_id", "product_id"]).size().rename("semanas_con_stock")
    demand = pd.concat([units, in_stock_weeks], axis=1).fillna(0)
    demand["demanda_semanal"] = demand["unidades"] / demand["semanas_con_stock"].clip(lower=1)

    s = sales.merge(products[["product_id", "unit_cost_eur"]], on="product_id")
    s["margen"] = s["unit_price_eur"] * s["units"] * (1 - s["discount_pct"] / 100) - s["unit_cost_eur"] * s["units"]
    um = (s.groupby("product_id")["margen"].sum() / s.groupby("product_id")["units"].sum()).clip(lower=0)

    zero = inv[(inv["week"] >= pd.Timestamp(since)) & (inv["on_hand_units"] == 0)]
    out = zero.groupby(["store_id", "product_id"]).size().rename("semanas_sin_stock").reset_index()
    out = out.merge(demand["demanda_semanal"].reset_index(), on=["store_id", "product_id"], how="left")
    out["margen_unitario_eur"] = out["product_id"].map(um)
    out["margen_perdido_eur"] = out["semanas_sin_stock"] * out["demanda_semanal"] * out["margen_unitario_eur"]
    out = out.merge(stores[["store_id", "region"]], on="store_id", how="left")
    out = out.merge(products[["product_id", "category"]], on="product_id", how="left")
    return out.sort_values("margen_perdido_eur", ascending=False).reset_index(drop=True)


def value_tree(orders: pd.DataFrame, sales: pd.DataFrame, products: pd.DataFrame, inventory: pd.DataFrame,
               tickets: pd.DataFrame, stores: pd.DataFrame, since: str = "2025-11-01",
               assumptions: dict | None = None) -> pd.DataFrame:
    """Árbol de valor: beneficio anual de decidir con datos fiables, en euros, con supuestos.

    Parámetros
    ----------
    orders, sales, inventory, stores, products : datos VALIDADOS (``data/clean``).
    tickets : tickets con columnas ``created_ts``, ``category`` y ``priority`` (esta última procede de
        ``data/_meta/tickets_labels.csv``; en producción vendría del clasificador de la Persona 4).
    since : inicio de los 12 meses de referencia (por defecto nov-2025 a oct-2026, que incluye la campaña).

    Devuelve una fila por rama con su importe base y el rango bajo-alto, más una fila TOTAL.
    """
    a = assumptions or ASSUMPTIONS
    o = prepare_orders(orders, stores)
    o = o[o["created_ts"] >= pd.Timestamp(since)]
    margin_pct = (o[o["status"] == "delivered"].groupby("channel")["margin_eur"].sum()
                  / o[o["status"] == "delivered"].groupby("channel")["revenue_eur"].sum())

    def rng(key: str, amount: float) -> dict:
        return {k: amount * a[key][k] for k in ("bajo", "base", "alto")}

    rows = []

    # 1) Estratégica: ventas con margen de contribución negativo (descuentos altos)
    neg = o[(o["status"] == "delivered") & (o["margin_eur"] < 0)]
    loss = -neg["margin_eur"].sum()
    rows.append(dict(rama="1. Descuentos que destruyen margen", nivel="Estratégico", responsable="Comercial / Finanzas",
                     exposicion_eur=loss, **rng("descuentos_recuperable", loss),
                     dato=f"{len(neg):,} pedidos entregados con margen < 0 (de ellos {int((neg['channel'] == 'marketplace').sum()):,} en marketplace)".replace(",", "."),
                     supuesto="Se evita o reprecia la parte indicada de esa pérdida (tope de descuento del 20-30 %)",
                     confianza="Alta (dato validado)"))

    # 2) Táctica: roturas de stock
    lost = stockout_lost_margin(inventory, sales, products, stores, since)
    rows.append(dict(rama="2. Roturas de stock", nivel="Táctico", responsable="Operaciones / Compras",
                     exposicion_eur=lost["margen_perdido_eur"].sum(), **rng("roturas_recuperable", lost["margen_perdido_eur"].sum()),
                     dato=f"{int(lost['semanas_sin_stock'].sum()):,} semanas tienda-producto sin stock".replace(",", "."),
                     supuesto="Reposición anticipada recupera la parte indicada del margen perdido (estimación simple; la Persona 3 calcula el margen en riesgo)",
                     confianza="Media (estimación de demanda)"))

    # 3) Operativa: cancelaciones
    canc = o[o["status"] == "cancelled"]
    potential = (canc["revenue_eur"] * canc["channel"].map(margin_pct)).sum()
    sunk = canc["variable_cost_eur"].sum()
    rows.append(dict(rama="3. Pedidos cancelados", nivel="Operativo", responsable="Operaciones",
                     exposicion_eur=potential + sunk, **rng("cancelaciones_recuperable", potential + sunk),
                     dato=f"{len(canc):,} cancelaciones; margen que habrían dado {potential:,.0f} EUR + coste ya incurrido {sunk:,.0f} EUR".replace(",", "."),
                     supuesto="Intervenir los pedidos en riesgo evita la parte indicada de las cancelaciones",
                     confianza="Media (causa de cancelación desconocida)"))

    # 4) Operativa: tickets de prioridad alta mal atendidos
    t = tickets.copy()
    t["created_ts"] = pd.to_datetime(t["created_ts"])
    t = t[t["created_ts"] >= pd.Timestamp(since)]
    alta = t[t["priority"] == "alta"]
    unit = alta["category"].map(a["coste_incidente_eur"]).fillna(0.0)
    cost_cat = unit.sum()  # coste si TODOS los tickets alta se atendieran mal
    rows.append(dict(rama="4. Tickets urgentes sin priorizar", nivel="Operativo", responsable="Soporte",
                     exposicion_eur=cost_cat, **rng("tickets_incidentes_evitables", cost_cat),
                     dato=f"{len(alta):,} tickets de prioridad alta en 12 meses ({len(t):,} en total)".replace(",", "."),
                     supuesto="Sin prioridad, la cola es FIFO; la clasificación reduce los incidentes en los puntos de % indicados",
                     confianza="Baja (costes unitarios supuestos)"))

    # 5) Operativa: tickets por retraso de entrega
    retraso = t[t["category"] == "retraso_entrega"]
    cost_ret = len(retraso) * a["coste_gestion_ticket_eur"]
    rows.append(dict(rama="5. Coste de gestionar retrasos", nivel="Operativo", responsable="Operaciones / Soporte",
                     exposicion_eur=cost_ret, **rng("tickets_retraso_evitables", cost_ret),
                     dato=f"{len(retraso):,} tickets por retraso en 12 meses".replace(",", "."),
                     supuesto="Reducir el P90 de entrega evita la parte indicada de esos tickets",
                     confianza="Media (coste de gestión supuesto)"))

    tree = pd.DataFrame(rows)
    total = {"rama": "TOTAL valor anual", "nivel": "", "responsable": "", "dato": "", "supuesto": "", "confianza": ""}
    for c in ("exposicion_eur", "bajo", "base", "alto"):
        total[c] = tree[c].sum()
    tree = pd.concat([tree, pd.DataFrame([total])], ignore_index=True)
    tree.attrs["margen_anual_referencia_eur"] = float(o["margin_eur"].sum())
    tree.attrs["desde"] = since
    return tree
