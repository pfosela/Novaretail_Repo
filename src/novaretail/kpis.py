"""KPIs de negocio y árbol de valor. Responsable: Persona 1.

Las mismas funciones se aplican a datos en bruto (KPI "ingenuo") y a datos validados
(KPI "validado") para la comparación M2.
"""
import pandas as pd


def order_kpis(orders: pd.DataFrame) -> pd.Series:
    """KPIs básicos de pedidos: nº pedidos, ingresos, margen de contribución, ticket medio.

    TODO(Persona 1): añadir tasa de devolución, cancelación y on-time delivery.
    """
    revenue = pd.to_numeric(orders["revenue_eur"], errors="coerce")
    cost = pd.to_numeric(orders["variable_cost_eur"], errors="coerce")
    return pd.Series({
        "pedidos": orders["order_id"].nunique(),
        "filas": len(orders),
        "ingresos_eur": revenue.sum(),
        "margen_contribucion_eur": (revenue - cost).sum(),
        "ticket_medio_eur": revenue.mean(),
    })


def kpis_by(orders: pd.DataFrame, by: str) -> pd.DataFrame:
    """KPIs agrupados (por canal, región, mes...)."""
    return orders.groupby(by).apply(order_kpis, include_groups=False)


# TODO(Persona 1): value_tree() -> cuantificar el valor de los datos (M1): ingresos perdidos por
# retrasos, por roturas de stock, coste de tickets mal priorizados, etc.
def value_tree(*args, **kwargs):
    raise NotImplementedError
