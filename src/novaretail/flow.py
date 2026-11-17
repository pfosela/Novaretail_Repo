"""Métricas de flujo a partir del event log. Responsable: Persona 3.

Lee los eventos ya validados (data/clean) y calcula:
- tiempo de cada etapa (created->picked->packed->shipped->delivered), P50 y P90
  por almacén, transportista y mes
- WIP diario (pedidos empezados y no entregados)
- cuello de botella (etapa con mayor P90 / mayor WIP)
- pedidos que requieren intervención hoy (decisión operativa)
"""
import pandas as pd

STAGES = [("created", "picked"), ("picked", "packed"), ("packed", "shipped"), ("shipped", "delivered")]


def stage_durations(events: pd.DataFrame) -> pd.DataFrame:
    """Una fila por pedido con las horas de cada etapa.

    TODO(Persona 3): pivotar por actividad (primer timestamp de cada una) y restar por pares.
    """
    raise NotImplementedError


def percentiles(durations: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """TODO(Persona 3): P50 y P90 de cada etapa agrupado por `by`."""
    raise NotImplementedError


def daily_wip(events: pd.DataFrame) -> pd.DataFrame:
    """TODO(Persona 3)."""
    raise NotImplementedError


def orders_needing_action(events: pd.DataFrame, orders: pd.DataFrame, today: pd.Timestamp) -> pd.DataFrame:
    """TODO(Persona 3): pedidos que van a incumplir la promesa si nadie actúa hoy."""
    raise NotImplementedError
