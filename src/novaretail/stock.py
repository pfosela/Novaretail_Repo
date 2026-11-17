"""Riesgo de rotura de stock priorizado por margen. Responsable: Persona 3.

Idea: cobertura = (on_hand + on_order) / demanda semanal esperada, comparada con el lead time.
Prioridad = probabilidad de rotura x margen unitario x demanda -> margen en riesgo (EUR).
Decisión táctica: qué combinaciones tienda-producto reponer esta semana.
"""
import pandas as pd


def stockout_risk(inventory: pd.DataFrame, sales: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """TODO(Persona 3)."""
    raise NotImplementedError
