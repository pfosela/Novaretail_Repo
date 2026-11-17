"""Análisis de cohortes de clientes. Responsable: Persona 3.

Cohorte = mes de alta (signup_date). Métrica = % de clientes que compran en el mes N tras el alta,
segmentado por canal de adquisición. Base para el clasificador de inactividad (entrega 3).
"""
import pandas as pd


def retention_matrix(orders: pd.DataFrame, customers: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """TODO(Persona 3)."""
    raise NotImplementedError
