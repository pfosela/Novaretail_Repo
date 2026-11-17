"""Fuente externa vía API con contrato pydantic. Responsable: Persona 2.

Propuesta: Open-Meteo (https://open-meteo.com, sin clave) con temperatura y precipitación
diarias por ciudad de tienda, o Nager.Date (festivos de España).
"""
from datetime import date

import pandas as pd
from pydantic import BaseModel, Field

from novaretail.config import RAW


class DailyWeather(BaseModel):
    """Contrato de cada registro. Si la API cambia de formato, falla aquí y no más adelante."""
    city: str
    day: date
    temp_max_c: float = Field(ge=-30, le=55)
    temp_min_c: float = Field(ge=-40, le=45)
    precipitation_mm: float = Field(ge=0)


CACHE = RAW / "external_weather.parquet"


def fetch_weather(city: str, lat: float, lon: float, start: date, end: date) -> list[DailyWeather]:
    """TODO(Persona 2): llamar a la API con `requests` (timeout, reintentos) y validar con DailyWeather."""
    raise NotImplementedError


def load_weather() -> pd.DataFrame:
    """TODO(Persona 2): si la API falla, usar la última copia en CACHE (fallback)."""
    raise NotImplementedError
