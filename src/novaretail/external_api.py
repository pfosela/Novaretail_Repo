"""Fuente externa vía API con contrato pydantic. Responsable: Persona 2.

Fuente: Open-Meteo Historical Weather API (https://open-meteo.com), gratuita y sin clave.
Variable: temperatura máxima/mínima y precipitación diarias en las ciudades con tienda.
Para qué: explicar y prever la demanda por región (lluvia y calor cambian el tráfico en tienda
y el peso del canal online) en el modelo de la entrega 3.

Contrato: cada registro se valida con `DailyWeather`. Si la API cambia de formato o devuelve
valores imposibles, el registro falla aquí y no contamina el análisis.
Fallback: si la API no responde, se usa la última copia guardada en CACHE.
"""
from __future__ import annotations

import time
from datetime import date

import pandas as pd
import requests
from pydantic import BaseModel, Field, ValidationError

from novaretail.config import DATA_END, RAW

API_URL = "https://archive-api.open-meteo.com/v1/archive"
CACHE = RAW / "external_weather.parquet"
START = date(2025, 1, 1)
END = pd.Timestamp(DATA_END).date()

# ciudades con tienda (stores.csv) -> coordenadas
CITIES = {
    "Bilbao": (43.263, -2.935), "Santander": (43.462, -3.810), "Oviedo": (43.361, -5.849),
    "Sevilla": (37.389, -5.984), "Málaga": (36.721, -4.421), "Granada": (37.177, -3.599),
    "Barcelona": (41.388, 2.170), "Valencia": (39.470, -0.376), "Zaragoza": (41.649, -0.889),
    "Madrid": (40.417, -3.704), "Toledo": (39.863, -4.027), "Valladolid": (41.652, -4.724),
}


class DailyWeather(BaseModel):
    """Contrato de un registro diario."""
    city: str
    day: date
    temp_max_c: float = Field(ge=-30, le=55)
    temp_min_c: float = Field(ge=-40, le=45)
    precipitation_mm: float = Field(ge=0, le=500)


class ExternalSourceError(RuntimeError):
    pass


def _get(params: dict, retries: int = 3, timeout: int = 20) -> dict:
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(API_URL, params=params, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except (requests.RequestException, ValueError) as exc:
            last = exc
            time.sleep(2 ** attempt)
    raise ExternalSourceError(f"Open-Meteo no responde tras {retries} intentos: {last}")


def parse_response(city: str, payload: dict) -> tuple[list[DailyWeather], int]:
    """Convierte la respuesta de la API en registros validados. Devuelve (válidos, nº rechazados)."""
    try:
        d = payload["daily"]
        days, tmax, tmin, prec = d["time"], d["temperature_2m_max"], d["temperature_2m_min"], d["precipitation_sum"]
    except (KeyError, TypeError) as exc:
        raise ExternalSourceError(f"La API ha cambiado de formato: falta {exc}") from exc
    ok, rejected = [], 0
    for t, hi, lo, pr in zip(days, tmax, tmin, prec):
        try:
            ok.append(DailyWeather(city=city, day=t, temp_max_c=hi, temp_min_c=lo, precipitation_mm=pr))
        except ValidationError:
            rejected += 1
    return ok, rejected


def fetch_weather(city: str, lat: float, lon: float, start: date = START, end: date = END) -> tuple[list[DailyWeather], int]:
    params = {"latitude": lat, "longitude": lon, "start_date": start.isoformat(), "end_date": end.isoformat(),
              "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum", "timezone": "Europe/Madrid"}
    return parse_response(city, _get(params))


def refresh_weather() -> pd.DataFrame:
    """Descarga todas las ciudades, valida y actualiza la caché."""
    records, rejected = [], 0
    for city, (lat, lon) in CITIES.items():
        ok, bad = fetch_weather(city, lat, lon)
        records += ok
        rejected += bad
    df = pd.DataFrame([r.model_dump() for r in records])
    df["day"] = pd.to_datetime(df["day"])
    df.attrs["rejected"] = rejected
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE, index=False)
    return df


def load_weather(refresh: bool = True) -> tuple[pd.DataFrame, str]:
    """Devuelve (datos, origen). Origen = 'api' o 'cache'. Si no hay ni API ni caché, error claro."""
    if refresh:
        try:
            return refresh_weather(), "api"
        except ExternalSourceError as exc:
            print(f"[aviso] {exc}. Se usa la caché.")
    if CACHE.exists():
        return pd.read_parquet(CACHE), "cache"
    raise ExternalSourceError("Sin conexión con Open-Meteo y sin caché en data/raw/external_weather.parquet")
