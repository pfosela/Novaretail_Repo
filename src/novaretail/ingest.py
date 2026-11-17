"""Carga de las fuentes en bruto (capa bronze). Responsable: Persona 2.

Aquí solo se LEE. No se limpia nada: la limpieza la hacen los contratos de `contracts.py`.
"""
import json

import pandas as pd

from novaretail.config import RAW


def load_orders() -> pd.DataFrame:
    # Todo como texto: las fechas vienen en formatos mezclados y hay que validarlas después.
    return pd.read_csv(RAW / "orders.csv", dtype=str)


def load_sales() -> pd.DataFrame:
    return pd.read_parquet(RAW / "sales.parquet")


def load_products() -> pd.DataFrame:
    return pd.read_csv(RAW / "products.csv")


def load_stores() -> pd.DataFrame:
    return pd.read_csv(RAW / "stores.csv")


def load_customers() -> pd.DataFrame:
    return pd.read_csv(RAW / "customers.csv", dtype={"phone": str, "postal_code": str})


def load_inventory() -> pd.DataFrame:
    return pd.read_parquet(RAW / "inventory_weekly.parquet")


def load_events() -> pd.DataFrame:
    return pd.read_csv(RAW / "fulfilment_events.csv", dtype=str)


def load_tickets() -> pd.DataFrame:
    with open(RAW / "tickets.json", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


# TODO(Persona 2): extracción del catálogo web (catalog/*.html) con BeautifulSoup
def load_catalog() -> pd.DataFrame:
    raise NotImplementedError


# TODO(Persona 2): extracción de facturas PDF (invoices/*.pdf) con pdfplumber
def load_invoices() -> pd.DataFrame:
    raise NotImplementedError
