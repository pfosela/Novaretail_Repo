"""Carga de las fuentes en bruto (capa bronze). Responsable: Persona 2.

Aquí solo se LEE. No se limpia nada: la limpieza la hacen los contratos de `contracts.py`.
"""
import json
import re

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


def parse_eur(text: str | None) -> float | None:
    """'1.234,56 €' -> 1234.56 (formato español)."""
    if text is None:
        return None
    t = re.sub(r"[^\d,.-]", "", text).replace(".", "").replace(",", ".")
    return float(t) if t else None


# Catálogo web (semiestructurado: HTML)
def load_catalog() -> pd.DataFrame:
    """Extrae las fichas de producto de catalog/page_XX.html (scraping con BeautifulSoup)."""
    from bs4 import BeautifulSoup

    rows = []
    for page in sorted((RAW / "catalog").glob("page_*.html")):
        soup = BeautifulSoup(page.read_text(encoding="utf-8"), "html.parser")
        for card in soup.select("div.product-card"):
            old = card.select_one(".old-price")
            rows.append({
                "product_id": card.get("data-sku"),
                "product_name": card.select_one(".name").get_text(strip=True),
                "category": card.select_one(".category").get_text(strip=True),
                "web_price_eur": parse_eur(card.select_one(".price").get_text()),
                "web_old_price_eur": parse_eur(old.get_text()) if old else None,
                "web_stock": card.select_one(".stock").get_text(strip=True),
                "source_page": page.name,
            })
    return pd.DataFrame(rows)


# Facturas de proveedor (semiestructurado: PDF)
_INVOICE_PATTERNS = {
    "invoice_number": r"FACTURA N\S*\s+(\S+)",
    "invoice_date": r"Fecha:\s*(\d{2}/\d{2}/\d{4})",
    "supplier_cif": r"^CIF:\s*([A-Z]\d{8})",
    "base_eur": r"Base imponible:\s*([\d.,]+)",
    "vat_rate": r"IVA \((\d+)%\)",
    "vat_eur": r"IVA \(\d+%\):\s*([\d.,]+)",
    "total_eur": r"TOTAL:\s*([\d.,]+)",
}


def parse_invoice_text(text: str) -> dict:
    out = {"supplier": text.splitlines()[0].strip() if text else None}
    for key, pat in _INVOICE_PATTERNS.items():
        m = re.search(pat, text, flags=re.MULTILINE)
        out[key] = m.group(1) if m else None
    for key in ["base_eur", "vat_eur", "total_eur"]:
        out[key] = parse_eur(out[key])
    out["vat_rate"] = int(out["vat_rate"]) / 100 if out["vat_rate"] else None
    if out["invoice_date"]:
        out["invoice_date"] = pd.to_datetime(out["invoice_date"], format="%d/%m/%Y").date()
    return out


def load_invoices() -> pd.DataFrame:
    """Extrae cabecera y totales de invoices/*.pdf con pdfplumber y comprueba que cuadran.

    `total_cuadra` = total == base + IVA (±0,01 €). Las que no cuadran van a revisión de Finanzas.
    """
    import pdfplumber

    rows = []
    for f in sorted((RAW / "invoices").glob("*.pdf")):
        with pdfplumber.open(f) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
            lines = sum(len(t) - 1 for p in pdf.pages for t in p.extract_tables())
        rec = parse_invoice_text(text)
        rec.update({"file": f.name, "n_lines": lines})
        rows.append(rec)
    df = pd.DataFrame(rows)
    df["total_cuadra"] = (df["base_eur"] + df["vat_eur"] - df["total_eur"]).abs() <= 0.01
    return df
