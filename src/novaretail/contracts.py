"""Contratos de calidad (bronze -> silver). Responsable: Persona 2.

Cada dataset pasa por `clean_<dataset>(raw, ...) -> (validos, cuarentena, informe)`:

1. **Correcciones seguras**: arreglos que no inventan datos (duplicados exactos, espacios,
   mayúsculas, formato de fecha europeo, transportista deducible del propio pedido). Se cuentan.
2. **Reglas de fila**: las filas que las incumplen NO se borran, van a cuarentena con su motivo
   para poder contarlas y explicarlas en la memoria (KPI ingenuo vs. validado, M2).
3. **Reglas críticas** (de dataset): si fallan, el pipeline se detiene (`cli.py validate` sale con
   código 1), porque el dataset no es utilizable: faltan columnas, está vacío o hay demasiada
   cuarentena (> MAX_QUARANTINE_RATE).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from novaretail.config import ACTIVITIES, DATA_END, VALID_CHANNELS, VALID_STATUS

MAX_QUARANTINE_RATE = 0.10          # más de un 10 % en cuarentena = dataset no fiable -> parar
RECONCILE_TOL_EUR = 0.05            # tolerancia pedido vs. suma de sus líneas (redondeos)
RECONCILE_TOL_PCT = 0.01

CHANNEL_FIXES = {"store": "tienda", "market_place": "marketplace"}


# ----------------------------------------------------------------------------
# Informe de calidad
# ----------------------------------------------------------------------------
@dataclass
class QualityReport:
    dataset: str
    rows_in: int = 0
    rows_valid: int = 0
    rows_quarantined: int = 0
    corrections: dict[str, int] = field(default_factory=dict)
    quarantine_by_reason: dict[str, int] = field(default_factory=dict)
    warnings: dict[str, int] = field(default_factory=dict)
    critical: list[str] = field(default_factory=list)

    @property
    def quarantine_rate(self) -> float:
        return self.rows_quarantined / self.rows_in if self.rows_in else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["quarantine_rate"] = round(self.quarantine_rate, 4)
        return d


def _check_critical(rep: QualityReport, df: pd.DataFrame, required: list[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        rep.critical.append(f"faltan columnas: {missing}")
    if len(df) == 0:
        rep.critical.append("dataset vacío")


def _finish(rep: QualityReport, valid: pd.DataFrame, bad: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    quarantine = pd.concat(bad, ignore_index=True) if bad else pd.DataFrame(columns=["quarantine_reason"])
    rep.rows_valid = len(valid)
    rep.rows_quarantined = len(quarantine)
    if len(quarantine):
        rep.quarantine_by_reason = (quarantine["quarantine_reason"].str.split(";").explode()
                                    .value_counts().astype(int).to_dict())
    if rep.quarantine_rate > MAX_QUARANTINE_RATE:
        rep.critical.append(f"cuarentena {rep.quarantine_rate:.1%} > {MAX_QUARANTINE_RATE:.0%}")
    return valid.reset_index(drop=True), quarantine


def _quarantine(df: pd.DataFrame, mask: pd.Series, reason: str, bad: list) -> pd.DataFrame:
    mask = mask.fillna(False).astype(bool)
    if mask.any():
        q = df[mask].copy()
        q["quarantine_reason"] = reason
        bad.append(q)
    return df[~mask]


def _apply_schema(df: pd.DataFrame, schema: pa.DataFrameSchema, bad: list) -> pd.DataFrame:
    """Valida con pandera y manda a cuarentena cada fila que falle, con motivo `columna:regla`."""
    try:
        return schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        fc = err.failure_cases.dropna(subset=["index"]).copy()
        row_checks = {c.name for c in schema.checks}
        is_row = fc["check"].isin(row_checks)
        fc["reason"] = fc["column"].fillna("fila").astype(str) + ":" + fc["check"].astype(str)
        fc.loc[is_row, "reason"] = "fila:" + fc.loc[is_row, "check"].astype(str)
        reasons = fc.groupby("index")["reason"].apply(lambda s: ";".join(sorted(set(s))))
        mask = df.index.isin(reasons.index)
        q = df[mask].copy()
        q["quarantine_reason"] = reasons.reindex(q.index).values
        bad.append(q)
        return df[~mask]


def _named(fn, name: str) -> pa.Check:
    return pa.Check(fn, name=name, element_wise=False)


# ----------------------------------------------------------------------------
# Utilidades de tipos
# ----------------------------------------------------------------------------
def parse_ts(s: pd.Series) -> pd.Series:
    """Fechas ISO (con o sin fracciones de segundo) o europeas (dd/mm/aaaa hh:mm).

    Ojo: no usar `format="mixed", dayfirst=True` sobre todo: interpretaría 2025-07-12 como 7 de diciembre.
    """
    iso = pd.to_datetime(s, format="ISO8601", errors="coerce")
    eu = pd.to_datetime(s, format="%d/%m/%Y %H:%M", errors="coerce")
    return iso.fillna(eu)


def _strip_lower(s: pd.Series) -> pd.Series:
    return s.astype("string").str.strip().str.lower()


# ----------------------------------------------------------------------------
# SALES (líneas de pedido)
# ----------------------------------------------------------------------------
SALES_SCHEMA = pa.DataFrameSchema(
    {
        "sale_id": pa.Column(str, unique=True),
        "order_id": pa.Column(str),
        "sale_ts": pa.Column("datetime64[ns]"),
        "product_id": pa.Column(str),
        "store_id": pa.Column(str),
        "customer_id": pa.Column(str),
        "channel": pa.Column(str, _named(lambda s: s.isin(VALID_CHANNELS), "canal_valido")),
        "units": pa.Column(int, _named(lambda s: s >= 1, "unidades_positivas")),
        "unit_price_eur": pa.Column(float, _named(lambda s: s > 0, "precio_positivo")),
        "discount_pct": pa.Column(int, _named(lambda s: s.between(0, 100), "descuento_0_100")),
    },
)


def clean_sales(raw: pd.DataFrame, products: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, QualityReport]:
    rep = QualityReport("sales", rows_in=len(raw))
    _check_critical(rep, raw, list(SALES_SCHEMA.columns))
    if rep.critical:
        return raw.iloc[0:0], raw.iloc[0:0], rep
    df, bad = raw.copy(), []

    n = len(df)
    df = df.drop_duplicates()
    rep.corrections["duplicados_exactos_eliminados"] = n - len(df)
    df["channel"] = _strip_lower(df["channel"]).replace(CHANNEL_FIXES)
    df["sale_ts"] = pd.to_datetime(df["sale_ts"], errors="coerce")

    # integridad referencial: el producto debe existir en el catálogo
    df = _quarantine(df, ~df["product_id"].isin(products["product_id"]), "producto_inexistente", bad)
    df = _apply_schema(df, SALES_SCHEMA, bad)
    valid, q = _finish(rep, df, bad)
    return valid, q, rep


def line_revenue(sales: pd.DataFrame) -> pd.Series:
    """Importe de cada línea, tal como lo calcula el sistema de ventas."""
    return (sales["unit_price_eur"] * sales["units"] * (1 - sales["discount_pct"] / 100)).round(2)


# ----------------------------------------------------------------------------
# ORDERS
# ----------------------------------------------------------------------------
ORDERS_SCHEMA = pa.DataFrameSchema(
    {
        "order_id": pa.Column(str, pa.Check.str_matches(r"^NR\d{7}$"), unique=True),
        "customer_id": pa.Column(str),
        "store": pa.Column(str, _named(lambda s: s.str.fullmatch(r"S\d{3}"), "tienda_valida")),
        "channel": pa.Column(str, _named(lambda s: s.isin(VALID_CHANNELS), "canal_valido")),
        "created_ts": pa.Column("datetime64[ns]"),
        "promised_ts": pa.Column("datetime64[ns]"),
        "delivered_ts": pa.Column("datetime64[ns]", nullable=True),
        "status": pa.Column(str, _named(lambda s: s.isin(VALID_STATUS), "estado_valido")),
        "revenue_eur": pa.Column(float, _named(lambda s: s > 0, "importe_positivo")),
        "variable_cost_eur": pa.Column(float, _named(lambda s: s >= 0, "coste_no_negativo")),
    },
    checks=[
        _named(lambda d: d["delivered_ts"].isna() | (d["delivered_ts"] >= d["created_ts"]), "entrega_tras_creacion"),
        _named(lambda d: d["promised_ts"] >= d["created_ts"], "promesa_tras_creacion"),
    ],
)


def clean_orders(raw: pd.DataFrame, valid_lines: pd.DataFrame | None = None,
                 quarantined_lines: pd.DataFrame | None = None,
                 as_of: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame, QualityReport]:
    """`valid_lines` / `quarantined_lines`: salida de `clean_sales`, para conciliar el importe del pedido
    con la suma de sus líneas. Sin ellas, no se concilia."""
    rep = QualityReport("orders", rows_in=len(raw))
    _check_critical(rep, raw, list(ORDERS_SCHEMA.columns))
    if rep.critical:
        return raw.iloc[0:0], raw.iloc[0:0], rep
    df, bad = raw.copy(), []

    # 1) correcciones seguras
    n = len(df)
    df = df.drop_duplicates()
    rep.corrections["duplicados_exactos_eliminados"] = n - len(df)
    ch = _strip_lower(df["channel"]).replace(CHANNEL_FIXES)
    rep.corrections["canal_normalizado"] = int((ch != df["channel"]).fillna(False).sum())
    df["channel"] = ch
    st = df["store"].astype("string").str.strip()
    rep.corrections["tienda_sin_espacios"] = int((st != df["store"]).fillna(False).sum())
    df["store"] = st
    df["status"] = df["status"].replace("", pd.NA)
    eu_dates = df["created_ts"].astype("string").str.match(r"^\d{2}/\d{2}/\d{4}").fillna(False)
    rep.corrections["fecha_formato_europeo"] = int(eu_dates.sum())
    for c in ["created_ts", "promised_ts", "delivered_ts"]:
        df[c] = parse_ts(df[c])
    for c in ["revenue_eur", "variable_cost_eur"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 2) reglas de fila
    as_of = as_of or pd.Timestamp(DATA_END)
    df = _quarantine(df, df["created_ts"].isna(), "fecha_creacion_invalida", bad)
    df = df.sort_values("created_ts", kind="stable")
    df = _quarantine(df, df["order_id"].duplicated(keep="first"), "order_id_reenviado", bad)
    df = _quarantine(df, df["created_ts"] > as_of, "fecha_futura", bad)
    df = _quarantine(df, df["variable_cost_eur"].isna(), "coste_ausente", bad)

    # conciliación con las líneas de venta: el importe del pedido debe ser la suma de sus líneas.
    # Solo se concilian pedidos cuyas líneas son todas válidas (si no, se avisa pero no se castiga).
    if valid_lines is not None:
        lines = valid_lines.assign(_amt=line_revenue(valid_lines)).groupby("order_id")["_amt"].sum()
        expected = df["order_id"].map(lines)
        if quarantined_lines is not None and len(quarantined_lines):
            incomplete = df["order_id"].isin(quarantined_lines["order_id"])
            rep.warnings["pedido_no_conciliable_por_lineas_en_cuarentena"] = int(incomplete.sum())
            expected = expected.mask(incomplete)
        diff = (df["revenue_eur"] - expected).abs()
        tol = (expected.abs() * RECONCILE_TOL_PCT).clip(lower=RECONCILE_TOL_EUR)
        mismatch = expected.notna() & (diff > tol) & (df["revenue_eur"] > 0)
        df = _quarantine(df, mismatch, "importe_no_cuadra_con_lineas", bad)

    # 3) contrato pandera (tipos, dominios y coherencia de fechas)
    df = _apply_schema(df, ORDERS_SCHEMA, bad)
    valid, q = _finish(rep, df, bad)
    return valid, q, rep


# ----------------------------------------------------------------------------
# FULFILMENT EVENTS
# ----------------------------------------------------------------------------
EVENTS_SCHEMA = pa.DataFrameSchema(
    {
        "order_id": pa.Column(str),
        "activity": pa.Column(str, _named(lambda s: s.isin(ACTIVITIES), "actividad_valida")),
        "event_ts": pa.Column("datetime64[ns]"),
        "resource": pa.Column(str),
        "warehouse": pa.Column(str),
        "carrier": pa.Column(str),
    },
)


def clean_events(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, QualityReport]:
    rep = QualityReport("fulfilment_events", rows_in=len(raw))
    _check_critical(rep, raw, list(EVENTS_SCHEMA.columns))
    if rep.critical:
        return raw.iloc[0:0], raw.iloc[0:0], rep
    df, bad = raw.copy(), []

    # 1) correcciones seguras
    act = _strip_lower(df["activity"])
    rep.corrections["actividad_en_minusculas"] = int((act != df["activity"]).fillna(False).sum())
    df["activity"] = act
    n = len(df)
    df = df.drop_duplicates()
    rep.corrections["eventos_duplicados_eliminados"] = n - len(df)
    df["event_ts"] = parse_ts(df["event_ts"])
    # el transportista es único por pedido: si falta en un evento, se toma del resto del pedido
    known = df.dropna(subset=["carrier"]).groupby("order_id")["carrier"].agg(lambda s: s.mode().iat[0])
    missing = df["carrier"].isna()
    df.loc[missing, "carrier"] = df.loc[missing, "order_id"].map(known)
    rep.corrections["transportista_deducido"] = int((missing & df["carrier"].notna()).sum())

    # 2) reglas de fila
    df = _quarantine(df, df["event_ts"].isna(), "fecha_evento_invalida", bad)
    # secuencia: created <= picked <= packed <= shipped <= delivered <= returned.
    # Si un pedido rompe el orden, se aparta el pedido entero (sus tiempos por etapa no son fiables).
    first = df.pivot_table(index="order_id", columns="activity", values="event_ts", aggfunc="min")
    first = first.reindex(columns=[a for a in ACTIVITIES if a in first.columns])
    broken = pd.Series(False, index=first.index)
    for a, b in zip(first.columns[:-1], first.columns[1:]):
        broken |= (first[b] < first[a]).fillna(False)
    rep.warnings["pedidos_con_secuencia_incoherente"] = int(broken.sum())
    df = _quarantine(df, df["order_id"].isin(broken[broken].index), "secuencia_incoherente", bad)

    df = _apply_schema(df, EVENTS_SCHEMA, bad)
    valid, q = _finish(rep, df, bad)
    return valid.sort_values(["order_id", "event_ts"]).reset_index(drop=True), q, rep


# ----------------------------------------------------------------------------
# INVENTORY (snapshot semanal)
# ----------------------------------------------------------------------------
INVENTORY_SCHEMA = pa.DataFrameSchema(
    {
        "store_id": pa.Column(str),
        "product_id": pa.Column(str),
        "week": pa.Column("datetime64[ns]"),
        "on_hand_units": pa.Column(int, _named(lambda s: s >= 0, "stock_no_negativo")),
        "on_order_units": pa.Column(int, _named(lambda s: s >= 0, "pendiente_no_negativo")),
        "lead_time_days": pa.Column(int, _named(lambda s: s > 0, "plazo_positivo")),
    },
    unique=["store_id", "product_id", "week"],
)


def clean_inventory(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, QualityReport]:
    rep = QualityReport("inventory_weekly", rows_in=len(raw))
    _check_critical(rep, raw, list(INVENTORY_SCHEMA.columns))
    if rep.critical:
        return raw.iloc[0:0], raw.iloc[0:0], rep
    df, bad = raw.copy(), []
    df["week"] = pd.to_datetime(df["week"], errors="coerce")
    n = len(df)
    df = df.drop_duplicates()
    rep.corrections["duplicados_exactos_eliminados"] = n - len(df)

    # completitud: cada combinación tienda-producto debería tener todas las semanas
    weeks = df["week"].nunique()
    per_combo = df.groupby(["store_id", "product_id"])["week"].nunique()
    rep.warnings["semanas_faltantes"] = int((weeks - per_combo).clip(lower=0).sum())

    df = _apply_schema(df, INVENTORY_SCHEMA, bad)
    valid, q = _finish(rep, df, bad)
    return valid, q, rep


# ----------------------------------------------------------------------------
# Cobertura: ¿detectamos todos los errores que inyectó el generador?
# ----------------------------------------------------------------------------
# error inyectado -> (cómo lo trata el contrato, motivo/corrección/aviso que lo recoge)
COVERAGE_MAP = {
    "orders": {
        "duplicate_exact_rows": ("corrección", "duplicados_exactos_eliminados"),
        "duplicate_order_id_resent": ("cuarentena", "order_id_reenviado"),
        "variable_cost_missing": ("cuarentena", "coste_ausente"),
        "delivered_before_created": ("cuarentena", "fila:entrega_tras_creacion"),
        "created_ts_alt_format": ("corrección", "fecha_formato_europeo"),
        "created_ts_future": ("cuarentena", "fecha_futura + fila:promesa_tras_creacion"),
        "channel_inconsistent_label": ("corrección", "canal_normalizado"),
        "revenue_negative": ("cuarentena", "revenue_eur:importe_positivo"),
        "revenue_outlier_x100": ("cuarentena", "importe_no_cuadra_con_lineas"),
        "store_null": ("cuarentena", "store:not_nullable"),
        "store_trailing_spaces": ("corrección", "tienda_sin_espacios"),
        "status_empty": ("cuarentena", "status:not_nullable"),
    },
    "sales": {
        "product_id_orphan": ("cuarentena", "producto_inexistente"),
        "discount_out_of_range": ("cuarentena", "discount_pct:descuento_0_100"),
        "units_zero": ("cuarentena", "units:unidades_positivas"),
    },
    "fulfilment_events": {
        "event_duplicated": ("corrección", "eventos_duplicados_eliminados"),
        "event_out_of_order": ("cuarentena", "pedidos_con_secuencia_incoherente"),
        "carrier_null": ("corrección", "transportista_deducido"),
        "activity_capitalized": ("corrección", "actividad_en_minusculas"),
    },
    "inventory_weekly": {
        "on_hand_negative": ("cuarentena", "on_hand_units:stock_no_negativo"),
        "snapshot_rows_missing": ("aviso", "semanas_faltantes"),
    },
}


def coverage_vs_injected(reports: dict[str, QualityReport], meta_file: Path) -> pd.DataFrame:
    """Compara lo detectado con `_meta/generation_info.json`. Solo para verificar el contrato."""
    injected = json.loads(Path(meta_file).read_text(encoding="utf-8"))["injected_quality_issues"]
    rows = []
    for ds, mapping in COVERAGE_MAP.items():
        rep = reports.get(ds)
        for err, (how, key) in mapping.items():
            found = None
            if rep is not None:
                pool = {**rep.corrections, **rep.quarantine_by_reason, **rep.warnings}
                keys = [k.strip() for k in key.split("+")]
                found = sum(pool.get(k, 0) for k in keys)
            rows.append({"dataset": ds, "error_inyectado": err, "inyectados": injected.get(ds, {}).get(err),
                         "tratamiento": how, "regla": key, "detectados": found})
    df = pd.DataFrame(rows)
    df["cubierto"] = df["detectados"].fillna(0) > 0
    return df
