"""Contratos de calidad (bronze -> silver). Responsable: Persona 2.

Patrón: cada dataset tiene una función `clean_<dataset>(df) -> (validos, cuarentena)`.
Las filas que incumplen una regla NO se borran: van a cuarentena con el motivo, para poder
contarlas en la memoria (KPI ingenuo vs. validado, M2).

`orders` está hecho como ejemplo. TODO(Persona 2): sales, events, inventory y fuente externa.
"""
import pandas as pd
import pandera.pandas as pa

from novaretail.config import DATA_END, VALID_CHANNELS, VALID_STATUS

ORDERS_SCHEMA = pa.DataFrameSchema(
    {
        "order_id": pa.Column(str, pa.Check.str_matches(r"^NR\d{7}$"), unique=True),
        "customer_id": pa.Column(str),
        "store": pa.Column(str, pa.Check.str_matches(r"^S\d{3}$")),
        "channel": pa.Column(str, pa.Check.isin(VALID_CHANNELS)),
        "created_ts": pa.Column("datetime64[ns]"),
        "promised_ts": pa.Column("datetime64[ns]"),
        "delivered_ts": pa.Column("datetime64[ns]", nullable=True),
        "status": pa.Column(str, pa.Check.isin(VALID_STATUS)),
        "revenue_eur": pa.Column(float, pa.Check.in_range(0, 20_000, include_min=False)),
        "variable_cost_eur": pa.Column(float, pa.Check.ge(0)),
    },
    checks=[
        pa.Check(lambda d: d["delivered_ts"].isna() | (d["delivered_ts"] >= d["created_ts"]),
                 name="delivered_after_created"),
        pa.Check(lambda d: d["promised_ts"] >= d["created_ts"], name="promised_after_created"),
    ],
    coerce=False,
)


def parse_ts(s: pd.Series) -> pd.Series:
    """Fechas en formato ISO (aaaa-mm-dd hh:mm:ss) o europeo (dd/mm/aaaa hh:mm).

    Ojo: no usar `format="mixed", dayfirst=True` sobre todo: interpretaría 2025-07-12 como 7 de diciembre.
    """
    iso = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    eu = pd.to_datetime(s, format="%d/%m/%Y %H:%M", errors="coerce")
    return iso.fillna(eu)


def _quarantine(df: pd.DataFrame, mask: pd.Series, reason: str, bad: list) -> pd.DataFrame:
    q = df[mask].copy()
    q["quarantine_reason"] = reason
    bad.append(q)
    return df[~mask]


def clean_orders(raw: pd.DataFrame, as_of: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Normaliza lo que se puede corregir sin inventar datos y pone en cuarentena el resto."""
    df = raw.copy()
    bad: list[pd.DataFrame] = []

    # 1) duplicados exactos: se quedan una vez (no es un error del pedido, es de la integración)
    df = df.drop_duplicates()

    # 2) correcciones seguras: espacios y mayúsculas en canal/tienda
    df["channel"] = df["channel"].str.strip().str.lower().replace({"store": "tienda", "market_place": "marketplace"})
    df["store"] = df["store"].str.strip()

    # 3) tipos (fechas: ISO o dd/mm/aaaa; cualquier otra cosa queda NaT y va a cuarentena)
    for c in ["created_ts", "promised_ts", "delivered_ts"]:
        df[c] = parse_ts(df[c])
    for c in ["revenue_eur", "variable_cost_eur"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 4) reglas que mandan a cuarentena (añadid aquí las que falten)
    as_of = as_of or pd.Timestamp(DATA_END)
    df = _quarantine(df, df["created_ts"].isna(), "created_ts_invalido", bad)
    # reenvíos: mismo order_id con distinta hora -> se conserva el primero
    df = df.sort_values("created_ts", kind="stable")
    df = _quarantine(df, df["order_id"].duplicated(keep="first"), "order_id_reenviado", bad)
    df = _quarantine(df, df["created_ts"] > as_of, "created_ts_futuro", bad)
    df = _quarantine(df, df["variable_cost_eur"].isna(), "coste_ausente", bad)

    # TODO(Persona 2): comparad la cuarentena con data/_meta/generation_info.json.
    # Todavía hay errores inyectados que estas reglas NO detectan (pista: importes atípicos).

    # 5) el resto de reglas las aplica el contrato pandera
    try:
        valid = ORDERS_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        fc = err.failure_cases.dropna(subset=["index"])
        reasons = fc.groupby("index")["check"].apply(lambda s: ";".join(sorted(set(map(str, s)))))
        mask = df.index.isin(reasons.index)
        q = df[mask].copy()
        q["quarantine_reason"] = reasons.reindex(q.index).values
        bad.append(q)
        valid = df[~mask]

    quarantine = pd.concat(bad, ignore_index=True) if bad else pd.DataFrame()
    return valid.reset_index(drop=True), quarantine
