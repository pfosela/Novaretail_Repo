"""Privacidad y triaje de tickets de NovaRetail (Persona 4).

Se mantiene la lógica reutilizable en este módulo. El notebook muestra únicamente
textos enmascarados y métricas agregadas. El conjunto original es sintético,
pero el tratamiento diseñado también minimiza la exposición de datos personales.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter

import pandas as pd


PATTERNS = {
    "EMAIL": re.compile(r"\b[\w.+%-]+@[\w.-]+\.[A-Za-z]{2,}\b", re.IGNORECASE),
    "IBAN": re.compile(r"(?<!\w)ES[ ]?\d{2}(?:[ ]?\d{4}){5}(?!\w)", re.IGNORECASE),
    "DNI": re.compile(r"\b\d{8}[A-Z]\b", re.IGNORECASE),
    "TELEFONO": re.compile(r"(?<!\d)(?:\+34[ -]?)?[6789](?:[ -]?\d){8}(?!\d)"),
    "TARJETA": re.compile(
        r"\b(tarjeta\s+(?:terminad[ao]\s+en|acabada\s+en)\s+)\d{4}\b", re.IGNORECASE
    ),
    "TARJETA_COMPLETA": re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
    "DIRECCION": re.compile(
        r"\b(?:calle|c\.|c/|vial|glorieta|v[ií]a|alameda|pasaje|cañada|avenida|avda\.?|plaza|paseo|camino|carretera|ronda|traves[ií]a|carrer)"
        r"\s+[^\n!?]{4,110}?,\s*\d{5}\s+[A-Za-zÁÉÍÓÚÜÑáéíóúüñ-]+",
        re.IGNORECASE,
    ),
}
# Los patrones específicos van antes que el genérico de teléfonos/tarjetas.
MASK_ORDER = ("EMAIL", "IBAN", "DNI", "TARJETA", "TARJETA_COMPLETA", "DIRECCION", "TELEFONO")
ORDER_ID = re.compile(r"\bNR\d{7}\b", re.IGNORECASE)
POSTAL_CITY = re.compile(r",\s*\d{5}\s+[A-Za-zÁÉÍÓÚÜÑáéíóúüñ-]+", re.IGNORECASE)
STREET_START = re.compile(
    r"\b(?:rúa|rua|calle|c\.|vial|v[ií]a|glorieta|alameda|pasaje|cañada|plaza|paseo|avenida|"
    r"traves[ií]a|ronda|camino|carrer|carretera)\s+", re.IGNORECASE
)
ADDRESS_INTRO = re.compile(
    r"(?:la nueva es|mi direcci[oó]n es|del pedido\s+NR\d{7}\s+a|pedido\s+NR\d{7}\s+a)\s+",
    re.IGNORECASE,
)


def _mask_postal_addresses(text: str) -> str:
    """El código postal permite detectar direcciones con tipos de vía no previstos."""
    for match in reversed(list(POSTAL_CITY.finditer(text))):
        before = text[:match.start()]
        intro = list(ADDRESS_INTRO.finditer(before))
        streets = list(STREET_START.finditer(before))
        intro_start = intro[-1].end() if intro else -1
        street_start = streets[-1].start() if streets else -1
        if intro_start > -1 and match.start() - intro_start <= 120:
            start = intro_start
        elif street_start > -1 and match.start() - street_start <= 120:
            start = street_start
        else:
            # Si falla la introducción por errores tipográficos, se enmascara
            # el fragmento desde el último separador y no solo el CP.
            start = max(before.rfind(". "), before.rfind("? "), before.rfind("! ")) + 2
        text = text[:start] + "[DIRECCION]" + text[match.end():]
    return text

NAME_AFTER_SOY = re.compile(
    r"(?i:\bsoy)\s+([A-ZÁÉÍÓÚÜÑ][\wÁÉÍÓÚÜÑáéíóúüñ'-]+(?:\s+"
    r"[A-ZÁÉÍÓÚÜÑ][\wÁÉÍÓÚÜÑáéíóúüñ'-]+){1,3})\b"
)


def _normalize(text: str) -> str:
    t = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in t if not unicodedata.combining(ch))


def _customer_index(customers: pd.DataFrame | None, orders: pd.DataFrame | None) -> tuple[dict, dict, dict, dict]:
    """Índices temporales para detectar nombres concretos; nunca se exportan."""
    if customers is None or customers.empty:
        return {}, {}, {}, {}
    required = {"customer_id", "first_name", "last_name"}
    if not required.issubset(customers.columns):
        raise ValueError(f"Faltan columnas de clientes: {required - set(customers.columns)}")
    names = {
        str(r.customer_id): f"{r.first_name} {r.last_name}".strip()
        for r in customers.itertuples(index=False)
    }
    emails = dict(zip(customers["email"].fillna("").str.casefold(), customers["customer_id"].astype(str))) \
        if "email" in customers.columns else {}
    phones = dict(zip(customers["phone"].fillna("").astype(str), customers["customer_id"].astype(str))) \
        if "phone" in customers.columns else {}
    ids = {}
    if orders is not None and {"order_id", "customer_id"}.issubset(orders.columns):
        ids = dict(zip(orders["order_id"].astype(str), orders["customer_id"].astype(str)))
    return names, emails, phones, ids


def _linked_customer(text: str, emails: dict, phones: dict, order_ids: dict) -> str | None:
    m = ORDER_ID.search(text)
    if m and m.group(0).upper() in order_ids:
        return order_ids[m.group(0).upper()]
    m = PATTERNS["EMAIL"].search(text)
    if m and m.group(0).casefold() in emails:
        return emails[m.group(0).casefold()]
    m = PATTERNS["TELEFONO"].search(text)
    if m and m.group(0).replace(" ", "").replace("-", "") in phones:
        return phones[m.group(0).replace(" ", "").replace("-", "")]
    return None


def mask_text(text: str, *, known_name: str | None = None) -> str:
    """Enmascara PII reconocible; conserva ID de pedido para permitir triaje.

    La regex sola no garantiza anonimización: faltas de ortografía y direcciones
    fuera de patrón requieren supervisión humana antes de publicar los datos.
    """
    if not isinstance(text, str):
        return ""
    out = _mask_postal_addresses(text)
    if known_name:
        out = re.sub(r"(?<!\w)" + re.escape(known_name) + r"(?!\w)", "[NOMBRE]", out, flags=re.I)
    # Fallback para nombres presentados como «soy Nombre Apellido».
    out = NAME_AFTER_SOY.sub(lambda m: m.group(0)[: m.start(1) - m.start(0)] + "[NOMBRE]", out)
    for label in MASK_ORDER:
        pattern = PATTERNS[label]
        if label == "TARJETA":
            out = pattern.sub(lambda m: m.group(1) + "[TARJETA]", out)
        else:
            out = pattern.sub("[TARJETA]" if label == "TARJETA_COMPLETA" else f"[{label}]", out)
    return out


def mask_tickets(
    tickets: pd.DataFrame,
    customers: pd.DataFrame | None = None,
    orders: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Devuelve tickets sin texto libre original, con `text_masked`.

    Clientes y pedidos se usan solo durante la transformación para localizar
    nombres completos sin necesitar NER externo.
    """
    if "text" not in tickets.columns:
        raise ValueError("Se espera la columna 'text' en tickets")
    names, emails, phones, order_ids = _customer_index(customers, orders)
    def one(text):
        raw = text if isinstance(text, str) else ""
        cid = _linked_customer(raw, emails, phones, order_ids)
        return mask_text(raw, known_name=names.get(cid))
    out = tickets.copy()
    out["text_masked"] = out["text"].map(one)
    return out.drop(columns=["text"])


# Indicadores identificables a través del lenguaje, no mediante datos de cliente.
CATEGORY_RULES = (
    ("proteccion_datos", (r"\brgpd\b", r"datos personales", r"supresi[oó]n", r"publicidad", r"mis datos", r"dadme de baja", r"daos de baja")),
    ("cobro_duplicado", (r"cobr(?:ado|o|ais).{0,16}dos veces", r"cargo duplicado", r"cobro duplicado", r"devolv(?:ed|er).{0,12}cargo")),
    ("cambio_direccion", (r"cambiar? la direcci[oó]n", r"nueva direcci[oó]n", r"direcci[oó]n de entrega", r"cambio de direcci[oó]n")),
    ("producto_defectuoso", (r"ha llegado roto", r"no funciona", r"producto defectuoso", r"quiero un cambio", r"pido una soluci[oó]n")),
    ("retraso_entrega", (r"no ha llegado", r"sin aparecer", r"d[ií]as esperando", r"d[ií]as de retraso", r"seguimiento del transportista", r"si no llega")),
    ("cancelacion_reembolso", (r"he cancelado", r"cancel[eé]", r"cancelad[oa].*reembolso", r"se puede recuperar")),
    ("devolucion", (r"quiero devolver", r"devolv[ií]", r"solicito la devoluci[oó]n", r"producto no es como", r"talla no es")),
    ("queja_atencion", (r"atenci[oó]n en la tienda", r"nadie me ayud[oó]", r"nadie responde", r"p[eé]simo servicio", r"esperando en el chat")),
    ("consulta_producto", (r"cu[aá]ndo volver", r"tener stock", r"est[aá] disponible", r"est[aá] agotado", r"en mi tienda", r"quer[ií]a saber")),
)
BASE_PRIORITY = {
    "proteccion_datos": "alta", "cobro_duplicado": "alta", "cambio_direccion": "alta",
    "producto_defectuoso": "media", "retraso_entrega": "media",
    "cancelacion_reembolso": "media", "devolucion": "baja", "queja_atencion": "baja",
    "consulta_producto": "baja", "sin_clasificar": "alta",  # revisión humana
}


def classify_ticket(text_masked: str) -> tuple[str, str]:
    """Clasifica solo el texto enmascarado (categoría y prioridad)."""
    if not isinstance(text_masked, str) or not text_masked.strip():
        return "sin_clasificar", "alta"
    txt = _normalize(text_masked)
    category = "sin_clasificar"
    for cat, terms in CATEGORY_RULES:
        if any(re.search(_normalize(regex), txt) for regex in terms):
            category = cat
            break
    priority = BASE_PRIORITY[category]
    if category == "retraso_entrega":
        days = re.findall(r"\b(\d{1,2})\s+dias\b", txt)
        if (days and max(map(int, days)) >= 5) or "lo necesito urgente" in txt:
            priority = "alta"
    return category, priority


def classify_tickets(masked: pd.DataFrame) -> pd.DataFrame:
    if "text_masked" not in masked.columns or "text" in masked.columns:
        raise ValueError("La clasificación solo acepta tickets previamente enmascarados")
    out = masked.copy()
    preds = out["text_masked"].map(classify_ticket)
    out["pred_category"] = preds.map(lambda pair: pair[0])
    out["pred_priority"] = preds.map(lambda pair: pair[1])
    out["needs_human_review"] = out["pred_priority"].eq("alta") | out["pred_category"].eq("proteccion_datos")
    return out


def link_ticket_segments(masked: pd.DataFrame, orders: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    """Asigna segmento solo cuando el identificador de pedido es recuperable.

    No se utiliza el segmento como variable de entrada del clasificador: se
    conserva únicamente para auditoría de equidad.
    """
    lookup = orders[["order_id", "customer_id"]].drop_duplicates("order_id").merge(
        customers[["customer_id", "segment"]], on="customer_id", how="left"
    )
    segment_by_order = lookup.set_index("order_id")["segment"].to_dict()
    out = masked.copy()
    out["segmento"] = out["text_masked"].map(
        lambda t: segment_by_order.get(m.group(0).upper(), "sin_vincular") if (m := ORDER_ID.search(t)) else "sin_vincular"
    )
    return out


def evaluate_classification(predicted: pd.DataFrame, labels: pd.DataFrame) -> dict[str, pd.DataFrame | float | int]:
    """Audita rendimiento contra labels sintéticos; nunca entrena con ellos."""
    frame = predicted.merge(labels[["ticket_id", "category", "priority"]], on="ticket_id", how="inner", validate="one_to_one")
    if len(frame) != len(predicted):
        raise ValueError("Faltan etiquetas para parte de los tickets")
    frame["error_categoria"] = frame["pred_category"] != frame["category"]
    frame["error_prioridad"] = frame["pred_priority"] != frame["priority"]
    cats = sorted(set(frame["category"]) | set(frame["pred_category"]))
    category_rows = []
    for cat in cats:
        truth, pred = frame["category"].eq(cat), frame["pred_category"].eq(cat)
        tp = (truth & pred).sum()
        category_rows.append({
            "categoria": cat, "soporte": int(truth.sum()),
            "precision": round(tp / pred.sum(), 4) if pred.sum() else 0.,
            "recall": round(tp / truth.sum(), 4) if truth.sum() else 0.,
        })
    columns = ["channel"] + (["segmento"] if "segmento" in frame else [])
    groups = []
    for field in columns:
        for group, subset in frame.groupby(field, dropna=False):
            groups.append({"dimension": field, "grupo": str(group), "n": len(subset),
                           "error_categoria_pct": round(100 * subset["error_categoria"].mean(), 2),
                           "error_prioridad_pct": round(100 * subset["error_prioridad"].mean(), 2)})
    high = frame["priority"].eq("alta")
    return {
        "n": len(frame),
        "acierto_categoria_pct": round(100 * (~frame["error_categoria"]).mean(), 2),
        "acierto_prioridad_pct": round(100 * (~frame["error_prioridad"]).mean(), 2),
        "recall_alta_pct": round(100 * (frame.loc[high, "pred_priority"] == "alta").mean(), 2) if high.any() else 0.,
        "por_categoria": pd.DataFrame(category_rows), "por_grupo": pd.DataFrame(groups),
        "comparacion": frame.drop(columns=["text_masked"], errors="ignore"),
    }


def audit_masking(raw: pd.DataFrame, masked: pd.DataFrame, customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    """Comprobación independiente: ocurrencias observables de PII sintética antes/después.

    Incluye valores exactos conocidos de los clientes vinculables y formatos
    reconocibles de DNI, IBAN, teléfono, dirección y tarjeta. No mide errores
    que también se escapan de los detectores de la auditoría.
    """
    names, emails, phones, order_ids = _customer_index(customers, orders)
    compare = raw[["ticket_id", "text"]].merge(
        masked[["ticket_id", "text_masked"]], on="ticket_id", validate="one_to_one"
    )
    counts = Counter(); leaks = Counter()
    for before, after in compare[["text", "text_masked"]].itertuples(index=False, name=None):
        before, after = str(before), str(after)
        cid = _linked_customer(before, emails, phones, order_ids)
        for addr in POSTAL_CITY.findall(before):
            counts["DIRECCION_POSTAL"] += 1
            leaks["DIRECCION_POSTAL"] += addr.casefold() in after.casefold()
        if cid and names.get(cid):
            person = names[cid]
            if person.casefold() in before.casefold():
                counts["NOMBRE"] += 1
                leaks["NOMBRE"] += person.casefold() in after.casefold()
        for label in ("EMAIL", "IBAN", "DNI", "TELEFONO", "DIRECCION", "TARJETA", "TARJETA_COMPLETA"):
            before_vals = [m.group(0) for m in PATTERNS[label].finditer(before)]
            # Un registro puede tener varias instancias. Conteo a nivel de ocurrencias.
            counts[label] += len(before_vals)
            for val in before_vals:
                leaks[label] += val.casefold() in after.casefold()
    result = []
    for label in sorted(counts):
        n = counts[label]
        result.append({"tipo": label, "detectadas_original": n, "sin_enmascarar": leaks[label],
                       "fuga_observable_pct": round(100 * leaks[label] / n, 2) if n else 0.})
    return pd.DataFrame(result)


def personal_data_inventory(datasets: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Inventario propuesto: retenciones sujetas a aprobación por el DPO.

    Si se pasan los datasets, filtra a columnas existentes en el proyecto.
    No genera una copia con información de clientes.
    """
    entries = [
        ("customers", "customer_id", "Identificador seudónimo", "Vincular pedidos y soporte", "Contrato, art. 6.1.b RGPD", "Vigencia de relación + revisión DPO", "Atención/Operaciones"),
        ("customers", "first_name", "Identificativo", "Identificación y atención", "Contrato, art. 6.1.b RGPD", "Vigencia de relación + revisión DPO", "Atención"),
        ("customers", "last_name", "Identificativo", "Identificación y atención", "Contrato, art. 6.1.b RGPD", "Vigencia de relación + revisión DPO", "Atención"),
        ("customers", "email", "Contacto", "Seguimiento de pedidos y soporte", "Contrato, art. 6.1.b RGPD", "Vigencia de relación + revisión DPO", "Atención"),
        ("customers", "phone", "Contacto", "Notificaciones de entrega", "Contrato, art. 6.1.b RGPD", "Vigencia de relación + revisión DPO", "Atención/Logística"),
        ("customers", "birth_year", "Dato demográfico", "Análisis agregado; revisar necesidad", "Interés legítimo solo si supera ponderación", "Eliminar si no es necesario", "Analítica restringida"),
        ("customers", "postal_code", "Localización indirecta", "Distribución y entrega", "Contrato, art. 6.1.b RGPD", "Vigencia de relación + revisión DPO", "Logística"),
        ("customers", "region", "Localización indirecta", "Planificación agregada", "Interés legítimo, previa ponderación", "Agregado sin identificadores", "Analítica"),
        ("customers", "segment", "Perfil comercial", "Análisis de equidad del servicio", "Interés legítimo, previa ponderación", "Revisión periódica; sin exportar PII", "DPO/Analítica"),
        ("customers", "acquisition_channel", "Perfil comercial", "Evaluar captación y retención", "Interés legítimo, previa ponderación", "Revisión periódica; agregado", "Marketing/Analítica"),
        ("orders", "customer_id", "Identificador seudónimo", "Relacionar compra y cliente", "Contrato, art. 6.1.b RGPD", "Contrato + obligaciones aplicables", "Operaciones/Finanzas"),
        ("sales", "customer_id", "Identificador seudónimo", "Conciliación ventas", "Contrato, art. 6.1.b RGPD", "Contrato + obligaciones aplicables", "Finanzas"),
        ("tickets", "text", "Texto libre: DNI, nombre, contacto y direcciones", "Resolver incidencias y derechos", "Contrato o obligación legal según solicitud", "Propuesta: 24 meses, revisión DPO", "Soporte autorizado; acceso auditado"),
        ("tickets", "channel", "Metadato que puede vincularse", "Enrutar y auditar servicio", "Contrato, art. 6.1.b RGPD", "Propuesta: 24 meses, revisión DPO", "Soporte/Analítica agregada"),
        ("invoices", "supplier", "Identificativo eventual (autónomos)", "Facturación y contabilidad", "Obligación legal, art. 6.1.c RGPD", "Plazo legal contable/fiscal aplicable", "Finanzas"),
        ("invoices", "supplier_cif", "Fiscal (eventualmente personal)", "Facturación y contabilidad", "Obligación legal, art. 6.1.c RGPD", "Plazo legal contable/fiscal aplicable", "Finanzas"),
    ]
    cols = ["conjunto", "campo", "categoria", "finalidad", "base_legal", "conservacion", "acceso"]
    df = pd.DataFrame(entries, columns=cols)
    if datasets:
        df = df[df.apply(lambda r: r["conjunto"] not in datasets or r["campo"] in datasets[r["conjunto"]].columns, axis=1)].reset_index(drop=True)
    return df
