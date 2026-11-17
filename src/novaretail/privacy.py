"""Enmascarado de datos personales y clasificación de tickets. Responsable: Persona 4.

El texto de los tickets contiene DNI, emails, teléfonos, IBAN, direcciones y nombres.
Nada sale de esta función sin enmascarar: el resto del proyecto solo usa `text_masked`.
"""
import re

import pandas as pd

# Ejemplos de patrones. TODO(Persona 4): completar (IBAN, tarjeta, direcciones, nombres) y medir
# cuántos se escapan usando data/_meta para comprobarlo.
PATTERNS = {
    "DNI": re.compile(r"\b\d{8}[A-Z]\b"),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b"),
    "TELEFONO": re.compile(r"\b[6789]\d{8}\b"),
}


def mask_text(text: str) -> str:
    for label, pattern in PATTERNS.items():
        text = pattern.sub(f"[{label}]", text)
    return text


def mask_tickets(tickets: pd.DataFrame) -> pd.DataFrame:
    out = tickets.copy()
    out["text_masked"] = out["text"].map(mask_text)
    return out.drop(columns=["text"])


def classify_ticket(text_masked: str) -> tuple[str, str]:
    """Devuelve (categoría, prioridad).

    TODO(Persona 4): reglas por palabras clave. Evaluar contra data/_meta/tickets_labels.csv
    y revisar si el error es mayor en algún canal o segmento (sesgo).
    """
    raise NotImplementedError


def personal_data_inventory() -> pd.DataFrame:
    """TODO(Persona 4): tabla campo | dataset | categoría | finalidad | base legal | retención | acceso."""
    raise NotImplementedError
