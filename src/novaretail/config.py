"""Rutas y constantes compartidas. Importad siempre las rutas desde aquí, nunca las escribáis a mano."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"            # bronze: ficheros tal cual salen del generador / fuentes
QUARANTINE = DATA / "quarantine"
CLEAN = DATA / "clean"        # silver: datos validados por los contratos
META = DATA / "_meta"         # control del generador: NO usar como dato de entrada
FIGURES = ROOT / "docs" / "figuras"

DATA_END = "2026-10-31 23:59:59"  # fin del periodo generado: lo posterior es una fecha futura
SEED = 42                     # todo el grupo usa la misma semilla
VALID_CHANNELS = ["tienda", "web", "app", "marketplace"]
VALID_STATUS = ["delivered", "returned", "cancelled", "shipped", "processing", "pending"]
ACTIVITIES = ["created", "picked", "packed", "shipped", "delivered", "returned"]
