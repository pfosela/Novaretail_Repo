# NovaRetail · Plataforma de datos y soporte a la decisión

Proyecto de grupo de la Unit 17 (Business Process Support), Pearson HND Applied Computing & AI, UCS 2026-2027.

## Puesta en marcha (desde cero)

```bash
git clone <url-del-repo> && cd novaretail
uv sync                       # instala Python y dependencias exactas (uv.lock)
uv run novaretail all         # genera data/raw y valida -> data/clean + data/quarantine
uv run pytest                 # tests
```

Para los notebooks, abrid `notebooks/` en VS Code o en Jupyter con el kernel del `.venv`.

## Estructura

```
├── scripts/generate_data.py   # generador reproducible (semilla 42) -> data/raw, data/_meta
├── src/novaretail/
│   ├── config.py              # rutas y constantes compartidas
│   ├── cli.py                 # `novaretail generate | validate | all`
│   ├── kpis.py                # P1: KPIs y árbol de valor
│   ├── ingest.py              # P2: carga de fuentes (bronze)
│   ├── contracts.py           # P2: contratos de calidad y cuarentena (bronze -> silver)
│   ├── external_api.py        # P2: fuente externa vía API + contrato pydantic
│   ├── flow.py                # P3: métricas de flujo del event log
│   ├── stock.py               # P3: riesgo de rotura por margen
│   ├── cohorts.py             # P3: cohortes de clientes
│   └── privacy.py             # P4: enmascarado y clasificación de tickets
├── notebooks/                 # un notebook por bloque de análisis
├── tests/
├── docs/
│   ├── ficha_decision.md      # plantilla de la sesión 5
│   ├── contribuciones.md      # tabla de contribución individual (obligatoria)
│   ├── uso_ia.md              # anexo de uso de IA (obligatorio)
│   └── figuras/               # gráficos que van a la memoria
└── data/                      # NO se sube: se regenera con `uv run novaretail all`
    ├── raw/                   # bronze
    ├── quarantine/            # filas rechazadas, con el motivo
    ├── clean/                 # silver
    └── _meta/                 # control del generador (no usar como entrada)
```

La lógica va en `src/novaretail/` y los notebooks solo la llaman y muestran resultados. Así el código se puede testear y reutilizar en el pipeline (entrega 2) y en el dashboard.

## Reparto (Entrega 1)

| Persona | Bloque | Módulos |
|---|---|---|
| 1 | Contexto, procesos y valor (apartados 1–2) | `kpis.py`, `01_contexto_kpis` |
| 2 | Calidad e ingesta (apartado 3) | `ingest.py`, `contracts.py`, `external_api.py`, `02_calidad_ingesta` |
| 3 | Event log, stock y cohortes (apartado 3) | `flow.py`, `stock.py`, `cohorts.py`, `03`, `04` |
| 4 | Legal, ética y seguridad (apartado 5) | `privacy.py`, `05_privacidad_tickets` |

Las tareas pendientes de cada uno están marcadas como `TODO(Persona N)` en el código.

## Forma de trabajar

1. Nunca se trabaja directamente en `main`. Cada uno usa su propia rama: `git switch -c p3/event-log`.
2. Commits pequeños y con un mensaje claro: `flow: P50/P90 por almacén`.
3. Pull request a `main` (con la plantilla) y revisión de otro miembro antes de hacer el merge.
4. Las librerías se añaden con `uv add <paquete>` y se suben `pyproject.toml` y `uv.lock`. Nunca `pip install`.
5. Antes de cada entrega: clon limpio, `uv sync`, `uv run novaretail all` y ejecutar todos los notebooks.
6. Las claves de API y las credenciales del warehouse van en `.env`, que ya está en `.gitignore`.
