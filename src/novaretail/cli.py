"""Puntos de entrada reproducibles.

    uv run novaretail generate      # genera data/raw (semilla del grupo)
    uv run novaretail validate      # bronze -> silver: data/clean + data/quarantine + informe
    uv run novaretail external      # descarga la fuente externa (Open-Meteo) con caché
    uv run novaretail all           # todo lo anterior
"""
import argparse
import json
import subprocess
import sys

from novaretail import contracts, ingest
from novaretail.config import CLEAN, DATA, QUARANTINE, ROOT, SEED


def generate(seed: int = SEED, scale: float = 1.0) -> None:
    script = ROOT / "scripts" / "generate_data.py"
    subprocess.run([sys.executable, str(script), "--seed", str(seed), "--scale", str(scale), "--out", str(DATA)],
                   check=True)


def _save(name: str, valid, quarantine) -> None:
    valid.to_parquet(CLEAN / f"{name}.parquet", index=False)
    q = quarantine.copy()
    for c in q.columns:  # la cuarentena guarda el dato tal cual llegó: todo a texto para no fallar al escribir
        if q[c].dtype == object:
            q[c] = q[c].astype("string")
    q.to_parquet(QUARANTINE / f"{name}.parquet", index=False)


def validate() -> int:
    """Valida todos los datasets. Devuelve 1 (y el pipeline se para) si falla alguna regla crítica."""
    CLEAN.mkdir(parents=True, exist_ok=True)
    QUARANTINE.mkdir(parents=True, exist_ok=True)
    products = ingest.load_products()

    # sales primero: sus líneas válidas sirven para conciliar el importe de los pedidos
    sales, sales_q, r_sales = contracts.clean_sales(ingest.load_sales(), products)
    orders, orders_q, r_orders = contracts.clean_orders(ingest.load_orders(), sales, sales_q)
    events, events_q, r_events = contracts.clean_events(ingest.load_events())
    inv, inv_q, r_inv = contracts.clean_inventory(ingest.load_inventory())

    reports = [r_orders, r_sales, r_events, r_inv]
    for name, v, q in [("orders", orders, orders_q), ("sales", sales, sales_q),
                       ("fulfilment_events", events, events_q), ("inventory_weekly", inv, inv_q)]:
        _save(name, v, q)
    # dimensiones: se copian a silver tal cual (no tienen errores inyectados)
    products.to_parquet(CLEAN / "products.parquet", index=False)
    ingest.load_stores().to_parquet(CLEAN / "stores.parquet", index=False)
    ingest.load_customers().to_parquet(CLEAN / "customers.parquet", index=False)

    (CLEAN / "quality_report.json").write_text(
        json.dumps([r.to_dict() for r in reports], ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'dataset':<20}{'entrada':>10}{'válidas':>10}{'cuarentena':>12}{'%':>7}")
    for r in reports:
        print(f"{r.dataset:<20}{r.rows_in:>10}{r.rows_valid:>10}{r.rows_quarantined:>12}{r.quarantine_rate:>7.1%}")
    critical = [(r.dataset, c) for r in reports for c in r.critical]
    if critical:
        for ds, c in critical:
            print(f"[CRÍTICO] {ds}: {c}", file=sys.stderr)
        print("Pipeline detenido: hay reglas críticas incumplidas.", file=sys.stderr)
        return 1
    print("Informe completo en data/clean/quality_report.json")
    return 0


def external() -> int:
    from novaretail.external_api import ExternalSourceError, load_weather
    try:
        df, origin = load_weather()
    except ExternalSourceError as exc:
        print(f"[aviso] fuente externa no disponible: {exc}", file=sys.stderr)
        return 0  # no es crítica para el resto del pipeline
    print(f"meteorología: {len(df)} registros ({origin})")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(prog="novaretail")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--seed", type=int, default=SEED)
    g.add_argument("--scale", type=float, default=1.0)
    sub.add_parser("validate")
    sub.add_parser("external")
    sub.add_parser("all")
    args = ap.parse_args()
    if args.cmd == "generate":
        generate(args.seed, args.scale)
    elif args.cmd == "validate":
        sys.exit(validate())
    elif args.cmd == "external":
        sys.exit(external())
    elif args.cmd == "all":
        generate()
        code = validate()
        if code:
            sys.exit(code)
        sys.exit(external())


if __name__ == "__main__":
    main()
