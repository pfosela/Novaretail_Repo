"""Puntos de entrada reproducibles.

    uv run novaretail generate      # genera data/raw (semilla del grupo)
    uv run novaretail validate      # bronze -> silver: data/clean + data/quarantine
    uv run novaretail all           # todo lo anterior
"""
import argparse
import subprocess
import sys

from novaretail import contracts, ingest
from novaretail.config import CLEAN, DATA, QUARANTINE, ROOT, SEED


def generate(seed: int = SEED, scale: float = 1.0) -> None:
    script = ROOT / "scripts" / "generate_data.py"
    subprocess.run([sys.executable, str(script), "--seed", str(seed), "--scale", str(scale), "--out", str(DATA)],
                   check=True)


def validate() -> None:
    CLEAN.mkdir(parents=True, exist_ok=True)
    QUARANTINE.mkdir(parents=True, exist_ok=True)
    raw = ingest.load_orders()
    valid, quarantine = contracts.clean_orders(raw)
    valid.to_parquet(CLEAN / "orders.parquet", index=False)
    quarantine.to_parquet(QUARANTINE / "orders.parquet", index=False)
    print(f"orders: {len(raw)} filas en bruto -> {len(valid)} válidas, {len(quarantine)} en cuarentena")
    if len(quarantine):
        print(quarantine["quarantine_reason"].value_counts().to_string())
    # TODO(Persona 2): resto de datasets. Si falla una regla CRÍTICA, sys.exit(1) para parar el pipeline.


def main() -> None:
    ap = argparse.ArgumentParser(prog="novaretail")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--seed", type=int, default=SEED)
    g.add_argument("--scale", type=float, default=1.0)
    sub.add_parser("validate")
    sub.add_parser("all")
    args = ap.parse_args()
    if args.cmd == "generate":
        generate(args.seed, args.scale)
    elif args.cmd == "validate":
        validate()
    elif args.cmd == "all":
        generate()
        validate()


if __name__ == "__main__":
    main()
