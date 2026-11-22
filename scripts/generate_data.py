"""
Generador reproducible de datos sintéticos de NovaRetail (Unit 17 · Business Process Support).

Uso:
    uv run python generate_data.py                 # semilla 42, escala 1.0, salida en data/
    uv run python generate_data.py --seed 7 --scale 0.5 --out data

Con la misma semilla y escala, la salida es idéntica en cualquier máquina.
Todos los datos personales son ficticios (Faker es_ES).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# Parámetros del negocio
START = pd.Timestamp("2025-01-01")
END = pd.Timestamp("2026-10-31 23:59:59")

REGIONS = {
    "Norte": ["Bilbao", "Santander", "Oviedo"],
    "Sur": ["Sevilla", "Málaga", "Granada"],
    "Este": ["Barcelona", "Valencia", "Zaragoza"],
    "Centro": ["Madrid", "Toledo", "Valladolid"],
}
WAREHOUSE_BY_REGION = {"Norte": "WH-BIO", "Sur": "WH-SVQ", "Este": "WH-BCN", "Centro": "WH-MAD"}
FORMATS = ["flagship", "estandar", "express", "outlet"]
FORMAT_SIZE = {"flagship": 3.0, "estandar": 1.6, "express": 0.8, "outlet": 1.0}
FORMAT_ASSORTMENT = {"flagship": 220, "estandar": 150, "express": 80, "outlet": 110}

# categoría: (precio medio, margen bruto, tasa de devolución, peso en catálogo, plazo proveedor días)
CATEGORIES = {
    "Electronica": (160, 0.22, 0.06, 0.12, 14),
    "Hogar": (42, 0.40, 0.05, 0.15, 10),
    "Moda": (34, 0.56, 0.18, 0.20, 21),
    "Deportes": (52, 0.42, 0.08, 0.12, 12),
    "Belleza": (21, 0.62, 0.03, 0.12, 7),
    "Juguetes": (27, 0.45, 0.04, 0.10, 18),
    "Gourmet": (14, 0.35, 0.01, 0.10, 5),
    "Libros": (17, 0.30, 0.02, 0.09, 6),
}
NOUNS = {
    "Electronica": ["Auriculares", "Altavoz", "Cargador", "Tablet", "Smartwatch", "Teclado", "Raton", "Camara"],
    "Hogar": ["Lampara", "Sarten", "Juego de toallas", "Cojin", "Organizador", "Vajilla", "Cafetera", "Manta"],
    "Moda": ["Camiseta", "Pantalon", "Chaqueta", "Vestido", "Zapatillas", "Bolso", "Sudadera", "Camisa"],
    "Deportes": ["Esterilla", "Mancuernas", "Botella termica", "Mochila", "Raqueta", "Balon", "Banda elastica", "Gorra"],
    "Belleza": ["Crema hidratante", "Serum", "Perfume", "Champu", "Protector solar", "Mascarilla", "Labial", "Gel"],
    "Juguetes": ["Puzle", "Peluche", "Juego de mesa", "Bloques", "Coche RC", "Muneca", "Kit ciencia", "Pelota"],
    "Gourmet": ["Aceite AOVE", "Cafe en grano", "Te verde", "Chocolate", "Jamon loncheado", "Queso curado", "Mermelada", "Turron"],
    "Libros": ["Novela", "Guia de viaje", "Libro de cocina", "Comic", "Ensayo", "Cuento infantil", "Agenda", "Atlas"],
}
BRANDS = ["Novatek", "Zentro", "Alba", "Kairo", "Lumen", "Brisa", "Sierra", "Marea"]
VARIANTS = ["", "Pro", "Lite", "Max", "Plus", "Mini", "Eco", "Classic"]

CHANNELS = ["tienda", "web", "app", "marketplace"]
ACQ_CHANNELS = ["tienda", "web", "app", "marketplace", "redes_sociales", "referido"]
ACQ_P = [0.38, 0.22, 0.16, 0.10, 0.09, 0.05]
ACQ_TO_PREF = {"tienda": "tienda", "web": "web", "app": "app", "marketplace": "marketplace",
               "redes_sociales": "app", "referido": "web"}

CARRIERS = ["RapidLog", "EnviaYa", "TransIberia", "PaqPlus"]
# probabilidad de transportista por almacén
CARRIER_P = {
    "WH-MAD": [0.35, 0.30, 0.20, 0.15],
    "WH-BCN": [0.30, 0.25, 0.30, 0.15],
    "WH-SVQ": [0.25, 0.35, 0.25, 0.15],
    "WH-BIO": [0.20, 0.25, 0.20, 0.35],
}
# mediana (horas) y dispersión del tramo shipped -> delivered
CARRIER_TRANSIT = {"RapidLog": (18, 0.35), "EnviaYa": (24, 0.40), "TransIberia": (28, 0.85), "PaqPlus": (22, 0.45)}

DISCOUNT_LEVELS = np.array([0, 5, 10, 15, 20, 30, 40, 50])
DISCOUNT_P = {
    ("tienda", 0): [0.70, 0.08, 0.10, 0.05, 0.05, 0.02, 0.00, 0.00],
    ("tienda", 1): [0.30, 0.05, 0.15, 0.10, 0.20, 0.12, 0.06, 0.02],
    ("web", 0): [0.60, 0.10, 0.12, 0.08, 0.06, 0.03, 0.01, 0.00],
    ("web", 1): [0.22, 0.05, 0.13, 0.12, 0.22, 0.15, 0.08, 0.03],
    ("app", 0): [0.50, 0.12, 0.18, 0.10, 0.06, 0.03, 0.01, 0.00],
    ("app", 1): [0.18, 0.05, 0.14, 0.13, 0.22, 0.16, 0.09, 0.03],
    ("marketplace", 0): [0.35, 0.08, 0.15, 0.15, 0.14, 0.08, 0.04, 0.01],
    ("marketplace", 1): [0.10, 0.03, 0.10, 0.12, 0.20, 0.22, 0.15, 0.08],
}


# Utilidades
def _norm(x):
    x = np.asarray(x, dtype=float)
    return x / x.sum()


def dni_letter(n: np.ndarray) -> np.ndarray:
    letters = np.array(list("TRWAGMYFPDXBNJZSQVHLCKE"))
    return letters[n % 23]


def promo_flag(dates: pd.DatetimeIndex) -> np.ndarray:
    """1 si la fecha cae en rebajas, Black Friday/Cyber Monday o Navidad."""
    m, d = dates.month, dates.day
    rebajas = ((m == 1) & (d >= 7)) | (m == 2) & (d <= 15) | (m == 7) | ((m == 8) & (d <= 15))
    bf = np.zeros(len(dates), dtype=bool)
    for y in (2025, 2026):
        nov = pd.date_range(f"{y}-11-01", f"{y}-11-30")
        black_friday = nov[(nov.day_of_week == 4)][-1]
        bf |= (dates.normalize() >= black_friday - pd.Timedelta(days=3)) & (
            dates.normalize() <= black_friday + pd.Timedelta(days=3))
    return (rebajas | bf).astype(int)


def daily_demand_factor(days: pd.DatetimeIndex) -> np.ndarray:
    weekday = np.array([0.9, 0.85, 0.9, 0.95, 1.15, 1.35, 0.9])[days.day_of_week]
    month = np.array([1.15, 0.85, 0.9, 0.95, 0.95, 1.0, 1.05, 0.8, 0.95, 1.0, 1.25, 1.55])[days.month - 1]
    t = (days - START).days / 365.0
    growth = 1 + 0.12 * t
    f = weekday * month * growth
    for y in (2025, 2026):
        nov = pd.date_range(f"{y}-11-01", f"{y}-11-30")
        bf = nov[(nov.day_of_week == 4)][-1]
        f = np.where(days == bf, f * 3.0, f)
        f = np.where((days > bf) & (days <= bf + pd.Timedelta(days=3)), f * 1.8, f)
    return f


def into_window(ts: pd.Series, start_h: int, end_h: int, rng: np.random.Generator) -> pd.Series:
    """Si una marca cae fuera del horario [start_h, end_h), la mueve al inicio del siguiente turno."""
    h = ts.dt.hour
    jitter = pd.to_timedelta(rng.integers(0, 90, len(ts)), unit="m")
    early = h < start_h
    late = h >= end_h
    out = ts.copy()
    out[early] = ts[early].dt.normalize() + pd.Timedelta(hours=start_h) + jitter[early.values]
    out[late] = ts[late].dt.normalize() + pd.Timedelta(days=1, hours=start_h) + jitter[late.values]
    return out


def skip_sunday(ts: pd.Series, hour: int, rng: np.random.Generator) -> pd.Series:
    sunday = ts.dt.day_of_week == 6
    out = ts.copy()
    jitter = pd.to_timedelta(rng.integers(0, 240, sunday.sum()), unit="m")
    out[sunday] = ts[sunday].dt.normalize() + pd.Timedelta(days=1, hours=hour) + jitter
    return out


# Dimensiones
def build_stores(rng):
    rows, i = [], 1
    for region, cities in REGIONS.items():
        for k in range(10):
            fmt = "flagship" if k == 0 else rng.choice(FORMATS[1:], p=[0.55, 0.30, 0.15])
            rows.append({
                "store_id": f"S{i:03d}",
                "store_name": f"NovaRetail {cities[k % 3]} {k + 1:02d}",
                "region": region,
                "city": cities[k % 3],
                "format": fmt,
                "opening_date": (pd.Timestamp("2012-01-01") + pd.Timedelta(days=int(rng.integers(0, 4400)))).date(),
                "sqm": int(FORMAT_SIZE[fmt] * rng.normal(600, 80)),
            })
            i += 1
    return pd.DataFrame(rows)


def build_products(rng, n=400):
    cats = list(CATEGORIES)
    weights = np.array([CATEGORIES[c][3] for c in cats])
    cat = rng.choice(cats, size=n, p=weights / weights.sum())
    rows = []
    for i, c in enumerate(cat):
        mean_price, margin, *_ = CATEGORIES[c]
        price = max(2.0, mean_price * rng.lognormal(0, 0.45))
        price = math.floor(price) + 0.99 if price > 5 else round(price, 2)
        m = float(np.clip(margin + rng.normal(0, 0.06), 0.08, 0.80))
        name = f"{rng.choice(NOUNS[c])} {rng.choice(BRANDS)} {rng.choice(VARIANTS)}".strip()
        rows.append({
            "product_id": f"P{i + 1:04d}",
            "product_name": name,
            "category": c,
            "list_price_eur": price,
            "unit_cost_eur": round(price * (1 - m), 2),
            "supplier_lead_time_days": int(CATEGORIES[c][4] + rng.integers(-2, 4)),
        })
    df = pd.DataFrame(rows)
    # popularidad tipo Zipf (oculta, solo se usa internamente)
    pop = 1 / np.power(rng.permutation(n) + 1, 0.8)
    return df, pop / pop.sum()


def build_customers(rng, fake: Faker, n=30000):
    early = int(n * 0.25)
    signup_early = START - pd.to_timedelta(rng.integers(1, 730, early), unit="D")
    span = (END.normalize() - START).days
    # crecimiento: más altas en el segundo año
    u = rng.random(n - early) ** 0.8
    signup_late = START + pd.to_timedelta((u * span).astype(int), unit="D")
    signup = pd.DatetimeIndex(np.concatenate([signup_early.values, signup_late.values]))
    acq = rng.choice(ACQ_CHANNELS, size=n, p=ACQ_P)
    region = rng.choice(list(REGIONS), size=n, p=[0.18, 0.24, 0.27, 0.31])
    segment = np.where(rng.random(n) < 0.18, "premium", "estandar")
    birth_year = rng.integers(1950, 2007, n)

    # ciclo de vida (oculto): unos se quedan, otros abandonan
    loyal = rng.random(n) < 0.35
    lifetime = rng.exponential(240, n).astype(int) + 20
    churn_date = np.where(loyal, pd.Timestamp("2030-01-01").value,
                          (signup + pd.to_timedelta(lifetime, unit="D")).values.astype("datetime64[ns]").astype(np.int64))
    activity = rng.lognormal(0, 0.7, n) * np.where(segment == "premium", 2.2, 1.0)

    first = [fake.first_name() for _ in range(n)]
    last = [f"{fake.last_name()} {fake.last_name()}" for _ in range(n)]
    domains = rng.choice(["gmail.com", "hotmail.com", "yahoo.es", "outlook.es", "icloud.com", "telefonica.net"], n)
    email = [
        f"{f.lower()}.{l.split()[0].lower()}{rng.integers(1, 999)}@{d}"
        .replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u").replace("ñ", "n")
        for f, l, d in zip(first, last, domains)
    ]
    phone = [f"6{rng.integers(10000000, 99999999)}" for _ in range(n)]
    df = pd.DataFrame({
        "customer_id": [f"C{i + 1:06d}" for i in range(n)],
        "first_name": first,
        "last_name": last,
        "email": email,
        "phone": phone,
        "birth_year": birth_year,
        "postal_code": [fake.postcode() for _ in range(n)],
        "region": region,
        "segment": segment,
        "acquisition_channel": acq,
        "signup_date": signup.date,
    })
    hidden = {"churn_ns": churn_date, "activity": activity, "signup_ns": signup.values.astype(np.int64)}
    return df, hidden


# Pedidos y ventas
def build_orders_and_sales(rng, stores, products, pop, customers, hidden, scale):
    days = pd.date_range(START, END.normalize(), freq="D")
    base = 105 * scale
    lam = base * daily_demand_factor(days)
    n_per_day = rng.poisson(lam)

    region_stores = {r: stores.loc[stores.region == r, "store_id"].to_numpy() for r in REGIONS}
    region_store_w = {}
    for r in REGIONS:
        s = stores.loc[stores.region == r]
        w = s["format"].map(FORMAT_SIZE).to_numpy()
        region_store_w[r] = w / w.sum()

    cust_region = customers["region"].to_numpy()
    cust_pref = customers["acquisition_channel"].map(ACQ_TO_PREF).to_numpy()
    rows_c, rows_d, rows_ch = [], [], []
    for d, n in zip(days, n_per_day):
        dn = d.value
        elig = np.nonzero((hidden["signup_ns"] <= dn) & (hidden["churn_ns"] >= dn))[0]
        w = hidden["activity"][elig]
        c = rng.choice(elig, size=n, p=w / w.sum())
        t = (d - START).days / 365.0
        online = 0.46 + 0.08 * t
        mix = np.array([1 - online, online * 0.45, online * 0.35, online * 0.20])
        use_pref = rng.random(n) < 0.55
        ch = np.where(use_pref, cust_pref[c], rng.choice(CHANNELS, size=n, p=mix))
        rows_c.append(c)
        rows_d.append(np.full(n, dn))
        rows_ch.append(ch)
    cidx = np.concatenate(rows_c)
    day_ns = np.concatenate(rows_d)
    channel = np.concatenate(rows_ch)
    n_orders = len(cidx)

    # hora de creación: tienda en horario comercial, online con pico nocturno
    hour_store = rng.choice(np.arange(10, 21), size=n_orders, p=np.array([4, 6, 8, 9, 7, 6, 7, 9, 11, 13, 10]) / 90)
    hour_online = rng.choice(np.arange(24), size=n_orders,
                             p=_norm([2, 1, 1, .5, .5, .5, 1, 2, 3, 4, 5, 5, 5, 5, 4, 4, 4, 5, 6, 7, 8, 9, 8, 4.5]))
    hour = np.where(channel == "tienda", hour_store, hour_online)
    created = pd.to_datetime(day_ns) + pd.to_timedelta(hour, unit="h") + pd.to_timedelta(rng.integers(0, 3600, n_orders), unit="s")

    order = pd.DataFrame({
        "order_id": [f"NR{i + 1:07d}" for i in range(n_orders)],
        "customer_idx": cidx,
        "channel": channel,
        "created_ts": created,
    }).sort_values("created_ts", kind="stable").reset_index(drop=True)
    order["order_id"] = [f"NR{i + 1:07d}" for i in range(n_orders)]
    order["customer_id"] = customers["customer_id"].to_numpy()[order.customer_idx]
    order["region"] = cust_region[order.customer_idx]
    order["store"] = [rng.choice(region_stores[r], p=region_store_w[r]) for r in order["region"]]

    # --- líneas de venta
    n_lines = np.minimum(1 + rng.poisson(0.8, n_orders), 6)
    oi = np.repeat(np.arange(n_orders), n_lines)
    is_online = order["channel"].to_numpy()[oi] != "tienda"
    cat = products["category"].to_numpy()
    p_online = pop * np.where(cat == "Electronica", 1.6, 1.0) * np.where(cat == "Gourmet", 0.6, 1.0)
    p_online /= p_online.sum()
    prod_idx = np.where(is_online,
                        rng.choice(len(products), size=len(oi), p=p_online),
                        rng.choice(len(products), size=len(oi), p=pop))
    units = 1 + rng.poisson(0.35, len(oi))
    line_ts = order["created_ts"].to_numpy()[oi]
    promo = promo_flag(pd.DatetimeIndex(line_ts))
    ch_line = order["channel"].to_numpy()[oi]
    disc = np.empty(len(oi), dtype=int)
    for (ch, pf), p in DISCOUNT_P.items():
        m = (ch_line == ch) & (promo == pf)
        disc[m] = rng.choice(DISCOUNT_LEVELS, size=m.sum(), p=p)
    years_before_end = (END - pd.DatetimeIndex(line_ts)).days.to_numpy() / 365.0
    inflation = 1 - 0.03 * years_before_end
    unit_price = products["list_price_eur"].to_numpy()[prod_idx] * inflation
    unit_cost = products["unit_cost_eur"].to_numpy()[prod_idx] * inflation
    line_rev = np.round(unit_price * units * (1 - disc / 100), 2)
    line_cost = unit_cost * units

    sales = pd.DataFrame({
        "sale_id": [f"SL{i + 1:08d}" for i in range(len(oi))],
        "order_id": order["order_id"].to_numpy()[oi],
        "sale_ts": line_ts,
        "product_id": products["product_id"].to_numpy()[prod_idx],
        "store_id": order["store"].to_numpy()[oi],
        "customer_id": order["customer_id"].to_numpy()[oi],
        "channel": ch_line,
        "units": units,
        "unit_price_eur": np.round(unit_price, 2),
        "discount_pct": disc,
    })

    rev = np.bincount(oi, weights=line_rev, minlength=n_orders)
    cogs = np.bincount(oi, weights=line_cost, minlength=n_orders)
    tot_units = np.bincount(oi, weights=units, minlength=n_orders)
    ret_rate = np.zeros(n_orders)
    np.maximum.at(ret_rate, oi, np.array([CATEGORIES[c][2] for c in cat])[prod_idx])
    order["revenue_eur"] = np.round(rev, 2)
    order["_cogs"] = cogs
    order["_units"] = tot_units
    order["_ret_rate"] = ret_rate
    return order, sales


def build_fulfilment(rng, order):
    """Event log de pedidos online + tiempos, estados y costes variables."""
    on = order["channel"] != "tienda"
    o = order.loc[on].copy()
    n = len(o)
    o["warehouse"] = o["region"].map(WAREHOUSE_BY_REGION)
    o["carrier"] = [rng.choice(CARRIERS, p=CARRIER_P[w]) for w in o["warehouse"]]

    # carga diaria por almacén -> multiplicador de retraso
    day = o["created_ts"].dt.normalize()
    load = o.groupby([o["warehouse"], day])["order_id"].transform("count")
    mean_load = o.groupby("warehouse")["order_id"].transform("count") / day.nunique()
    rel = (load / mean_load).to_numpy()
    is_bcn = (o["warehouse"] == "WH-BCN").to_numpy()

    def hours(median, sigma, size):
        return rng.lognormal(np.log(median), sigma, size)

    created = o["created_ts"]
    pick_h = hours(3, 0.6, n) * np.maximum(rel, 1) ** 0.6 * np.where(is_bcn, 1.3, 1.0)
    picked = into_window(created + pd.to_timedelta(pick_h, unit="h"), 7, 23, rng)
    # cuello de botella deliberado: embalaje en WH-BCN se satura con la carga
    pack_med = np.where(is_bcn, 4.5, 1.0)
    pack_h = hours(1, 0.5, n) * pack_med * np.maximum(rel, 1) ** np.where(is_bcn, 2.2, 0.4)
    packed = into_window(picked + pd.to_timedelta(pack_h, unit="h"), 7, 23, rng)

    # recogida del transportista: 12:00 y 18:00, sin recogidas en domingo
    ph = packed.dt.hour + packed.dt.minute / 60
    base_day = packed.dt.normalize()
    slot = np.where(ph < 12, 12, np.where(ph < 18, 18, 36))
    shipped = base_day + pd.to_timedelta(slot, unit="h") + pd.to_timedelta(rng.integers(0, 45, n), unit="m")
    shipped = skip_sunday(shipped, 12, rng)

    med = o["carrier"].map({k: v[0] for k, v in CARRIER_TRANSIT.items()}).to_numpy().astype(float)
    sig = o["carrier"].map({k: v[1] for k, v in CARRIER_TRANSIT.items()}).to_numpy().astype(float)
    med = np.where((o["carrier"] == "PaqPlus").to_numpy() & (o["region"] == "Norte").to_numpy(), med * 1.9, med)
    transit = rng.lognormal(np.log(med), sig)
    delivered = into_window(shipped + pd.to_timedelta(transit, unit="h"), 9, 21, rng)
    delivered = skip_sunday(delivered, 10, rng)

    promise_h = np.where(o["channel"] == "marketplace", 96, 72)
    promised = o["created_ts"] + pd.to_timedelta(promise_h, unit="h")

    cancelled = rng.random(n) < 0.025
    cancel_after_pick = rng.random(n) < 0.4
    p_ret = o["_ret_rate"].to_numpy() * np.where(o["channel"] == "marketplace", 1.25, 1.0) * 1.3
    returned = (~cancelled) & (rng.random(n) < p_ret)
    returned_ts = delivered + pd.to_timedelta(rng.uniform(2, 25, n), unit="D")

    ev = []
    sysres = o["channel"].map({"web": "web-checkout", "app": "app-checkout", "marketplace": "mkt-api"}).to_numpy()
    wh = o["warehouse"].to_numpy()
    pickers = np.char.add(np.char.add(wh.astype(str), "-PK"), np.char.zfill(rng.integers(1, 26, n).astype(str), 2))
    packers = np.char.add(np.char.add(wh.astype(str), "-EM"), np.char.zfill(rng.integers(1, 13, n).astype(str), 2))
    carriers = o["carrier"].to_numpy()

    def add(mask, activity, ts, res):
        ev.append(pd.DataFrame({
            "order_id": o["order_id"].to_numpy()[mask],
            "activity": activity,
            "event_ts": ts.to_numpy()[mask],
            "resource": res[mask],
            "warehouse": wh[mask],
            "carrier": carriers[mask],
        }))

    all_ = np.ones(n, dtype=bool)
    add(all_, "created", created, sysres)
    add(~cancelled | cancel_after_pick, "picked", picked, pickers)
    add(~cancelled, "packed", packed, packers)
    add(~cancelled, "shipped", shipped, carriers)
    add(~cancelled, "delivered", delivered, carriers)
    add(returned, "returned", returned_ts, carriers)
    events = pd.concat(ev, ignore_index=True)
    # recorte al final del periodo: lo que aún no ha pasado no existe
    events = events[events["event_ts"] <= END].copy()

    last = events.sort_values("event_ts").groupby("order_id")["activity"].last()
    status_map = {"created": "pending", "picked": "processing", "packed": "processing",
                  "shipped": "shipped", "delivered": "delivered", "returned": "returned"}
    st = o["order_id"].map(last).map(status_map)
    st = np.where(cancelled, "cancelled", st)
    o["status"] = st
    dl = events[events.activity == "delivered"].set_index("order_id")["event_ts"]
    o["delivered_ts"] = o["order_id"].map(dl)
    o["promised_ts"] = promised
    events = events.sort_values(["order_id", "event_ts"], kind="stable").reset_index(drop=True)
    return o[["order_id", "status", "promised_ts", "delivered_ts", "warehouse", "carrier"]], events


def finish_orders(rng, order, ful):
    order = order.merge(ful, on="order_id", how="left")
    store = order["channel"] == "tienda"
    order.loc[store, "promised_ts"] = order.loc[store, "created_ts"]
    order.loc[store, "delivered_ts"] = order.loc[store, "created_ts"]
    store_ret = store & (rng.random(len(order)) < order["_ret_rate"] * 0.6)
    order.loc[store, "status"] = np.where(store_ret[store], "returned", "delivered")

    ch = order["channel"]
    units = order["_units"]
    rev = order["revenue_eur"]
    fulfil = np.select(
        [ch == "tienda", ch.isin(["web", "app"]), ch == "marketplace"],
        [0.8 * units, 3.9 + 0.5 * units + 0.015 * rev, 3.9 + 0.5 * units + 0.15 * rev],
    )
    reverse = np.where(order["status"] == "returned", np.where(ch == "tienda", 1.5, 5.5), 0)
    order["variable_cost_eur"] = np.round(order["_cogs"] + fulfil + reverse, 2)
    order.loc[order["status"] == "cancelled", "variable_cost_eur"] = np.round(
        3.0 + 0.5 * units[order["status"] == "cancelled"], 2)
    cols = ["order_id", "customer_id", "store", "channel", "created_ts", "promised_ts", "delivered_ts",
            "status", "revenue_eur", "variable_cost_eur"]
    return order[cols].copy()


# Errores de calidad deliberados
def inject_order_errors(rng, orders):
    df = orders.copy()
    for c in ["created_ts", "promised_ts", "delivered_ts"]:
        df[c] = pd.to_datetime(df[c]).dt.strftime("%Y-%m-%d %H:%M:%S")
    df = df.astype({"store": object, "channel": object, "status": object, "created_ts": object,
                    "delivered_ts": object, "promised_ts": object})
    df["revenue_eur"] = df["revenue_eur"].astype(float)
    df["variable_cost_eur"] = df["variable_cost_eur"].astype(float)
    n = len(df)
    issues = {}

    def pick(frac, mask=None):
        pool = np.arange(n) if mask is None else np.nonzero(mask)[0]
        k = int(round(frac * n)) if mask is None else int(round(frac * len(pool)))
        return rng.choice(pool, size=k, replace=False)

    # coste ausente (más frecuente en marketplace)
    i1 = np.concatenate([pick(0.02), pick(0.08, (df["channel"] == "marketplace").to_numpy())])
    i1 = np.unique(i1)
    df.loc[i1, "variable_cost_eur"] = np.nan
    issues["variable_cost_missing"] = len(i1)

    # entrega anterior a la creación
    online_deliv = (df["channel"] != "tienda").to_numpy() & df["delivered_ts"].notna().to_numpy()
    i2 = pick(0.01, online_deliv)
    d = pd.to_datetime(df.loc[i2, "created_ts"]) - pd.to_timedelta(rng.integers(2, 72, len(i2)), unit="h")
    df.loc[i2, "delivered_ts"] = d.dt.strftime("%Y-%m-%d %H:%M:%S")
    issues["delivered_before_created"] = len(i2)

    # formato de fecha alternativo (dd/mm/aaaa)
    i3 = pick(0.01)
    df.loc[i3, "created_ts"] = pd.to_datetime(df.loc[i3, "created_ts"]).dt.strftime("%d/%m/%Y %H:%M")
    issues["created_ts_alt_format"] = len(i3)

    # fechas en el futuro
    i4 = pick(0.002)
    df.loc[i4, "created_ts"] = (pd.to_datetime(df.loc[i4, "created_ts"], format="mixed", dayfirst=False)
                                + pd.DateOffset(years=1)).dt.strftime("%Y-%m-%d %H:%M:%S")
    issues["created_ts_future"] = len(i4)

    # variantes de canal
    i5 = pick(0.02)
    variants = {"web": ["Web", "WEB ", " web"], "app": ["APP", "App"], "marketplace": ["Marketplace", "market_place"],
                "tienda": ["Tienda", "TIENDA", "store"]}
    df.loc[i5, "channel"] = [rng.choice(variants[c]) for c in df.loc[i5, "channel"]]
    issues["channel_inconsistent_label"] = len(i5)

    # importes imposibles
    i6 = pick(0.003)
    df.loc[i6, "revenue_eur"] = -df.loc[i6, "revenue_eur"]
    i7 = pick(0.001)
    df.loc[i7, "revenue_eur"] = df.loc[i7, "revenue_eur"] * 100
    issues["revenue_negative"] = len(i6)
    issues["revenue_outlier_x100"] = len(i7)

    # tienda nula o con espacios, estado vacío
    i8 = pick(0.005)
    df.loc[i8, "store"] = None
    i9 = pick(0.01)
    df.loc[i9, "store"] = df.loc[i9, "store"].astype(str) + "  "
    i10 = pick(0.003)
    df.loc[i10, "status"] = ""
    issues["store_null"] = len(i8)
    issues["store_trailing_spaces"] = len(i9)
    issues["status_empty"] = len(i10)

    # duplicados exactos y reenvíos con el mismo order_id
    i11 = pick(0.015)
    dup_exact = df.loc[i11].copy()
    i12 = pick(0.005)
    dup_resent = df.loc[i12].copy()
    dup_resent["created_ts"] = (pd.to_datetime(dup_resent["created_ts"], format="mixed")
                                + pd.to_timedelta(rng.integers(1, 600, len(i12)), unit="s")).dt.strftime("%Y-%m-%d %H:%M:%S")
    issues["duplicate_exact_rows"] = len(i11)
    issues["duplicate_order_id_resent"] = len(i12)

    df = pd.concat([df, dup_exact, dup_resent], ignore_index=True)
    df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    return df, issues


def inject_sales_errors(rng, sales):
    df = sales.copy()
    n = len(df)
    issues = {}
    i1 = rng.choice(n, int(n * 0.003), replace=False)
    df.loc[i1, "product_id"] = [f"P9{rng.integers(100, 999)}" for _ in i1]
    issues["product_id_orphan"] = len(i1)
    i2 = rng.choice(n, int(n * 0.002), replace=False)
    df.loc[i2, "discount_pct"] = rng.choice([-10, 110, 150, 500], len(i2))
    issues["discount_out_of_range"] = len(i2)
    i3 = rng.choice(n, int(n * 0.001), replace=False)
    df.loc[i3, "units"] = 0
    issues["units_zero"] = len(i3)
    return df, issues


def inject_event_errors(rng, events):
    df = events.copy().astype({"activity": object, "carrier": object})
    n = len(df)
    issues = {}
    i1 = rng.choice(n, int(n * 0.004), replace=False)
    dup = df.loc[i1].copy()
    issues["event_duplicated"] = len(i1)
    deliv = np.nonzero((df["activity"] == "delivered").to_numpy())[0]
    i2 = rng.choice(deliv, int(len(deliv) * 0.003), replace=False)
    df.loc[i2, "event_ts"] = df.loc[i2, "event_ts"] - pd.to_timedelta(rng.integers(60, 120, len(i2)), unit="h")
    issues["event_out_of_order"] = len(i2)
    i3 = rng.choice(n, int(n * 0.003), replace=False)
    df.loc[i3, "carrier"] = None
    issues["carrier_null"] = len(i3)
    i4 = rng.choice(n, int(n * 0.002), replace=False)
    df.loc[i4, "activity"] = df.loc[i4, "activity"].str.capitalize()
    issues["activity_capitalized"] = len(i4)
    df = pd.concat([df, dup], ignore_index=True).sort_values(["order_id", "event_ts"], kind="stable")
    return df.reset_index(drop=True), issues


# Inventario semanal
def build_inventory(rng, stores, products, sales):
    combos = []
    pop_rank = sales.groupby("product_id")["units"].sum().reindex(products["product_id"]).fillna(0)
    w = (pop_rank + 1).to_numpy()
    w = w / w.sum()
    for _, s in stores.iterrows():
        k = FORMAT_ASSORTMENT[s["format"]]
        prods = rng.choice(products["product_id"], size=k, replace=False, p=w)
        combos.append(pd.DataFrame({"store_id": s["store_id"], "product_id": prods, "region": s["region"]}))
    combos = pd.concat(combos, ignore_index=True)

    st = sales[sales["channel"] == "tienda"].copy()
    st["week"] = pd.to_datetime(st["sale_ts"]).dt.to_period("W-SUN").dt.start_time
    weeks = pd.date_range(START - pd.Timedelta(days=START.day_of_week), END, freq="W-MON")
    # `sales` solo recoge ventas identificadas (cliente con cuenta). La demanda total de tienda
    # es ~TRAFFIC_MULT veces mayor y mantiene la misma estacionalidad, mix por tienda y por producto.
    TRAFFIC_MULT = 20
    week_units = st.groupby("week")["units"].sum().reindex(weeks).fillna(0).to_numpy()
    store_share = st.groupby("store_id")["units"].sum()
    store_share = (store_share / store_share.sum()).reindex(combos["store_id"]).fillna(0).to_numpy()
    prod_units = st.groupby("product_id")["units"].sum().reindex(products["product_id"]).fillna(0) + 0.5
    # dentro de cada tienda, el reparto entre productos del surtido sigue la popularidad global
    pu = prod_units.reindex(combos["product_id"]).to_numpy()
    within = pu / pd.Series(pu).groupby(combos["store_id"].to_numpy()).transform("sum").to_numpy()
    lam = TRAFFIC_MULT * np.outer(store_share * within, week_units)
    dem = rng.poisson(lam).astype(float)
    n, W = dem.shape

    prod = products.set_index("product_id")
    lead = prod.loc[combos["product_id"], "supplier_lead_time_days"].to_numpy()
    margin = 1 - prod.loc[combos["product_id"], "unit_cost_eur"].to_numpy() / prod.loc[combos["product_id"], "list_price_eur"].to_numpy()
    lead_var = np.maximum(2, lead + rng.integers(-2, 5, n))
    lead_w = np.ceil(lead_var / 7).astype(int)

    avg = dem.mean(axis=1)            # el planificador usa la media anual: ignora la estacionalidad
    std = dem.std(axis=1)
    # política mal calibrada en alto margen de Este y Sur (semilla de roturas que importan)
    safety = np.where((margin > 0.5) & combos["region"].isin(["Este", "Sur"]).to_numpy(), 0.2, 1.1)
    s_point = avg * (lead_w + 0.5) + safety * std * np.sqrt(lead_w)
    S = s_point + avg * 2 + 2
    on_hand = np.ceil(S).astype(float)
    pipeline = np.zeros((n, W + 12))
    oh_rec = np.zeros((n, W))
    oo_rec = np.zeros((n, W))
    for t in range(W):
        on_hand = on_hand + pipeline[:, t]
        on_hand = np.maximum(on_hand - dem[:, t], 0)
        outstanding = pipeline[:, t + 1:].sum(axis=1)
        pos = on_hand + outstanding
        q = np.where(pos < s_point, np.ceil(S - pos), 0)
        arr = t + lead_w
        pipeline[np.arange(n), arr] += q
        oh_rec[:, t] = on_hand
        oo_rec[:, t] = pipeline[:, t + 1:].sum(axis=1)

    inv = pd.DataFrame({
        "store_id": np.repeat(combos["store_id"].to_numpy(), W),
        "product_id": np.repeat(combos["product_id"].to_numpy(), W),
        "week": np.tile(weeks.date, n),
        "on_hand_units": oh_rec.ravel().astype(int),
        "on_order_units": oo_rec.ravel().astype(int),
        "lead_time_days": np.repeat(lead_var, W),
    })
    issues = {}
    i1 = rng.choice(len(inv), int(len(inv) * 0.001), replace=False)
    inv.loc[i1, "on_hand_units"] = -inv.loc[i1, "on_hand_units"] - 1
    issues["on_hand_negative"] = len(i1)
    drop = rng.choice(len(inv), int(len(inv) * 0.004), replace=False)
    inv = inv.drop(index=drop).reset_index(drop=True)
    issues["snapshot_rows_missing"] = len(drop)
    return inv, issues


# Tickets de soporte (texto libre con datos personales sintéticos)
TEMPLATES = {
    "retraso_entrega": [
        "Hola, soy {nombre}. Mi pedido {pedido} tenía que llegar hace {dias} días y sigue sin aparecer. Mi teléfono es {telefono}. ¿Qué está pasando?",
        "Buenas, el pedido {pedido} no ha llegado todavía. Lo necesito urgente, por favor llamadme al {telefono}. {nombre}",
        "Llevo {dias} días esperando el pedido {pedido}. El seguimiento del transportista no se actualiza. Mi email es {email}.",
        "Es la tercera vez que escribo. Pedido {pedido}, {dias} días de retraso. Si no llega mañana quiero cancelarlo y que me devolváis el dinero. {nombre}, DNI {dni}",
    ],
    "devolucion": [
        "Quiero devolver el pedido {pedido}, la talla no es la correcta. ¿Cómo lo hago? Mi dirección es {direccion}.",
        "Hola, devolví el pedido {pedido} hace {dias} días y aún no tengo el reembolso de {importe} euros. {nombre}, {email}",
        "Solicito la devolución del pedido {pedido}. El producto no es como en la foto. Contacto: {telefono}",
    ],
    "cancelacion_reembolso": [
        "He cancelado el pedido {pedido} y todavía no veo el reembolso en la tarjeta terminada en {tarjeta}. {nombre}",
        "Cancelé el pedido {pedido} por error, ¿se puede recuperar? Mi correo es {email}",
    ],
    "producto_defectuoso": [
        "El producto del pedido {pedido} ha llegado roto. Adjunto fotos. Quiero un cambio. {nombre}, tel. {telefono}",
        "Lo que he recibido en el pedido {pedido} no funciona desde el primer día. Pido una solución. {email}",
    ],
    "cobro_duplicado": [
        "Me habéis cobrado dos veces el pedido {pedido} ({importe} euros). Mi IBAN es {iban}, devolvedme el cargo por favor. {nombre}",
        "Cargo duplicado en la tarjeta terminada en {tarjeta} por el pedido {pedido}. Necesito que lo reviséis ya. DNI {dni}",
    ],
    "cambio_direccion": [
        "Necesito cambiar la dirección de entrega del pedido {pedido} a {direccion}. Gracias, {nombre}",
        "¿Puedo cambiar la dirección del pedido {pedido}? La nueva es {direccion}. Teléfono {telefono}",
    ],
    "consulta_producto": [
        "¿Cuándo volveréis a tener stock de {producto}? En mi tienda está agotado desde hace semanas. {email}",
        "Hola, quería saber si {producto} está disponible en la tienda de {ciudad}. Gracias",
    ],
    "proteccion_datos": [
        "Solicito el acceso a todos mis datos personales y la supresión de mi cuenta conforme al RGPD. {nombre}, DNI {dni}, {email}",
        "No quiero recibir más publicidad. Daos de baja mi email {email} y mi teléfono {telefono}. Quiero saber de dónde habéis sacado mis datos.",
    ],
    "queja_atencion": [
        "La atención en la tienda de {ciudad} fue lamentable. Nadie me ayudó con el pedido {pedido}. {nombre}",
        "Llevo una hora esperando en el chat y nadie responde. Pedido {pedido}. Pésimo servicio.",
    ],
}
PRIORITY_BASE = {"retraso_entrega": "media", "devolucion": "baja", "cancelacion_reembolso": "media",
                 "producto_defectuoso": "media", "cobro_duplicado": "alta", "cambio_direccion": "alta",
                 "consulta_producto": "baja", "proteccion_datos": "alta", "queja_atencion": "baja"}


def add_typos(text: str, rng) -> str:
    chars = list(text)
    for _ in range(rng.integers(1, 4)):
        j = int(rng.integers(1, len(chars) - 1))
        chars[j], chars[j - 1] = chars[j - 1], chars[j]
    return "".join(chars)


def build_tickets(rng, fake, orders_clean, customers, products, stores, scale):
    o = orders_clean[orders_clean["channel"] != "tienda"].copy()
    late_h = (o["delivered_ts"].fillna(END) - o["promised_ts"]).dt.total_seconds() / 3600
    o["late_days"] = np.maximum(0, np.ceil(late_h / 24)).astype(int)
    w = 1 + 6 * (o["late_days"] > 0) + 3 * o["status"].isin(["returned", "cancelled"])
    k = int(len(o) * 0.08)
    sel = o.sample(n=k, weights=w, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    cust = customers.set_index("customer_id").loc[sel["customer_id"]].reset_index()

    cats = []
    for st, late in zip(sel["status"], sel["late_days"]):
        if late > 0 and rng.random() < 0.75:
            cats.append("retraso_entrega")
        elif st == "returned" and rng.random() < 0.8:
            cats.append("devolucion")
        elif st == "cancelled" and rng.random() < 0.8:
            cats.append("cancelacion_reembolso")
        else:
            cats.append(rng.choice(["producto_defectuoso", "cobro_duplicado", "cambio_direccion", "consulta_producto",
                                    "proteccion_datos", "queja_atencion", "devolucion"],
                                   p=[0.22, 0.08, 0.10, 0.20, 0.07, 0.18, 0.15]))
    channels = rng.choice(["email", "chat", "telefono", "formulario_web", "redes_sociales"], size=k,
                          p=[0.38, 0.27, 0.15, 0.12, 0.08])
    store_city = dict(zip(stores["store_id"], stores["city"]))
    order_city = sel["store"].map(store_city)

    rows, labels = [], []
    for i in range(k):
        c = cats[i]
        dni_n = int(rng.integers(10000000, 99999999))
        vals = {
            "nombre": f"{cust.loc[i, 'first_name']} {cust.loc[i, 'last_name']}",
            "pedido": sel.loc[i, "order_id"],
            "email": cust.loc[i, "email"],
            "telefono": cust.loc[i, "phone"],
            "dni": f"{dni_n}{dni_letter(np.array([dni_n]))[0]}",
            "direccion": f"{fake.street_address()}, {cust.loc[i, 'postal_code']} {order_city.iloc[i]}",
            "dias": max(2, int(sel.loc[i, "late_days"]) or int(rng.integers(2, 12))),
            "importe": f"{sel.loc[i, 'revenue_eur']:.2f}".replace(".", ","),
            "tarjeta": f"{rng.integers(1000, 9999)}",
            "iban": fake.iban(),
            "producto": rng.choice(products["product_name"]),
            "ciudad": order_city.iloc[i],
        }
        text = rng.choice(TEMPLATES[c]).format(**vals)
        ch = channels[i]
        if ch == "redes_sociales":
            text = text.lower().replace("hola, ", "").replace("buenas, ", "") + rng.choice([" 😡", " 🙄", " !!!", " #NovaRetail"])
        elif ch == "telefono":
            text = "Transcripción de llamada: " + text
        if rng.random() < 0.12:
            text = add_typos(text, rng)

        created = sel.loc[i, "promised_ts"] + pd.to_timedelta(rng.uniform(2, 120), unit="h")
        if c in ("devolucion", "producto_defectuoso") and pd.notna(sel.loc[i, "delivered_ts"]):
            created = sel.loc[i, "delivered_ts"] + pd.to_timedelta(rng.uniform(4, 240), unit="h")
        created = min(created, END)

        prio = PRIORITY_BASE[c]
        if c == "retraso_entrega" and vals["dias"] >= 5:
            prio = "alta"
        if cust.loc[i, "segment"] == "premium" and prio == "baja":
            prio = "media"
        rows.append({"ticket_id": f"TK{i + 1:06d}", "created_ts": created.strftime("%Y-%m-%dT%H:%M:%S"),
                     "channel": ch, "text": text})
        labels.append({"ticket_id": f"TK{i + 1:06d}", "category": c, "priority": prio,
                       "needs_human_review": prio == "alta" or c == "proteccion_datos"})

    tickets = pd.DataFrame(rows).sort_values("created_ts").reset_index(drop=True)
    old_to_new = {old: f"TK{j + 1:06d}" for j, old in enumerate(tickets["ticket_id"])}
    tickets["ticket_id"] = tickets["ticket_id"].map(old_to_new)
    lab = pd.DataFrame(labels)
    lab["ticket_id"] = lab["ticket_id"].map(old_to_new)
    lab = lab.sort_values("ticket_id").reset_index(drop=True)
    return tickets, lab


# Facturas PDF y catálogo web
SUPPLIERS = [
    ("Distribuciones Tecnova S.L.", "B12345674", ["Electronica"]),
    ("Textiles del Levante S.A.", "A23456785", ["Moda"]),
    ("Casa y Confort S.L.", "B34567896", ["Hogar"]),
    ("Deporte Total Iberia S.L.", "B45678907", ["Deportes"]),
    ("Cosmética Natural del Sur S.L.", "B56789018", ["Belleza"]),
    ("Juguetería Mediterránea S.L.", "B67890129", ["Juguetes"]),
    ("Delicias de la Tierra S.Coop.", "F78901230", ["Gourmet"]),
    ("Editorial Horizonte S.A.", "A89012341", ["Libros"]),
]
VAT = {"Gourmet": 0.10, "Libros": 0.04}


def build_invoices(rng, products, out_dir: Path, n=48):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    out_dir.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    truth = []
    bad = set(rng.choice(n, 4, replace=False).tolist())
    for i in range(n):
        sup, cif, cats = SUPPLIERS[i % len(SUPPLIERS)]
        cat = cats[0]
        date = START + pd.Timedelta(days=int(rng.integers(0, (END - START).days)))
        number = f"F{date.year}-{rng.integers(1000, 9999)}"
        prods = products[products["category"] == cat].sample(n=int(rng.integers(3, 9)), random_state=int(rng.integers(1e9)))
        lines = [["Ref.", "Descripción", "Uds.", "Precio unit.", "Importe"]]
        base = 0.0
        for _, p in prods.iterrows():
            u = int(rng.integers(10, 200))
            amt = round(u * p["unit_cost_eur"], 2)
            base += amt
            lines.append([p["product_id"], p["product_name"], str(u), f"{p['unit_cost_eur']:.2f} €".replace(".", ","),
                          f"{amt:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")])
        base = round(base, 2)
        rate = VAT.get(cat, 0.21)
        vat = round(base * rate, 2)
        total = round(base + vat, 2)
        shown_total = total
        if i in bad:  # factura con descuadre deliberado
            shown_total = round(total + float(rng.choice([-50, 12.5, 100, 0.99])), 2)

        def eur(x):
            return f"{x:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")

        path = out_dir / f"factura_{number}.pdf"
        doc = SimpleDocTemplate(str(path), pagesize=A4, invariant=1)
        story = [
            Paragraph(f"<b>{sup}</b><br/>CIF: {cif}", styles["Normal"]),
            Spacer(1, 12),
            Paragraph(f"<b>FACTURA</b> Nº {number}<br/>Fecha: {date.strftime('%d/%m/%Y')}", styles["Heading2"]),
            Paragraph("Cliente: NovaRetail S.A. · CIF A87654321 · Calle Ficticia 1, 28001 Madrid", styles["Normal"]),
            Spacer(1, 12),
        ]
        t = Table(lines, colWidths=[50, 210, 40, 80, 90])
        t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
                               ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
        story += [t, Spacer(1, 12),
                  Paragraph(f"Base imponible: {eur(base)}<br/>IVA ({int(rate * 100)}%): {eur(vat)}<br/>"
                            f"<b>TOTAL: {eur(shown_total)}</b>", styles["Normal"])]
        doc.build(story)
        truth.append({"file": path.name, "invoice_number": number, "invoice_date": date.date(), "supplier": sup,
                      "supplier_cif": cif, "base_eur": base, "vat_rate": rate, "vat_eur": vat,
                      "total_shown_eur": shown_total, "total_correct": shown_total == total})
    return pd.DataFrame(truth)


def build_catalog(rng, products, out_dir: Path, per_page=50):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = products.copy()
    web_price = p["list_price_eur"] * np.where(rng.random(len(p)) < 0.12, rng.uniform(0.7, 0.9, len(p)), 1.0)
    p["web_price"] = np.round(web_price, 2)
    p["stock"] = np.where(rng.random(len(p)) < 0.07, "Agotado", "Disponible")
    pages = math.ceil(len(p) / per_page)
    for k in range(pages):
        chunk = p.iloc[k * per_page:(k + 1) * per_page]
        cards = []
        for _, r in chunk.iterrows():
            price = f"{r['web_price']:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
            old = ""
            if r["web_price"] < r["list_price_eur"]:
                old = f'<span class="old-price">{r["list_price_eur"]:.2f} €</span>'.replace(".", ",")
            cards.append(
                f'<div class="product-card" data-sku="{r["product_id"]}">\n'
                f'  <h3 class="name">{r["product_name"]}</h3>\n'
                f'  <span class="category">{r["category"]}</span>\n'
                f'  <span class="price">{price} €</span>{old}\n'
                f'  <span class="stock">{r["stock"]}</span>\n</div>')
        nav = " ".join(f'<a href="page_{j + 1:02d}.html">{j + 1}</a>' for j in range(pages))
        html = ("<!DOCTYPE html>\n<html lang=\"es\"><head><meta charset=\"utf-8\"><title>NovaRetail · Catálogo</title></head>\n"
                f"<body><h1>Catálogo NovaRetail</h1><nav class=\"pagination\">{nav}</nav>\n"
                f"<main class=\"catalog\">\n{chr(10).join(cards)}\n</main></body></html>\n")
        (out_dir / f"page_{k + 1:02d}.html").write_text(html, encoding="utf-8")
    return pages


# Main
def main():
    ap = argparse.ArgumentParser(description="Generador de datos sintéticos de NovaRetail")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scale", type=float, default=1.0, help="1.0 ≈ 80.000 pedidos")
    ap.add_argument("--out", type=Path, default=Path("data"))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    Faker.seed(args.seed)
    fake = Faker("es_ES")
    raw = args.out / "raw"
    meta = args.out / "_meta"
    raw.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)

    print("· dimensiones")
    stores = build_stores(rng)
    products, pop = build_products(rng)
    customers, hidden = build_customers(rng, fake, n=int(30000 * max(args.scale, 0.2)))

    print("· pedidos y ventas")
    order, sales = build_orders_and_sales(rng, stores, products, pop, customers, hidden, args.scale)
    print("· event log de fulfilment")
    ful, events = build_fulfilment(rng, order)
    orders_clean = finish_orders(rng, order, ful)

    print("· inventario semanal")
    inventory, inv_issues = build_inventory(rng, stores, products, sales)
    print("· tickets de soporte")
    tickets, ticket_labels = build_tickets(rng, fake, orders_clean, customers, products, stores, args.scale)
    print("· facturas PDF y catálogo web")
    invoices_truth = build_invoices(rng, products, raw / "invoices")
    n_pages = build_catalog(rng, products, raw / "catalog")

    print("· errores de calidad")
    orders_dirty, ord_issues = inject_order_errors(rng, orders_clean)
    sales_dirty, sal_issues = inject_sales_errors(rng, sales)
    events_dirty, ev_issues = inject_event_errors(rng, events)

    print("· escritura")
    stores.to_csv(raw / "stores.csv", index=False)
    products.to_csv(raw / "products.csv", index=False)
    customers.to_csv(raw / "customers.csv", index=False)
    orders_dirty.to_csv(raw / "orders.csv", index=False)
    sales_dirty.to_parquet(raw / "sales.parquet", index=False)
    inventory.to_parquet(raw / "inventory_weekly.parquet", index=False)
    events_dirty.to_csv(raw / "fulfilment_events.csv", index=False)
    with open(raw / "tickets.json", "w", encoding="utf-8") as f:
        json.dump(tickets.to_dict(orient="records"), f, ensure_ascii=False, indent=1)

    ticket_labels.to_csv(meta / "tickets_labels.csv", index=False)
    invoices_truth.to_csv(meta / "invoices_ground_truth.csv", index=False)
    info = {
        "seed": args.seed, "scale": args.scale,
        "period": [str(START.date()), str(END.date())],
        "rows": {"stores": len(stores), "products": len(products), "customers": len(customers),
                 "orders_raw": len(orders_dirty), "orders_unique": len(orders_clean), "sales": len(sales_dirty),
                 "inventory_weekly": len(inventory), "fulfilment_events": len(events_dirty),
                 "tickets": len(tickets), "invoices_pdf": len(invoices_truth), "catalog_pages": n_pages},
        "injected_quality_issues": {"orders": ord_issues, "sales": sal_issues, "fulfilment_events": ev_issues,
                                    "inventory_weekly": inv_issues,
                                    "invoices": {"total_mismatch": int((~invoices_truth["total_correct"]).sum())}},
    }
    (meta / "generation_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(info["rows"], indent=2))


if __name__ == "__main__":
    main()
