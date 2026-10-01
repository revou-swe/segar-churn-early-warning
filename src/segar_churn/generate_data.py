"""Synthetic stand-in for the PT Segar Nusantara anchor dataset.

The program provides the real anchor dataset to enrolled participants. This
generator produces a *structurally identical* fictional dataset so the repo can
be cloned and run cold by anyone (Checkpoint 2B requirement) without sharing
confidential data. Swap the CSVs in data/raw/ for the program dataset and the
rest of the pipeline runs unchanged.

Tables written to data/raw/:
    outlets.csv      one row per retail partner (master data)
    orders.csv       one row per order / invoice
    visits.csv       one row per sales-rep visit
    complaints.csv   one row per complaint or product return

A handful of realistic data-quality defects are injected on purpose
(duplicates, negative values, future dates, missing city) so that
data_checks.py has something real to catch.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

REGIONS = {
    #  name               share  cities                                         depot_km_scale
    "Jabodetabek":       (0.28, ["Jakarta Barat", "Jakarta Timur", "Bekasi", "Tangerang", "Depok", "Bogor"], 9),
    "Jawa Barat":        (0.19, ["Bandung", "Cimahi", "Cirebon", "Tasikmalaya", "Karawang"], 16),
    "Jawa Tengah":       (0.16, ["Semarang", "Solo", "Magelang", "Pekalongan", "Purwokerto"], 18),
    "Jawa Timur":        (0.19, ["Surabaya", "Sidoarjo", "Malang", "Kediri", "Gresik"], 15),
    "Sumatera Utara":    (0.10, ["Medan", "Binjai", "Pematangsiantar", "Deli Serdang"], 22),
    "Sulawesi Selatan":  (0.08, ["Makassar", "Gowa", "Maros", "Parepare"], 24),
}
REGION_CODE = {"Jabodetabek": "JKT", "Jawa Barat": "JBR", "Jawa Tengah": "JTG",
               "Jawa Timur": "JTM", "Sumatera Utara": "SMU", "Sulawesi Selatan": "SLS"}
OUTLET_TYPES = {
    # name           share  orders/mo  log(order value IDR)  skus/order  base discount  hazard shift
    "Warung":        (0.44, 2.2,       np.log(650_000),      5,          0.020,          0.30),
    "Minimarket":    (0.25, 3.6,       np.log(2_400_000),    11,         0.030,          0.00),
    "Grosir":        (0.19, 4.8,       np.log(7_500_000),    16,         0.045,         -0.15),
    "Supermarket":   (0.12, 6.5,       np.log(14_000_000),   24,         0.035,         -0.45),
}
NAME_PREFIX = {"Warung": ["Warung", "Toko"], "Minimarket": ["Mini Market", "Toko"],
               "Grosir": ["Grosir", "Toko Grosir", "UD"], "Supermarket": ["Swalayan", "Supermarket"]}
NAME_WORDS = ["Maju Jaya", "Sumber Rezeki", "Berkah", "Makmur", "Sinar Abadi", "Sentosa", "Barokah",
              "Lancar", "Mulia", "Harapan Baru", "Sejahtera", "Indah", "Bintang Timur", "Cahaya",
              "Mitra Usaha", "Karya Mandiri", "Murah Meriah", "Segar Jaya", "Rukun", "Damai",
              "Tunas Baru", "Anugerah", "Bahagia", "Pelita", "Mekar Sari", "Subur", "Setia Kawan"]
REP_FIRST = ["Andi", "Budi", "Citra", "Dewi", "Eko", "Fajar", "Gita", "Hendra", "Indra", "Joko", "Kartika",
             "Lukman", "Maya", "Nanda", "Putri", "Rizky", "Sari", "Taufik", "Wulan", "Yoga", "Agus",
             "Bayu", "Dimas", "Fitri", "Hadi", "Irma", "Rina", "Teguh", "Yudi", "Lestari"]

COMPETITOR_REGION = "Jawa Timur"   # a rival distributor enters this region ...
COMPETITOR_FROM = 15               # ... from Jan-2026 (month index 15)


def _month_starts() -> pd.DatetimeIndex:
    return pd.date_range(C.START_MONTH + "-01", periods=C.N_MONTHS + 1, freq="MS")


def generate(n_outlets: int = 2600, seed: int = C.SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    months = _month_starts()
    T = C.N_MONTHS

    # ---------------- outlet master ----------------
    reg_names = list(REGIONS)
    region = rng.choice(reg_names, n_outlets, p=[REGIONS[r][0] for r in reg_names])
    type_names = list(OUTLET_TYPES)
    otype = rng.choice(type_names, n_outlets, p=[OUTLET_TYPES[t][0] for t in type_names])
    city = np.array([rng.choice(REGIONS[r][1]) for r in region])
    distance = np.round(np.array([rng.gamma(2.0, REGIONS[r][2] / 2) for r in region]) + 0.5, 1)

    # onboarding: 78% already partners before the window, rest join during it
    pre = rng.random(n_outlets) < 0.78
    onboard_month = np.where(pre, -rng.integers(1, 60, n_outlets), rng.integers(0, T - 3, n_outlets))

    # sales reps: ~45 outlets each, nested in region; each rep has a visit habit
    rep_id = np.empty(n_outlets, dtype=object)
    rep_rate = {}
    rep_name = {}
    fi = 0
    for r in reg_names:
        idx = np.where(region == r)[0]
        n_rep = max(2, int(round(len(idx) / 45)))
        code = REGION_CODE[r]
        reps = [f"SR-{code}-{k+1:02d}" for k in range(n_rep)]
        for rp in reps:
            rep_rate[rp] = float(np.clip(rng.normal(2.2, 0.8), 0.5, 4.0))
            rep_name[rp] = REP_FIRST[fi % len(REP_FIRST)]
            fi += 1
        rep_id[idx] = rng.choice(reps, len(idx))

    names = []
    seen = set()
    for i in range(n_outlets):
        while True:
            nm = f"{rng.choice(NAME_PREFIX[otype[i]])} {rng.choice(NAME_WORDS)}"
            if rng.random() < 0.5:
                nm += f" {city[i].split()[-1]}"
            if rng.random() < 0.25:
                nm += f" {rng.integers(1, 9)}"
            if nm not in seen:
                seen.add(nm)
                break
        names.append(nm)

    outlets = pd.DataFrame({
        "outlet_id": [f"OUT-{i+10001}" for i in range(n_outlets)],
        "outlet_name": names,
        "outlet_type": otype,
        "region": region,
        "city": city,
        "rep_id": rep_id,
        "rep_name": [rep_name[r] for r in rep_id],
        "onboard_date": [(months[0] + pd.DateOffset(months=int(m))).strftime("%Y-%m-%d") for m in onboard_month],
        "distance_km": distance,
    })

    # ---------------- monthly simulation ----------------
    typ = {k: np.array([OUTLET_TYPES[t][j] for t in otype]) for j, k in
           enumerate(["share", "lam", "mu", "sku", "disc", "haz"])}
    rrate = np.array([rep_rate[r] for r in rep_id])
    comp = (region == COMPETITOR_REGION)

    h = rng.normal(0, 1, n_outlets)                 # latent "relationship health"
    state = np.zeros(n_outlets, dtype=int)          # 0 active, 1-2 declining (months left), 9 lost
    state[:] = 0
    order_rows, visit_rows, compl_rows = [], [], []
    oid = 1
    for t in range(T):
        live = onboard_month <= t
        tenure = t - onboard_month
        comp_on = comp & (t >= COMPETITOR_FROM)
        # health dynamics
        h = 0.86 * h + rng.normal(0, 0.42, n_outlets) - np.where(comp_on, 0.12, 0.0)

        # rep visits this month (exogenous habit, fewer for far outlets)
        vis_lam = rrate * np.where(otype == "Warung", 0.7, 1.0) * np.exp(-distance / 80)
        n_vis = rng.poisson(vis_lam) * live * (state != 9)
        # monthly churn hazard
        logit = (-4.35 - 1.05 * h + typ["haz"] + 0.45 * (tenure < 6) + 0.55 * (n_vis == 0)
                 + 0.85 * comp_on + 0.006 * distance)
        p_churn = 1 / (1 + np.exp(-logit))
        start_decline = live & (state == 0) & (rng.random(n_outlets) < p_churn)
        state = np.where(start_decline, rng.integers(1, 3, n_outlets), state)
        # reactivation of lost outlets (win-backs happen, rarely)
        react = (state == 9) & (rng.random(n_outlets) < 0.035)
        h = np.where(react, rng.normal(-0.3, 0.5, n_outlets), h)
        state = np.where(react, 0, state)

        declining = (state >= 1) & (state <= 2)
        intensity = np.where(declining, 0.38, 1.0) * np.where(state == 9, 0.0, 1.0) * live
        lam = typ["lam"] * np.exp(0.33 * h) * intensity
        n_ord = rng.poisson(lam)

        m0 = months[t]
        days_in = (months[t + 1] - m0).days
        for i in np.nonzero(n_ord)[0]:
            k = n_ord[i]
            days = np.sort(rng.integers(0, days_in, k))
            val = np.exp(rng.normal(typ["mu"][i] + 0.12 * h[i], 0.45, k))
            skus = np.maximum(1, rng.poisson(typ["sku"][i] * np.exp(0.22 * h[i]) * (0.65 if declining[i] else 1.0), k))
            late_mu = max(0.5, 6.0 - 5.0 * h[i] + (7 if declining[i] else 0))
            late = np.maximum(0, np.round(rng.gamma(1.6, late_mu / 1.6, k) - 2)).astype(int)
            disc = np.clip(rng.normal(typ["disc"][i] + 0.012 * max(0, -h[i]), 0.012, k), 0, 0.2)
            for j in range(k):
                order_rows.append((f"INV-{oid:07d}", outlets.outlet_id.iat[i],
                                   (m0 + pd.Timedelta(days=int(days[j]))).strftime("%Y-%m-%d"),
                                   int(round(val[j], -3)), int(skus[j]), round(float(disc[j]), 3), int(late[j])))
                oid += 1
        for i in np.nonzero(n_vis)[0]:
            for d in rng.integers(0, days_in, n_vis[i]):
                visit_rows.append((outlets.outlet_id.iat[i], rep_id[i],
                                   (m0 + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d")))
        n_c = rng.poisson(0.11 * np.exp(-0.75 * h) * (intensity > 0) * live
                          * np.where(declining, 2.0, 1.0))
        for i in np.nonzero(n_c)[0]:
            for d in rng.integers(0, days_in, n_c[i]):
                compl_rows.append((outlets.outlet_id.iat[i],
                                   (m0 + pd.Timedelta(days=int(d))).strftime("%Y-%m-%d"),
                                   rng.choice(["Late delivery", "Damaged goods", "Short shipment",
                                               "Expired stock", "Price dispute"],
                                              p=[0.3, 0.2, 0.2, 0.1, 0.2])))
        # advance decline state
        state = np.where(declining, state - 1, state)
        state = np.where(declining & (state == 0), 9, state)

    orders = pd.DataFrame(order_rows, columns=["invoice_id", "outlet_id", "order_date", "gross_value_idr",
                                               "n_skus", "discount_pct", "days_paid_late"])
    visits = pd.DataFrame(visit_rows, columns=["outlet_id", "rep_id", "visit_date"]).sort_values("visit_date")
    complaints = pd.DataFrame(compl_rows, columns=["outlet_id", "complaint_date", "category"]).sort_values("complaint_date")

    # ---------------- inject realistic defects ----------------
    dup = orders.sample(frac=0.003, random_state=seed)
    orders = pd.concat([orders, dup]).sort_values(["order_date", "invoice_id"]).reset_index(drop=True)
    neg = orders.sample(12, random_state=seed + 1).index
    orders.loc[neg, "gross_value_idr"] *= -1                    # credit notes keyed as orders
    fut = orders.sample(6, random_state=seed + 2).index
    orders.loc[fut, "order_date"] = orders.loc[fut, "order_date"].str.replace("2025", "2027", n=1)
    miss = outlets.sample(frac=0.01, random_state=seed + 3).index
    outlets.loc[miss, "city"] = np.nan

    return {"outlets": outlets, "orders": orders, "visits": visits, "complaints": complaints}


def main() -> None:
    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    tables = generate()
    for name, df in tables.items():
        df.to_csv(C.DATA_RAW / f"{name}.csv", index=False)
        print(f"wrote data/raw/{name}.csv  ({len(df):,} rows)")


if __name__ == "__main__":
    main()
