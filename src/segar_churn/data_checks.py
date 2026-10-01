"""Data validation + cleaning (Checkpoint 1: "validated analytical dataset").

Every check is a named rule with a severity:
    BLOCK  -> pipeline stops; the data cannot be trusted
    FIX    -> a known, documented cleaning rule is applied and the count is logged
    WARN   -> reported for a human to look at, data passes through unchanged

The log is written to reports/data_quality_log.md so the reviewer can see
exactly what was removed and why.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from . import config as C


@dataclass
class CheckResult:
    table: str
    rule: str
    severity: str
    n_rows: int
    action: str

    @property
    def passed(self) -> bool:
        return self.n_rows == 0


class DataQualityError(RuntimeError):
    pass


def load_raw() -> dict[str, pd.DataFrame]:
    return {
        "outlets": pd.read_csv(C.DATA_RAW / "outlets.csv"),
        "orders": pd.read_csv(C.DATA_RAW / "orders.csv"),
        "visits": pd.read_csv(C.DATA_RAW / "visits.csv"),
        "complaints": pd.read_csv(C.DATA_RAW / "complaints.csv"),
    }


REQUIRED_COLUMNS = {
    "outlets": ["outlet_id", "outlet_name", "outlet_type", "region", "city", "rep_id", "onboard_date", "distance_km"],
    "orders": ["invoice_id", "outlet_id", "order_date", "gross_value_idr", "n_skus", "discount_pct", "days_paid_late"],
    "visits": ["outlet_id", "rep_id", "visit_date"],
    "complaints": ["outlet_id", "complaint_date", "category"],
}


def validate_and_clean(raw: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], list[CheckResult]]:
    log: list[CheckResult] = []
    t = {k: v.copy() for k, v in raw.items()}
    data_end = pd.Timestamp(C.START_MONTH + "-01") + pd.DateOffset(months=C.N_MONTHS) - pd.Timedelta(days=1)

    # --- schema (BLOCK) ---
    for name, cols in REQUIRED_COLUMNS.items():
        missing = [c for c in cols if c not in t[name].columns]
        log.append(CheckResult(name, f"required columns present ({len(cols)})", "BLOCK", len(missing),
                               f"missing: {missing}" if missing else "ok"))

    o = t["outlets"]
    log.append(CheckResult("outlets", "outlet_id unique", "BLOCK", int(o.outlet_id.duplicated().sum()), "ok"))
    bad_type = ~o.outlet_type.isin(["Warung", "Minimarket", "Grosir", "Supermarket"])
    log.append(CheckResult("outlets", "outlet_type in allowed set", "BLOCK", int(bad_type.sum()), "ok"))
    n = int(o.city.isna().sum())
    o["city"] = o.city.fillna("Unknown")
    log.append(CheckResult("outlets", "city not null", "FIX", n, "filled with 'Unknown' (city is not a model feature)"))

    # --- orders ---
    od = t["orders"]
    od["order_date"] = pd.to_datetime(od.order_date, errors="coerce")
    n = int(od.order_date.isna().sum())
    od = od[od.order_date.notna()]
    log.append(CheckResult("orders", "order_date parseable", "FIX", n, "dropped"))

    m = od.gross_value_idr <= 0
    log.append(CheckResult("orders", "gross_value_idr > 0", "FIX", int(m.sum()),
                           "dropped (credit notes keyed as orders; finance confirmed they belong in returns)"))
    od = od[~m]

    m = od.order_date > data_end
    log.append(CheckResult("orders", f"order_date <= {data_end.date()}", "FIX", int(m.sum()),
                           "dropped (year typo, e.g. 2027 for 2025)"))
    od = od[~m]

    n = int(od.duplicated().sum())
    od = od.drop_duplicates()
    log.append(CheckResult("orders", "no exact duplicate rows", "FIX", n, "dropped duplicates (ERP double-posting)"))
    log.append(CheckResult("orders", "invoice_id unique after de-dup", "BLOCK", int(od.invoice_id.duplicated().sum()), "ok"))

    orphan = ~od.outlet_id.isin(o.outlet_id)
    log.append(CheckResult("orders", "outlet_id exists in outlet master", "BLOCK", int(orphan.sum()), "ok"))
    log.append(CheckResult("orders", "discount_pct between 0 and 0.5", "WARN",
                           int((~od.discount_pct.between(0, 0.5)).sum()), "reported only"))
    log.append(CheckResult("orders", "days_paid_late >= 0", "BLOCK", int((od.days_paid_late < 0).sum()), "ok"))
    big = od.gross_value_idr > od.gross_value_idr.quantile(0.999) * 5
    log.append(CheckResult("orders", "no extreme order values (>5x p99.9)", "WARN", int(big.sum()), "reported only"))

    # --- visits / complaints ---
    v = t["visits"]
    v["visit_date"] = pd.to_datetime(v.visit_date)
    log.append(CheckResult("visits", "outlet_id exists in outlet master", "BLOCK", int((~v.outlet_id.isin(o.outlet_id)).sum()), "ok"))
    cp = t["complaints"]
    cp["complaint_date"] = pd.to_datetime(cp.complaint_date)
    log.append(CheckResult("complaints", "outlet_id exists in outlet master", "BLOCK", int((~cp.outlet_id.isin(o.outlet_id)).sum()), "ok"))

    # --- coverage sanity (WARN): every month should have orders ---
    per_month = od.groupby(od.order_date.dt.to_period("M")).size()
    thin = per_month[per_month < per_month.median() * 0.5]
    log.append(CheckResult("orders", "no month with <50% of median order volume", "WARN", len(thin), "reported only"))

    blocking = [r for r in log if r.severity == "BLOCK" and not r.passed]
    if blocking:
        raise DataQualityError("; ".join(f"{r.table}: {r.rule} ({r.n_rows})" for r in blocking))

    clean = {"outlets": o, "orders": od.reset_index(drop=True), "visits": v, "complaints": cp}
    return clean, log


def write_log(log: list[CheckResult], raw: dict, clean: dict) -> None:
    lines = ["# Data quality log", "",
             "Generated by `make checks` (src/segar_churn/data_checks.py). Re-run after any data refresh.", "",
             "| Table | Rule | Severity | Rows affected | Action |", "|---|---|---|---:|---|"]
    for r in log:
        status = "✅" if r.passed else ("🛠️" if r.severity == "FIX" else ("⚠️" if r.severity == "WARN" else "⛔"))
        lines.append(f"| {r.table} | {r.rule} | {status} {r.severity} | {r.n_rows:,} | {r.action} |")
    lines += ["", "## Row counts", "", "| Table | Raw | Clean |", "|---|---:|---:|"]
    for k in raw:
        lines.append(f"| {k} | {len(raw[k]):,} | {len(clean[k]):,} |")
    (C.REPORTS / "data_quality_log.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    raw = load_raw()
    clean, log = validate_and_clean(raw)
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    for k, df in clean.items():
        df.to_csv(C.DATA_PROCESSED / f"{k}.csv", index=False)
    write_log(log, raw, clean)
    for r in log:
        print(f"[{'PASS' if r.passed else r.severity:5}] {r.table:10} {r.rule:45} {r.n_rows:>6}")
    print("clean tables written to data/processed/, log -> reports/data_quality_log.md")


if __name__ == "__main__":
    main()
