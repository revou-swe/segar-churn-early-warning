# Data

## Source

`data/raw/` is produced by `make data` (`src/segar_churn/generate_data.py`): a **fictional, synthetic** stand-in for the PT Segar Nusantara anchor dataset, with the same structure. It is deterministic (seed 42), so every run produces identical numbers.

Enrolled participants replace the four CSVs with the program's anchor dataset (or their own anonymised workplace export with the same columns) and run `PYTHONPATH=src python -m segar_churn.pipeline --use-existing-raw`.

Defects are injected on purpose so the data checks have real work to do: ~0.3% duplicate invoices, 12 negative-value invoices (credit notes), future-dated invoices (year typos), 1% of outlets missing a city.

Built into the simulation, for the analysis to discover: a rival distributor enters **Jawa Timur in January 2026**, raising churn there.

## Data dictionary

### `outlets.csv` — one row per retail partner
| Column | Type | Description |
|---|---|---|
| outlet_id | str | `OUT-10001` … primary key |
| outlet_name | str | trading name (fictional) |
| outlet_type | str | Warung · Minimarket · Grosir · Supermarket |
| region | str | Jabodetabek · Jawa Barat · Jawa Tengah · Jawa Timur · Sumatera Utara · Sulawesi Selatan |
| city | str | city / kabupaten |
| rep_id, rep_name | str | assigned sales rep (`SR-JTM-06`) |
| onboard_date | date | first month as a partner |
| distance_km | float | road distance from serving depot |

### `orders.csv` — one row per invoice
| Column | Type | Description |
|---|---|---|
| invoice_id | str | primary key |
| outlet_id | str | FK → outlets |
| order_date | date | date order placed |
| gross_value_idr | int | invoice value, IDR |
| n_skus | int | distinct SKUs on the invoice |
| discount_pct | float | average discount, 0–1 |
| days_paid_late | int | days paid after due date (0 = on time) |

### `visits.csv` — one row per sales-rep visit
`outlet_id`, `rep_id`, `visit_date`

### `complaints.csv` — one row per complaint or return
`outlet_id`, `complaint_date`, `category` (Late delivery · Damaged goods · Short shipment · Expired stock · Price dispute)

## Derived tables (`data/processed/`)

| File | Grain | Built by |
|---|---|---|
| `outlets/orders/visits/complaints.csv` | cleaned copies | `data_checks.py` |
| `panel.csv` | outlet × month-end, Apr-2025 → Jul-2026, with label `churned` | `features.py` |
| `scoring_snapshot.csv` | active outlets at 30-Sep-2026, no label | `features.py` |

## Privacy

The analytical dataset has no personal data: no owner names, phone numbers, NIK or addresses. Do not add them. Real partner exports go in `data/private/`, which is git-ignored.
