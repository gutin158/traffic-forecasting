# Downloading post-2022 California traffic data (PeMS Clearinghouse)

LargeST only covers **2017–2021**. To get genuinely post-pandemic data (2022–2025)
we have to pull it straight from Caltrans PeMS. Caltrans **disallows scripted/automated
downloading**, so this is a manual (browser) process. These steps get District 4
(Greater Bay Area — the same region as our LargeST `gba_*` data) hourly station data.

---

## 1. Create a PeMS account (one-time, free)
1. Go to <https://pems.dot.ca.gov/>.
2. Click **Register** (or "Create account"). Approval is usually 1–2 business days.
3. Log in.

## 2. Open the Data Clearinghouse
- Navigate to <https://pems.dot.ca.gov/?dnode=Clearinghouse>
  (or top menu: **Tools → Data Clearinghouse**).

## 3. Select what to download
On the Clearinghouse page there are two dropdowns:
1. **Type** → choose **`Station Hour`**  (already hourly — no 5-min aggregation needed).
   - If `Station Hour` isn't offered for the year, use **`Station 5-Minute`** and we'll
     aggregate to hourly with the same reshape-mean we used for LargeST.
2. **District** → choose **`District 4`** (San Francisco Bay Area).

Click **Submit**. The page then lists **downloadable `.txt.gz` files grouped by year**,
typically **one file per day** (so ~365 files/year).

## 4. Batch-download the files (use a browser extension — do NOT script wget/curl)
Caltrans blocks automated tools, but a browser download-manager extension acting on the
page you've loaded is fine:
- **Chrome/Edge:** "Chrono Download Manager" or "Simple mass downloader" → "find all links",
  filter to `.gz`, download.
- **Firefox:** "DownThemAll!" → select all `.txt.gz` links.

Download the years you want (**2022, 2023, 2024, 2025**) into:
```
~/projects/traffic_forecasting/data/ca_post2022/d04_station_hour/
```

## 5. Also grab the station metadata (for lat/lng + sensor matching)
1. **Type** → **`Station Metadata`**, **District** → **`District 4`**, Submit.
2. Download the most recent `d04_text_meta_YYYY_MM_DD.txt` into
   `~/projects/traffic_forecasting/data/ca_post2022/meta/`.

This lets us match post-2022 Station IDs back to the LargeST GBA sensor IDs (join on
Station ID; most District-4 mainline stations persist across years).

---

## File format reference (so the loader can parse it)

**Station Hour** files are gzipped CSV, **no header row**. Columns:

| # | Column | Notes |
|---|--------|-------|
| 1 | Timestamp | `MM/DD/YYYY HH:MM:SS` (local PST/PDT) |
| 2 | Station | sensor ID (matches LargeST IDs) |
| 3 | District | = 4 |
| 4 | Freeway # | |
| 5 | Direction | N/S/E/W |
| 6 | Lane Type | ML = mainline, etc. |
| 7 | Station Length | |
| 8 | Samples | # raw samples |
| 9 | % Observed | data quality; <100 means some imputation |
| 10 | **Total Flow** | **veh/hr — our target** |
| 11 | Avg Occupancy | |
| 12 | Avg Speed | mph |
| 13+ | per-lane columns | (Lane N Samples, Flow, Occ, Speed) — usually ignore |

We mainly need columns **1 (Timestamp), 2 (Station), 10 (Total Flow)** — and optionally
12 (Avg Speed) if we want a speed target too.

---

## When the files are in place
Tell me, and I'll write `src/process_pems_post2022.py` to:
1. Read all `d04_station_hour/*.txt.gz`, keep cols [Timestamp, Station, Total Flow].
2. Pivot to a `(hours × sensors)` matrix matching our LargeST `gba_hourly_*.npz` layout.
3. Filter to the LargeST GBA sensor set (or keep all District-4 stations — your call).
4. Cache as `gba_hourly_2022.npz`, … so it drops straight into the existing backtest.

### Caveats to expect
- **% Observed < 100**: PeMS imputes missing sensors; flag/keep as needed.
- **Sensor turnover**: some 2017-era stations are decommissioned by 2024 and new ones
  appear — the overlapping set is what allows cross-period comparison.
- **DST**: PeMS timestamps are local; watch the spring/fall hour near the DST switch.
- **Volume**: ~365 files/year/district; the batch extension is the bottleneck, not disk.
