# Weather join — scoping

Goal: test whether exogenous weather adds week-ahead forecast skill for GBA hourly
**flow**, beyond the EWMA + holiday baseline (currently R²=0.9493 on 2019).

## Train/serve consistency (the key methodological point)
At inference we only have **weather forecasts**, not actuals. Training on actuals
but serving on forecasts = train/serve skew (model over-trusts weather). For a
*production* model the feature must be "the week-ahead forecast as of the origin"
in BOTH training and serving. But we measure the **ceiling on actuals first**, and
only source forecasts if the ceiling justifies it.

Also: weather-forecast skill itself decays with horizon (day 1-3 sharp, day 5-7
weak for precip), so the *achievable* weather gain shrinks toward h=168 and is
diluted by our week-aggregate metric. The *ceiling* (perfect weather) is all-horizon.

## Staged plan
- **Stage 0 — perfect-weather ceiling.** ERA5 actuals, train+test. "If we had a
  perfect 7-day forecast, how much would R² improve?" If ~0, STOP.
- **Stage 1a — emulated forecasts.** Degrade actuals with realistic 1-7 day
  forecast error; gives an achievable estimate without new data.
- **Stage 1b — real forecasts.** Archived week-ahead forecasts + traffic, which
  only coexist **post-2022** -> reuses the post-2022 PeMS pull
  (`docs/post2022_pems_download.md`).

## Data sources (verified)
- **Actuals:** Open-Meteo ERA5 archive API — free, no key, hourly precip/temp/wind,
  any lat/lng, back to 1940. Confirmed working for 2017 (SFO storm = 6.8 mm/h).
  `https://archive-api.open-meteo.com/v1/archive?latitude=..&longitude=..&start_date=..&end_date=..&hourly=precipitation,temperature_2m,wind_speed_10m&timezone=America/Los_Angeles`
- **Archived forecasts (Stage 1b):** Open-Meteo previous-runs / ECMWF TIGGE,
  effectively from ~2022. Pre-2022 "historical forecast" is reanalysis backfill,
  NOT real forecasts — do not use it to claim a forecast-based number.

## Join design (Stage 0)
- Spatial: ~5 anchor stations (SFO, OAK, SJC, Santa Rosa, Concord); assign each of
  the 2,352 sensors to nearest by lat/lng. Storms are regional, so this suffices
  for the ceiling. Per-sensor grid = Stage 1 refinement.
- Temporal: ERA5 hourly aligns to our hourly flow grid (PST, DST-aware).
- Target: seasonal residual r = y - b. Features: precip, temp, wind + interactions
  (rain × hour-of-day: wet commute ≠ wet midday).
- Eval: rolling-origin 2019 backtest, bucketed wet vs dry hours; report aggregate
  R² lift AND conditional improvement on wet hours.

## Mechanism caveat
We model FLOW (volume). Rain's effect on volume is more modest than on speed
(commute demand inelastic); real drops are big storms + discretionary trips. Expect
a small ceiling concentrated in the wet winter (Jan–Feb 2017 storms). Measuring the
ceiling first is exactly why this ordering matters.
