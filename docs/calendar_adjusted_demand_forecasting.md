# Calendar-Adjusted Week-Ahead Forecasting: A Simple, Strong Baseline

*A pedagogical note for week-ahead **electricity demand** forecasting. Distilled
from a traffic-forecasting project, but every idea transfers — both series are
dominated by the same weekly human-activity cycle plus holiday/calendar effects.*

---

## 1. The setup

We forecast an hourly series $y_t$ (say, system demand in MW) **one week ahead**:
at origin $t$ we want $\hat y_{t+1},\dots,\hat y_{t+168}$ (168 = hours in a week).

The single most important fact about demand is that it is **periodic with a one-week
period**: the same hour of the same weekday looks remarkably similar week to week
(Tuesday 6pm ≈ last Tuesday 6pm). Define the **hour-of-week** index

$$\phi(t) = t \bmod 168 \in \{0,1,\dots,167\},$$

which encodes *both* the day-of-week and the hour-of-day. Everything below is built
on the empirical observation that $y_t$ is, to first order, a function of $\phi(t)$
times a slowly-moving level — plus weather, holidays, and noise.

---

## 2. The baseline: last-4-weeks hour-of-week average (L4W)

The natural predictor of any hour is its recent average **at the same hour of
week**:

$$\boxed{\;\hat y_t^{\text{(L4W)}} = \frac{1}{4}\sum_{k=1}^{4} y_{t-168k}\;}$$

i.e. average the **same weekday-and-hour** over the previous 4 weeks.

**Worked example.** Forecast next **Wednesday 18:00**. Look at the last four
Wednesdays at 18:00:

| weeks ago | demand (MW) |
|---|---|
| 4 | 820 |
| 3 | 845 |
| 2 | 805 |
| 1 | 830 |

$$\hat y = \tfrac14(820+845+805+830) = \mathbf{825\ \text{MW}}.$$

**Why it's valid a week ahead (no leakage).** Every term $y_{t-168k}$ is at least one
week old, so at origin $t$ the *entire* next week's forecast is computable from data
you already have. The same formula works at any horizon that is a whole number of
weeks; the lags just get older.

**Why 4 weeks?** Averaging cancels week-to-week noise (incidents, one-off weather)
while staying recent enough to track the slowly-moving level. One week alone is too
noisy; many weeks lag real changes. Four is a good default.

### 2a. A one-line upgrade: decaying weights (EWMA)

If the level is trending (load creeping up, a cold snap building), weight recent
weeks more, with weights $w_k \propto \alpha^{\,k-1}$ (e.g. $\alpha = 0.8$),
normalized to sum to 1:

$$\hat y_t^{\text{(EWMA)}} = \sum_{k=1}^{4} w_k\, y_{t-168k},\qquad
w \approx (0.34,\,0.27,\,0.22,\,0.17)\ \text{[recent}\to\text{old]}.$$

**Worked example with a rising trend.** Last four Wednesdays at 18:00 =
780, 800, 820, 840 (oldest→newest).

- Flat L4W: $\tfrac14(780+800+820+840) = 810$ MW.
- EWMA: $0.34(840)+0.27(820)+0.22(800)+0.17(780) = \mathbf{815.5}$ MW.

The EWMA leans toward the recent, higher values — tracking the upward drift the flat
average lags. (In a flat period the two agree.)

---

## 3. The calendar/holiday adjustment

The baseline has one glaring blind spot: **holidays**. To forecast Christmas Day, L4W
averages the four *normal* weeks before it — and badly over-predicts, because demand
on Christmas behaves nothing like a normal weekday (it looks like a quiet Sunday).
Holidays are a small fraction of hours but a large fraction of total error, and they
are **known in advance** — so model them explicitly.

The fix is a **multiplicative calendar factor**, learned per *(holiday × hour-of-day)*
and applied on special days:

$$\boxed{\;\hat y_t^{\text{(adj)}} = \hat y_t^{\text{(L4W)}}\times f\big[\text{holiday}(t),\,\text{hour}(t)\big]\;}$$

where, estimated from history,

$$f[\text{hol},h] = \frac{\displaystyle\sum_{\text{past instances of that holiday at hour }h} \text{actual}}{\displaystyle\sum \text{baseline}}
\;=\;\text{average ratio }\frac{\text{actual demand}}{\text{L4W baseline}}.$$

A factor of $0.82$ means "this holiday-hour runs at 82% of what the seasonal baseline
predicts."

**Worked example — estimating the factor.** Christmas, 18:00, two past years:

| Christmas | actual (MW) | that year's L4W baseline | ratio |
|---|---|---|---|
| year 1 | 640 | 790 | 0.810 |
| year 2 | 690 | 810 | 0.852 |

$$f[\text{Christmas},18{:}00] = \tfrac12(0.810+0.852) = \mathbf{0.831}.$$

**Worked example — applying it.** This year, the L4W baseline for Dec 25, 18:00 (from
the four normal December weeks before it) is 800 MW. The adjusted forecast:

$$\hat y = 800 \times 0.831 = \mathbf{665\ \text{MW}},$$

versus the unadjusted 800 — a ~17% correction exactly where the baseline fails.

**Practical notes that matter a lot:**
- **Use multiplicative, not additive, factors.** The holiday *depression* scales with
  the level (a cold Christmas still draws more than a mild one); a ratio captures that,
  a fixed MW offset doesn't.
- **Per hour-of-day, not one number per holiday.** Holidays reshape the *daily profile*
  — the morning ramp flattens, the evening peak softens — so $f$ varies across the 24
  hours.
- **Pool across years** (and, if you have them, across regions/feeders) to denoise the
  factor — you only get one instance of each holiday per year.
- **Include the shoulder days.** Christmas Eve, the day after, the dead week between
  Christmas and New Year, "bridge" days between a holiday and a weekend — treat each as
  its own special-day key. Define a clean special-day calendar; the factor machinery
  then handles them uniformly.
- **It stays valid at any horizon.** Holidays come from the calendar, so the factor is
  known as far ahead as you like — unlike weather, it needs no forecast.

### Putting it together — a holiday day, hour by hour

| hour | L4W baseline | factor $f$ | adjusted forecast |
|---|---|---|---|
| 08:00 | 740 | 0.70 | 518 |
| 12:00 | 810 | 0.85 | 689 |
| 18:00 | 800 | 0.83 | 664 |
| 22:00 | 690 | 0.88 | 607 |

Same baseline, reshaped by the learned holiday profile. That's the whole method:
**recency-weighted hour-of-week average, times a known-ahead calendar factor.**

---

## 4. Why this works (and what's left over)

Think of the series as

$$y_t \;=\; \underbrace{L_t}_{\text{slow level}}\;\times\;\underbrace{S_{\phi(t)}}_{\text{weekly shape}}\;+\;\underbrace{\varepsilon_t}_{\text{noise}}.$$

- The **weekly shape** $S$ is deterministic given the calendar — known equally well one
  hour or one week ahead. The L4W average *is* an estimate of $L_t \times S_{\phi(t)}$.
- The **calendar factor** patches the days where the shape itself changes (holidays).
- What remains, $\varepsilon_t$, is genuine noise — and, for the parts driven by
  weather, a *systematic* signal the calendar can't see (more on this next).

In our traffic study this simple rule explained ~94% of hourly variance one week
ahead, and the holiday adjustment removed the single largest chunk of the remaining
error. Electricity demand has the same skeleton, so expect similar behavior on the
calendar/seasonal part.

---

## 5. Is there scope for "fancy" modeling to help further?

Short answer: **for the calendar/seasonal/holiday structure, very little — but
electricity has one big exogenous driver that traffic lacks, and *that* is where
modeling genuinely pays.** Let me separate the two cleanly.

**(a) On the calendar/seasonal part: simple rules are near-optimal.** Once you've
removed the hour-of-week shape and the holiday factor, the residual is mostly
unpredictable from the series' own past. In our project we threw the usual escalation
at it — autoregressive models on the residual, gradient-boosted trees with calendar
and unit-type features, pooled/panel models across thousands of sensors — and **none
beat a one- or two-parameter rule**. The reason is structural, not a failure of
effort: calendar features that the baseline already encodes carry ~zero extra signal
by construction, and what's left is noise. This matches the electricity-forecasting
record: the **GEFCom** load-forecasting competitions are repeatedly won by *structured
regression* (calendar dummies + temperature splines), not deep networks. So if "fancy"
means sequence models / deep nets on the demand history to squeeze the calendar part —
expect little, and a well-built simple rule as a stubbornly strong baseline.

**(b) The electricity-specific twist: temperature.** This is the crucial difference
from traffic. Demand is **weather-driven** — heating when cold, cooling when hot — and
the load–temperature relationship is **large, nonlinear, and not visible to any
calendar rule**. This is exactly where modeling earns its keep:
- The classic **load–temperature curve** is U-/hockey-stick-shaped (high load at low
  temps for heating, high at high temps for A/C, a minimum around ~18 °C). Model it with
  **piecewise-linear or spline terms in temperature** (or heating/cooling degree-days),
  *interacted with hour-of-day and season* (a hot afternoon ≠ a hot night).
- A semiparametric regression / **GAM** of the form
  *demand ~ hour-of-week effects + holiday factor + smooth(temperature, by hour) +
  smooth(temperature lags)* is the field's workhorse and is very hard to beat.
- Practically: keep the calendar machinery from §2–§3 as-is, and add the temperature
  response as the main modeled component.

**(c) The honest catch — temperature must be *forecast*.** At week-ahead horizons you
don't know next Tuesday's temperature; you have a weather forecast, whose error grows
with horizon and often becomes the *dominant* source of demand-forecast error a few
days out. Two consequences: (i) train the temperature response on the *forecasts*
you'll actually have at inference (or accept the optimism of training on actuals), and
(ii) this makes a strong case for **probabilistic** forecasting — propagate weather
uncertainty into a predictive *distribution* of demand, which is what peak/capacity and
risk decisions actually need. Distributional modeling (quantile regression, ensembles)
is a place where added sophistication pays off — not for a better point mean, but for
calibrated risk.

**Bottom line for your friend.** The calendar/holiday part is essentially a solved,
simple-rule problem — use the recency-weighted hour-of-week average times a known-ahead
calendar factor, and don't expect deep learning to beat it. Spend the modeling budget
where electricity actually differs from traffic: the **nonlinear temperature response**
(structured regression / GAM, not black-box sequence models) and **distributional**
forecasts that carry weather uncertainty through to demand. Complexity should follow
*information* (temperature) and *objective* (risk), not precede them.

---

### One-paragraph recipe

> Build the hour-of-week average of the last 4 weeks (EWMA-weighted if the level
> trends). Learn a multiplicative factor per (special-day × hour-of-day) as the
> historical ratio of actual to baseline, pooled over years, and multiply it in on
> holidays and their shoulder days. That is a strong, leakage-safe, interpretable
> week-ahead baseline. Then add a smooth temperature-response term (splines/degree-days
> interacted with hour) as the one component worth real modeling effort, and report a
> predictive distribution so weather-forecast uncertainty is visible. Benchmark anything
> fancier against this — it will be harder to beat than it looks.
