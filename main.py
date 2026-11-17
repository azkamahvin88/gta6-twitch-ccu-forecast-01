"""gta6-twitch-ccu-forecast-01 entry: launch-day Twitch CCU forecaster (Python 3.13, stdlib only).

Model: launch-day shape curves from comparable console launches, anchored on the
case's hour-1 print.

For every comparable launch in the case file we take its release-day hourly curve
and express each hour as a log-ratio to that launch's own hour-1 print. Across
launches, the log-ratio of an hour is a strongly decreasing, concave function of
the launch's size (log hour-1 print): small launches explode through the day as
time zones unlock (up to +7 log units), huge launches barely grow. So per hour
h = 2..24 we fit a quadratic regression of log-ratio on log(hour-1 print) over
the comparable launches, and take the fitted value at the target's hour-1 print
(clipped to the training range, so an unprecedented launch is treated like the
largest launch seen, never extrapolated down a parabola). Quantiles come from
the empirical distribution of that hour's regression residuals, scaled by 0.8
(the residuals overstate the true conditional spread, which the practice cases
and leave-one-out over the comparables both confirm).

Only comparables tiered "comparable" enter training (early-access-contaminated
and weak launches have distorted release-day curves). A comparable counts only
when its release day has an hour-1 print >= 100 viewers and at least 20 of its
24 hourly points present, so the target's own pre-release history (no release
day yet) can never leak into training.

The final median is a 50/50 blend (in log space) of that regression forecast
with a nearest-neighbour forecast: the 8 comparables whose hour-1 print is
closest to the target's in log space, whose empirical log-ratio quantiles per
hour are used directly. The blend keeps the regression's smooth size law and
the neighbours' local shape, and it is the best of the tested variants both on
the practice cases and by leave-one-out over the comparables. The blended
curve is lightly smoothed across neighbouring hours (3-hour moving average on
log quantiles), which the same validation supports.

Deterministic: sorted iteration, no randomness, no network, no wall clock, no
files outside the repo. Falls back to the pooled median curve when too few
comparables qualify, and to the hour-1 print when there is no usable hour 1.
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# Training quality gates for a comparable launch.
MIN_HOUR1 = 100.0
MIN_HOURS_PRESENT = 20
MIN_TRAIN_ROWS = 8
# Residual quantiles are scaled by this: the raw residuals overstate the true
# conditional spread (validated on the practice cases and by leave-one-out).
SPREAD_SCALE = 0.8


def tier_of(comp):
    """Tier from the comparable's unlock note; unmarked counts as comparable."""
    note = comp.get("unlock") or ""
    if "Tier: early-access-contaminated" in note:
        return "ea"
    if "Tier: weak" in note:
        return "weak"
    if "Tier: target-history" in note:
        return "target"
    return "comp"


def release_day_curve(comp):
    """The comparable's release-day hourly viewers, hours 1..24, or None.

    A launch qualifies only with an hour-1 print of at least MIN_HOUR1 and at
    least MIN_HOURS_PRESENT of its 24 hourly points present and positive.
    """
    day = (comp.get("day_start") or "")[:10]
    if not day:
        return None
    by_hour = {}
    for row in comp.get("hourly") or []:
        frm = row.get("from") or ""
        if frm.startswith(day):
            try:
                h = int(frm[11:13])
            except ValueError:
                continue
            v = row.get("viewers")
            if v is not None:
                by_hour[h] = v
    if 0 not in by_hour or by_hour[0] is None or by_hour[0] < MIN_HOUR1:
        return None
    vals = [by_hour.get(h) for h in range(24)]
    if sum(1 for v in vals if v is not None and v > 0) < MIN_HOURS_PRESENT:
        return None
    return vals


def training_rows(case):
    """[(log hour-1 print, release-day curve)] from tier-comparable launches."""
    rows = []
    for comp in case.get("comparables") or []:
        if tier_of(comp) != "comp":
            continue
        curve = release_day_curve(comp)
        if curve is None:
            continue
        rows.append((math.log(curve[0]), curve))
    return rows


def ols(xs, ys):
    """Least squares y ~ a + b*x; returns (a, b, residuals)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    a = my - b * mx
    return a, b, [y - (a + b * x) for x, y in zip(xs, ys)]


def ols_quad(xs, ys):
    """Least squares y ~ a + b1*x + b2*x^2; returns ((a, b1, b2), residuals)."""
    n = len(xs)
    mx1 = sum(xs) / n
    mx2 = sum(x * x for x in xs) / n
    my = sum(ys) / n
    s11 = sum((x - mx1) ** 2 for x in xs)
    s12 = sum((x - mx1) * (x * x - mx2) for x in xs)
    s22 = sum((x * x - mx2) ** 2 for x in xs)
    t1 = sum((x - mx1) * (y - my) for x, y in zip(xs, ys))
    t2 = sum((x * x - mx2) * (y - my) for x, y in zip(xs, ys))
    det = s11 * s22 - s12 * s12
    if abs(det) < 1e-12:
        a, b, resid = ols(xs, ys)
        return (a, b, 0.0), resid
    b1 = (t1 * s22 - t2 * s12) / det
    b2 = (t2 * s11 - t1 * s12) / det
    a = my - b1 * mx1 - b2 * mx2
    resid = [y - (a + b1 * x + b2 * x * x) for x, y in zip(xs, ys)]
    return (a, b1, b2), resid


def quantile(sorted_vals, q):
    """Linear-interpolation quantile of an ascending list."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def median(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def quad_forecast(rows, p1):
    """Per-hour quadratic-OLS forecast: {hour: [q05, q25, q50, q75, q95]}."""
    x0 = math.log(p1)
    xs_all = [r[0] for r in rows]
    xmin, xmax = min(xs_all), max(xs_all)
    xx = min(max(x0, xmin), xmax)
    out = {}
    for h in range(2, 25):
        xs, ys = [], []
        for logp1, curve in rows:
            v = curve[h - 1]
            if v is not None and v > 0:
                xs.append(logp1)
                ys.append(math.log(v) - math.log(curve[0]))
        if len(xs) < MIN_TRAIN_ROWS:
            continue
        (a, b1, b2), resid = ols_quad(xs, ys)
        mu = a + b1 * xx + b2 * xx * xx
        rs = sorted(resid)
        med = median(resid)
        out[h] = [
            math.exp(math.log(p1) + mu + SPREAD_SCALE * (quantile(rs, tau) - med) + med)
            for tau in LEVELS
        ]
    return out


def knn_forecast(rows, p1, k=8):
    """Nearest-neighbour forecast: the k comparables closest in log hour-1.

    Per hour, the empirical quantiles of the neighbours' log-ratios, scaled by
    SPREAD_SCALE around their median, exactly as in the regression forecast.
    """
    x0 = math.log(p1)
    near = sorted(rows, key=lambda r: abs(r[0] - x0))[:k]
    out = {}
    for h in range(2, 25):
        lrs = []
        for logp1, curve in near:
            v = curve[h - 1]
            if v is not None and v > 0:
                lrs.append(math.log(v) - math.log(curve[0]))
        if len(lrs) < 4:
            continue
        lrs.sort()
        med = median(lrs)
        out[h] = [
            math.exp(math.log(p1) + med + SPREAD_SCALE * (quantile(lrs, tau) - med))
            for tau in LEVELS
        ]
    return out


def blend_forecasts(a, b, weight=0.5):
    """Geometric-mean blend of two {hour: quantiles} forecasts."""
    out = {}
    for h in sorted(set(a) | set(b)):
        if h in a and h in b:
            out[h] = [math.exp(weight * math.log(x) + (1.0 - weight) * math.log(y))
                      for x, y in zip(a[h], b[h])]
        elif h in a:
            out[h] = a[h]
        else:
            out[h] = b[h]
    return out


def smooth_hours(pred, window=3):
    """3-hour moving average of log quantiles across neighbouring hours."""
    hours = sorted(pred)
    out = {}
    half = window // 2
    for i, h in enumerate(hours):
        neigh = [pred[hours[j]] for j in range(max(0, i - half), min(len(hours), i + half + 1))]
        out[h] = [math.exp(sum(math.log(row[t]) for row in neigh) / len(neigh))
                  for t in range(5)]
    return out


def pooled_median_forecast(rows, p1):
    """Fallback: pooled empirical log-ratio quantiles per hour."""
    out = {}
    for h in range(2, 25):
        lrs = []
        for logp1, curve in rows:
            v = curve[h - 1]
            if v is not None and v > 0:
                lrs.append(math.log(v) - math.log(curve[0]))
        if not lrs:
            continue
        lrs.sort()
        out[h] = [math.exp(math.log(p1) + quantile(lrs, tau)) for tau in LEVELS]
    return out


def forecast_case(case, name):
    if case.get("levels") != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    first = next((k for k in case.get("known") or [] if k.get("hour") == 1), None)
    p1 = max(1.0, float(first["viewers"])) if first and first.get("viewers") else 1.0
    rows = training_rows(case)
    if len(rows) >= MIN_TRAIN_ROWS:
        quad = quad_forecast(rows, p1)
        near = knn_forecast(rows, p1)
        pred = smooth_hours(blend_forecasts(quad, near))
    else:
        pred = {}
    if len(pred) < 23:
        fallback = pooled_median_forecast(rows, p1) if rows else {}
        for h in range(2, 25):
            pred.setdefault(h, fallback.get(h) or [p1] * 5)
    hours = []
    for row in case["forecast"]:
        h = row["hour"]
        qs = pred.get(h) or [p1] * 5
        # Clamp to (0, 1e9] and keep non-decreasing, as the rules require.
        qs = [min(max(q, 1.0), 1e9) for q in qs]
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        hours.append({"hour": h, "q": qs})
    hours.sort(key=lambda r: r["hour"])
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
