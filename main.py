"""Forecast GTA VI PS5 trophy earn rates on Exophase at a case's target.

Model (per trophy, fit on the case's own history):
  logit(rate) ~ a + b * ln(1 + hours since release), weighted least squares
  with recency weights (24h half-life), slope shrunk toward the tier median.
  The median is the line read at the target; the spread combines the fit
  residual, a random-walk term growing with the horizon, and the slope's
  standard error times the extrapolation distance. This beats the house
  baseline (last rate + 0.5*sqrt(days) spread) in two ways: the median moves
  with the trophy's actual trajectory instead of staying frozen, and the
  spread widens where the fit is uncertain instead of being one-size-fits-all.

Usage: python3 main.py  (reads $MAKEFUN_INPUT_DIR or /input, writes
$MAKEFUN_OUTPUT_DIR or /output). Standard library only, no network.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
RW_SIGMA = 0.5          # random-walk spread, logit units per sqrt(day)
HALF_LIFE_H = 24.0      # recency weight half-life for history reads
TAU_SLOPE = 0.5         # prior sd of slopes (logit per ln-hour) for shrinkage
MIN_SPREAD = 0.35
MAX_SPREAD = 2.5


def clamp(x, lo=0.01, hi=99.99):
    return min(hi, max(lo, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(y):
    if y > 20:
        return 99.99
    if y < -20:
        return 0.01
    return 100.0 / (1.0 + math.exp(-y))


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    v = whole / 10000
    return int(v) if float(v).is_integer() else v


def parse_time(s):
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def wls(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b, resid_var, se_b)."""
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return my, 0.0, 0.0, 0.0
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(1, len(xs) - 2)
    var = sum(w * r * r for w, r in zip(ws, resid)) / (sw * dof / len(xs)) if sw > 0 else 0.0
    se_b = math.sqrt(var / sxx) if sxx > 0 else 0.0
    return a, b, var, se_b


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name}: levels {case['levels']} != {LEVELS}")
    history = case["history"]
    if not history:
        raise ValueError(f"{name}: no history")
    release = parse_time(case["game"].get("release_date", "2026-11-19T00:00:00Z"))
    target = parse_time(case["target"])
    last = history[-1]
    t_last = parse_time(last["at"])
    d_days = case["horizon_minutes"] / 1440.0
    x_target = math.log1p((target - release).total_seconds() / 3600.0)
    x_last = math.log1p((t_last - release).total_seconds() / 3600.0)

    # Per-trophy series: (x, y, weight) with recency weights.
    series = []
    for i in range(len(case["trophies"])):
        pts = []
        for read in history:
            rate = read["rates"][i]
            if rate is None:
                continue
            t = parse_time(read["at"])
            age_h = max(0.0, (t_last - t).total_seconds() / 3600.0)
            w = 0.5 ** (age_h / HALF_LIFE_H)
            pts.append((math.log1p((t - release).total_seconds() / 3600.0),
                        logit(rate), w))
        series.append(pts)

    # Tier-median slopes for shrinkage.
    tier_slopes = {}
    fits = []
    for i, trophy in enumerate(case["trophies"]):
        pts = series[i]
        if len(pts) >= 3:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ws = [p[2] for p in pts]
            a, b, var, se_b = wls(xs, ys, ws)
        else:
            a = logit(last["rates"][i]) if last["rates"][i] is not None else logit(0)
            b, var, se_b = 0.0, 0.0, 0.0
        fits.append((a, b, var, se_b))
        tier_slopes.setdefault(trophy["tier"], []).append(b)
    tier_med = {}
    for tier, bs in tier_slopes.items():
        s = sorted(bs)
        tier_med[tier] = s[len(s) // 2] if s else 0.0

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        a, b, var, se_b = fits[i]
        # Empirical-Bayes shrinkage of the slope toward the tier median:
        # a confident fit keeps its own slope, an uncertain one borrows the
        # tier's; a trophy with too few points falls back to the tier median.
        med = tier_med.get(trophy["tier"], 0.0)
        if len(series[i]) < 3:
            b = med
        elif se_b <= 0:
            pass
        else:
            pd = 1.0 / (se_b * se_b)
            pp = 1.0 / (TAU_SLOPE * TAU_SLOPE)
            b = (b * pd + med * pp) / (pd + pp)
        y_med = a + b * x_target
        spread = math.sqrt(var + (RW_SIGMA * math.sqrt(d_days)) ** 2
                           + (se_b * max(0.0, x_target - x_last)) ** 2)
        spread = min(MAX_SPREAD, max(MIN_SPREAD, spread))
        qs = [round4(clamp(inv_logit(y_med + z * spread), 0, 100)) for z in Z]
        qs = [max(qs[0], q) if k else q for k, q in enumerate(qs)]
        # enforce non-decreasing
        for k in range(1, 5):
            qs[k] = max(qs[k], qs[k - 1])
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        out = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(out, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
