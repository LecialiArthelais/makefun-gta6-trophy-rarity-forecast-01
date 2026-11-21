# GTA VI Trophy Rarity Forecast

Forecast, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn rate
on Exophase at a case's target, as five quantiles per trophy — an entry for
make.fun competition `gta6-trophy-rarity-forecast-01`.

## Program

`python3 main.py` (Python 3.13, standard library only, no network). It reads
every `*.in` case in `$MAKEFUN_INPUT_DIR` (else `/input`) and writes
`<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (else `/output`):
`{"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}`.

## Model

Per trophy, weighted least squares of `logit(rate)` on `ln(1 + hours since
release)`, with recency weights (24h half-life) over the case's history. The
slope is shrunk toward its tier's median slope by empirical-Bayes weighting
(a confident fit keeps its slope; an uncertain one borrows the tier's; a
trophy with fewer than 3 reads uses the tier median). The median is the fitted
line read at the target. The spread combines the fit residual, a random-walk
term `0.5 * sqrt(horizon in days)`, and the slope's standard error times the
extrapolation distance in log-time (floored at 0.35, capped at 2.5 logit
units). Quantiles are the median plus the standard normal quantiles times the
spread, mapped back to percent and rounded to 4 decimals.

This beats a frozen last-rate forecast (the house baseline) because the median
follows each trophy's actual trajectory, and the spread widens exactly where
the fit is uncertain instead of being one-size-fits-all.

## Local run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## Competition

https://make.fun/bounty/gta6-trophy-rarity-forecast-01 — scored by mean
pinball loss on the logit scale over all trophies and quantiles; a prize needs
a loss strictly below the best house entry.
