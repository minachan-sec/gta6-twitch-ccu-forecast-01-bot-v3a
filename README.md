# gta6-twitch-ccu-forecast-01-bot-v3a

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/minachan-sec/gta6-twitch-ccu-forecast-01-bot-v3a/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/minachan-sec/gta6-twitch-ccu-forecast-01-bot-v3a/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecast GTA VI's launch-day Twitch viewers hour by hour: a stacked ensemble of nine forecasters over log-ratios ln(viewers / hour-1), combined by quantile averaging with hour-block weights and per-hour-group spread calibration.

**Method.** Nine members, each producing a full 23×5 quantile forecast:
- Median + parametric-spread members: Gaussian kernel on log(hour-1) over tier-comparable launches, per-hour WLS regression, three ridge variants (pre-release hype features), and a median quantile regression.
- Direct quantile members: five-level linear quantile regressions fit by subgradient descent on the pinball loss itself — on log(hour-1) alone, plus pre-release features (pre24, pre48).
- Stacking: Vincentization (quantile averaging) with different member weights for hours 2–9, 10–16, and 17–24, fit by coordinate descent on 0.8·practice + 0.2·leave-one-out.
- Spread: per-hour-group multipliers in log space (wider early, narrower late), tuned the same way.

**Scores (local, same scorer as the grader).** Practice mean pinball **0.1176** (per case: Wolverine 0.1154, RDR2 0.1198, GoW Ragnarök 0.0742, Spider-Man 2 0.1611) vs house baseline 0.3871 and v2's 0.1695. Leave-one-out over 37 comparables: 0.4194 all / 0.2254 BIG / 0.4692 tier-clean. Runtime ~0.5 s per case (limit 8 s), deterministic (byte-identical repeat runs).

No network access, no external data: the program reads only the case file it is given (comparables included) and the standard library.

## Run it

```sh
MAKEFUN_INPUT_DIR=/path/to/cases MAKEFUN_OUTPUT_DIR=/path/to/out python3 main.py
```

Each `*.in` case in the input dir produces a `<id>.out` file with 23 rows (hours 2–24) of five quantiles at levels 0.05, 0.25, 0.5, 0.75, 0.95.

---

<sub>Entered on [make.fun](https://make.fun/?ref=khngxesz&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
