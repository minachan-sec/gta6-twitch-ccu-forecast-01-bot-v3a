"""v3a — GTA VI Twitch launch-day forecast.

A stacked ensemble of nine full-forecast members over the log-ratio
ln(viewers / hour-1 print), combined by quantile averaging (Vincentization)
with hour-block weights, then reshaped by per-hour-group spread multipliers.

Members
-------
Median + parametric-spread members (spread from asymmetric residual std,
smoothed across hours, scaled, floored):
  * kernel-tier   Gaussian kernel on log(hour-1) over tier='comparable' launches
  * wls-tier      per-hour weighted least squares on log(hour-1), tier pool
  * ridge-pre24   per-hour ridge on [log(hour-1), pre24]
  * ridge-all     per-hour ridge on [log(hour-1), pre24, prepk], lambda 3
  * ridge-l1      same features as ridge-all, lambda 1
  * qreg          per-hour median (tau=0.5) quantile regression on log(hour-1)

Direct quantile members (five levels fit by subgradient descent on pinball):
  * qreg5         linear quantile regression on log(hour-1)
  * qreg5f        quantile regression on [log(hour-1), pre24]
  * qreg5f-pre48  quantile regression on [log(hour-1), pre48]

Stacking weights (fit by coordinate descent on 0.8*practice + 0.2*LOO):
  hours  2- 9: wls-tier .5, ridge-pre24 .4, qreg5f .6, kernel-tier .3
  hours 10-16: ridge-pre24 .2, qreg5 .8, qreg5f .6, qreg .6, qreg5f-pre48 .5
  hours 17-24: wls-tier .7, ridge-pre24 .5, ridge-l1 .8, ridge-all .2

Spread multipliers on the stacked forecast (log-space, around the median):
  hours 2-5: x1.5, 6-9: x0.7, 10-13: x1.0, 14-18: x0.7, 19-24: x0.4

Practice (4 cases): 0.1176 mean pinball (v2: 0.1695, house: 0.3871).
LOO over 37 comparables: 0.4200 all / 0.2253 BIG / 0.4702 tier-clean.
"""
import json
import math
import re
import statistics
from datetime import datetime

Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# --- tuned constants -------------------------------------------------------
KERNEL_BW = 1.0          # Gaussian bandwidth on log(hour-1)
UNLOCK_BOOST = 3.0       # weight multiplier for same unlock class
SPREAD_SMOOTH = 0.7      # MA-3 smoothing of per-hour sigma
SPREAD_SCALE = 0.6       # sigma scale for median members
SPREAD_FLOOR = 0.05      # sigma floor coefficient: floor * sqrt(h-1)
QR_ITERS = 30            # subgradient iterations for 5-level QR
QR_LR = 0.08             # subgradient step for 5-level QR
QREG_ITERS = 200         # median-only QR iterations
QREG_LR = 0.02
SLOPE_MIN, SLOPE_MAX = -2.0, 1.0

BLOCKS = [
    (2, 9, {'wls_tier': 0.5, 'ridge_pre24': 0.4, 'qreg5f': 0.6, 'k10_tier': 0.3}),
    (10, 16, {'ridge_pre24': 0.2, 'qreg5': 0.8, 'qreg5f': 0.6, 'qreg': 0.6,
              'qreg5f_pre48': 0.5}),
    (17, 24, {'wls_tier': 0.7, 'ridge_pre24': 0.5, 'ridge_l1': 0.8,
              'ridge_all': 0.2}),
]
MULT_GROUPS = [(2, 5, 1.5), (6, 9, 0.7), (10, 13, 1.0), (14, 18, 0.7),
               (19, 24, 0.4)]


# --- parsing ---------------------------------------------------------------

def _parse(t):
    return datetime.fromisoformat(t.replace('Z', '+00:00'))


def _tier_of(text):
    m = re.search(r'Tier:\s*([a-z-]+)', text or '')
    return m.group(1) if m else 'none'


def _unlock_class(text):
    u = (text or '').lower()
    if 'roll' in u or 'midnight local' in u:
        return 'rolling'
    if 'simultaneous' in u or 'global unlock' in u or 'single global' in u:
        return 'simultaneous'
    return 'other'


def _window_mean(by_hour, lo, hi):
    vals = [v for h, v in by_hour.items() if lo <= h <= hi and v >= 1]
    return statistics.mean(vals) if vals else None


def comparable_record(c):
    """One comparable launch -> log-ratio shape + pre-release features."""
    ds = _parse(c['day_start'])
    by_hour = {}
    for p in c.get('hourly') or []:
        h = int((_parse(p['from']) - ds).total_seconds() // 3600)
        if -400 <= h < 96 and p.get('viewers') is not None:
            by_hour[h] = p['viewers']
    if 0 not in by_hour or by_hour[0] < 1:
        return None
    v1 = by_hour[0]
    ratios = []
    for h in range(1, 24):
        if h in by_hour and by_hour[h] >= 1:
            ratios.append(math.log(by_hour[h] / v1))
        else:
            ratios.append(None)
    if sum(1 for r in ratios if r is not None) < 18:
        return None
    pre24 = _window_mean(by_hour, -30, -18)
    pre48 = _window_mean(by_hour, -54, -42)
    prepk = max([v for h, v in by_hour.items()
                 if -336 <= h <= -1 and v >= 1], default=None)
    feats = {
        'pre24': math.log(v1 / pre24) if pre24 else None,
        'pre48': math.log(v1 / pre48) if pre48 else None,
        'trend48': math.log(pre24 / pre48) if (pre24 and pre48) else None,
        'prepk': math.log(v1 / prepk) if prepk else None,
    }
    return {'name': c['name'], 'v1': v1, 'ratios': ratios,
            'tier': _tier_of(c.get('unlock') or ''),
            'uclass': _unlock_class(c.get('unlock') or ''),
            'feats': feats}


def target_features(case):
    """Pre-release features for the target case from its history_hourly."""
    ds = _parse(case['day_start'])
    by_hour = {}
    for p in case.get('history_hourly') or []:
        h = int((_parse(p['from']) - ds).total_seconds() // 3600)
        if p.get('viewers') is not None:
            by_hour[h] = p['viewers']
    p1 = [k for k in case['known'] if k['hour'] == 1][0]['viewers']
    pre24 = _window_mean(by_hour, -30, -18)
    pre48 = _window_mean(by_hour, -54, -42)
    prepk = max([v for h, v in by_hour.items()
                 if -336 <= h <= -1 and v >= 1], default=None)
    return {'p1': p1,
            'pre24': math.log(p1 / pre24) if pre24 else None,
            'pre48': math.log(p1 / pre48) if pre48 else None,
            'trend48': math.log(pre24 / pre48) if (pre24 and pre48) else None,
            'prepk': math.log(p1 / prepk) if prepk else None,
            'uclass': _unlock_class(case.get('notes') or '')}


# --- small numeric helpers -------------------------------------------------

def _weights(lv, tf, pool):
    """Gaussian kernel on log(hour-1), boosted for same unlock class."""
    ws = []
    for o in pool:
        w = math.exp(-((lv - math.log(o['v1'])) ** 2) / (2 * KERNEL_BW ** 2))
        if o['uclass'] == tf['uclass']:
            w *= UNLOCK_BOOST
        ws.append(w)
    return ws


def _wquantiles(pts, qs=(0.05, 0.25, 0.5, 0.75, 0.95)):
    pts = sorted(pts)
    tot = sum(w for _, w in pts)
    out = []
    for q in qs:
        acc = 0.0
        val = pts[-1][0]
        for r, w in pts:
            acc += w
            if acc >= q * tot:
                val = r
                break
        out.append(val)
    return out


def _wls(xs, ys, ws):
    """Weighted least squares y ~ a + b x. Returns (a, b) or None."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 3:
        return None
    mx = sum(w * x for x, w in zip(xs, ws)) / sw
    my = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - mx) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-9:
        return (my, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for x, y, w in zip(xs, ys, ws))
    return (my - (sxy / sxx) * mx, sxy / sxx)


def _solve(A, b):
    """Gaussian elimination with partial pivoting. Returns x or None."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(n):
        p = max(range(c, n), key=lambda r: abs(M[r][c]))
        if abs(M[p][c]) < 1e-12:
            return None
        M[c], M[p] = M[p], M[c]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c] / M[c][c]
                for k in range(c, n + 1):
                    M[r][k] -= f * M[c][k]
    return [M[i][n] / M[i][i] for i in range(n)]


def _tier_pool(others):
    pool = [o for o in others if o['tier'] == 'comparable']
    return pool if len(pool) >= 5 else others


# --- median members --------------------------------------------------------

def _median_kernel(p1, tf, pool):
    lv = math.log(p1)
    ws = _weights(lv, tf, pool)
    out = []
    for i in range(23):
        pts = [(o['ratios'][i], w) for o, w in zip(pool, ws)
               if o['ratios'][i] is not None]
        out.append(_wquantiles(pts)[2] if pts else 0.0)
    return out


def _median_wls(p1, tf, pool):
    lv = math.log(p1)
    ws = _weights(lv, tf, pool)
    out = []
    for i in range(23):
        xs, ys, w2 = [], [], []
        for o, w in zip(pool, ws):
            if o['ratios'][i] is not None:
                xs.append(math.log(o['v1']))
                ys.append(o['ratios'][i])
                w2.append(w)
        fit = _wls(xs, ys, w2)
        if fit is None:
            out.append(0.0)
            continue
        a, b = fit
        b = min(max(b, SLOPE_MIN), SLOPE_MAX)
        out.append(a + b * lv)
    return out


def _median_ridge(p1, tf, others, lam, feats):
    """Per-hour ridge on [1, log(v1), feats...]; WLS fallback when a feature
    is missing for the target or too few comparables carry it."""
    lv = math.log(p1)
    fv = [tf.get(f) for f in feats]
    wls_fallback = None

    def wls_hour(i):
        nonlocal wls_fallback
        if wls_fallback is None:
            wls_fallback = _median_wls(p1, tf, others)
        return wls_fallback[i]

    if any(v is None for v in fv):
        return _median_wls(p1, tf, others)
    ws = _weights(lv, tf, others)
    out = []
    for i in range(23):
        rows = [(o, w) for o, w in zip(others, ws)
                if o['ratios'][i] is not None
                and all(o['feats'].get(f) is not None for f in feats)]
        if len(rows) < 6:
            out.append(wls_hour(i))
            continue
        k = 2 + len(feats)
        A = [[0.0] * k for _ in range(k)]
        b = [0.0] * k
        for o, w in rows:
            x = [1.0, math.log(o['v1'])] + [o['feats'][f] for f in feats]
            y = o['ratios'][i]
            for a in range(k):
                b[a] += w * x[a] * y
                for c in range(k):
                    A[a][c] += w * x[a] * x[c]
        for a in range(1, k):
            A[a][a] += lam
        beta = _solve(A, b)
        if beta is None:
            out.append(wls_hour(i))
            continue
        xt = [1.0, lv] + fv
        out.append(sum(beta[j] * xt[j] for j in range(k)))
    return out


def _median_qreg(p1, tf, others):
    """Per-hour median (tau=0.5) quantile regression on log(v1) by
    subgradient descent, warm-started from WLS, x centered per hour."""
    lv = math.log(p1)
    ws = _weights(lv, tf, others)
    out = []
    for i in range(23):
        pts = [(math.log(o['v1']), o['ratios'][i], w)
               for o, w in zip(others, ws) if o['ratios'][i] is not None]
        if len(pts) < 5:
            out.append(0.0)
            continue
        sw = sum(w for _, _, w in pts)
        mx = sum(w * x for x, _, w in pts) / sw
        fit = _wls([x for x, _, _ in pts], [y for _, y, _ in pts],
                   [w for _, _, w in pts])
        a, b = fit if fit else (0.0, 0.0)
        a = a + b * mx  # reparam: pred = a + b*(x - mx)
        for _ in range(QREG_ITERS):
            ga, gb = 0.0, 0.0
            for x, y, w in pts:
                r = y - (a + b * (x - mx))
                g = 0.5 if r >= 0 else -0.5
                ga += w * g
                gb += w * g * (x - mx)
            a += QREG_LR * ga / sw
            b += QREG_LR * gb / sw
            b = min(max(b, SLOPE_MIN), SLOPE_MAX)
        out.append(a + b * (lv - mx))
    return out


# --- parametric spread for median members ----------------------------------

def _parametric_spread(p1, tf, others, med):
    """Per-hour asymmetric sigma from weighted residual std above/below the
    median, MA-3 smoothed, scaled, with a sqrt(h-1) floor. The spread is
    always computed over the FULL pool, even for tier-pool medians."""
    lv = math.log(p1)
    ws = _weights(lv, tf, others)
    sig_up, sig_dn = [], []
    for i in range(23):
        up, dn = [], []
        for o, w in zip(others, ws):
            if o['ratios'][i] is None:
                continue
            r = o['ratios'][i] - med[i]
            (up if r >= 0 else dn).append((r, w))

        def wstd(pts):
            sw = sum(w for _, w in pts)
            if sw <= 0:
                return 0.0
            return math.sqrt(sum(w * r * r for r, w in pts) / sw)

        sig_up.append(wstd(up))
        sig_dn.append(wstd(dn))

    def smooth(arr):
        out = []
        for i in range(23):
            lo, hi = max(0, i - 1), min(22, i + 1)
            avg = statistics.mean(arr[lo:hi + 1])
            out.append((1 - SPREAD_SMOOTH) * arr[i] + SPREAD_SMOOTH * avg)
        return out

    sig_up, sig_dn = smooth(sig_up), smooth(sig_dn)
    rows = []
    for i in range(23):
        h = i + 2
        fl = SPREAD_FLOOR * math.sqrt(h - 1)
        out = []
        for z in Z:
            if z == 0:
                out.append(med[i])
                continue
            s = sig_up[i] if z > 0 else sig_dn[i]
            out.append(med[i] + z * max(s * SPREAD_SCALE, fl))
        for j in range(1, 5):
            out[j] = max(out[j], out[j - 1])
        rows.append([p1 * math.exp(q) for q in out])
    return rows


# --- direct quantile members -----------------------------------------------

def _qreg5(p1, tf, others, feats=()):
    """Five-level linear quantile regression per hour on
    [log(v1), feats...], fit by subgradient descent on the pinball loss
    (kernel-weighted), warm-started from a ridge fit. Falls back to the
    no-feature version when a target feature is missing."""
    lv = math.log(p1)
    fv = [tf.get(f) for f in feats]
    if feats and any(v is None for v in fv):
        # A target feature is missing: fall back to the no-feature version.
        return _qreg5(p1, tf, others, ())
    ws = _weights(lv, tf, others)
    rows = []
    for i in range(23):
        pts = []
        for o, w in zip(others, ws):
            if o['ratios'][i] is None:
                continue
            if any(o['feats'].get(f) is None for f in feats):
                continue
            pts.append(([math.log(o['v1'])] + [o['feats'][f] for f in feats],
                        o['ratios'][i], w))
        if len(pts) < (6 if not feats else 8):
            rows.append([p1] * 5)
            continue
        k = 1 + len(feats)
        sw = sum(w for _, _, w in pts)
        mx = [sum(w * x[j] for x, _, w in pts) / sw for j in range(k)]
        # warm start: ridge solve on centered features
        A = [[0.0] * (k + 1) for _ in range(k + 1)]
        b = [0.0] * (k + 1)
        for x, y, w in pts:
            xc = [1.0] + [x[j] - mx[j] for j in range(k)]
            for a in range(k + 1):
                b[a] += w * xc[a] * y
                for c in range(k + 1):
                    A[a][c] += w * xc[a] * xc[c]
        for a in range(1, k + 1):
            A[a][a] += 3.0
        beta0 = _solve(A, b) or [0.0] * (k + 1)
        xt = [lv - mx[0]] + [fv[j] - mx[j + 1] for j in range(len(feats))]
        xc_pts = [[1.0] + [x[j] - mx[j] for j in range(k)] for x, _, _ in pts]
        qs = []
        for tau in LEVELS:
            beta = beta0[:]
            for _ in range(QR_ITERS):
                g = [0.0] * (k + 1)
                for (x, y, w), xc in zip(pts, xc_pts):
                    pred = sum(beta[j] * xc[j] for j in range(k + 1))
                    r = y - pred
                    gg = tau if r >= 0 else tau - 1.0
                    for j in range(k + 1):
                        g[j] += w * gg * xc[j]
                for j in range(k + 1):
                    beta[j] += QR_LR * g[j] / sw
                beta[1] = min(max(beta[1], SLOPE_MIN), SLOPE_MAX)
            qs.append(beta[0] + sum(beta[j + 1] * xt[j] for j in range(k)))
        for j in range(1, 5):
            qs[j] = max(qs[j], qs[j - 1])
        qs = [min(max(q, -6.0), 6.0) for q in qs]
        rows.append([p1 * math.exp(q) for q in qs])
    return rows


def _house_forecast(p1):
    rows = []
    for h in range(2, 25):
        s = 0.15 * math.sqrt(h - 1)
        rows.append([p1 * math.exp(z * s) for z in Z])
    return rows


# --- the stack ---------------------------------------------------------------

def _member_forecasts(p1, tf, others):
    """All nine members, each a 23x5 forecast of absolute viewers."""
    tier = _tier_pool(others)
    fc = {}
    fc['k10_tier'] = _parametric_spread(
        p1, tf, others, _median_kernel(p1, tf, tier))
    fc['wls_tier'] = _parametric_spread(
        p1, tf, others, _median_wls(p1, tf, tier))
    fc['ridge_pre24'] = _parametric_spread(
        p1, tf, others, _median_ridge(p1, tf, others, 3.0, ('pre24',)))
    fc['ridge_all'] = _parametric_spread(
        p1, tf, others, _median_ridge(p1, tf, others, 3.0, ('pre24', 'prepk')))
    fc['ridge_l1'] = _parametric_spread(
        p1, tf, others, _median_ridge(p1, tf, others, 1.0, ('pre24', 'prepk')))
    fc['qreg'] = _parametric_spread(
        p1, tf, others, _median_qreg(p1, tf, others))
    fc['qreg5'] = _qreg5(p1, tf, others, ())
    fc['qreg5f'] = _qreg5(p1, tf, others, ('pre24',))
    fc['qreg5f_pre48'] = _qreg5(p1, tf, others, ('pre48',))
    return fc


def _stack(fc):
    """Quantile averaging with hour-block weights."""
    rows = []
    for i in range(23):
        h = i + 2
        w = None
        for lo, hi, bw in BLOCKS:
            if lo <= h <= hi:
                w = bw
                break
        tot = sum(w.values())
        rows.append([sum(w[n] * fc[n][i][k] for n in w) / tot
                     for k in range(5)])
    return rows


def _rescale(rows):
    """Per-hour-group spread multipliers, in log space around the median."""
    out = []
    for i, r in enumerate(rows):
        h = i + 2
        m = 1.0
        for lo, hi, mult in MULT_GROUPS:
            if lo <= h <= hi:
                m = mult
                break
        med = r[2]
        lm = math.log(max(med, 1e-9))
        out.append([med * math.exp(m * (math.log(max(q, 1e-9)) - lm))
                    for q in r])
    return out


def _clamp(rows):
    out = []
    for r in rows:
        q = [min(max(v, 1.0), 1e9) if math.isfinite(v) else 1.0 for v in r]
        for j in range(1, 5):
            q[j] = max(q[j], q[j - 1])
        out.append(q)
    return out


def forecast(case):
    """Full pipeline for one case dict. Returns 23 rows of 5 quantiles."""
    tf = target_features(case)
    p1 = max(tf['p1'], 1)
    others = []
    for c in case.get('comparables') or []:
        r = comparable_record(c)
        if r:
            others.append(r)
    if len(others) < 5:
        return _clamp(_house_forecast(p1))
    fc = _member_forecasts(p1, tf, others)
    return _clamp(_rescale(_stack(fc)))


if __name__ == '__main__':
    import sys
    case = json.load(open(sys.argv[1]))
    rows = forecast(case)
    print(json.dumps({'hours': [{'hour': i + 2, 'q': rows[i]}
                                for i in range(23)]}))
