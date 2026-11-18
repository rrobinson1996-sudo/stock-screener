"""
Fund rotation backtest: can a simple, rules-based fund rotation beat holding QQQ,
or hold up much better in bad markets?

Rule (decided before seeing any results):
  - At each month-end, rank the funds by 12-1 momentum (12-month return, skipping the
    latest month).
  - Only funds above their 200-day average qualify.
  - Hold the top N qualifying funds in equal amounts from the next trading day.
    Empty slots sit in cash (T-bills, BIL).
  - Variant: an SPY regime gate (SPY above its 200-day and 50-day above 200-day,
    checked at month-end). When it is off, everything goes to cash.
  - Variant: 50/50 mix of QQQ buy-and-hold and the rotation, rebalanced monthly.
  - Trading cost: 0.10% of the amount traded.
  - Prices are adjusted for dividends (total return).

Pass rule (set in advance): in BOTH periods (2018-21 and 2022-26), the strategy must
either
  (a) beat QQQ on return per unit of worst drawdown (CAGR / |max drawdown|), or
  (b) cut QQQ's max drawdown by at least a third while giving up no more than
      2 points a year of return.
If it fails, it gets dropped.
Added after a random-data check (before any real results): a rotation must ALSO beat
"Equal weight, all funds" on return per drawdown in both periods. Otherwise the ranking
added nothing, and simply holding the equal mix is the better choice.

Colab:
    !pip -q install yfinance
    !git clone -q https://github.com/rrobinson1996-sudo/stock-screener.git
    %cd stock-screener
    import fund_rotation as F
    F.run()
"""

import numpy as np
import pandas as pd

FUNDS = [
    "QQQ", "IWM", "MDY", "RSP",
    "XLK", "SMH", "XLF", "XLE", "XLV", "XLI", "XLY",
    "XLP", "XLU", "XLB", "XLRE", "XLC",
    "EFA", "EEM",
    "MTUM", "PDP",
    "GLD", "SLV", "DBC",
    "TLT", "IEF",
]                       # the report's fund list; QQQ stands in for QQQM (2020), IBIT left out (2024)
CASH = "BIL"
BENCH = ["QQQ", "SPY"]
PERIODS = [("2018-21", "2018-01-01", "2021-12-31"),
           ("2022-26 (incl. 2022 bear)", "2022-01-01", None),
           ("All 2018-26", "2018-01-01", None)]
COST = 0.001
HISTORY_START = "2015-06-01"     # room for 12-month momentum and the 200-day before 2018


# ------------------------------------------------------------------ data
def download(start=HISTORY_START):
    import yfinance as yf
    syms = list(dict.fromkeys(FUNDS + BENCH + [CASH]))
    raw = yf.download(syms, start=start, auto_adjust=True, progress=False, group_by="column")
    px = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
    px = px.dropna(how="all").ffill()
    missing = [s for s in syms if s not in px or px[s].dropna().empty]
    if missing:
        print("No data for:", ", ".join(missing))
    return px


# ------------------------------------------------------------------ strategy
def month_ends(idx):
    s = pd.Series(idx, index=idx)
    return set(s.groupby([idx.year, idx.month]).max())


def rotation_weights(px, top_n=3, gate=False, funds=FUNDS):
    """Daily target weights (decided at month-end closes, applied from the next day)."""
    funds = [f for f in funds if f in px]
    mom = px[funds].shift(21) / px[funds].shift(252) - 1          # 12-1 momentum
    above = px[funds] > px[funds].rolling(200).mean()
    spy = px["SPY"]
    risk_on = (spy > spy.rolling(200).mean()) & (spy.rolling(50).mean() > spy.rolling(200).mean())
    ends = month_ends(px.index)
    w = pd.DataFrame(0.0, index=px.index, columns=funds + [CASH])
    cur = pd.Series(0.0, index=w.columns)
    cur[CASH] = 1.0
    for d in px.index:
        if d in ends and mom.loc[d].notna().sum() >= 5:
            cur = pd.Series(0.0, index=w.columns)
            if gate and not bool(risk_on.loc[d]):
                cur[CASH] = 1.0
            else:
                ok = mom.loc[d][above.loc[d] & mom.loc[d].notna()].sort_values(ascending=False)
                picks = list(ok.index[:top_n])
                for p in picks:
                    cur[p] = 1.0 / top_n
                cur[CASH] = 1.0 - len(picks) / top_n
        w.loc[d] = cur
    return w.shift(1).fillna(0.0)          # decide at the close, hold from the next day


def equal_weights(px, funds=FUNDS):
    """Reference line: every fund with data, equal amounts, rebalanced monthly (no ranking)."""
    funds = [f for f in funds if f in px]
    ends = month_ends(px.index)
    w = pd.DataFrame(0.0, index=px.index, columns=funds + [CASH])
    cur = pd.Series(0.0, index=w.columns)
    cur[CASH] = 1.0
    for d in px.index:
        if d in ends:
            have = [f for f in funds if pd.notna(px.at[d, f]) and pd.notna(px[f].shift(252).at[d])]
            cur = pd.Series(0.0, index=w.columns)
            for f in have:
                cur[f] = 1.0 / len(have)
        w.loc[d] = cur
    return w.shift(1).fillna(0.0)


def run_weights(px, w):
    """Daily returns of a weight table, after trading costs."""
    rets = px[w.columns].pct_change().fillna(0.0)
    if CASH in rets and rets[CASH].abs().sum() == 0:
        rets[CASH] = 0.0
    gross = (w * rets).sum(axis=1)
    turnover = w.diff().abs().sum(axis=1).fillna(0.0)
    return gross - turnover * COST


def mix(r_a, r_b, a=0.5):
    """Monthly-rebalanced mix of two daily return streams (rebalancing cost ignored)."""
    out, wa = [], a
    ends = month_ends(r_a.index)
    va, vb = a, 1 - a
    for d in r_a.index:
        va *= 1 + r_a.loc[d]
        vb *= 1 + r_b.loc[d]
        out.append(va + vb)
        if d in ends:
            tot = va + vb
            va, vb = tot * a, tot * (1 - a)
    v = pd.Series(out, index=r_a.index)
    return v.pct_change().fillna(v.iloc[0] - 1)


# ------------------------------------------------------------------ measures
def stats(r):
    r = r.dropna()
    if len(r) < 60:
        return None
    eq = (1 + r).cumprod()
    yrs = len(r) / 252
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    mdd = (eq / eq.cummax() - 1).min()
    yearly = (1 + r).groupby(r.index.year).prod() - 1
    return dict(CAGR=cagr, MaxDD=mdd, WorstYear=yearly.min(), Vol=r.std() * np.sqrt(252),
                RetPerDD=cagr / abs(mdd) if mdd < 0 else np.nan)


def table(streams, start, end, qqq):
    rows = []
    q = qqq.loc[start:end]
    me = "ME" if pd.__version__ >= "2.2" else "M"
    qm = (1 + q).resample(me).prod() - 1
    for name, r in streams.items():
        s = stats(r.loc[start:end])
        if s is None:
            continue
        m = (1 + r.loc[start:end]).resample(me).prod() - 1
        s["BeatQQQ_months%"] = float((m > qm.reindex(m.index)).mean() * 100) if name != "QQQ buy & hold" else np.nan
        rows.append(dict(Strategy=name, **s))
    t = pd.DataFrame(rows).set_index("Strategy")
    return t


def passes(t, name):
    q = t.loc["QQQ buy & hold"]
    s = t.loc[name]
    a = s["RetPerDD"] > q["RetPerDD"]
    b = (abs(s["MaxDD"]) <= abs(q["MaxDD"]) * 2 / 3) and (s["CAGR"] >= q["CAGR"] - 0.02)
    return bool(a or b), ("(a) better return per drawdown" if a else "(b) much smaller drawdown" if b else "neither")


def fmt(t):
    out = t.copy()
    for c in ["CAGR", "MaxDD", "WorstYear", "Vol"]:
        out[c] = (out[c] * 100).map(lambda v: f"{v:+.1f}%" if c != "Vol" else f"{v:.1f}%")
    out["RetPerDD"] = out["RetPerDD"].map(lambda v: f"{v:.2f}")
    out["BeatQQQ_months%"] = out["BeatQQQ_months%"].map(lambda v: "" if pd.isna(v) else f"{v:.0f}%")
    return out.to_string()


# ------------------------------------------------------------------ main
def run(px=None):
    px = download() if px is None else px
    if CASH not in px:
        px[CASH] = px["QQQ"] * 0 + 1.0           # no T-bill data: cash earns 0
    strategies = {"QQQ buy & hold": px["QQQ"].pct_change().fillna(0.0),
                  "SPY buy & hold": px["SPY"].pct_change().fillna(0.0)}
    strategies["Equal weight, all funds"] = run_weights(px, equal_weights(px))
    for n in (2, 3, 5):
        for gate in (False, True):
            name = f"Top {n}" + (" + SPY gate" if gate else "")
            strategies[name] = run_weights(px, rotation_weights(px, n, gate))
    for gate in (False, True):
        name = "50/50 QQQ + Top 3" + (" + SPY gate" if gate else "")
        strategies[name] = mix(strategies["QQQ buy & hold"], strategies["Top 3" + (" + SPY gate" if gate else "")])

    print("=" * 100)
    print(" FUND ROTATION BACKTEST | monthly, 12-1 momentum, above 200-day, equal weight, 0.10% cost per trade")
    print(" Pass rule (set in advance): in BOTH periods, (a) better return per drawdown than QQQ, or")
    print("                             (b) max drawdown at least 1/3 smaller with no more than 2 pts/yr less return")
    print("=" * 100)
    tables = {}
    for label, start, end in PERIODS:
        t = table(strategies, start, end, strategies["QQQ buy & hold"])
        tables[label] = t
        print(f"\n {label}")
        print(fmt(t))

    print("\n VERDICT (both periods must pass; rotations must also beat the equal-weight mix)")
    p1, p2 = PERIODS[0][0], PERIODS[1][0]
    EW = "Equal weight, all funds"
    for name in strategies:
        if "buy & hold" in name:
            continue
        ok1, why1 = passes(tables[p1], name)
        ok2, why2 = passes(tables[p2], name)
        skill = ""
        if name != EW:
            b1 = tables[p1].at[name, "RetPerDD"] > tables[p1].at[EW, "RetPerDD"]
            b2 = tables[p2].at[name, "RetPerDD"] > tables[p2].at[EW, "RetPerDD"]
            skill = "beats equal mix in both" if b1 and b2 else "does NOT beat the equal mix"
            ok = ok1 and ok2 and b1 and b2
        else:
            ok = ok1 and ok2
        print(f"  {name:28s} {'PASS' if ok else 'FAIL'}   vs QQQ 2018-21: {why1:30s} 2022-26: {why2:30s} {skill}")

    print("\n Read 'Equal weight, all funds' as the no-skill line: if a rotation doesn't beat it,")
    print(" the momentum ranking added nothing beyond simply spreading money across the funds.")
    w = rotation_weights(px, 3, False).iloc[-1]
    held = [f"{k} {v:.0%}" for k, v in w.items() if v > 0]
    print(f"\n Top 3 holdings as of {px.index[-1].date()} (for reference only, not a signal): {', '.join(held)}")
    print("\nCAVEATS: one fixed rule set, no tuning (tuning to this data would overstate results);")
    print("         2018-26 includes only one real bear market (2022) plus the 2020 crash; before taxes.")
    return tables
