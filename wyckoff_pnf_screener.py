"""
NASDAQ-100 Relative-Strength Leadership Screener + Walk-Forward Backtest
=========================================================================
P&F relative strength vs QQQ, trend/momentum filters, Wyckoff volume
confirmation, an earnings-date filter, and a backtest that runs the exact
same rules on past dates.

HOW TO USE IN COLAB
  1. (If needed)  !pip install -q yfinance lxml
  2. Paste this whole file into a cell and run it. It runs the live screener.
  3. To test the rules on history, run in a new cell:
         bt = run_backtest()
     (takes a few minutes: ~100 tickers x ~150 weekly dates)
  4. Check any stocks (in the NASDAQ-100 or not) against the full system:
         check_stocks("MANH, FAST, COST")
     Red-flag check only (runs automatically on BUY + top watch names):
         red_flag_check(["AAPL", "FAST"])
  5. To test ATR-scaled boxes against fixed 1%/2% boxes on the same data:
         runs = compare_box_modes()

THE STRATEGY (a stock must clear every GATE to be a buy candidate)
  Market regime   QQQ above its 200-day average and 50-day above 200-day.
                  Breakouts fail far more often in falling markets.
  Trend           Close > 50-day > 200-day, 200-day rising, within 25% of 52-week high.
  Momentum        12-month return (skipping the latest month) in the top half of the index.
  RS P&F          Relative-strength P&F chart vs QQQ is on a BUY signal.
  Not extended    No more than 5 ATRs above the 50-day average (don't chase).
  Earnings        No earnings report in the next 7 calendar days (avoid a coin-flip gap).
  Volatility      ATR(20) no more than 6% of price.
  Liquidity       50-day average dollar volume >= $25M.

  Then a 0-100 SCORE ranks the survivors:
    Momentum rank 30 | RS P&F quality 20 | Price P&F buy 10 |
    Wyckoff 30 (Secondary Test 10, Sign of Strength 15, SOS above rally high 5;
               earnings reactions and 4%+ opening gaps never count as a SOS) |
    Proximity to 52-week high 10
  BUY CANDIDATE = all gates passed AND score >= 60 (calibrate with the backtest).

  Each candidate gets a trade plan: stop = close - 3.5 x ATR(20), risk %,
  optional share count for a 1% account risk, and a price objective = 1x measured
  move (thrust-column height projected from the P&F breakout level).

P&F BOX SIZE (box_mode="atr", the default)
  Each stock gets its own box: 1 x ATR(20) as a % of price for the price chart,
  1.5 x the RS line's average daily move for the RS chart, rounded to 0.25%
  steps and kept between 0.5% and 6%. Boxes stay percentage-based (log scale),
  so a stock that doubles over two years is charted consistently.

BACKTEST CAVEATS (printed with results too)
  * Uses TODAY's NASDAQ-100 members, so it has survivorship bias: past
    losers that dropped out of the index are missing. Absolute returns are
    flattered. Compare the signal groups with the "All evaluated" baseline,
    which has the same bias.
  * Weekly observations with 60-day holds overlap, so sample sizes overstate
    independence. The "New entries" row counts each fresh signal only once.
"""

import re
from io import StringIO
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# =====================================================================  CONFIG
CONFIG = dict(
    benchmark="QQQ",
    history="3y",              # live screener download (indicators + 2y P&F window)
    pnf_window=504,            # bars of history used to build each P&F chart (~2 years)

    # Point & Figure box sizing
    #   "atr": box = volatility-scaled percent, set per stock on each evaluation date
    #          price box = px_atr_mult x ATR(20)/close
    #          RS box    = rs_vol_mult x average daily % move of the RS line (100 days)
    #   "pct": fixed percentages below (the previous behavior; keep for comparison)
    box_mode="atr",
    px_atr_mult=1.0, rs_vol_mult=1.5,
    box_step=0.0025, box_min=0.005, box_max=0.06,   # round to 0.25% steps to stop daily flicker
    rs_box_pct=0.02, rs_reversal=3, rs_fresh_bars=20,
    px_box_pct=0.01, px_reversal=3,

    # Wyckoff SC -> AR -> ST -> SOS
    wy_lookback=90, sc_vol_mult=2.0, ar_min_rally=0.05,
    st_band=0.03, st_vol_ratio=0.6, st_min_gap=3,
    sos_vol_mult=1.5, sos_range_mult=1.2, sos_close_pos=0.30,
    event_gap=0.04,            # an up-gap >= 4% at the open = news/earnings event, not a SOS

    # Gates
    min_dollar_vol=25e6,
    max_off_high=0.25,
    min_mom_pct=0.50,
    max_extension_atr=5.0,
    max_atr_pct=0.06,          # average daily range above 6% of price = too volatile
    earn_block_days=7,
    require_risk_on=True,

    # Score threshold and risk
    buy_score=60,
    stop_atr_mult=5.0,         # 2021-26 backtest: tighter stops were hit on normal pullbacks
    account_size=None,         # e.g. 100_000 to get share counts
    risk_per_trade=0.01,

    # Red-flag check (headlines, analyst downgrades, estimate cuts)
    news_check=True,
    news_days=14,              # how far back to look at headlines and analyst actions
    news_watch_top=5,          # also check the top N watch-list names

    # Data handling
    use_earnings=True,
    drop_partial_bar=True,

    # Backtest
    bt_history="8y",           # 8 years so the test window includes the 2022 bear market
    bt_years=5,
    bt_step=5,                 # evaluate every 5 trading days (weekly)
    bt_horizons=(20, 60),
    bt_stop_mults=(2.5, 3.5, 5.0, None),   # stop widths to compare (None = no stop)
    # stop styles to compare on BUY signals (all start at stop_atr_mult x ATR):
    #   fixed      - never moves (what the strategy uses)
    #   atr_trail  - raised each day to close - stop_atr_mult x ATR, never lowered
    #   breakeven  - moved to the entry price once the gain equals the initial risk
    #   pnf_trail  - raised to one box below each completed O column on the price P&F chart
    bt_stop_styles=("fixed", "atr_trail", "breakeven", "pnf_trail"),
)

FALLBACK_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "GOOGL", "GOOG", "AVGO", "PEP",
    "COST", "CSCO", "TMUS", "ADBE", "TXN", "NFLX", "AMD", "CMCSA", "INTU", "QCOM",
    "AMGN", "HON", "INTC", "ISRG", "SBUX", "BKNG", "MDLZ", "GILD", "LRCX", "VRTX",
]
MIN_BARS = 260


# =====================================================================  DATA
_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


def _clean(symbols):
    out = sorted({str(t).strip().upper().replace(".", "-") for t in symbols
                  if isinstance(t, str) and 1 <= len(str(t).strip()) <= 6})
    if not 90 <= len(out) <= 110:
        raise ValueError(f"parsed {len(out)} symbols, expected ~100")
    return out


def _from_nasdaq_api():
    r = requests.get("https://api.nasdaq.com/api/quote/list-type/nasdaq100",
                     headers=_BROWSER_HEADERS, timeout=15)
    r.raise_for_status()
    rows = r.json()["data"]["data"]["rows"]
    return _clean(row["symbol"] for row in rows)


def _from_wikipedia(url):
    r = requests.get(url, headers=_BROWSER_HEADERS, timeout=15)
    r.raise_for_status()
    # Don't rely on the table id (Wikipedia changes it). Take any ~100-row
    # table that has a Ticker or Symbol column.
    for tbl in pd.read_html(StringIO(r.text)):
        col = next((c for c in tbl.columns if str(c).strip() in ("Ticker", "Symbol")), None)
        if col is not None and 90 <= len(tbl) <= 110:
            return _clean(tbl[col].dropna())
    raise ValueError("no ~100-row Ticker/Symbol table on page")


def load_nasdaq100():
    sources = [
        ("Nasdaq.com", _from_nasdaq_api),
        ("Wikipedia", lambda: _from_wikipedia("https://en.wikipedia.org/wiki/Nasdaq-100")),
        ("Wikipedia list page",
         lambda: _from_wikipedia("https://en.wikipedia.org/wiki/List_of_Nasdaq-100_companies")),
    ]
    for name, fn in sources:
        try:
            tickers = fn()
            print(f"Loaded {len(tickers)} NASDAQ-100 tickers from {name}.")
            return tickers
        except Exception as e:
            print(f"  {name} failed: {type(e).__name__}: {e}")
    print(f"*** All roster sources failed. Using fallback list of {len(FALLBACK_TICKERS)} "
          f"mega-caps. Momentum ranks and backtest results will NOT reflect the full index. ***")
    return list(FALLBACK_TICKERS)


def _drop_partial_bar(df):
    if df.empty:
        return df
    now = datetime.now(ZoneInfo("America/New_York"))
    if df.index[-1].date() == now.date() and (now.hour, now.minute) < (16, 15):
        return df.iloc[:-1]
    return df


def download_data(tickers, cfg, period):
    symbols = list(dict.fromkeys(list(tickers) + [cfg["benchmark"]]))
    print(f"Downloading {period} of daily data for {len(symbols)} symbols...")
    raw = yf.download(symbols, period=period, interval="1d", group_by="ticker",
                      auto_adjust=True, threads=True, progress=False)
    data = {}
    for t in symbols:
        try:
            df = raw[t].dropna(subset=["Close"])
        except KeyError:
            continue
        if cfg["drop_partial_bar"]:
            df = _drop_partial_bar(df)
        if not df.empty:
            df.index = pd.DatetimeIndex(df.index).tz_localize(None) \
                if df.index.tz is not None else pd.DatetimeIndex(df.index)
            data[t] = df
    return data


def fetch_earnings(tickers, limit=30, workers=4):
    """Past and scheduled earnings dates per ticker (None if unavailable)."""
    def one(t):
        tk = yf.Ticker(t)
        try:
            ed = tk.get_earnings_dates(limit=limit)
            if ed is not None and len(ed):
                idx = pd.DatetimeIndex(ed.index)
                if idx.tz is not None:
                    idx = idx.tz_convert("America/New_York").tz_localize(None)
                return t, np.array(sorted(set(idx.normalize())), dtype="datetime64[ns]")
        except Exception:
            pass
        try:
            cal = tk.calendar
            dts = cal.get("Earnings Date", []) if isinstance(cal, dict) else []
            if dts:
                return t, np.array(sorted(pd.to_datetime(dts)), dtype="datetime64[ns]")
        except Exception:
            pass
        return t, None

    print(f"Fetching earnings dates for {len(tickers)} tickers...")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        out = dict(ex.map(one, tickers))
    today = np.datetime64(pd.Timestamp.now().normalize())
    ok = sum(v is not None for v in out.values())
    past = sum(v is not None and (v < today).any() for v in out.values())
    print(f"  earnings dates found for {ok}/{len(tickers)} "
          f"({past} with PAST report dates, needed to spot earnings-driven moves)")
    if past < ok:
        print("  NOTE: tickers without past dates rely on the price-gap check "
              "to exclude event-driven Signs of Strength.")
    return out


def earnings_ahead(ed, d, days):
    """True if a report falls in (d, d+days]. None if unknown."""
    if ed is None:
        return None
    d0 = np.datetime64(pd.Timestamp(d).normalize())
    return bool(((ed > d0) & (ed <= d0 + np.timedelta64(days, "D"))).any())


def next_earnings(ed, d):
    if ed is None:
        return None
    d0 = np.datetime64(pd.Timestamp(d).normalize())
    fut = ed[ed > d0]
    return pd.Timestamp(fut[0]).date().isoformat() if len(fut) else None


def on_earnings_reaction(ed, day):
    """True if `day` is the report day or within 3 calendar days after it."""
    if ed is None or day is None:
        return False
    d0 = np.datetime64(pd.Timestamp(day).normalize())
    delta = (d0 - ed) / np.timedelta64(1, "D")
    return bool(((delta >= 0) & (delta <= 3)).any())


# =====================================================================  INDICATORS
def add_indicators(df):
    """All causal (rolling / shifted), so values at date d use only data <= d."""
    d = df.copy()
    c = d["Close"]
    d["sma50"] = c.rolling(50).mean()
    d["sma200"] = c.rolling(200).mean()
    d["sma200_up"] = d["sma200"] > d["sma200"].shift(20)
    prev = c.shift(1)
    tr = pd.concat([d["High"] - d["Low"], (d["High"] - prev).abs(),
                    (d["Low"] - prev).abs()], axis=1).max(axis=1)
    d["atr20"] = tr.rolling(20).mean()
    d["hi252"] = d["High"].rolling(252, min_periods=200).max()
    d["mom"] = c.shift(21) / c.shift(252) - 1            # 12-1 month momentum
    d["dvol50"] = (c * d["Volume"]).rolling(50).mean()
    d["vol50"] = d["Volume"].rolling(50).mean()
    d["vol20"] = d["Volume"].rolling(20).mean()
    d["rng"] = d["High"] - d["Low"]
    d["rng20"] = d["rng"].rolling(20).mean()
    d["chg"] = c.diff()
    return d


def prepare(tickers, data, cfg):
    bench = add_indicators(data[cfg["benchmark"]])
    bench["risk_on"] = (bench["Close"] > bench["sma200"]) & (bench["sma50"] > bench["sma200"])
    prepared, moms = {}, {}
    for t in tickers:
        df = data.get(t)
        if df is None or len(df) < MIN_BARS:
            continue
        df = add_indicators(df)
        rs = (df["Close"] / bench["Close"].reindex(df.index) * 100).dropna()
        prepared[t] = (df, rs)
        moms[t] = df["mom"]
    mom_pct = pd.DataFrame(moms).reindex(bench.index).rank(axis=1, pct=True)
    return bench, prepared, mom_pct


# =====================================================================  POINT & FIGURE
def build_pnf(series, box_pct, reversal):
    """Close-only P&F, log-scaled integer boxes. Returns (columns, signals, n)."""
    s = series.dropna()
    lv = np.log(s.to_numpy(dtype=float)) / np.log1p(box_pct)
    dates = s.index
    cols, signals = [], []
    base = int(np.floor(lv[0]))
    for i, v in enumerate(lv):
        up, dn = int(np.floor(v)), int(np.ceil(v))
        if not cols:
            if up >= base + 1:
                cols.append(dict(type="X", bottom=base, top=up, sig=False, i0=i, ie=i))
            elif dn <= base - 1:
                cols.append(dict(type="O", top=base, bottom=dn, sig=False, i0=i, ie=i))
            continue
        c = cols[-1]
        if c["type"] == "X":
            if up > c["top"]:
                c["top"], c["ie"] = up, i
            elif dn <= c["top"] - reversal:
                cols.append(dict(type="O", top=c["top"] - 1, bottom=dn, sig=False, i0=i, ie=i))
        else:
            if dn < c["bottom"]:
                c["bottom"], c["ie"] = dn, i
            elif up >= c["bottom"] + reversal:
                cols.append(dict(type="X", bottom=c["bottom"] + 1, top=up, sig=False, i0=i, ie=i))
        c = cols[-1]
        if len(cols) >= 3 and not c["sig"]:
            prior = cols[-3]
            if c["type"] == "X" and c["top"] > prior["top"]:
                c["sig"], c["si"] = True, i
                signals.append(("BUY", dates[i], i, prior["top"] + 1))   # breakout level
            elif c["type"] == "O" and c["bottom"] < prior["bottom"]:
                c["sig"], c["si"] = True, i
                signals.append(("SELL", dates[i], i, prior["bottom"] - 1))
    return cols, signals, len(s)


def pnf_chart_data(series, box_pct, reversal, max_cols=60):
    """Compact P&F chart data for drawing: [[1=X/0=O, bottom, top, 'YYYY-MM', signal], ...]."""
    s = series.dropna()
    cols, sigs, n = build_pnf(s, box_pct, reversal)
    if not cols:
        return None
    dates = s.index
    def day(i):
        return pd.Timestamp(dates[i]).strftime("%Y-%m-%d") if i is not None else ""
    # [X=1/O=0, bottom, top, start month, signal?, signal date, date of last box]
    out = [[1 if c["type"] == "X" else 0, int(c["bottom"]), int(c["top"]),
            pd.Timestamp(dates[c.get("i0", 0)]).strftime("%Y-%m"), 1 if c["sig"] else 0,
            day(c.get("si")), day(c.get("ie"))]
           for c in cols[-max_cols:]]
    last = sigs[-1] if sigs else None
    return dict(box=round(float(box_pct), 5), rev=int(reversal), cols=out,
                last=round(float(s.iloc[-1]), 4),
                signal=last[0] if last else None,
                bars_ago=int(n - 1 - last[2]) if last else None)


def box_sizes(df, rs, pos, rpos, cfg):
    """Box sizes (as fractions) for the price and RS charts on this date.
    Uses only data up to the evaluation date, so it is backtest-safe."""
    if cfg["box_mode"] != "atr":
        return cfg["px_box_pct"], cfg["rs_box_pct"]

    def snap(x):
        x = round(x / cfg["box_step"]) * cfg["box_step"]
        return float(min(cfg["box_max"], max(cfg["box_min"], x)))

    r = df.iloc[pos]
    px = snap(cfg["px_atr_mult"] * r["atr20"] / r["Close"])
    rs_move = rs.iloc[max(0, rpos - 99): rpos + 1].pct_change().abs().mean()
    rsb = snap(cfg["rs_vol_mult"] * rs_move) if pd.notna(rs_move) else cfg["rs_box_pct"]
    return px, rsb


def pnf_status(series, box_pct, reversal):
    cols, sigs, n = build_pnf(series, box_pct, reversal)
    if len(cols) < 3:
        return None
    last = sigs[-1] if sigs else None
    st = dict(column=cols[-1]["type"],
              signal=last[0] if last else None,
              bars_ago=(n - 1 - last[2]) if last else None,
              objective=None)
    if st["signal"] == "BUY":
        # 1x measured move: height of the thrust column (first X column off the
        # most recent low) projected up from the breakout level. More conservative
        # than the classic 3x vertical count, which overstates targets badly.
        recent = cols[-7:]
        o_idx = [k for k, c in enumerate(recent) if c["type"] == "O"]
        if o_idx:
            j = min(o_idx, key=lambda k: recent[k]["bottom"])
            if j + 1 < len(recent):
                th = recent[j + 1]
                boxes = th["top"] - th["bottom"] + 1
                st["objective"] = (1 + box_pct) ** (last[3] + boxes)
    return st


# =====================================================================  WYCKOFF
def wyckoff_check(w, cfg, ed=None):
    """w = trailing window with indicator columns. Events must occur in order.
    Earnings reactions and large opening gaps are not counted as a Sign of
    Strength: they reflect news, not gradual absorption of supply."""
    out = dict(sc=None, st=None, sos=None, sos_above_ar=False, sos_event_skipped=False)
    sc_c = w[(w["chg"] < 0) & (w["Volume"] >= cfg["sc_vol_mult"] * w["vol50"])]
    if sc_c.empty:
        return out
    sc_date = sc_c["Low"].idxmin()
    sc_pos = w.index.get_loc(sc_date)
    sc_low, sc_vol = w.at[sc_date, "Low"], w.at[sc_date, "Volume"]
    out["sc"] = sc_date
    after = w.iloc[sc_pos + 1:]
    if after.empty or after["Low"].min() < sc_low * (1 - cfg["st_band"]):
        return out
    st_pos, ar_high = None, None
    for k in range(sc_pos + cfg["st_min_gap"], len(w)):
        bar = w.iloc[k]
        ar = w["High"].iloc[sc_pos + 1:k].max()
        if (abs(bar["Low"] / sc_low - 1) <= cfg["st_band"]
                and bar["Volume"] <= cfg["st_vol_ratio"] * sc_vol
                and ar >= sc_low * (1 + cfg["ar_min_rally"])):
            st_pos, ar_high = k, ar
            out["st"] = w.index[k]
            break
    if st_pos is None:
        return out
    post = w.iloc[st_pos + 1:]
    rng = post["rng"].replace(0, np.nan)
    sos = post[(post["chg"] > 0) & (post["Close"] > post["Open"])
               & ((post["High"] - post["Close"]) / rng <= cfg["sos_close_pos"])
               & (post["rng"] >= cfg["sos_range_mult"] * post["rng20"])
               & (post["Volume"] >= cfg["sos_vol_mult"] * post["vol20"])]
    if not sos.empty:
        gap = sos["Open"] / (sos["Close"] - sos["chg"]) - 1
        event = (gap >= cfg["event_gap"]) | pd.Series(
            [on_earnings_reaction(ed, dt) for dt in sos.index], index=sos.index)
        out["sos_event_skipped"] = bool(event.any())
        sos = sos[~event]
    if not sos.empty:
        above = sos[sos["Close"] > ar_high]
        out["sos_above_ar"] = not above.empty
        out["sos"] = (above if not above.empty else sos).index[-1]
    return out


# =====================================================================  EVALUATION
def evaluate(ticker, df, rs, d, mom_pct_val, risk_on, ed, cfg):
    """Point-in-time evaluation of one ticker on date d. Used by BOTH the
    screener and the backtest, so the backtest tests exactly the live rules."""
    pos = df.index.get_indexer([d])[0]
    rpos = rs.index.get_indexer([d])[0]
    if pos < MIN_BARS - 1 or rpos < 0:
        return None
    r = df.iloc[pos]
    need = ["sma50", "sma200", "atr20", "hi252", "dvol50"]
    if r[need].isna().any() or pd.isna(mom_pct_val):
        return None

    W = cfg["pnf_window"]
    px_box, rs_box = box_sizes(df, rs, pos, rpos, cfg)
    rs_st = pnf_status(rs.iloc[max(0, rpos - W + 1): rpos + 1], rs_box, cfg["rs_reversal"])
    px_st = pnf_status(df["Close"].iloc[max(0, pos - W + 1): pos + 1], px_box, cfg["px_reversal"])
    if rs_st is None or px_st is None:
        return None
    wy = wyckoff_check(df.iloc[max(0, pos - cfg["wy_lookback"] + 1): pos + 1], cfg, ed)

    close, atr = float(r["Close"]), float(r["atr20"])
    if atr <= 0 or close <= 0:          # flat/stale price data (e.g. halted or bad bars)
        return None
    earn_soon = earnings_ahead(ed, d, cfg["earn_block_days"]) if cfg["use_earnings"] else None
    off_high = 1 - close / r["hi252"]

    gates = {
        "Liquidity": r["dvol50"] >= cfg["min_dollar_vol"],
        "Trend": bool(close > r["sma50"] > r["sma200"] and r["sma200_up"]
                      and off_high <= cfg["max_off_high"]),
        "Momentum": mom_pct_val >= cfg["min_mom_pct"],
        "RS_PnF": rs_st["signal"] == "BUY",
        "NotExtended": (close - r["sma50"]) / atr <= cfg["max_extension_atr"],
        "NoEarnings": earn_soon is not True,
        "Volatility": atr / close <= cfg["max_atr_pct"],
    }
    if cfg["require_risk_on"]:
        gates["Regime"] = bool(risk_on)
    failed = [k for k, v in gates.items() if not v]

    # Weights (total 100). These are the ORIGINAL weights, set before any
    # backtest. A reweighting fitted to the 2023-26 results did worse on 2021-26,
    # so the original weights are kept. Don't tune these to the backtest.
    s = 30 * mom_pct_val
    if rs_st["signal"] == "BUY":
        s += 10
        if rs_st["bars_ago"] is not None and rs_st["bars_ago"] <= cfg["rs_fresh_bars"]:
            s += 5
    if rs_st["column"] == "X":
        s += 5
    if px_st["signal"] == "BUY":
        s += 10
    if wy["st"] is not None:
        s += 10
    if wy["sos"] is not None:
        s += 15
        if wy["sos_above_ar"]:
            s += 5
    s += 10 if off_high <= 0.10 else 5 if off_high <= 0.15 else 0
    score = round(s, 1)

    if not failed and score >= cfg["buy_score"]:
        tier = "BUY"
    elif (not failed and score >= 40) or (len(failed) == 1 and score >= cfg["buy_score"]):
        tier = "WATCH"
    else:
        tier = ""

    stop = close - cfg["stop_atr_mult"] * atr
    wy_tag = "".join([
        "SC" if wy["sc"] is not None else "",
        ">ST" if wy["st"] is not None else "",
        (">SOS" + ("*" if wy["sos_above_ar"] else "")) if wy["sos"] is not None else "",
        " (event skipped)" if wy["sos_event_skipped"] and wy["sos"] is None else "",
    ])
    row = dict(
        Date=pd.Timestamp(d).date().isoformat(), Ticker=ticker, Tier=tier, Score=score,
        Close=round(close, 2), ATR=round(atr, 4), MomPct=round(mom_pct_val * 100),
        RS=f"{rs_st['column']}/{rs_st['signal'] or '-'}",
        RS_BarsAgo=rs_st["bars_ago"],
        Px=f"{px_st['column']}/{px_st['signal'] or '-'}",
        Wyckoff=wy_tag, OffHigh=round(off_high * 100, 1),
        NextEarn=next_earnings(ed, d) if cfg["use_earnings"] else None,
        Stop=round(stop, 2), RiskPct=round((close - stop) / close * 100, 1),
        Objective=round(px_st["objective"], 2) if px_st["objective"] else None,
        Upside=round((px_st["objective"] / close - 1) * 100, 1) if px_st["objective"] else None,
        PxBox=round(px_box * 100, 2), RSBox=round(rs_box * 100, 2),
        Failed=",".join(failed),
    )
    if cfg["account_size"]:
        row["Shares"] = int(cfg["account_size"] * cfg["risk_per_trade"] // (close - stop))
    return row


# =====================================================================  LIVE SCREENER
NEEDS_TEXT = {
    "Regime": "the market regime to turn RISK-ON (QQQ back above trend)",
    "Trend": "a confirmed uptrend (above a rising 200-day and its 50-day, within 25% of its high)",
    "Momentum": "12-month momentum to reach the top half of the NASDAQ-100",
    "RS_PnF": "a relative-strength buy signal vs QQQ",
    "NotExtended": "a pullback toward its 50-day average (it's too stretched to buy now)",
    "NoEarnings": "its earnings report to pass (due within 7 days)",
    "Volatility": "volatility to calm down (daily range over 6% of price)",
    "Liquidity": "more trading volume",
}


def watch_needs(row, cfg=CONFIG):
    """Plain-English reason a WATCH stock isn't a BUY yet, and what would change that."""
    failed = [f for f in str(row.get("Failed") or "").split(",") if f]
    score = float(row.get("Score") or 0)
    parts = [NEEDS_TEXT.get(f, f) for f in failed]
    if score < cfg["buy_score"]:
        gap = cfg["buy_score"] - score
        rs_col, rs_sig = (str(row.get("RS") or "-/-").split("/") + ["-"])[:2]
        px_sig = str(row.get("Px") or "-/-").split("/")[-1]
        wy = str(row.get("Wyckoff") or "")
        bars = row.get("RS_BarsAgo")
        off = float(row.get("OffHigh") or 100)
        options = []
        if "SOS" not in wy:
            options.append(("a Wyckoff sign of strength", 15))
        if ">ST" not in wy:
            options.append(("a Wyckoff secondary test", 10))
        if px_sig != "BUY":
            options.append(("a price P&F breakout", 10))
        if off > 10:
            options.append(("a move to within 10% of its high", 10 if off > 15 else 5))
        if rs_col != "X":
            options.append(("the RS chart turning back up", 5))
        options.sort(key=lambda o: -o[1])
        hint = ", ".join(f"{n} (+{p})" for n, p in options[:2])
        need = max(1, int(np.ceil(gap - 1e-9)))
        now = f"{score:.1f}".rstrip("0").rstrip(".")
        parts.append(f"{need} more score point{'s' if need > 1 else ''} (now {now})"
                     + (f", e.g. from {hint}" if hint else ""))
    if not parts:
        return ""
    if failed == ["Regime"] and score >= cfg["buy_score"]:
        return "Ready except for the market: needs " + parts[0] + "."
    return "Needs " + "; ".join(parts) + "."


def run_screener(cfg=CONFIG, tickers=None, data=None, earnings=None):
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] RS leadership screener vs {cfg['benchmark']}")
    tickers = tickers or load_nasdaq100()
    data = data or download_data(tickers, cfg, cfg["history"])
    if cfg["benchmark"] not in data:
        print("Benchmark download failed.")
        return None
    if earnings is None:
        earnings = fetch_earnings(tickers) if cfg["use_earnings"] else {}

    bench, prepared, mom_pct = prepare(tickers, data, cfg)
    d = bench.index[-1]
    risk_on = bool(bench["risk_on"].iloc[-1])
    print(f"\nAs of {d.date()}  |  Market regime: "
          f"{'RISK-ON' if risk_on else 'CAUTION (QQQ below trend) - no new buys'}")

    rows, skipped = [], [t for t in tickers if t not in prepared]
    for t, (df, rs) in prepared.items():
        try:
            row = evaluate(t, df, rs, d, mom_pct.at[d, t] if t in mom_pct else np.nan,
                           risk_on, earnings.get(t), cfg)
            if row:
                rows.append(row)
            else:
                skipped.append(t)
        except Exception as e:
            skipped.append(f"{t} ({type(e).__name__})")
    if not rows:
        print("No tickers could be evaluated.")
        return None

    res = pd.DataFrame(rows).sort_values("Score", ascending=False).reset_index(drop=True)
    res["Needs"] = [watch_needs(r, cfg) if r["Tier"] == "WATCH" else "" for _, r in res.iterrows()]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    show = ["Ticker", "Score", "Close", "MomPct", "RS", "RS_BarsAgo", "Px", "Wyckoff",
            "OffHigh", "NextEarn", "Stop", "RiskPct", "Objective", "Upside", "PxBox", "RSBox"]
    if cfg["account_size"]:
        show.append("Shares")
    buys = res[res["Tier"] == "BUY"]
    print("\n" + "=" * 90)
    print(f" BUY CANDIDATES  (all gates passed, score >= {cfg['buy_score']})")
    print("=" * 90)
    print(buys[show].to_string(index=False) if not buys.empty else "None today.")

    watch = res[res["Tier"] == "WATCH"].head(20)
    print("\n" + "=" * 90)
    print(" WATCH LIST  (passes gates with lower score, or one gate short)")
    print("=" * 90)
    print(watch[show + ["Failed"]].to_string(index=False) if not watch.empty else "None.")
    for _, r in watch.iterrows():
        print(f"  {r['Ticker']:6s} {r['Needs']}")

    gate_cols = ["Liquidity", "Trend", "Momentum", "RS_PnF", "NotExtended", "NoEarnings",
                 "Volatility", "Regime"]
    fails = {g: res["Failed"].str.contains(g).sum() for g in gate_cols}
    print(f"\nEvaluated {len(res)} | skipped {len(skipped)} | gate failures: "
          + ", ".join(f"{k} {v}" for k, v in fails.items() if v))
    if cfg.get("news_check"):
        check = list(buys["Ticker"]) + list(watch["Ticker"].head(cfg["news_watch_top"]))
        if check:
            flags = red_flag_check(check, cfg["news_days"])
            res = res.merge(flags, on="Ticker", how="left")
            print("\nVETO? = strong reason to skip or wait | REVIEW = read the headlines first "
                  "| CLEAN = nothing found | NO DATA = check manually")

    fname = f"screener_{d:%Y%m%d}.csv"
    res.drop(columns=["Headlines"], errors="ignore").to_csv(fname, index=False)
    print(f"Full results saved to {fname}")
    res.attrs.update(asof=d.date().isoformat(), risk_on=risk_on, benchmark=cfg["benchmark"],
                     universe=len(tickers), evaluated=len(res), gate_failures=fails)
    return res


# =====================================================================  CHECK ANY STOCK
GATE_TEXT = {
    "Liquidity": "too thinly traded (under $25M a day)",
    "Trend": "not in an uptrend (needs close > 50-day > rising 200-day, within 25% of its high)",
    "Momentum": "12-month momentum is in the bottom half vs the NASDAQ-100",
    "RS_PnF": "not outperforming QQQ (no relative-strength P&F buy signal)",
    "NotExtended": "too far above its 50-day average - wait for a pullback",
    "NoEarnings": "earnings report within 7 days",
    "Volatility": "too volatile (average daily range over 6% of price)",
    "Regime": "market regime is CAUTION (QQQ below trend)",
}


def plain_verdict(row, cfg=CONFIG):
    if row.get("Tier") == "n/a":
        return row.get("Failed") or "No data."
    failed = [f for f in str(row.get("Failed") or "").split(",") if f]
    score = row.get("Score") or 0
    if not failed and score >= cfg["buy_score"]:
        return "Passes everything: BUY candidate. Take it if the red-flag check is CLEAN."
    if failed == ["Regime"] and score >= cfg["buy_score"]:
        return "Setup qualifies, but the market regime is CAUTION - wait for it to turn on."
    parts = [GATE_TEXT.get(f, f) for f in failed if f != "Regime"]
    if score < cfg["buy_score"]:
        parts.append(f"score {score:.0f} is below {cfg['buy_score']}")
    if "Regime" in failed:
        parts.append("and the market regime is CAUTION")
    return "Not a buy: " + "; ".join(parts) + "."


def check_stocks(symbols, cfg=CONFIG, universe=None, data=None, news=True):
    """Run any tickers (in the NASDAQ-100 or not) through the full system:
    gates, score, red-flag check, with a plain-English verdict.
    Momentum is ranked against the NASDAQ-100 so scores are comparable."""
    if isinstance(symbols, str):
        symbols = symbols.replace(";", ",").replace(" ", ",").split(",")
    symbols = list(dict.fromkeys(s.strip().upper().replace(".", "-") for s in symbols if s.strip()))
    if not symbols:
        print("No tickers given.")
        return None
    universe = universe or load_nasdaq100()
    allsyms = list(dict.fromkeys(list(universe) + symbols))
    if data is None:
        data = download_data(allsyms, cfg, cfg["history"])
    else:
        missing = [t for t in symbols if t not in data]
        if missing:
            data = {**data, **download_data(missing, cfg, cfg["history"])}
    if cfg["benchmark"] not in data:
        print("Benchmark download failed.")
        return None
    earnings = fetch_earnings(symbols) if cfg["use_earnings"] else {}
    bench, prepared, mom_pct = prepare(allsyms, data, cfg)
    d = bench.index[-1]
    risk_on = bool(bench["risk_on"].iloc[-1])

    rows = []
    for t in symbols:
        row = None
        if t in prepared:
            df, rs = prepared[t]
            try:
                row = evaluate(t, df, rs, d, mom_pct.at[d, t], risk_on, earnings.get(t), cfg)
            except Exception:
                row = None
        if row is None:
            why = ("no price data (check the symbol)" if t not in data
                   else "not enough history (needs about a year of trading)")
            row = dict(Ticker=t, Tier="n/a", Failed=why)
        if not row.get("Tier"):
            row["Tier"] = "NO"
        row["Verdict"] = watch_needs(row, cfg) if row.get("Tier") == "WATCH" else plain_verdict(row, cfg)
        rows.append(row)
    res = pd.DataFrame(rows)

    print("\n" + "=" * 90)
    print(f" STOCK CHECK  |  as of {d.date()}  |  market regime: {'RISK-ON' if risk_on else 'CAUTION'}")
    print("=" * 90)
    for _, r in res.iterrows():
        if r["Tier"] == "n/a":
            print(f"\n{r['Ticker']:6s}  {r['Verdict']}")
            continue
        print(f"\n{r['Ticker']:6s}  score {r['Score']:.0f}  |  {r['Tier']}  |  close ${r['Close']:,.2f}  "
              f"|  stop ${r['Stop']:,.2f} ({r['RiskPct']:.1f}%)  |  next earnings {r.get('NextEarn') or 'n/a'}")
        print(f"        RS {r['RS']}  price {r['Px']}  momentum rank {r['MomPct']:.0f}  "
              f"off high {r['OffHigh']}%  Wyckoff {r['Wyckoff'] or '-'}")
        print(f"        -> {r['Verdict']}")
    ok = [t for t in res.loc[res["Tier"] != "n/a", "Ticker"]]
    if news and ok and cfg.get("news_check", True):
        flags = red_flag_check(ok, cfg["news_days"])
        res = res.merge(flags, on="Ticker", how="left")
    res.attrs.update(asof=d.date().isoformat(), risk_on=risk_on)
    return res


# =====================================================================  RED-FLAG CHECK
# Looks for reasons NOT to buy. It can flag or veto a candidate, never add one.
NEG_RE = re.compile(
    r"\b(downgrad\w*|lowers? (?:guidance|outlook|forecast|target|price target)\w*|"
    r"cut to (?:hold|neutral|sell|underperform|underweight|equal[- ]weight)|"
    r"cuts? (?:guidance|outlook|forecast|price target|target)\w*|"
    r"miss(?:es|ed)?|lawsuit\w*|sued|sues|class action|probe\w*|investigat\w*|subpoena\w*|"
    r"SEC|DOJ|FTC|antitrust|recall\w*|fraud\w*|short[- ]seller\w*|plunge\w*|tumble\w*|"
    r"slump\w*|sinks?|sank|layoffs?|job cuts|resign\w*|steps down|ousted|bankrupt\w*|"
    r"defaults?|delist\w*|halt\w*|warns?|warning|restat\w*|rejects?|rejected|delay\w*|"
    r"setbacks?|weak demand|tariffs?)\b", re.I)
EVENT_RE = re.compile(
    r"\b(acquir\w*|acquisition|takeover|buyout|merger|deal talks|bid for|tender offer|"
    r"activist|spin[- ]?off)\b", re.I)


def _news_items(tk):
    """Normalize yfinance news across its old and new formats."""
    try:
        raw = tk.news or []
    except Exception:
        return None                      # fetch failed (distinct from "no news")
    out = []
    for it in raw:
        c = it.get("content", it) if isinstance(it, dict) else {}
        title = c.get("title") or ""
        ts = None
        if c.get("pubDate") or c.get("displayTime"):
            ts = pd.to_datetime(c.get("pubDate") or c.get("displayTime"), utc=True, errors="coerce")
        elif it.get("providerPublishTime"):
            ts = pd.to_datetime(it["providerPublishTime"], unit="s", utc=True, errors="coerce")
        prov = c.get("provider")
        src = prov.get("displayName", "") if isinstance(prov, dict) else it.get("publisher", "")
        link = c.get("canonicalUrl") or c.get("clickThroughUrl")
        url = link.get("url", "") if isinstance(link, dict) else it.get("link", "")
        out.append(dict(ts=ts, title=title, source=src or "", url=url or "",
                        text=f"{title} {c.get('summary') or ''}"))
    return out


def _estimate_revisions(tk):
    """(up, down) EPS estimate revisions for the current fiscal year, last 30 days."""
    try:
        r = tk.get_eps_revisions()
        if r is None or r.empty:
            return None
        row = r.loc["0y"] if "0y" in r.index else r.iloc[min(2, len(r) - 1)]
        cols = {c.lower(): c for c in r.columns}
        up = row[cols["uplast30days"]] if "uplast30days" in cols else np.nan
        dn = row[cols["downlast30days"]] if "downlast30days" in cols else np.nan
        return float(up) if pd.notna(up) else 0.0, float(dn) if pd.notna(dn) else 0.0
    except Exception:
        return None


def _analyst_actions(tk, days):
    """(upgrades, downgrades) in the last `days` days."""
    try:
        ud = tk.upgrades_downgrades
        if ud is None or ud.empty:
            return None
        idx = pd.DatetimeIndex(ud.index)
        idx = idx.tz_localize(None) if idx.tz is not None else idx
        recent = ud[idx >= pd.Timestamp.now().normalize() - pd.Timedelta(days=days)]
        act = recent["Action"].astype(str).str.lower()
        return int((act == "up").sum()), int((act == "down").sum())
    except Exception:
        return None


def red_flag_check(tickers, days=14, show_links=True):
    """Scan recent headlines, analyst actions and estimate revisions for each ticker.
    Verdicts:
      VETO?   estimates for this year are being cut, or the news is a takeover /
              merger type event (the price move is event-driven, not trend-driven)
      REVIEW  negative headlines or recent downgrades - read them before buying
      CLEAN   nothing found
    This only covers Yahoo Finance's recent headlines, so it is a screen, not a guarantee."""
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    rows = []
    print("\n" + "=" * 90)
    print(f" RED-FLAG CHECK  (last {days} days: headlines, analyst actions, estimate revisions)")
    print("=" * 90)
    for t in tickers:
        tk = yf.Ticker(t)
        items = _news_items(tk)
        news = [n for n in (items or []) if n["ts"] is None or n["ts"] >= cutoff]
        neg, evt = [], []
        for n in news:
            hits = sorted({m.group(0).lower() for m in NEG_RE.finditer(n["text"])})
            ev = sorted({m.group(0).lower() for m in EVENT_RE.finditer(n["text"])})
            if hits:
                neg.append((n, hits))
            if ev:
                evt.append((n, ev))
        rev = _estimate_revisions(tk)
        acts = _analyst_actions(tk, max(days, 30))

        flags, verdict = [], "CLEAN"
        if rev and rev[1] > rev[0]:
            flags.append(f"EPS estimates being cut ({rev[0]:.0f} up / {rev[1]:.0f} down, 30d)")
            verdict = "VETO?"
        if evt:
            flags.append(f"{len(evt)} event headline(s) (deal/takeover type)")
            verdict = "VETO?"
        if acts and acts[1] > 0:
            flags.append(f"{acts[1]} downgrade(s) vs {acts[0]} upgrade(s), 30d")
            verdict = "REVIEW" if verdict == "CLEAN" else verdict
        if neg:
            flags.append(f"{len(neg)} negative headline(s)")
            verdict = "REVIEW" if verdict == "CLEAN" else verdict

        if items is None and rev is None and acts is None:
            verdict = "NO DATA"
            flags.append("Yahoo returned nothing - check manually")
        scanned = "headlines unavailable" if items is None else f"{len(news)} headlines scanned"
        print(f"\n{t:6s} -> {verdict}   ({scanned})")
        print(f"   EPS revisions this FY (30d): "
              + (f"{rev[0]:.0f} up / {rev[1]:.0f} down" if rev else "n/a"))
        print(f"   Analyst actions (30d):       "
              + (f"{acts[0]} up / {acts[1]} down" if acts else "n/a"))
        for n, hits in (evt + neg)[:8]:
            when = n["ts"].strftime("%Y-%m-%d") if n["ts"] is not None and pd.notna(n["ts"]) else "    ?     "
            print(f"   ! {when}  {n['source'][:18]:18s} {n['title'][:95]}  [{', '.join(hits)}]")
            if show_links and n["url"]:
                print(f"       {n['url']}")
        heads = [dict(date=(n["ts"].strftime("%Y-%m-%d") if n["ts"] is not None and pd.notna(n["ts"]) else ""),
                      source=n["source"], title=n["title"], url=n["url"], hits=", ".join(h))
                 for n, h in (evt + neg)[:8]]
        rows.append(dict(Ticker=t, News=verdict, NewsFlags="; ".join(flags), Headlines=heads,
                         EPS_up30=rev[0] if rev else None, EPS_down30=rev[1] if rev else None))
    return pd.DataFrame(rows)


# =====================================================================  BACKTEST
def _fwd(df, pos, h):
    """Return from next-day open to close h bars later."""
    if pos + h >= len(df):
        return np.nan
    return df["Close"].iloc[pos + h] / df["Open"].iloc[pos + 1] - 1


def _trade(df, pos, stop, H):
    """Enter next open, exit at stop (gap-aware) or at close after H bars."""
    if pos + H >= len(df):
        return np.nan
    entry = df["Open"].iloc[pos + 1]
    if entry <= stop:
        return 0.0
    seg = df.iloc[pos + 1: pos + H + 1]
    hit = np.where(seg["Low"].to_numpy() <= stop)[0]
    if len(hit):
        k = hit[0]
        exit_px = min(seg["Open"].iloc[k], stop) if k > 0 else stop
        return exit_px / entry - 1
    return seg["Close"].iloc[-1] / entry - 1


def _pnf_trail_events(df, pos, H, box, rev, W):
    """(bar index when known, stop price) for each completed O column. An O column's low
    is final only once the next X column has started, so there is no look-ahead."""
    start = max(0, pos - W + 1)
    s = df["Close"].iloc[start: pos + H + 1]
    cols, _, _ = build_pnf(s, box, rev)
    ev = []
    for k in range(len(cols) - 1):
        if cols[k]["type"] == "O":
            ev.append((start + cols[k + 1]["i0"], (1 + box) ** (cols[k]["bottom"] - 1)))
    return ev


def _trade_style(df, pos, H, stop0, style, atr_mult=5.0, events=None):
    """Simulate one trade: enter next open, exit when the low touches the stop (at the open
    if it gapped below), else at the close after H bars. Returns (return, bars held)."""
    if pos + H >= len(df):
        return np.nan, np.nan
    o, lo, c = (df[k].to_numpy() for k in ("Open", "Low", "Close"))
    atr = df["atr20"].to_numpy()
    entry = o[pos + 1]
    if entry <= stop0:
        return 0.0, 1
    stop, risk = stop0, entry - stop0
    ev = sorted(events or [])
    for k in range(1, H + 1):
        j = pos + k
        if lo[j] <= stop:
            px = stop if k == 1 else min(o[j], stop)
            return px / entry - 1, k
        # update the stop with information known at today's close (used from tomorrow)
        if style == "atr_trail" and not np.isnan(atr[j]):
            stop = max(stop, c[j] - atr_mult * atr[j])
        elif style == "breakeven" and c[j] >= entry + risk:
            stop = max(stop, entry)
        elif style == "pnf_trail":
            while ev and ev[0][0] <= j:
                stop = max(stop, ev.pop(0)[1])
    return c[pos + H] / entry - 1, H


def _summary(df, mask, name, horizons):
    H = max(horizons)
    g = df[mask]
    if g.empty:
        return dict(Group=name, N=0)
    out = dict(Group=name, N=len(g), Tickers=g["Ticker"].nunique())
    for h in horizons:
        out[f"Excess{h}d_avg%"] = round(g[f"ex{h}"].mean() * 100, 2)
    out[f"Excess{H}d_med%"] = round(g[f"ex{H}"].median() * 100, 2)
    out[f"Return{H}d_avg%"] = round(g[f"ret{H}"].mean() * 100, 2)
    out[f"Beat_QQQ_{H}d%"] = round((g[f"ex{H}"] > 0).mean() * 100, 1)
    out["StopTrade_avg%"] = round(g["trade"].mean() * 100, 2)
    out["StopTrade_win%"] = round((g["trade"] > 0).mean() * 100, 1)
    out["Worst_trade%"] = round(g["trade"].min() * 100, 1)
    return out



def run_backtest(cfg=CONFIG, tickers=None, data=None, earnings=None):
    H = max(cfg["bt_horizons"])
    cfg = dict(cfg, drop_partial_bar=True)
    tickers = tickers or load_nasdaq100()
    data = data or download_data(tickers, cfg, cfg["bt_history"])
    if earnings is None:
        earnings = fetch_earnings(tickers) if cfg["use_earnings"] else {}
    bench, prepared, mom_pct = prepare(tickers, data, cfg)

    first = max(MIN_BARS, len(bench) - cfg["bt_years"] * 252 - H)
    dates = bench.index[first: len(bench) - H - 1: cfg["bt_step"]]
    print(f"Backtesting {len(prepared)} tickers on {len(dates)} dates "
          f"({dates[0].date()} to {dates[-1].date()})...")

    rows = []
    for i, d in enumerate(dates):
        bpos = bench.index.get_loc(d)
        risk_on = bool(bench["risk_on"].iloc[bpos])
        bret = {h: _fwd(bench, bpos, h) for h in cfg["bt_horizons"]}
        for t, (df, rs) in prepared.items():
            try:
                row = evaluate(t, df, rs, d, mom_pct.at[d, t], risk_on, earnings.get(t), cfg)
            except Exception:
                row = None
            if not row:
                continue
            pos = df.index.get_loc(d)
            for h in cfg["bt_horizons"]:
                row[f"ret{h}"] = _fwd(df, pos, h)
                row[f"ex{h}"] = row[f"ret{h}"] - bret[h]
            row["trade"] = _trade(df, pos, row["Stop"], H)
            for m in cfg["bt_stop_mults"]:
                stop_m = row["Close"] - m * row["ATR"] if m else -np.inf
                row[f"stop_{m or 'none'}"] = _trade(df, pos, stop_m, H)
            if row["Tier"] == "BUY":
                stop0 = row["Close"] - cfg["stop_atr_mult"] * row["ATR"]
                ev = None
                if "pnf_trail" in cfg["bt_stop_styles"]:
                    try:
                        rpos = rs.index.get_loc(d)
                        pxb, _ = box_sizes(df, rs, pos, rpos, cfg)
                        ev = _pnf_trail_events(df, pos, H, pxb, cfg["px_reversal"], cfg["pnf_window"])
                    except Exception:
                        ev = []
                for st in cfg["bt_stop_styles"]:
                    r_, k_ = _trade_style(df, pos, H, stop0, st, cfg["stop_atr_mult"], ev)
                    row[f"style_{st}"], row[f"days_{st}"] = r_, k_
            row["RiskOn"] = risk_on
            rows.append(row)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(dates)} dates done")

    bt = pd.DataFrame(rows).dropna(subset=[f"ex{H}"])
    bt = bt.sort_values(["Ticker", "Date"])
    bt["NewEntry"] = (bt["Tier"] == "BUY") & (bt.groupby("Ticker")["Tier"].shift(1) != "BUY")
    gates_ok = bt["Failed"].isin(["", "Regime"])

    groups = [
        ("All evaluated (baseline)", pd.Series(True, index=bt.index)),
        ("Passed gates (ignoring regime)", gates_ok),
        ("Passed gates, score >= 40", gates_ok & (bt["Score"] >= 40)),
        ("BUY candidates", bt["Tier"] == "BUY"),
        ("BUY: new entries only", bt["NewEntry"]),
        ("Gates ok but regime CAUTION", gates_ok & ~bt["RiskOn"]
         & (bt["Score"] >= cfg["buy_score"])),
        ("Any Wyckoff SOS (no gates)", bt["Wyckoff"].str.contains("SOS")),
    ]
    summ = pd.DataFrame([_summary(bt, m, n, cfg["bt_horizons"]) for n, m in groups])

    bt["ScoreBucket"] = pd.cut(bt["Score"], [0, 30, 45, 60, 75, 101], right=False)
    buckets = pd.DataFrame([_summary(bt, bt["ScoreBucket"] == b, f"Score {b}", cfg["bt_horizons"])
                            for b in bt["ScoreBucket"].cat.categories])

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("\n" + "=" * 100)
    print(f" BACKTEST RESULTS  |  excess = return minus QQQ over the same window  |  "
          f"StopTrade = {H}d hold with {cfg['stop_atr_mult']}xATR stop")
    print("=" * 100)
    print(summ.to_string(index=False))
    print("\n By score bucket (all rows, gates ignored):")
    print(buckets.to_string(index=False))

    stop_rows = []
    for name, m in groups[:1] + groups[3:6]:
        g = bt[m]
        r = dict(Group=name, N=len(g))
        for sm in cfg["bt_stop_mults"]:
            c = g[f"stop_{sm or 'none'}"]
            lab = f"{sm}xATR" if sm else "NoStop"
            r[f"{lab}_avg%"] = round(c.mean() * 100, 2)
            r[f"{lab}_win%"] = round((c > 0).mean() * 100, 1)
            r[f"{lab}_worst%"] = round(c.min() * 100, 1)
        stop_rows.append(r)
    print(f"\n Stop-width comparison ({H}-day hold, enter next open, exit at stop or day {H}):")
    print(pd.DataFrame(stop_rows).to_string(index=False))

    styles = [st for st in cfg["bt_stop_styles"] if f"style_{st}" in bt.columns]
    if styles:
        style_rows = []
        for name, m in (("BUY candidates", bt["Tier"] == "BUY"), ("BUY: new entries only", bt["NewEntry"])):
            g = bt[m]
            for st in styles:
                c = g[f"style_{st}"].dropna()
                if c.empty:
                    continue
                style_rows.append(dict(
                    Group=name, Style=st, N=len(c),
                    Avg_return_pct=round(c.mean() * 100, 2),
                    Median_pct=round(c.median() * 100, 2),
                    Win_pct=round((c > 0).mean() * 100, 1),
                    Worst_pct=round(c.min() * 100, 1),
                    Best_pct=round(c.max() * 100, 1),
                    Avg_days_held=round(g[f"days_{st}"].dropna().mean(), 1),
                    Stopped_out_pct=round((g[f"days_{st}"].dropna() < H).mean() * 100, 1)))
        print(f"\n Stop-STYLE comparison (BUY signals, {cfg['stop_atr_mult']:g}xATR starting stop, {H}-day max hold):")
        print("   fixed = never moves | atr_trail = raised to close - "
              f"{cfg['stop_atr_mult']:g}xATR daily | breakeven = to entry once up 1x risk | "
              "pnf_trail = under each completed P&F O column")
        print(pd.DataFrame(style_rows).to_string(index=False))
    # ---- simple P&F rules vs the full system (each counted at entry only)
    def entries(mask):
        """First week a stock enters the group (not every week it stays in it)."""
        m = mask.astype("boolean").fillna(False).astype(bool)
        prev = m.groupby(bt["Ticker"]).shift(1, fill_value=False).astype(bool)
        return m & ~prev
    px_buy = bt["Px"].astype(str).str.endswith("/BUY")
    rs_buy = bt["RS"].astype(str).str.endswith("/BUY")
    fresh_rs = rs_buy & (bt["RS_BarsAgo"].fillna(999) <= 5)
    on = bt["RiskOn"].astype(bool)
    simple = [
        ("Baseline (every stock, every week)", pd.Series(True, index=bt.index), False),
        ("Price P&F on a buy signal", px_buy, True),
        ("RS P&F (vs QQQ) on a buy signal", rs_buy, True),
        ("Both P&F charts on buy signals", px_buy & rs_buy, True),
        ("Both P&F buys + RISK-ON regime", px_buy & rs_buy & on, True),
        ("Fresh RS buy (<=5 days) + price buy + RISK-ON", fresh_rs & px_buy & on, True),
        ("FULL SYSTEM: BUY new entries", bt["NewEntry"], False),
    ]
    rows_s = []
    for name, m, as_entry in simple:
        mm = entries(m) if as_entry else m
        r = _summary(bt, mm, name, cfg["bt_horizons"])
        rows_s.append({k: r.get(k) for k in ("Group", "N", "Tickers", f"Excess{H}d_avg%", f"Excess{H}d_med%",
                                             f"Return{H}d_avg%", f"Beat_QQQ_{H}d%", "StopTrade_avg%",
                                             "StopTrade_win%", "Worst_trade%")})
    print(f"\n Simple P&F rules vs the full system ({H}-day hold, {cfg['stop_atr_mult']:g}xATR stop; "
          "P&F groups counted once when a stock enters the group):")
    print(pd.DataFrame(rows_s).to_string(index=False))

    # ---- does the P&F upside target (or reward-to-risk) predict results among BUY signals?
    up = pd.to_numeric(bt["Upside"], errors="coerce")
    rk = pd.to_numeric(bt["RiskPct"], errors="coerce")
    rr = up / rk
    for label, series, edges, names in (
        ("Upside to P&F target", up, [0, 10, 20, 35, np.inf], ["0-10%", "10-20%", "20-35%", "35%+"]),
        ("Reward-to-risk (Upside / Risk)", rr, [0, 1, 2, 3, np.inf], ["under 1", "1-2", "2-3", "3+"]),
    ):
        rows_u = []
        for grp_name, gmask in (("BUY candidates", bt["Tier"] == "BUY"), ("BUY new entries", bt["NewEntry"])):
            met = gmask & ~(series > 0)
            r = _summary(bt, met, f"{grp_name}: target met / none", cfg["bt_horizons"])
            rows_u.append(r)
            for lo, hi, nm in zip(edges[:-1], edges[1:], names):
                m = gmask & (series > lo) & (series <= hi)
                rows_u.append(_summary(bt, m, f"{grp_name}: {nm}", cfg["bt_horizons"]))
        cols_u = ["Group", "N", f"Excess{H}d_avg%", f"Excess{H}d_med%", f"Beat_QQQ_{H}d%",
                  "StopTrade_avg%", "StopTrade_win%", "Worst_trade%"]
        print(f"\n {label}: does more upside mean better results? (BUY signals, {H}-day hold)")
        print(pd.DataFrame(rows_u).reindex(columns=cols_u).to_string(index=False))

    print("\nCAVEATS: current index members only (survivorship bias - compare against the "
          "baseline row, not zero);\n         overlapping weekly windows overstate sample size; "
          "no commissions/slippage.")
    bt.to_csv("backtest_signals.csv", index=False)
    print("All signal rows saved to backtest_signals.csv")
    return bt


def compare_box_modes(cfg=CONFIG):
    """Backtest fixed-% boxes and ATR boxes on the same downloaded data."""
    tickers = load_nasdaq100()
    data = download_data(tickers, cfg, cfg["bt_history"])
    earnings = fetch_earnings(tickers) if cfg["use_earnings"] else {}
    runs = {}
    for mode in ("pct", "atr"):
        print(f"\n{'#' * 30}  box_mode = {mode}  {'#' * 30}")
        runs[mode] = run_backtest(dict(cfg, box_mode=mode), tickers, data, earnings)
    return runs


# =====================================================================  RUN
if __name__ == "__main__":
    results = run_screener()
    # bt = run_backtest()      # uncomment, or run in its own cell
