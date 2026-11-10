"""
Daily check of the positions you hold, from your private Google Sheet.

Every tab laid out like "Positions" (A1 = Ticker) is an account: your real one,
a "Paper" one for pretend money, or any others you add. Columns:

  A Ticker | B Account | C Entry date | D Entry price | E Shares | F Stop | G Notes      (you)
  H Last close | I Days held | J Health | K Action | L Why | M Updated                  (system)
  N Gain % | O Value | P $ at risk to stop                                            (formulas)
  Q Exit date | R Exit price                                                          (you, when you sell)
  S QQQ same period                                                                   (system)
  T Trade return | U vs QQQ                                                           (formulas)

A row with an exit is a closed trade: it is kept for the record, marked CLOSED,
and compared with QQQ over the same days. Open rows get exactly one action:
  1. Stop hit            -> SELL
  2. Health broken       -> SELL
  3. Earnings in 7 days  -> HOLD THROUGH (gain covers stop distance) or SELL BEFORE EARNINGS
  4. Red flag VETO?      -> HOLD (health strong) or SELL (health fading)
  5. 60-day review       -> EXTEND 60 DAYS (strong) or SELL (fading)
  6. Target met          -> HOLD (stop unchanged)
  7. Fresh BUY again     -> ADD (OPTIONAL), within the 1% risk budget
  8. Otherwise           -> No action, with a status line
The stop is never moved: the backtest showed a fixed stop beat every trailing style.
Tabs with "paper" or "test" in the name never raise the SELL alert in the title.

The Cash tab is a register of deposits, withdrawals and interest per account;
the system fills in QQQ's close that day and what the same money would be worth
in QQQ today, so the Compare tab can measure each account against the index.

Secrets (GitHub repo settings): GOOGLE_SA_JSON (service-account key JSON) and
POSITIONS_SHEET_ID (the long ID in the sheet's URL). If either is missing the
check is skipped.
"""

import json
import os
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

SKIP_TABS = {"how to use", "compare", "cash"}
CASH_TAB = "Cash"
HOLD_DAYS = 60
STRONG_OK_FAILS = {"Regime", "NotExtended", "NoEarnings", "RoomToTarget"}   # entry-only gates; irrelevant once held
STAMP_TZ = "America/Indiana/Indianapolis"


# ------------------------------------------------------------------ sheet I/O
def open_book():
    key, sid = os.environ.get("GOOGLE_SA_JSON", "").strip(), os.environ.get("POSITIONS_SHEET_ID", "").strip()
    if not key or not sid:
        return None
    import gspread
    gc = gspread.service_account_from_dict(json.loads(key))
    return gc.open_by_key(sid)


def is_paper(title):
    t = title.lower()
    return "paper" in t or "test" in t


def _num(x):
    s = re.sub(r"[,$\s]", "", str(x or ""))
    try:
        return float(s)
    except ValueError:
        return None


def _date(x):
    x = str(x or "").strip()
    if not x:
        return None
    d = pd.to_datetime(x, errors="coerce")
    return None if pd.isna(d) else d.normalize()


def read_positions(vals):
    """Rows with a ticker, keeping their sheet row number for write-back."""
    out = []
    for i, row in enumerate(vals[1:], start=2):
        row = (row + [""] * 18)[:18]
        t = row[0].strip().upper().replace(".", "-")
        if not t:
            continue
        out.append(dict(row=i, ticker=t, account=row[1].strip(), entry_date=_date(row[2]),
                        entry=_num(row[3]), shares=_num(row[4]), stop=_num(row[5]), notes=row[6],
                        exit_date=_date(row[16]), exit=_num(row[17])))
    return out


def _blank(v):
    return "" if v is None else v


def write_results(ws, results, n_rows):
    """Fill H:M and S for every sheet row from 2 to the last used row (blank where no position)."""
    if n_rows < 2:
        return
    by_row = {r["row"]: r for r in results}
    stamp = datetime.now(ZoneInfo(STAMP_TZ)).strftime("%Y-%m-%d %-I:%M %p %Z")
    hm, s = [], []
    for i in range(2, n_rows + 1):
        r = by_row.get(i)
        if r is None:
            hm.append([""] * 6)
            s.append([""])
            continue
        hm.append([_blank(r["close"]), _blank(r["days"]), r["health"], r["action"], r["why"], "'" + stamp])
        s.append([_blank(r.get("qqq"))])
    ws.batch_update([{"range": f"H2:M{n_rows}", "values": hm}, {"range": f"S2:S{n_rows}", "values": s}],
                    value_input_option="USER_ENTERED")


# ------------------------------------------------------------------ QQQ comparisons
def _close_on(df, d):
    """Close on date d, or the last close before it."""
    if df is None or df.empty or d is None:
        return None
    s = df["Close"][df.index <= pd.Timestamp(d) + pd.Timedelta(hours=23)]
    return float(s.iloc[-1]) if len(s) else None


def qqq_change(bench, start, end=None):
    """QQQ % change from the close on `start` to the close on `end` (or the latest close)."""
    a = _close_on(bench, start)
    b = _close_on(bench, end) if end is not None else (float(bench["Close"].iloc[-1]) if bench is not None and not bench.empty else None)
    if not a or b is None:
        return None
    return round(b / a - 1, 4)


def update_cash(sh, bench):
    """Cash tab: F = QQQ close that day, G = the same money in QQQ today (deposits +, withdrawals -)."""
    try:
        ws = sh.worksheet(CASH_TAB)
    except Exception:
        return
    vals = ws.get_all_values()
    n = len(vals)
    if n < 2 or bench is None or bench.empty:
        return
    now = float(bench["Close"].iloc[-1])
    out = []
    for row in vals[1:]:
        row = (row + [""] * 4)[:4]
        d, kind, amt = _date(row[0]), row[2].strip().lower(), _num(row[3])
        then = _close_on(bench, d) if d is not None else None
        if not then or amt is None or kind not in ("deposit", "withdrawal", "interest"):
            out.append(["", ""])
            continue
        same = "" if kind == "interest" else round(amt * now / then * (-1 if kind == "withdrawal" else 1), 2)
        out.append([round(then, 2), same])
    ws.update(range_name=f"F2:G{n}", values=out, value_input_option="USER_ENTERED")


# ------------------------------------------------------------------ assessment
def health_of(chk):
    if chk is None or chk.get("Tier") == "n/a":
        return "Unknown"
    failed = {f for f in str(chk.get("Failed") or "").split(",") if f}
    rs_sig = str(chk.get("RS") or "-/-").split("/")[-1]
    if "Trend" in failed or rs_sig == "SELL":
        return "Broken"
    if failed <= STRONG_OK_FAILS and float(chk.get("Score") or 0) >= 60:
        return "Strong"
    return "Fading"


def _broken_reason(chk):
    failed = {f for f in str(chk.get("Failed") or "").split(",") if f}
    rs_sig = str(chk.get("RS") or "-/-").split("/")[-1]
    parts = []
    if rs_sig == "SELL":
        parts.append("its relative-strength chart vs QQQ flipped to SELL (it's lagging the index)")
    if "Trend" in failed:
        parts.append("its uptrend broke (no longer above a rising 200-day and its 50-day)")
    return " and ".join(parts) or "leadership has faded"


def assess(p, chk, df, today):
    """One position -> dict(row, close, days, health, action, why)."""
    res = dict(row=p["row"], ticker=p["ticker"], close=None, days=None, health="Unknown")
    if p["entry"] is None or p["stop"] is None or p["entry_date"] is None or pd.isna(p["entry_date"]):
        return dict(res, action="FIX ROW", why="Entry date, entry price and stop are all needed to check this position.")
    if df is None or df.empty:
        return dict(res, action="CHECK", why="No price data for this ticker today. Check the symbol.")

    close = float(df["Close"].iloc[-1])
    after = df[df.index > pd.Timestamp(p["entry_date"]).normalize()]
    days = int(len(after))
    gain = close / p["entry"] - 1
    risk = (p["entry"] - p["stop"]) / p["entry"]
    above = close / p["stop"] - 1 if p["stop"] > 0 else np.nan
    health = health_of(chk)
    res.update(close=round(close, 2), days=days, health=health)
    score = float(chk.get("Score") or 0) if chk is not None else 0
    flags = chk.get("NewsFlags") if chk is not None and isinstance(chk.get("NewsFlags"), str) else ""

    def out(action, why):
        return dict(res, action=action, why=why)

    # 1. stop hit (any low since entry at or below the stop)
    hit = after[after["Low"] <= p["stop"]]
    if not hit.empty:
        d = hit.index[0]
        return out("SELL", f"Stop ${p['stop']:,.2f} was hit on {d.date()} (low ${hit['Low'].iloc[0]:,.2f}). "
                           "If your stop order filled, enter the Exit date and Exit price; if not, sell at the next open.")
    if health == "Unknown":
        return out("CHECK", "The system couldn't evaluate this ticker today (not enough history or bad data). "
                            "Check it manually; your stop still applies.")
    # 2. health broken
    if health == "Broken":
        return out("SELL", f"Leadership has faded: {_broken_reason(chk)}. Gain {gain:+.1%}.")
    # 3. earnings within 7 days
    ne = chk.get("NextEarn") if chk is not None else None
    if ne:
        until = (pd.Timestamp(ne) - pd.Timestamp(today).normalize()).days
        if 0 <= until <= 7:
            if gain >= risk:
                return out("HOLD THROUGH EARNINGS",
                           f"Reports {ne}. Your {gain:+.1%} gain is at least the {risk:.1%} stop distance, "
                           "so a bad gap is cushioned. Stop stays where it is.")
            return out("SELL BEFORE EARNINGS",
                       f"Reports {ne}. Your {gain:+.1%} gain is less than the {risk:.1%} stop distance, "
                       "so a bad gap could cost more than you planned to risk.")
    # 4. red flag
    if chk is not None and chk.get("News") == "VETO?":
        if health == "Strong":
            return out("HOLD", f"Red flag ({flags}), but health is still strong (score {score:.0f}). "
                               "Keep the stop where it is and read the flagged headlines in the report.")
        return out("SELL", f"Red flag ({flags}) and fading health (score {score:.0f}).")
    # 5. 60-day review
    if days >= HOLD_DAYS:
        if health == "Strong":
            return out("EXTEND 60 DAYS", f"Day {days} review: still qualifies as a leader (score {score:.0f}). "
                                         f"Keep the stop at ${p['stop']:,.2f}. Gain {gain:+.1%}.")
        return out("SELL", f"Day {days} review: health is fading (score {score:.0f}, below the 60 needed to "
                           f"qualify). Gain {gain:+.1%}.")
    # 6. target met
    tgt = chk.get("Objective") if chk is not None else None
    if tgt and not pd.isna(tgt) and close >= float(tgt):
        return out("HOLD", f"Target ${float(tgt):,.2f} reached ({gain:+.1%}). Winners carry the results, so "
                           f"keep holding; the stop stays at ${p['stop']:,.2f}.")
    # 7. fresh BUY again
    if chk is not None and chk.get("Tier") == "BUY" and days >= 20:
        return out("ADD (OPTIONAL)", f"Fresh BUY again (score {score:.0f}). Add only if the extra shares keep "
                                     "total risk to your stop within 1% of the account.")
    # 8. all good
    return out("No action", f"Day {days} of {HOLD_DAYS} · {gain:+.1%} · {above:.0%} above the stop · "
                            f"health {health.lower()}.")


def closed_result(p, df, qqq):
    """A closed trade: no advice, just the result."""
    res = dict(row=p["row"], ticker=p["ticker"], close=None, days=None, health="", qqq=qqq, action="CLOSED")
    if p["entry"] is None or p["exit"] is None or p["entry_date"] is None:
        return dict(res, action="FIX ROW", why="A closed trade needs Entry date, Entry price and Exit price.")
    ret = p["exit"] / p["entry"] - 1
    if df is not None and not df.empty:
        res["days"] = int(((df.index > p["entry_date"]) & (df.index <= p["exit_date"] + pd.Timedelta(hours=23))).sum())
    vs = f", QQQ {qqq:+.1%} over the same days, so {ret - qqq:+.1%} vs QQQ" if qqq is not None else ""
    word = "beat" if qqq is not None and ret > qqq else "lagged" if qqq is not None else ""
    return dict(res, why=f"Closed {p['exit_date'].date()}: {ret:+.1%}{vs}." + (f" The trade {word} the index." if word else ""))


def summary_line(results, label="Positions"):
    """Short text for the notification."""
    if results is None:
        return ""
    live = [r for r in results if r["action"] != "CLOSED"]
    if not live:
        return f"{label}: no open positions."
    acts = [f"{r['action']} {r['ticker']}" for r in live if r["action"] != "No action"]
    ok = sum(r["action"] == "No action" for r in live)
    bits = acts + ([f"{ok} OK"] if ok else [])
    return f"{label}: " + " · ".join(bits)


def account_tabs(sh):
    """Every tab laid out like Positions (A1 = Ticker), except the helper tabs."""
    tabs = []
    for ws in sh.worksheets():
        if ws.title.strip().lower() in SKIP_TABS:
            continue
        vals = ws.get_all_values()
        if vals and vals[0] and vals[0][0].strip().lower() == "ticker":
            tabs.append((ws, vals))
    return tabs


def run_positions(S, cfg, data, universe):
    """Checks every account tab. Returns (results or None if not configured, real-account sells, summary text).
    Never raises."""
    try:
        sh = open_book()
    except Exception as e:
        print(f"Positions sheet unavailable: {type(e).__name__}: {e}")
        return [], [], f"Positions: couldn't open your sheet ({type(e).__name__}). Check the sharing and secrets."
    if sh is None:
        print("Positions check not configured (GOOGLE_SA_JSON / POSITIONS_SHEET_ID not set).")
        return None, [], ""
    try:
        tabs = account_tabs(sh)
    except Exception as e:
        print(f"Couldn't list the sheet's tabs: {e}")
        return [], [], f"Positions: couldn't read your sheet ({type(e).__name__})."

    bench = data.get(cfg.get("benchmark", "QQQ"))
    books = [(ws, vals, read_positions(vals)) for ws, vals in tabs]
    open_tickers = list(dict.fromkeys(p["ticker"] for _, _, pos in books for p in pos if p["exit"] is None))
    all_tickers = list(dict.fromkeys(p["ticker"] for _, _, pos in books for p in pos))
    missing = [t for t in all_tickers if t not in data]
    if missing:
        try:
            data.update(S.download_data(missing, cfg, cfg["history"]))
        except Exception as e:
            print(f"  position download failed: {e}")
    chk = {}
    if open_tickers:
        chk_df = S.check_stocks(open_tickers, cfg, universe=universe, data=data)
        chk = {} if chk_df is None else {r["Ticker"]: r.to_dict() for _, r in chk_df.iterrows()}
    today = pd.Timestamp.now().normalize()

    every, sells, lines = [], [], []
    for ws, vals, pos in books:
        results = []
        for p in pos:
            df = data.get(p["ticker"])
            try:
                qqq = qqq_change(bench, p["entry_date"], p["exit_date"]) if p["entry_date"] is not None else None
                if p["exit"] is not None or p["exit_date"] is not None:
                    if p["exit_date"] is None or p["exit"] is None:
                        r = dict(row=p["row"], ticker=p["ticker"], close=None, days=None, health="", qqq=None,
                                 action="FIX ROW", why="Fill in both Exit date and Exit price to close this trade.")
                    else:
                        r = closed_result(p, df, qqq)
                else:
                    r = dict(assess(p, chk.get(p["ticker"]), df, today), qqq=qqq)
            except Exception as e:
                r = dict(row=p["row"], ticker=p["ticker"], close=None, days=None, health="Unknown", qqq=None,
                         action="CHECK", why=f"Couldn't evaluate ({type(e).__name__}). Check manually.")
            results.append(r)
        try:
            write_results(ws, results, len(vals))
        except Exception as e:
            print(f"Couldn't write results back to {ws.title}: {e}")
        paper = is_paper(ws.title)
        if not paper:
            sells += [r["ticker"] for r in results if r["action"].startswith("SELL")]
        lines.append(summary_line(results, ws.title))
        print(f"[{ws.title}]{' (paper)' if paper else ''}")
        for r in results:
            print(f"  {r['ticker']:6s} {r['action']:22s} {r['why']}")
        every += [dict(r, tab=ws.title, paper=paper) for r in results]
    try:
        update_cash(sh, bench)
    except Exception as e:
        print(f"Couldn't update the Cash tab: {e}")
    if not books:
        lines = ["Positions: no account tabs found in your sheet."]
    return every, list(dict.fromkeys(sells)), "\n".join(lines)
