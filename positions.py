"""
Daily check of the positions you hold, from your private Google Sheet.

The sheet ("Positions" tab) has your inputs in columns A-G and the system's
results in H-M:

  A Ticker | B Account | C Entry date | D Entry price | E Shares | F Stop | G Notes
  H Last close | I Days held | J Health | K Action | L Why | M Updated

Every position gets exactly one recommended action, chosen in this order:
  1. Stop hit            -> SELL
  2. Health broken       -> SELL
  3. Earnings in 7 days  -> HOLD THROUGH (gain covers stop distance) or SELL BEFORE EARNINGS
  4. Red flag VETO?      -> HOLD (health strong) or SELL (health fading)
  5. 60-day review       -> EXTEND 60 DAYS (strong) or SELL (fading)
  6. Target met          -> HOLD (stop unchanged)
  7. Fresh BUY again     -> ADD (OPTIONAL), within the 1% risk budget
  8. Otherwise           -> No action, with a status line
The stop is never moved: the backtest showed a fixed stop beat every trailing style.

Secrets (GitHub repo settings): GOOGLE_SA_JSON (service-account key JSON) and
POSITIONS_SHEET_ID (the long ID in the sheet's URL). If either is missing the
check is skipped.
"""

import json
import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd

TAB = "Positions"
HOLD_DAYS = 60
STRONG_OK_FAILS = {"Regime", "NotExtended", "NoEarnings"}   # entry-timing gates; irrelevant once held


# ------------------------------------------------------------------ sheet I/O
def open_sheet():
    key, sid = os.environ.get("GOOGLE_SA_JSON", "").strip(), os.environ.get("POSITIONS_SHEET_ID", "").strip()
    if not key or not sid:
        return None
    import gspread
    gc = gspread.service_account_from_dict(json.loads(key))
    return gc.open_by_key(sid).worksheet(TAB)


def _num(x):
    s = re.sub(r"[,$\s]", "", str(x or ""))
    try:
        return float(s)
    except ValueError:
        return None


def read_positions(ws):
    """Rows with a ticker, keeping their sheet row number for write-back."""
    vals = ws.get_all_values()
    out = []
    for i, row in enumerate(vals[1:], start=2):
        row = (row + [""] * 7)[:7]
        t = row[0].strip().upper().replace(".", "-")
        if not t:
            continue
        dt = pd.to_datetime(row[2], errors="coerce") if row[2].strip() else None
        out.append(dict(row=i, ticker=t, account=row[1].strip(), entry_date=dt,
                        entry=_num(row[3]), shares=_num(row[4]), stop=_num(row[5]), notes=row[6]))
    return out


def write_results(ws, results, n_rows):
    """Fill H:M for every sheet row from 2 to the last used row (blank where no position)."""
    if n_rows < 2:
        return
    by_row = {r["row"]: r for r in results}
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    grid = []
    for i in range(2, n_rows + 1):
        r = by_row.get(i)
        grid.append(["", "", "", "", "", ""] if r is None else
                    [r["close"] if r["close"] is not None else "", r["days"] if r["days"] is not None else "",
                     r["health"], r["action"], r["why"], "'" + stamp])
    ws.update(range_name=f"H2:M{n_rows}", values=grid, value_input_option="USER_ENTERED")


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
                           "If your stop order filled, delete this row; if not, sell at the next open.")
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


def summary_line(results):
    """Short text for the notification."""
    if results is None:
        return ""
    if not results:
        return "Positions: none in your sheet."
    acts = [f"{r['action']} {r['ticker']}" for r in results if r["action"] != "No action"]
    ok = sum(r["action"] == "No action" for r in results)
    bits = acts + ([f"{ok} OK"] if ok else [])
    return "Positions: " + " · ".join(bits)


def run_positions(S, cfg, data, universe):
    """Returns (results or None if not configured, sells list). Never raises."""
    try:
        ws = open_sheet()
    except Exception as e:
        print(f"Positions sheet unavailable: {type(e).__name__}: {e}")
        return [], [], f"Positions: couldn't open your sheet ({type(e).__name__}). Check the sharing and secrets."
    if ws is None:
        print("Positions check not configured (GOOGLE_SA_JSON / POSITIONS_SHEET_ID not set).")
        return None, [], ""
    pos = read_positions(ws)
    n_rows = len(ws.get_all_values())
    if not pos:
        print("Positions: none in the sheet.")
        write_results(ws, [], n_rows)
        return [], [], "Positions: none in your sheet."

    tickers = list(dict.fromkeys(p["ticker"] for p in pos))
    missing = [t for t in tickers if t not in data]
    if missing:
        try:
            data.update(S.download_data(missing, cfg, cfg["history"]))
        except Exception as e:
            print(f"  position download failed: {e}")
    chk_df = S.check_stocks(tickers, cfg, universe=universe, data=data)
    chk = {} if chk_df is None else {r["Ticker"]: r.to_dict() for _, r in chk_df.iterrows()}
    today = pd.Timestamp.now().normalize()

    results = []
    for p in pos:
        try:
            results.append(assess(p, chk.get(p["ticker"]), data.get(p["ticker"]), today))
        except Exception as e:
            results.append(dict(row=p["row"], ticker=p["ticker"], close=None, days=None, health="Unknown",
                                action="CHECK", why=f"Couldn't evaluate ({type(e).__name__}). Check manually."))
    try:
        write_results(ws, results, n_rows)
    except Exception as e:
        print(f"Couldn't write results back to the sheet: {e}")
    sells = [r["ticker"] for r in results if r["action"].startswith("SELL")]
    for r in results:
        print(f"  {r['ticker']:6s} {r['action']:22s} {r['why']}")
    return results, sells, summary_line(results)
