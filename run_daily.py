"""
Daily runner for the RS leadership screener.

Runs on GitHub Actions every weekday after the close:
  1. downloads data (with retries - Yahoo sometimes throttles cloud servers)
  2. runs the stock screener (NASDAQ-100 vs QQQ) with the red-flag check
  3. runs fund mode (ETFs vs SPY)
  4. writes a phone-friendly report to docs/index.html (served by GitHub Pages),
     keeps a dated copy in docs/reports/, and logs every BUY signal to
     docs/data/signals.csv
  5. sends a push notification through ntfy.sh

Environment variables (set in the GitHub repo settings):
  NTFY_TOPIC          secret - your private ntfy topic name (no notification if unset)
  REPORT_URL          link opened when you tap the notification (set by the workflow)
  NOTIFY_ONLY_ON_BUY  "true" = only notify when there is a BUY candidate
"""

import html
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

import wyckoff_pnf_screener as S

ETFS = [
    "QQQM", "IWM", "MDY", "RSP",
    "XLK", "SMH", "XLF", "XLE", "XLV", "XLI", "XLY",
    "XLP", "XLU", "XLB", "XLRE", "XLC",
    "EFA", "EEM",
    "MTUM", "PDP",
    "GLD", "SLV", "DBC", "IBIT",
    "TLT", "IEF",
]
FUND_CFG = dict(S.CONFIG, benchmark="SPY", use_earnings=False, require_risk_on=False,
                min_dollar_vol=10e6, news_check=False)

DOCS = Path("docs")
REPORTS = DOCS / "reports"
DATA = DOCS / "data"


# ------------------------------------------------------------------ data
def download_with_retry(tickers, cfg, period, min_coverage=0.8, tries=3, wait=90):
    best, cov = {}, 0.0
    for attempt in range(1, tries + 1):
        try:
            data = S.download_data(tickers, cfg, period)
        except Exception as e:
            print(f"  download error: {type(e).__name__}: {e}")
            data = {}
        cov = sum(t in data for t in tickers) / max(len(tickers), 1)
        if len(data) > len(best):
            best = data
        if cfg["benchmark"] in data and cov >= min_coverage:
            return data, cov
        if attempt < tries:
            print(f"  coverage {cov:.0%} (attempt {attempt}/{tries}) - retrying in {wait}s")
            time.sleep(wait)
    cov = sum(t in best for t in tickers) / max(len(tickers), 1)
    return best, cov


# ------------------------------------------------------------------ html helpers
def esc(x):
    return html.escape("" if x is None or (isinstance(x, float) and pd.isna(x)) else str(x))


def fmt(v, kind=None):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if kind == "pct":
        return f"{v:.1f}%" if isinstance(v, (int, float)) else str(v)
    if kind == "money":
        return f"${v:,.2f}"
    if kind == "int":
        return f"{v:.0f}"
    return str(v)


BADGE = {"BUY": "good", "CLEAN": "good", "RISK-ON": "good", "REVIEW": "warn", "NO": "muted",
         "WATCH": "warn", "VETO?": "bad", "CAUTION": "bad", "NO DATA": "muted"}


def badge(text):
    if not text or (isinstance(text, float) and pd.isna(text)):
        return ""
    return f'<span class="badge {BADGE.get(str(text), "muted")}">{esc(text)}</span>'


def table(df, cols):
    """cols = [(column, header, kind)] ; kind in None/pct/money/int/badge/ticker"""
    if df is None or df.empty:
        return '<p class="empty">None.</p>'
    head = "".join(f"<th>{esc(h)}</th>" for _, h, _ in cols)
    body = []
    for _, r in df.iterrows():
        cells = []
        for c, _, kind in cols:
            v = r.get(c)
            if kind == "badge":
                cells.append(f"<td>{badge(v)}</td>")
            elif kind == "need":
                cells.append(f'<td class="need">{esc(v)}</td>')
            elif kind == "ticker":
                cells.append(f'<td class="tk"><a href="chart.html?t={esc(v)}">{esc(v)}</a></td>')
            else:
                cells.append(f'<td class="{"num" if kind else ""}">{esc(fmt(v, kind))}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'


def red_flag_cards(res):
    if res is None or "News" not in res.columns:
        return '<p class="empty">No candidates to check today.</p>'
    checked = res[res["News"].notna()]
    if checked.empty:
        return '<p class="empty">No candidates to check today.</p>'
    out = []
    for _, r in checked.iterrows():
        eps = (f'{fmt(r.get("EPS_up30"), "int")} up / {fmt(r.get("EPS_down30"), "int")} down'
               if pd.notna(r.get("EPS_up30")) else "n/a")
        items = "".join(
            f'<li><span class="muted">{esc(h["date"])} · {esc(h["source"])}</span><br>'
            + (f'<a href="{esc(h["url"])}">{esc(h["title"])}</a>' if h["url"] else esc(h["title"]))
            + f' <span class="hits">[{esc(h["hits"])}]</span></li>'
            for h in (r.get("Headlines") or []))
        out.append(
            f'<div class="card"><div class="row"><a class="tk big" href="chart.html?t={esc(r["Ticker"])}">{esc(r["Ticker"])}</a>'
            f'{badge(r["News"])}</div>'
            f'<div class="muted small">EPS estimate revisions, this fiscal year (30d): {esc(eps)}</div>'
            f'<div class="small">{esc(r.get("NewsFlags") or "Nothing flagged.")}</div>'
            + (f"<ul>{items}</ul>" if items else "") + "</div>")
    return "".join(out)


def headline_list(r):
    items = "".join(
        f'<li><span class="muted">{esc(h["date"])} · {esc(h["source"])}</span><br>'
        + (f'<a href="{esc(h["url"])}">{esc(h["title"])}</a>' if h["url"] else esc(h["title"]))
        + f' <span class="hits">[{esc(h["hits"])}]</span></li>'
        for h in (r.get("Headlines") if isinstance(r.get("Headlines"), list) else []))
    return f"<ul>{items}</ul>" if items else ""


def watchlist_cards(mine):
    if mine is None:
        return ('<p class="empty">No watchlist yet. Add tickers to <b>watchlist.txt</b> in the repo '
                '(one per line) and they will be checked every day.</p>')
    out = []
    for _, r in mine.iterrows():
        if r.get("Tier") == "n/a":
            out.append(f'<div class="card"><div class="row"><a class="tk big" href="chart.html?t={esc(r["Ticker"])}">{esc(r["Ticker"])}</a>'
                       f'{badge("NO DATA")}</div><div class="small">{esc(r["Verdict"])}</div></div>')
            continue
        flags = r.get("NewsFlags") if isinstance(r.get("NewsFlags"), str) else ""
        out.append(
            f'<div class="card"><div class="row"><a class="tk big" href="chart.html?t={esc(r["Ticker"])}">{esc(r["Ticker"])}</a>'
            f'<span>{badge(r["Tier"])} {badge(r.get("News"))}</span></div>'
            f'<div class="small muted">Score {fmt(r.get("Score"), "int")} · close {fmt(r.get("Close"), "money")} · '
            f'stop {fmt(r.get("Stop"), "money")} ({fmt(r.get("RiskPct"), "pct")}) · '
            f'next earnings {esc(r.get("NextEarn") or "n/a")}</div>'
            f'<div class="small" style="margin-top:6px">{esc(r["Verdict"])}</div>'
            + (f'<div class="small muted" style="margin-top:4px">Red flags: {esc(flags)}</div>' if flags else "")
            + headline_list(r) + "</div>")
    return "".join(out)


CSS = """
:root{--bg:#f7f7f5;--card:#fff;--fg:#1d1d1b;--muted:#6b6b66;--line:#e4e3de;
--good:#1f7a4d;--goodbg:#e3f3ea;--warn:#8a5a00;--warnbg:#fbf0d9;--bad:#a8322b;--badbg:#f9e3e1;--mutedbg:#ecebe7}
@media (prefers-color-scheme:dark){:root{--bg:#141413;--card:#1e1e1c;--fg:#ecebe6;--muted:#9c9b95;--line:#2f2f2c;
--good:#6fcf9b;--goodbg:#173326;--warn:#f0c46b;--warnbg:#3a2f14;--bad:#f08a80;--badbg:#3d1d1a;--mutedbg:#2a2a27}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
main{max-width:860px;margin:0 auto;padding:16px}
h1{font-size:22px;margin:4px 0 2px}h2{font-size:17px;margin:26px 0 8px}
.muted{color:var(--muted)}.small{font-size:13px}
.status{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin-bottom:10px}
.status .card{margin:0}.label{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
.value{font-size:18px;font-weight:600;margin-top:4px}
.row{display:flex;justify-content:space-between;align-items:center;gap:8px}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;font-weight:600}
.good{color:var(--good);background:var(--goodbg)}.warn{color:var(--warn);background:var(--warnbg)}
.bad{color:var(--bad);background:var(--badbg)}.muted.badge{color:var(--muted);background:var(--mutedbg)}
.scroll{overflow-x:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{padding:7px 9px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap}
th{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.04em}
tr:last-child td{border-bottom:0}.wrap td{white-space:normal}.wrap td.tk{white-space:nowrap}td.num{text-align:right;font-variant-numeric:tabular-nums}
.tk{font-weight:700}.big{font-size:17px}.empty{color:var(--muted);font-style:italic}
ul{padding-left:18px;margin:8px 0 0}li{margin-bottom:6px;font-size:13px}.hits{color:var(--bad);font-size:12px}
a{color:inherit}.warnbox{background:var(--warnbg);color:var(--warn);border-radius:10px;padding:10px 12px;margin-top:12px}
footer{margin:30px 0 10px;font-size:12px;color:var(--muted)}
pre{background:var(--mutedbg);border-radius:8px;padding:10px;overflow-x:auto;font-size:12px;white-space:pre}
code{background:var(--mutedbg);border-radius:4px;padding:1px 4px;font-size:12px}
.toc{columns:2;column-gap:18px;margin:6px 0 0;padding-left:20px}.toc li{margin-bottom:4px}
h2[id]{scroll-margin-top:12px}
.funnel{display:flex;flex-direction:column;gap:6px;margin-top:6px}
.funnel .step{display:flex;gap:10px;align-items:flex-start;border:1px solid var(--line);border-radius:8px;padding:8px 10px}
.funnel .step:nth-child(1){margin:0 0}.funnel .step:nth-child(2){margin:0 3%}.funnel .step:nth-child(3){margin:0 6%}
.funnel .step:nth-child(4){margin:0 9%}.funnel .step:nth-child(5){margin:0 12%}.funnel .step:nth-child(6){margin:0 12%}
.funnel .step:nth-child(7){margin:0 12%}
.hero{background:var(--goodbg);border-radius:10px;padding:12px 14px;margin:4px 0 14px}.hero .label{color:var(--good);font-weight:700}.hero p{margin:6px 0 0}.hero .scroll{background:var(--card)}.growth td{vertical-align:top;line-height:1.35}.growth th,.growth td{padding:6px 5px;font-size:12.5px}.growth th{font-size:10px}.growth td.tk{white-space:nowrap}td.need{white-space:normal;min-width:210px;max-width:280px}.bad-text{color:var(--bad);font-weight:600}.warn-text{color:var(--warn);font-weight:600}.warn-card{border-color:var(--warn);background:var(--warnbg)}.bad-card{border-color:var(--bad);background:var(--badbg)}.edge{color:var(--good)}.allin{color:var(--warn)}.put{font-size:11px;color:var(--muted)}
.test{border-top:1px solid var(--line);padding:8px 0}.test .row{margin-bottom:2px}.test .badge{margin-left:4px;font-size:11px}
.funnel .n{flex:none;width:22px;height:22px;border-radius:50%;background:var(--goodbg);color:var(--good);
font-weight:700;font-size:12px;display:flex;align-items:center;justify-content:center}.links a{margin-left:12px;white-space:nowrap}
main>.row:first-child{flex-wrap:wrap;row-gap:2px}main>.row:first-child h1{margin-right:auto}
"""


def build_html(stocks, funds, notes, recent_links, cheat_href="cheatsheet.html", mine=None):
    a = stocks.attrs if stocks is not None else {}
    in_ndx = set(stocks["Ticker"]) if stocks is not None else set()
    n_mine = 0 if mine is None else int((~mine["Ticker"].isin(in_ndx)).sum())
    tracked = (a.get("universe") or 0) + n_mine
    cov = (a.get("evaluated", 0) / a["universe"]) if a.get("universe") else 0
    if cov >= 0.90:
        health_cls, health_tag = "muted", ""
    elif cov >= 0.75:
        health_cls, health_tag = "warn-text", " · partial data"
    else:
        health_cls, health_tag = "bad-text", " · DATA PROBLEM: results incomplete"
    card_cls = {"muted": "card", "warn-text": "card warn-card", "bad-text": "card bad-card"}[health_cls]
    fa = funds.attrs if funds is not None else {}
    asof = a.get("asof", "?")
    regime = "RISK-ON" if a.get("risk_on") else "CAUTION"
    fregime = "RISK-ON" if fa.get("risk_on") else "CAUTION"
    gen = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    buys = stocks[stocks["Tier"] == "BUY"].copy() if stocks is not None else None
    if buys is not None and "Upside" in buys:
        # a measured-move target already below the price has been met - say so
        buys["Upside"] = buys["Upside"].map(lambda u: "met" if pd.notna(u) and u <= 0 else u)
    watch = stocks[stocks["Tier"] == "WATCH"].head(10) if stocks is not None else None
    fund_top = funds.head(8) if funds is not None else None

    buy_cols = [("Ticker", "Ticker", "ticker"), ("News", "Red flags", "badge"), ("Score", "Score", "int"),
                ("Close", "Close", "money"), ("Stop", "Stop", "money"), ("RiskPct", "Risk", "pct"),
                ("Objective", "Target", "money"), ("Upside", "Upside", "pct"),
                ("NextEarn", "Next earnings", None), ("Wyckoff", "Wyckoff", None)]
    watch_cols = [("Ticker", "Ticker", "ticker"), ("Score", "Score", "int"), ("Needs", "What it needs to become a BUY", "need"),
                  ("News", "Red flags", "badge"), ("Close", "Close", "money"), ("RS", "RS P&F", None),
                  ("MomPct", "Mom rank", "int"), ("OffHigh", "Off high", "pct"), ("Wyckoff", "Wyckoff", None)]
    fund_cols = [("Ticker", "Fund", "ticker"), ("Tier", "Tier", "badge"), ("Score", "Score", "int"),
                 ("Close", "Close", "money"), ("RS", "RS vs SPY", None), ("MomPct", "Mom rank", "int"),
                 ("OffHigh", "Off high", "pct"), ("Failed", "Missing", None)]

    if buys is None or buys.empty:
        why = ("The stock regime is CAUTION (QQQ below trend), so no new buys are issued."
               if regime == "CAUTION" else "No stock passed every gate with a score of 60+.")
        buy_html = f'<p class="empty">None today. {esc(why)}</p>'
    else:
        buy_html = table(buys, buy_cols)

    notes_html = "".join(f'<div class="warnbox">{esc(n)}</div>' for n in notes)
    links = " · ".join(f'<a href="{esc(href)}">{esc(lbl)}</a>' for lbl, href in recent_links) or "—"

    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Screener {esc(asof)}</title><style>{CSS}</style></head><body><main>
<div class="row"><h1>Daily Screener</h1><span class="small links"><a href="{esc(cheat_href.replace("cheatsheet", "scores"))}">All scores</a><a href="{esc(cheat_href)}">Cheat sheet</a><a href="{esc(cheat_href.replace("cheatsheet", "manual"))}">Manual</a></span></div>
<div class="muted small">Market data as of close {esc(asof)} · generated {esc(gen)}</div>
{notes_html}
<div class="status">
  <div class="card"><div class="label">Stock regime (QQQ)</div><div class="value">{badge(regime)}</div></div>
  <div class="card"><div class="label">Fund regime (SPY)</div><div class="value">{badge(fregime)}</div></div>
  <div class="card"><div class="label">Buy candidates</div><div class="value">{0 if buys is None else len(buys)}</div></div>
  <div class="{card_cls}"><div class="label">Stocks tracked</div><div class="value"><a href="{esc(cheat_href.replace("cheatsheet", "scores"))}">{esc(tracked)}</a></div>
  <div class="small {health_cls}">{esc(a.get("evaluated", "?"))} of {esc(a.get("universe", "?"))} NASDAQ-100 evaluated{f" · {n_mine} watchlist" if n_mine else ""}{health_tag}</div></div>
</div>

<h2>Buy candidates</h2>
{buy_html}

<h2>Watch list</h2>
<p class="muted small">On deck, not buys. The "What it needs" column says what each one is still missing.</p>
{table(watch, watch_cols)}

<h2>Your watchlist</h2>
<p class="muted small">Tickers from watchlist.txt, run through the same gates, score and red-flag check.</p>
{watchlist_cards(mine)}

<h2>Red-flag check</h2>
<p class="muted small">Looks for reasons <b>not</b> to buy. VETO? = estimates being cut or deal-driven news.
REVIEW = read the headlines. Never adds a stock the screener didn't flag.</p>
{red_flag_cards(stocks)}

<h2>Funds (vs SPY)</h2>
<p class="muted small">Which part of the market is leading. Edge vs SPY is thin in backtests - use as context, not a trading signal.</p>
{table(fund_top, fund_cols)}

<h2>History</h2>
<p class="small">{links} · <a href="scores.html">All scores</a> · <a href="data/signals.csv">BUY signal log (CSV)</a></p>

<footer>Rules: buy only when the stock regime is RISK-ON, every gate passes and score ≥ 60.
Stop = close − {S.CONFIG["stop_atr_mult"]}×ATR(20). Hold ~60 trading days. Research tool, not financial advice.</footer>
</main></body></html>"""


CHEAT = """
<h2>1. Check the stock regime first</h2>
<div class="card"><p>{caution} <b>CAUTION</b> (QQQ below its trend): no new stock buys. In the 2021&ndash;26 backtest,
signals taken in CAUTION markets returned about 0% over 60 days and trailed QQQ by about 4%.</p>
<p>{on} <b>RISK-ON</b>: buy candidates are live.</p>
<p class="muted small">This only governs the stock screener. Monthly buys of your index funds continue either way.</p></div>

<h2>2. Which candidates to take</h2>
<div class="card"><ul>
<li><b>Score 60 or more and every gate passed</b> = BUY candidate. Below 60 did no better than a random stock.</li>
<li><b>Don't favor 75+ over 60&ndash;75.</b> In the backtest, scores of 60&ndash;75 beat QQQ by about 4% over 60 days and 75+ by about 2.6%. A high score isn't a stronger buy.</li>
<li><b>Red flags decide, not the score:</b><br>
{clean} take it &nbsp;·&nbsp; {review} read the headlines (skip ones about other companies) &nbsp;·&nbsp;
{veto} skip or wait &mdash; estimates are being cut or deal news is driving the price &nbsp;·&nbsp; {nodata} check the news yourself</li>
<li><b>WATCH = on deck, not a buy.</b> These stocks are close: they pass the gates with a lower score, or score 60+ but miss exactly one gate. The <b>"What it needs"</b> column says what's missing (for example "the market regime to turn RISK-ON" or "a pullback toward its 50-day"). Do nothing until it becomes a BUY.</li>
<li><b>Never add a stock the screener didn't flag</b>, however good the press is.</li>
</ul></div>

<h2>3. How to place the trade</h2>
<div class="card"><ul>
<li><b>Entry:</b> at the next market open.</li>
<li><b>Stop:</b> the Stop column (close &minus; {stop}&times;ATR). Set it when you enter and don't move it down.</li>
<li><b>Size:</b> small. Risk about 1% of the account per trade: shares = (account &times; 1%) &divide; (entry &minus; stop). Fractional shares are fine.</li>
<li><b>Hold:</b> about 60 trading days (~3 months), then reassess. Exit early only if the stop is hit.</li>
<li><b>Target / Upside:</b> a rough measured-move target, not a forecast. Use it to check reward vs risk: Upside &divide; Risk of 1.5&ndash;2 or better is worth taking. "met" = target already reached.</li>
<li><b>Earnings:</b> the screener won't flag a stock within 7 days of its report. Note the Next earnings date; it will usually fall inside your hold.</li>
</ul></div>

<h2>4. What to expect</h2>
<div class="card"><ul>
<li>New BUY signals beat QQQ about <b>51%</b> of the time, by about <b>+3 to +5 points on average</b> over 60 days (backtest, before costs).</li>
<li>The median trade only barely beats QQQ. <b>A few big winners carry the results</b>, so don't sell winners early.</li>
<li>Expect losing streaks and a worst trade around <b>&minus;30%</b>. That's normal, not a reason to change the settings.</li>
</ul></div>

<h2>5. Funds section</h2>
<div class="card"><p>Context only: it shows which part of the market is leading. Ranking funds beat SPY by only 1&ndash;2 points in the backtest, and stops didn't help funds, so don't trade funds off it.</p></div>

<h2>6. Habits</h2>
<div class="card"><ul>
<li>Every BUY signal is logged automatically in <a href="data/signals.csv">signals.csv</a>. Note what you actually did.</li>
<li>Don't tweak settings after a bad week. Review once, after 6&ndash;12 months, against QQQ.</li>
<li>Selling puts: only with cash to cover 100 shares at the strike; a strike near the Stop column fits the system.</li>
</ul></div>
<h2>7. Phone commands</h2>
<div class="card"><p class="small muted">Type these in the Message box of your <b>command</b> topic in ntfy
(not the report topic), or use GitHub &rarr; Actions &rarr; Watchlist commands &rarr; Run workflow.</p>
<div class="scroll" style="margin:8px 0"><table class="wrap"><thead><tr><th>Send</th><th>What it does</th></tr></thead><tbody>
<tr><td class="tk">add MANH</td><td>Adds to your watchlist and replies with the full check</td></tr>
<tr><td class="tk">add MANH, COST</td><td>Several at once (commas or spaces)</td></tr>
<tr><td class="tk">remove COST</td><td>Takes it off your watchlist</td></tr>
<tr><td class="tk">check NVDA FAST</td><td>Replies with the check; watchlist unchanged</td></tr>
<tr><td class="tk">NVDA</td><td>Same as check</td></tr>
<tr><td class="tk">list</td><td>Replies with your current watchlist</td></tr>
<tr><td class="tk">help</td><td>Replies with this command list</td></tr>
</tbody></table></div>
<p class="small"><b>Timing:</b> ntfy commands are picked up every 15 minutes, 6 AM&ndash;10 PM Central; replies usually take 15&ndash;30 minutes.
The GitHub form replies in about 2 minutes. Messages sent overnight are picked up in the morning, but ntfy only keeps messages for about 12 hours.</p>
</div>

<h2>8. Command cautions</h2>
<div class="card"><ul>
<li><b>remove only edits your watchlist.</b> It never touches the NASDAQ-100 scan, which is downloaded fresh every run. A NASDAQ-100 stock you remove still appears in the main lists.</li>
<li><b>Limits:</b> up to 25 stocks on the watchlist and 10 checks per command run. Each check downloads the whole NASDAQ-100 (about a minute), and a long list slows the daily run and invites Yahoo to throttle it.</li>
<li><b>Don't send the same command over and over.</b> Messages are batched, so five quick messages become one run. Resending won't make the reply come faster.</li>
<li><b>Typos get added as-is.</b> "add APPL" puts APPL on the list and the report will say "no price data". Send <span class="tk">remove APPL</span> to fix it.</li>
<li><b>Funds don't belong on the watchlist.</b> ETFs get judged against QQQ there, which isn't meaningful; use the Funds section instead.</li>
<li><b>Same rules apply.</b> A watchlist stock is only a buy if the regime is RISK-ON, every gate passes and the score is 60+. Adding it doesn't lower the bar.</li>
<li><b>Commands never trade.</b> They only change the list and send checks. Buying and selling stay with you.</li>
<li><b>Keep your topic names private.</b> Anyone who knows the command topic could edit your watchlist (nothing else). Every change is saved in watchlist.txt's history on GitHub, so mistakes can be undone.</li>
</ul></div>

<footer>Numbers come from the 2021&ndash;26 backtest on current NASDAQ-100 members (flattered by survivorship, no costs). Research tool, not financial advice.</footer>
"""


def build_manual(src="manual_body.html"):
    body = Path(src).read_text(encoding="utf-8")
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Screener manual</title><style>{CSS}</style></head><body><main>
{body}</main></body></html>"""


def all_scores(stocks, mine):
    """Every stock tracked today with its score and a plain-English reason."""
    rows = []
    wl = set(mine["Ticker"]) if mine is not None else set()
    if stocks is not None:
        for _, r in stocks.iterrows():
            tier = r.get("Tier") or "NO"
            reason = r.get("Needs") if tier == "WATCH" else S.plain_verdict(dict(r, Tier=tier))
            rows.append(dict(Ticker=r["Ticker"], Score=r.get("Score"), Tier=tier,
                             Source="NASDAQ-100 + watchlist" if r["Ticker"] in wl else "NASDAQ-100",
                             Reason=reason, RedFlags=r.get("News") if isinstance(r.get("News"), str) else "",
                             Close=r.get("Close"), Failed=r.get("Failed") or "",
                             Stop=r.get("Stop"), Objective=r.get("Objective")))
    if mine is not None:
        have = {x["Ticker"] for x in rows}
        for _, r in mine.iterrows():
            if r["Ticker"] in have:
                continue
            rows.append(dict(Ticker=r["Ticker"], Score=r.get("Score"), Tier=r.get("Tier"), Source="Watchlist",
                             Reason=S.watch_needs(r) if r.get("Tier") == "WATCH" else r.get("Verdict"), RedFlags=r.get("News") if isinstance(r.get("News"), str) else "",
                             Close=r.get("Close"), Failed=r.get("Failed") or "",
                             Stop=r.get("Stop"), Objective=r.get("Objective")))
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    order = {"BUY": 0, "WATCH": 1, "NO": 2, "n/a": 3}
    df["_o"] = df["Tier"].map(order).fillna(2)
    return df.sort_values(["_o", "Score"], ascending=[True, False], na_position="last").drop(columns="_o")


def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and (pd.isna(v) or v != v):
        return None
    if hasattr(v, "item"):
        v = v.item()
    return round(v, 4) if isinstance(v, float) else v


def build_chart_data(scores, data, cfg, asof):
    """One small JSON with P&F columns for every tracked stock (drawn on the phone)."""
    syms = [t for t in scores["Ticker"] if t in data]
    _, prepared, _ = S.prepare(syms, data, cfg)
    W = cfg["pnf_window"]
    out = {}
    for _, r in scores.iterrows():
        t = r["Ticker"]
        if t not in prepared:
            continue
        df, rs = prepared[t]
        try:
            pxb, rsb = S.box_sizes(df, rs, len(df) - 1, len(rs) - 1, cfg)
            px = S.pnf_chart_data(df["Close"].iloc[-W:], pxb, cfg["px_reversal"])
            rsd = S.pnf_chart_data(rs.iloc[-W:], rsb, cfg["rs_reversal"])
        except Exception as e:
            print(f"  chart data skipped for {t}: {e}")
            continue
        out[t] = {k: _clean(v) for k, v in dict(
            tier=r["Tier"], score=r["Score"], reason=r["Reason"], flags=r["RedFlags"], src=r["Source"],
            close=r["Close"], stop=r.get("Stop"), target=r.get("Objective")).items()}
        out[t].update(px=px, rs=rsd)
    return dict(asof=asof, benchmark=cfg["benchmark"], stocks=out)


CHART_JS = r"""
const C=12,L=58,T=8,B=28;
const q=new URLSearchParams(location.search);let tk=(q.get('t')||'').toUpperCase();
function fmtP(v){return v>=1000?v.toFixed(0):v>=100?v.toFixed(1):v.toFixed(2);}
function svg(ch,marks){
 const cols=ch.cols,lb=Math.log(1+ch.box);let lo=1e9,hi=-1e9;
 cols.forEach(c=>{lo=Math.min(lo,c[1]);hi=Math.max(hi,c[2]);});
 marks=marks.filter(m=>m.price>0).map(m=>Object.assign({lvl:Math.floor(Math.log(m.price)/lb+1e-9)},m))
  .filter(m=>m.lvl>=lo-15&&m.lvl<=hi+15);
 marks.forEach(m=>{lo=Math.min(lo,m.lvl);hi=Math.max(hi,m.lvl);});lo-=1;hi+=1;
 const n=hi-lo+1,gw=cols.length*C,W=L+gw+50,H=T+n*C+B;
 let s=`<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg">`;
 for(let r=0;r<=n;r++){const y=T+r*C;s+=`<line x1="${L}" y1="${y}" x2="${L+gw}" y2="${y}" class="g"/>`;}
 for(let i=0;i<=cols.length;i++){const x=L+i*C;s+=`<line x1="${x}" y1="${T}" x2="${x}" y2="${T+n*C}" class="g"/>`;}
 const step=Math.max(1,Math.ceil(n/16));
 for(let l=hi;l>=lo;l--){if((hi-l)%step)continue;const y=T+(hi-l)*C+C/2+3;
  s+=`<text x="${L-5}" y="${y}" text-anchor="end" class="ax">${fmtP(Math.pow(1+ch.box,l))}</text>`;}
 let prev=null;
 cols.forEach((c,i)=>{const x=L+i*C;
  for(let l=c[1];l<=c[2];l++){const y=T+(hi-l)*C;
   if(c[0])s+=`<path d="M${x+2.5} ${y+2.5}L${x+C-2.5} ${y+C-2.5}M${x+C-2.5} ${y+2.5}L${x+2.5} ${y+C-2.5}" class="${c[4]?'xs':'x'}"/>`;
   else s+=`<circle cx="${x+C/2}" cy="${y+C/2}" r="${C/2-2.5}" class="${c[4]?'os':'o'}"/>`;}
  if(c[3]!==prev){const mo=+c[3].slice(5),code=mo<10?mo:'ABC'[mo-10];
   s+=`<text x="${x+C/2}" y="${T+n*C+11}" text-anchor="middle" class="mo">${code}</text>`;
   if(prev===null||c[3].slice(0,4)!==prev.slice(0,4))s+=`<text x="${x}" y="${T+n*C+23}" class="yr">${c[3].slice(0,4)}</text>`;prev=c[3];}
 });
 marks.forEach(m=>{const y=T+(hi-m.lvl)*C+C/2;
  s+=`<line x1="${L}" y1="${y}" x2="${L+gw+3}" y2="${y}" class="m ${m.cls}"/><text x="${L+gw+6}" y="${y+3}" class="ml ${m.cls}">${m.label}</text>`;});
 return s+'</svg>';
}
function colWord(ch){const c=ch.cols[ch.cols.length-1];return c[0]?'X column (rising)':'O column (falling)';}
function sigWord(ch){return ch.signal?`${ch.signal}${ch.bars_ago!=null?`, ${ch.bars_ago} trading day${ch.bars_ago==1?'':'s'} ago`:''}`:'none yet';}
function badge(t){const m={BUY:'good',CLEAN:'good',WATCH:'warn',REVIEW:'warn','VETO?':'bad'};return t?`<span class="badge ${m[t]||'muted'}">${t}</span>`:'';}
function show(d){
 const s=d.stocks[tk];const el=document.getElementById('c');
 if(!s){el.innerHTML=`<p class="empty">${tk?tk+' is not tracked today.':'Pick a stock.'} Add it with <b>add ${tk||'TICKER'}</b> in your command topic.</p>`;return;}
 document.title=tk+' P&F';
 let h=`<div class="card"><div class="row"><span class="tk big" style="font-size:22px">${tk}</span><span>${badge(s.tier)} ${badge(s.flags)} <b style="font-size:18px;margin-left:4px">${s.score!=null?Math.round(s.score):''}</b></span></div>
 <div class="small muted">${s.src} · close $${s.close} · as of ${d.asof}</div><div class="small" style="margin-top:6px">${s.reason||''}</div></div>`;
 if(s.px){const mk=[{price:s.close,label:'Now',cls:'now'}];if(s.stop)mk.push({price:s.stop,label:'Stop',cls:'stop'});if(s.target&&s.target>s.close)mk.push({price:s.target,label:'Target',cls:'tgt'});
  h+=`<h2>Price chart</h2><div class="small muted">${colWord(s.px)} · last signal: ${sigWord(s.px)} · box ${(s.px.box*100).toFixed(2)}% (1&times; its average daily range) · ${s.px.rev}-box reversal</div><div class="scroll ch">${svg(s.px,mk)}</div>`;}
 if(s.rs){h+=`<h2>Relative strength vs ${d.benchmark}</h2><div class="small muted">${colWord(s.rs)} · last signal: ${sigWord(s.rs)} · box ${(s.rs.box*100).toFixed(2)}% (1.5&times; the ratio&#39;s average daily move) · rising X's = beating ${d.benchmark}</div><div class="scroll ch">${svg(s.rs,[{price:s.rs.last,label:'Now',cls:'now'}])}</div>`;}
 el.innerHTML=h;document.querySelectorAll('.ch').forEach(x=>x.scrollLeft=x.scrollWidth);
}
fetch('data/pnf.json',{cache:'no-store'}).then(r=>r.json()).then(d=>{
 const sel=document.getElementById('pick');const names=Object.keys(d.stocks).sort();
 sel.innerHTML='<option value="">Choose a stock…</option>'+names.map(n=>`<option ${n===tk?'selected':''}>${n}</option>`).join('');
 sel.onchange=()=>{tk=sel.value;history.replaceState(null,'','?t='+tk);show(d);};show(d);
}).catch(()=>{document.getElementById('c').innerHTML='<p class="empty">Chart data not available yet. It appears after the next daily run.</p>';});
"""


def build_chart_page():
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>P&amp;F chart</title><style>{CSS}
.ch{{padding:6px 0;background:var(--card)}}.ch svg{{display:block}}
.g{{stroke:var(--line);stroke-width:.6}}.x,.xs{{stroke:var(--good);stroke-width:1.6;fill:none;stroke-linecap:round}}.xs{{stroke-width:2.6}}
.o,.os{{stroke:var(--bad);stroke-width:1.6;fill:none}}.os{{stroke-width:2.6}}
.ax,.mo,.yr,.ml{{fill:var(--muted);font-size:9px;font-family:-apple-system,Segoe UI,Roboto,sans-serif}}.yr{{font-weight:700}}
.m{{stroke-dasharray:3 3;stroke-width:1.2}}.m.now{{stroke:var(--fg)}}.ml.now{{fill:var(--fg)}}
.m.stop{{stroke:var(--bad)}}.ml.stop{{fill:var(--bad)}}.m.tgt{{stroke:var(--good)}}.ml.tgt{{fill:var(--good)}}
select{{width:100%;padding:9px 12px;border:1px solid var(--line);border-radius:10px;background:var(--card);color:var(--fg);font-size:15px;margin:10px 0}}
</style></head><body><main>
<div class="row"><h1>P&amp;F Chart</h1><span class="small links"><a href="index.html">Report</a><a href="scores.html">All scores</a></span></div>
<select id="pick"></select>
<div id="c"><p class="empty">Loading…</p></div>
<div class="card" style="margin-top:16px"><div class="label">How to read it</div><ul class="small">
<li><b>X</b> = price rising one box; <b>O</b> = falling one box. A new column starts only after a 3-box reversal, so small wiggles are ignored. Time isn't on the axis.</li>
<li><b>Bold column</b> = that column triggered a signal: an X column above the previous X high is a BUY (double-top breakout); an O column below the previous O low is a SELL.</li>
<li><b>Numbers along the bottom</b> = month each column started (1&ndash;9, A=Oct, B=Nov, C=Dec), with the year marked where it changes.</li>
<li><b>Lines:</b> Now = latest close · <span style="color:var(--bad)">Stop</span> = {S.CONFIG["stop_atr_mult"]:g}&times;ATR stop · <span style="color:var(--good)">Target</span> = measured-move objective.</li>
<li><b>Relative-strength chart:</b> the stock divided by QQQ. Rising X's mean it's beating the index, the core leadership test.</li>
<li><b>Why the two charts use different box sizes:</b> each box is sized to how much that line normally moves, so a signal means the same thing on a calm stock and a volatile one. The price box is 1&times; the stock's average daily range (ATR over 20 days) as a percent of price. The relative-strength line has no daily high or low, so its box is 1.5&times; its average daily move over 100 days. It usually moves less than the price, so its boxes are often smaller. Both are rounded to 0.25% and kept between 0.5% and 6%, and they're set fresh each day from the latest data.</li>
<li>The chart opens at the latest column; swipe left to see history.</li></ul></div>
</main><script>{CHART_JS}</script></body></html>"""


def build_scores_page(df, asof):
    items = []
    for _, r in df.iterrows():
        sc = fmt(r["Score"], "int")
        flags = badge(r["RedFlags"]) if r["RedFlags"] else ""
        items.append(
            f'<div class="srow" data-t="{esc(r["Ticker"]).lower()}" data-tier="{esc(r["Tier"])}" '
            f'data-src="{"w" if "watchlist" in str(r["Source"]).lower() else "n"}">'
            f'<div class="row"><span><a class="tk big" href="chart.html?t={esc(r["Ticker"])}">{esc(r["Ticker"])}</a> '
            f'<span class="muted small">{esc(r["Source"])}</span></span>'
            f'<span>{badge(r["Tier"])} {flags} <b class="sc">{esc(sc)}</b></span></div>'
            f'<div class="small">{esc(r["Reason"])}</div></div>')
    counts = df["Tier"].value_counts().to_dict()
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>All scores {esc(asof)}</title><style>{CSS}
.srow{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px;margin-bottom:8px}}
.sc{{font-size:16px;margin-left:6px}}.filters{{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0}}
.filters button{{border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:999px;padding:5px 12px;font-size:13px}}
.filters button.on{{background:var(--fg);color:var(--bg)}}
#q{{width:100%;padding:9px 12px;border:1px solid var(--line);border-radius:10px;background:var(--card);color:var(--fg);font-size:15px}}
</style></head><body><main>
<div class="row"><h1>All Scores</h1><span class="small links"><a href="index.html">Report</a><a href="data/scores_latest.csv">CSV</a></span></div>
<div class="muted small">Every stock tracked on {esc(asof)}: {len(df)} total · {counts.get("BUY", 0)} BUY · {counts.get("WATCH", 0)} WATCH. Sorted BUY, then WATCH, then the rest, highest score first.</div>
<input id="q" placeholder="Find a ticker, e.g. MANH" oninput="f()" style="margin-top:12px">
<div class="filters"><button class="on" data-f="all">All</button><button data-f="BUY">BUY</button>
<button data-f="WATCH">WATCH</button><button data-f="w">My watchlist</button><button data-f="NO">Not a buy</button></div>
<div id="list">{"".join(items)}</div>
<footer>Score cutoff for BUY is 60, with every gate passed and the regime RISK-ON. Research tool, not financial advice.</footer>
</main><script>
var mode='all';
function f(){{var q=document.getElementById('q').value.trim().toLowerCase();
document.querySelectorAll('.srow').forEach(function(e){{
var ok=(!q||e.dataset.t.indexOf(q)===0)&&(mode==='all'||(mode==='w'?e.dataset.src==='w':e.dataset.tier===mode));
e.style.display=ok?'':'none';}});}}
document.querySelectorAll('.filters button').forEach(function(b){{b.onclick=function(){{
document.querySelectorAll('.filters button').forEach(function(x){{x.classList.remove('on')}});
b.classList.add('on');mode=b.dataset.f;f();}};}});
</script></body></html>"""


def build_cheatsheet():
    body = CHEAT.format(caution=badge("CAUTION"), on=badge("RISK-ON"), clean=badge("CLEAN"),
                        review=badge("REVIEW"), veto=badge("VETO?"), nodata=badge("NO DATA"),
                        stop=S.CONFIG["stop_atr_mult"])
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Screener cheat sheet</title><style>{CSS}</style></head><body><main>
<div class="row"><h1>Cheat sheet</h1><span class="small links"><a href="index.html">Report</a><a href="manual.html">Manual</a></span></div>
<div class="muted small">What to do with the daily report.</div>
{body}</main></body></html>"""


# ------------------------------------------------------------------ notify
def notify(title, body, priority=3, tags="chart_with_upwards_trend", click=None, cheat=None):
    topic = os.environ.get("NTFY_TOPIC", "").strip()
    if not topic:
        print("NTFY_TOPIC not set - skipping notification.")
        return
    headers = {"Title": title.encode("ascii", "ignore").decode(), "Priority": str(priority), "Tags": tags}
    if click:
        headers["Click"] = click
        buttons = [f"view, Open report, {click}"]
        if cheat:
            buttons.append(f"view, Cheat sheet, {cheat}")
        headers["Actions"] = "; ".join(buttons)
    try:
        r = requests.post(f"https://ntfy.sh/{topic}", data=body.encode("utf-8"), headers=headers, timeout=20)
        print(f"Notification sent ({r.status_code}).")
    except Exception as e:
        print(f"Notification failed: {e}")


def summary(stocks, funds, mine=None):
    regime = "RISK-ON" if stocks.attrs.get("risk_on") else "CAUTION"
    buys = stocks[stocks["Tier"] == "BUY"]
    if mine is not None:
        extra = mine[(mine["Tier"] == "BUY") & ~mine["Ticker"].isin(buys["Ticker"])]
        buys = pd.concat([buys, extra])
    mine_line = ""
    if mine is not None:
        mine_line = "\nYour list: " + ", ".join(
            f"{r.Ticker} {r.Tier}" + (f" {r.Score:.0f}" if pd.notna(r.get('Score')) else "")
            for _, r in mine.iterrows())
    lead = ", ".join(funds[funds["Tier"] == "BUY"]["Ticker"].head(4)) if funds is not None else ""
    if not buys.empty:
        lines = [f"{r.Ticker}: score {r.Score:.0f}, red flags {r.get('News') or 'n/a'}, "
                 f"stop ${r.Stop:,.2f} ({r.RiskPct:.1f}%)" for _, r in buys.head(5).iterrows()]
        if len(buys) > 5:
            lines.append(f"...and {len(buys) - 5} more in the report")
        title = f"Screener: {len(buys)} BUY candidate{'s' if len(buys) > 1 else ''}"
        clean = [r.Ticker for _, r in buys.iterrows() if r.get("News") == "CLEAN"]
        do = ("Do: buy CLEAN names at the next open with the stop shown"
              + (f" ({', '.join(clean[:4])})" if clean else "")
              + ". Read REVIEW headlines first. Skip VETO?. Score 60-75 is as good as 75+.")
        body = do + "\n" + "\n".join(lines) + mine_line + (f"\nFunds leading: {lead}" if lead else "")
        return title, body, 4, True
    watch = stocks[stocks["Tier"] == "WATCH"].head(3)
    w = ", ".join(f"{r.Ticker} {r.Score:.0f}" for _, r in watch.iterrows()) or "none"
    title = f"Screener: no buys - regime {regime}"
    if regime == "CAUTION":
        do = "Do: no new stock buys while QQQ is below trend. Keep your monthly fund buys going."
    else:
        do = "Do: nothing new today. Keep your monthly fund buys going."
    ready = [r.Ticker for _, r in watch.iterrows() if r.get("News") == "CLEAN"]
    if ready:
        do += f" First in line if it turns on: {', '.join(ready)}."
    body = do + f"\nTop watch: {w}" + mine_line + (f"\nFunds leading: {lead}" if lead else "")
    return title, body, 2, False


# ------------------------------------------------------------------ main
def log_signals(stocks):
    buys = stocks[stocks["Tier"] == "BUY"]
    if buys.empty:
        return
    DATA.mkdir(parents=True, exist_ok=True)
    keep = ["Date", "Ticker", "Score", "Close", "Stop", "RiskPct", "Objective", "Upside",
            "NextEarn", "News", "NewsFlags", "Wyckoff"]
    new = buys[[c for c in keep if c in buys.columns]].copy()
    path = DATA / "signals.csv"
    if path.exists():
        old = pd.read_csv(path)
        new = pd.concat([old, new]).drop_duplicates(subset=["Date", "Ticker"], keep="last")
    new.to_csv(path, index=False)


def read_watchlist(path="watchlist.txt"):
    p = Path(path)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        out += [t for t in re.split(r"[,\s]+", line.split("#")[0]) if t]
    return list(dict.fromkeys(t.upper() for t in out))


def main():
    notes = []
    cfg = dict(S.CONFIG)

    tickers = S.load_nasdaq100()
    if len(tickers) < 90:
        notes.append(f"Index roster download failed - only {len(tickers)} fallback tickers were screened.")

    data, cov = download_with_retry(tickers, cfg, cfg["history"])
    if cfg["benchmark"] not in data or cov < 0.5:
        raise RuntimeError(f"Price download failed (coverage {cov:.0%}). Yahoo may be blocking - "
                           "re-run later from the Actions tab.")
    if cov < 0.75:
        notes.append(f"DATA PROBLEM: only {cov:.0%} of NASDAQ-100 stocks downloaded. Today's lists are "
                     "incomplete - re-run later from the Actions tab before acting on them.")
    elif cov < 0.9:
        notes.append(f"Partial data: {cov:.0%} of NASDAQ-100 stocks downloaded. Some candidates may be missing.")

    earnings = S.fetch_earnings(tickers)
    if sum(v is not None for v in earnings.values()) == 0:
        notes.append("Earnings dates unavailable - the earnings gate was not applied today.")

    stocks = S.run_screener(cfg, tickers=tickers, data=data, earnings=earnings)
    if stocks is None:
        raise RuntimeError("Stock screener returned no results.")

    mine = None
    wl = read_watchlist()
    if wl:
        extra = [t for t in wl if t not in data]
        if extra:
            try:
                data.update(S.download_data(extra, cfg, cfg["history"]))
            except Exception as e:
                print(f"  watchlist download failed: {e}")
        try:
            mine = S.check_stocks(wl, cfg, universe=tickers, data=data)
        except Exception as e:
            notes.append(f"Watchlist check failed: {type(e).__name__}: {e}")

    fdata, fcov = download_with_retry(ETFS, FUND_CFG, FUND_CFG["history"])
    funds = S.run_screener(FUND_CFG, tickers=ETFS, data=fdata, earnings={}) if FUND_CFG["benchmark"] in fdata else None
    if funds is None:
        notes.append("Fund screen unavailable today.")

    asof = stocks.attrs["asof"]
    REPORTS.mkdir(parents=True, exist_ok=True)
    log_signals(stocks)

    past = sorted(REPORTS.glob("*.html"), reverse=True)
    names = sorted({p.stem for p in past} | {asof}, reverse=True)[:15]
    page = build_html(stocks, funds, notes, [(n, f"reports/{n}.html") for n in names], mine=mine)
    (DOCS / "index.html").write_text(page, encoding="utf-8")
    scores = all_scores(stocks, mine)
    if not scores.empty:
        DATA.mkdir(parents=True, exist_ok=True)
        scores.assign(Date=asof).to_csv(DATA / "scores_latest.csv", index=False)
        (DOCS / "scores.html").write_text(build_scores_page(scores, asof), encoding="utf-8")
        try:
            cd = build_chart_data(scores, data, cfg, asof)
            (DATA / "pnf.json").write_text(json.dumps(cd, separators=(",", ":")), encoding="utf-8")
        except Exception as e:
            print(f"Chart data failed: {type(e).__name__}: {e}")
    (DOCS / "chart.html").write_text(build_chart_page(), encoding="utf-8")
    (DOCS / "cheatsheet.html").write_text(build_cheatsheet(), encoding="utf-8")
    if Path("manual_body.html").exists():
        (DOCS / "manual.html").write_text(build_manual(), encoding="utf-8")
    # dated copies link back up one level
    dated = (page.replace('href="reports/', 'href="').replace('href="data/', 'href="../data/')
             .replace('href="cheatsheet.html"', 'href="../cheatsheet.html"')
             .replace('href="manual.html"', 'href="../manual.html"')
             .replace('href="scores.html"', 'href="../scores.html"')
             .replace('href="chart.html', 'href="../chart.html'))
    (REPORTS / f"{asof}.html").write_text(dated, encoding="utf-8")
    print(f"Report written: docs/index.html and docs/reports/{asof}.html")

    title, body, prio, has_buy = summary(stocks, funds, mine)

    # Positions you hold (private Google Sheet). Results go to the sheet and the
    # notification only, never to the public report.
    try:
        import positions as P
        pres, sells, pline = P.run_positions(S, cfg, data, tickers)
    except Exception as e:
        print(f"Positions check failed: {type(e).__name__}: {e}")
        pres, sells, pline = None, [], f"Positions: check failed ({type(e).__name__})."
    if pline:
        body = pline + "\n" + body
    if sells:
        title = f"SELL {', '.join(sells[:3])} | " + title
        prio = max(prio, 5)
    pos_action = bool(pres) and any(r["action"] != "No action" for r in pres)

    if notes:
        body += "\nNote: " + " ".join(notes)
    if has_buy or pos_action or os.environ.get("NOTIFY_ONLY_ON_BUY", "").lower() != "true":
        url = os.environ.get("REPORT_URL")
        notify(title, body, prio, click=url, cheat=(url.rstrip("/") + "/cheatsheet.html") if url else None)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        traceback.print_exc()
        run_url = ""
        if os.environ.get("GITHUB_RUN_ID"):
            run_url = (f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/"
                       f"{os.environ.get('GITHUB_REPOSITORY')}/actions/runs/{os.environ['GITHUB_RUN_ID']}")
        notify("Screener FAILED", f"{type(e).__name__}: {e}", 4, tags="warning", click=run_url or None)
        sys.exit(1)
