"""
Watchlist commands from your phone.

Send a message to your ntfy COMMAND topic (or use the "Watchlist commands"
workflow form on GitHub) such as:

    add MANH            add to watchlist.txt and check it
    add MANH, COST      several at once
    remove COST         take it off the watchlist
    check NVDA FAST     run the full check without changing the watchlist
    MANH                a bare ticker means "check"
    list                reply with the current watchlist (stocks and funds)
    add fund QQQM       add a fund to funds_watchlist.txt (scored vs SPY in the Funds section)
    remove fund QQQM    take a fund off
    check fund QQQM     check a fund vs SPY without adding it

Replies arrive on your normal ntfy topic.

Two steps so the 15-minute polling stays cheap:
  python commands.py fetch   - standard library only; collects new messages and
                               tells the workflow whether there is anything to do
  python commands.py apply   - runs the screener on the collected commands
"""

import json
import os
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

WATCHLIST = Path("watchlist.txt")
FUND_WATCHLIST = Path("funds_watchlist.txt")
FUND_WORDS = {"fund", "funds", "etf", "etfs"}
MAX_FUNDS = 10
STATE = Path(".state/ntfy_since")
PENDING = Path("pending_commands.txt")
TICKER_RE = re.compile(r"^[A-Z][A-Z0-9.\-]{0,9}$")
MAX_WATCHLIST = 25   # keeps the daily run fast and Yahoo-friendly
MAX_CHECK = 10       # tickers checked per command run
HELP = ("Commands (send to your command topic):\n"
        "add MANH - add to watchlist + check\n"
        "remove MANH - take off watchlist (never affects the NASDAQ-100 scan)\n"
        "check NVDA FAST - check only (or just send NVDA)\n"
        "add fund QQQM / remove fund QQQM / check fund QQQM - funds, judged vs SPY\n"
        "list - show watchlist and your funds\n"
        f"Limits: {MAX_WATCHLIST} on the watchlist, {MAX_CHECK} checks per run. "
        "Replies come here (not in the command topic) in 15-30 min, 6am-10pm CT. "
        "Added stocks also show in the daily report under Your watchlist. Full guide: cheat sheet.")
VERBS = {"add": "add", "+": "add", "remove": "remove", "rm": "remove", "del": "remove",
         "delete": "remove", "-": "remove", "check": "check", "?": "check", "list": "list",
         "ls": "list", "help": "help"}


# ------------------------------------------------------------------ fetch (stdlib only)
def poll_ntfy(topic):
    since = STATE.read_text().strip() if STATE.exists() else "30m"
    url = f"https://ntfy.sh/{urllib.parse.quote(topic)}/json?poll=1&since={urllib.parse.quote(since)}"
    with urllib.request.urlopen(url, timeout=20) as r:
        lines = r.read().decode("utf-8").splitlines()
    msgs = [json.loads(l) for l in lines if l.strip()]
    msgs = [m for m in msgs if m.get("event") == "message"]
    if msgs:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(msgs[-1]["id"])
    return [m.get("message", "") for m in msgs]


def fetch():
    cmds = []
    manual = os.environ.get("CMD", "").strip()
    if manual:
        cmds.append(manual)
    topic = os.environ.get("NTFY_CMD_TOPIC", "").strip()
    if topic and not manual:
        try:
            cmds += poll_ntfy(topic)
        except Exception as e:
            print(f"ntfy poll failed: {e}")
    cmds = [c for c in cmds if c.strip()]
    PENDING.write_text("\n".join(cmds), encoding="utf-8")
    has = "true" if cmds else "false"
    print(f"{len(cmds)} command(s): {cmds}")
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write(f"has={has}\n")


# ------------------------------------------------------------------ parsing / watchlist
def parse(text):
    """'add MANH, cost' -> ('add', ['MANH','COST'], [], False); 'NVDA' -> ('check', ['NVDA'], [], False);
    'add fund QQQM' -> ('add', ['QQQM'], [], True)"""
    words = [w for w in re.split(r"[,\s]+", text.strip()) if w]
    if not words:
        return None, [], [], False
    verb = VERBS.get(words[0].lower())
    rest = words[1:] if verb else words
    verb = verb or "check"
    fund = bool(rest) and rest[0].lower() in FUND_WORDS
    if fund:
        rest = rest[1:]
    tickers, bad = [], []
    for w in rest:
        t = w.upper().replace("$", "")
        (tickers if TICKER_RE.match(t) else bad).append(t)
    return verb, list(dict.fromkeys(tickers)), bad, fund


def read_watchlist(path=WATCHLIST):
    if not path.exists():
        return [], []
    header, tickers = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("#") or not line.strip():
            if not tickers:
                header.append(line)
            continue
        tickers += [t.upper() for t in re.split(r"[,\s]+", line.split("#")[0]) if t]
    return header, list(dict.fromkeys(tickers))


def write_watchlist(header, tickers, path=WATCHLIST):
    path.write_text("\n".join(header + tickers) + "\n", encoding="utf-8")


FUND_HEADER = ["# Your funds - scored every day against SPY in the report's Funds section.",
               "# One ticker per line. Anything after # is ignored. Context only, not trade signals."]


# ------------------------------------------------------------------ apply
def describe_fund(r):
    from run_daily import fund_text
    if r.get("Tier") == "n/a":
        return f"{r['Ticker']} (fund): {r['Verdict']}"
    return (f"{r['Ticker']} (fund vs SPY): {r['Tier']} - score {r['Score']:.0f}, RS {r.get('RS')}, "
            f"momentum rank {r.get('MomPct', 0):.0f}, {r.get('OffHigh')}% off high. {fund_text(r['Verdict'])}")


def describe(r):
    if r.get("Tier") == "n/a":
        return f"{r['Ticker']}: {r['Verdict']}"
    flags = r.get("News") if isinstance(r.get("News"), str) else "n/a"
    return (f"{r['Ticker']}: {r['Tier']} - score {r['Score']:.0f}, red flags {flags}, "
            f"stop ${r['Stop']:,.2f} ({r['RiskPct']:.1f}%). {r['Verdict']}")


def apply():
    import wyckoff_pnf_screener as S
    from run_daily import notify

    report = os.environ.get("REPORT_URL")
    texts = [t for t in PENDING.read_text(encoding="utf-8").splitlines() if t.strip()] \
        if PENDING.exists() else []
    header, wl = read_watchlist()
    fheader, fl = read_watchlist(FUND_WATCHLIST)
    fheader = fheader or FUND_HEADER
    changed, fchanged, to_check, f_check, replies = False, False, [], [], []

    for text in texts:
        verb, tickers, bad, fund = parse(text)
        if bad:
            replies.append(f"Ignored (not a ticker): {', '.join(bad)}")
        if verb == "help" or (verb != "list" and not tickers):
            replies.append(HELP)
            continue
        if verb == "list":
            replies.append("Watchlist: " + (", ".join(wl) or "empty"))
            replies.append("Your funds: " + (", ".join(fl) or "none"))
        elif fund and verb == "add":
            new = [t for t in tickers if t not in fl]
            room = max(MAX_FUNDS - len(fl), 0)
            if len(new) > room:
                replies.append(f"Your fund list is capped at {MAX_FUNDS}. Not added: {', '.join(new[room:])}.")
                new = new[:room]
            fl += new
            fchanged |= bool(new)
            dup = [t for t in tickers if t not in new and t in fl]
            replies.append(" ".join(filter(None, [
                f"Added fund {', '.join(new)}." if new else "",
                f"Already on your fund list: {', '.join(dup)}." if dup else ""])))
            f_check += tickers
        elif fund and verb == "remove":
            gone = [t for t in tickers if t in fl]
            fl = [t for t in fl if t not in tickers]
            fchanged |= bool(gone)
            replies.append(f"Removed fund {', '.join(gone)}." if gone else f"Not on your fund list: {', '.join(tickers)}.")
        elif fund:
            f_check += tickers
        elif verb == "add":
            new = [t for t in tickers if t not in wl]
            dup = [t for t in tickers if t in wl]
            room = max(MAX_WATCHLIST - len(wl), 0)
            if len(new) > room:
                replies.append(f"Watchlist is capped at {MAX_WATCHLIST}. Not added: "
                               f"{', '.join(new[room:])}. Remove something first.")
                new = new[:room]
            wl += new
            changed |= bool(new)
            replies.append(" ".join(filter(None, [
                f"Added {', '.join(new)}." if new else "",
                f"Already on the list: {', '.join(dup)}." if dup else ""])))
            to_check += tickers
        elif verb == "remove":
            gone = [t for t in tickers if t in wl]
            wl = [t for t in wl if t not in tickers]
            changed |= bool(gone)
            replies.append(f"Removed {', '.join(gone)}." if gone else f"Not on the list: {', '.join(tickers)}.")
        elif verb == "check":
            to_check += tickers

    if changed:
        write_watchlist(header, wl)
        replies.append("Watchlist now: " + (", ".join(wl) or "empty"))
    if fchanged:
        write_watchlist(fheader, fl, FUND_WATCHLIST)
        replies.append("Your funds now: " + (", ".join(fl) or "none"))

    to_check = list(dict.fromkeys(to_check))
    if len(to_check) > MAX_CHECK:
        replies.append(f"Only {MAX_CHECK} checks per run - skipped: {', '.join(to_check[MAX_CHECK:])}.")
        to_check = to_check[:MAX_CHECK]
    if to_check:
        res = S.check_stocks(to_check)
        if res is None:
            replies.append("Check failed: price data unavailable. Try again later.")
        else:
            regime = "RISK-ON" if res.attrs.get("risk_on") else "CAUTION"
            replies.append(f"Market regime {regime} (data as of {res.attrs.get('asof')}).")
            replies += [describe(r) for _, r in res.iterrows()]

    f_check = list(dict.fromkeys(f_check))[:MAX_CHECK]
    if f_check:
        from run_daily import ETFS, FUND_CFG
        res = S.check_stocks(f_check, FUND_CFG, universe=ETFS, news=False)
        if res is None:
            replies.append("Fund check failed: price data unavailable. Try again later.")
        else:
            replies += [describe_fund(r) for _, r in res.iterrows()]

    checked = to_check + f_check
    title = "Check: " + ", ".join(checked[:4]) if checked else "Watchlist update"
    body = "\n".join(replies) or "Nothing to do."
    print(title, "\n" + body)
    # Replies go to both topics, so whoever sent the command sees the result.
    # They never contain positions.
    notify(title, body, 3, tags="mag", click=report)
    if os.environ.get("NTFY_PUBLIC_TOPIC", "").strip() != os.environ.get("NTFY_TOPIC", "").strip():
        notify(title, body, 3, tags="mag", click=report, env="NTFY_PUBLIC_TOPIC")


if __name__ == "__main__":
    {"fetch": fetch, "apply": apply}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()
