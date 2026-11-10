# Daily Screener

**User manual:** https://rrobinson1996-sudo.github.io/stock-screener/manual.html (source: `manual_body.html`)

Runs every weekday after the US market close on GitHub Actions and publishes a
phone-friendly report to GitHub Pages, with an optional push notification via
[ntfy](https://ntfy.sh).

- `wyckoff_pnf_screener.py`: the screener (NASDAQ-100 vs QQQ, plus the red-flag check)
- `run_daily.py`: builds the report and sends the notification
- `.github/workflows/daily.yml`: the schedule (21:40 UTC, Monday to Friday)
- `docs/`: the published report (`index.html`), dated copies (`reports/`) and
  the BUY signal log (`data/signals.csv`)

## Your watchlist

Edit `watchlist.txt` (one ticker per line, any US stock) and commit. Every
daily report then includes a "Your watchlist" section with each ticker's
gates, score, red-flag check and a plain-English verdict.

## Commands from your phone

Send a message to your ntfy **command** topic, or use Actions → **Watchlist
commands** → Run workflow:

| Message | Does |
|---|---|
| `add MANH` | adds to `watchlist.txt` and replies with the full check |
| `remove COST` | removes it |
| `check NVDA FAST` (or just `NVDA`) | replies with the check, list unchanged |
| `list` | replies with the current watchlist |

ntfy messages are picked up every 15 minutes, 6 AM to 10 PM Central (GitHub
can start late, so allow up to ~30 minutes). The workflow form replies in about
2 minutes.

## Settings (repo → Settings)

| Where | Name | Value |
|---|---|---|
| Secrets and variables → Actions → **Secrets** | `NTFY_TOPIC` | your private ntfy topic name (reports and replies) |
| Secrets and variables → Actions → **Secrets** | `NTFY_CMD_TOPIC` | a second private topic you send commands to |
| Secrets and variables → Actions → **Variables** (optional) | `NOTIFY_ONLY_ON_BUY` | `true` to notify only when there is a BUY candidate |
| Pages | Source | Deploy from a branch → `main` → `/docs` |

## Run it now

Actions tab → **Daily screener** → **Run workflow**.

## Changing the strategy

Edit `CONFIG` at the top of `wyckoff_pnf_screener.py` and commit. The backtest
(`run_backtest()`) still runs in Colab, not here.

Research tool, not financial advice.
