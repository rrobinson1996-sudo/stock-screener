# Daily Screener

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

## Settings (repo → Settings)

| Where | Name | Value |
|---|---|---|
| Secrets and variables → Actions → **Secrets** | `NTFY_TOPIC` | your private ntfy topic name |
| Secrets and variables → Actions → **Variables** (optional) | `NOTIFY_ONLY_ON_BUY` | `true` to notify only when there is a BUY candidate |
| Pages | Source | Deploy from a branch → `main` → `/docs` |

## Run it now

Actions tab → **Daily screener** → **Run workflow**.

## Changing the strategy

Edit `CONFIG` at the top of `wyckoff_pnf_screener.py` and commit. The backtest
(`run_backtest()`) still runs in Colab, not here.

Research tool, not financial advice.
