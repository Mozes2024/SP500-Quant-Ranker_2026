name: S&P 500 Quant Ranker

on:
  schedule:
    - cron: "0 6 * * 1-5"   # Every weekday at 06:00 UTC (after US market close)
  workflow_dispatch:          # Manual trigger from GitHub UI

jobs:
  rank:
    runs-on: ubuntu-latest
    timeout-minutes: 90       # Full run ~40-60 min

    steps:
      - name: Checkout repo
        uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Restore cache (7 days)
        id: restore-cache
        uses: actions/cache@v4
        with:
          path: sp500_cache_v5.pkl
          key: sp500-cache-v5-${{ runner.os }}-${{ steps.date.outputs.date }}
          restore-keys: |
            sp500-cache-v5-${{ runner.os }}-

      - name: Get current date (for cache key)
        id: date
        run: echo "date=$(date +'%Y-%m-%d')" >> $GITHUB_OUTPUT

      - name: Run ranking pipeline
        run: python sp500_github.py
        env:
          PYTHONUNBUFFERED: "1"

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: sp500-ranking-${{ github.run_number }}
          path: artifacts/
          retention-days: 30

      - name: Save updated cache
        uses: actions/cache/save@v4
        if: always()
        with:
          path: sp500_cache_v5.pkl
          key: sp500-cache-v5-${{ runner.os }}-${{ steps.date.outputs.date }}
