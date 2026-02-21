# ============================================================
# S&P 500 ADVANCED RANKING SYSTEM v5.2 – GitHub Edition
# Headless + Persistent Cache + Dynamic Valuation + Multi-Timeframe Momentum
# Runs automatically every day via GitHub Actions
# ============================================================

import subprocess, sys, os, pickle, time, shutil
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

# Install dependencies
subprocess.check_call([sys.executable, "-m", "pip", "install",
    "yfinance", "pandas", "numpy", "openpyxl==3.1.2", "requests",
    "beautifulsoup4", "matplotlib", "seaborn", "tqdm", "scipy", "-q"])

import yfinance as yf
import pandas as pd
import numpy as np
import requests
import warnings
from bs4 import BeautifulSoup
from tqdm import tqdm
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
import seaborn as sns
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.formatting.rule import ColorScaleRule

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.2f}".format)

# ====================== CONFIG ======================
CFG = {
    "weights": {
        "valuation": 0.19, "profitability": 0.16, "growth": 0.13,
        "earnings_quality": 0.08, "fcf_quality": 0.13,
        "financial_health": 0.09, "momentum": 0.09,
        "analyst": 0.11, "piotroski": 0.02,
    },
    "min_coverage": 0.45,
    "min_market_cap": 5_000_000_000,
    "min_avg_volume": 500_000,
    "cache_hours": 24,
    "sleep_tr": 0.35,
    "batch_size_tr": 10,
    "max_workers_yf": 20,
    "output_file": "artifacts/sp500_ranking_v5.2.xlsx",
}
assert abs(sum(CFG["weights"].values()) - 1.0) < 1e-6

CACHE_FILE = "sp500_cache_v5.pkl"
os.makedirs("artifacts", exist_ok=True)

# ====================== TIPRANKS ======================
TR_URL = "https://mobile.tipranks.com/api/stocks/stockAnalysisOverview"
TR_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.tipranks.com/",
    "Origin": "https://www.tipranks.com",
    "Connection": "keep-alive",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
}

_CONSENSUS = {"StrongBuy": 5, "Buy": 4, "Moderate Buy": 3.5, "Hold": 3, "Moderate Sell": 2, "Sell": 1, "StrongSell": 0}
_TREND = {"Increased": 1, "Unchanged": 0, "Decreased": -1, "BoughtShares": 1, "SoldShares": -1}
_SENTIMENT = {"VeryBullish": 5, "Bullish": 4, "Neutral": 3, "Bearish": 2, "VeryBearish": 1,
              "VeryPositive": 5, "Positive": 4, "Negative": 2, "VeryNegative": 1}
_SMA = {"Positive": 1, "Neutral": 0, "Negative": -1}

def _parse_tipranks(item: dict) -> dict:
    return {
        "tr_smart_score": item.get("smartScore"),
        "tr_price_target": item.get("convertedPriceTarget"),
        "tr_insider_3m_usd": item.get("insidersLast3MonthsSum"),
        "tr_hedge_fund_value": item.get("hedgeFundTrendValue"),
        "tr_investor_chg_7d": item.get("investorHoldingChangeLast7Days"),
        "tr_investor_chg_30d": item.get("investorHoldingChangeLast30Days"),
        "tr_momentum_12m": item.get("technicalsTwelveMonthsMomentum"),
        "tr_roe": item.get("fundamentalsReturnOnEquity"),
        "tr_asset_growth": item.get("fundamentalsAssetGrowth"),
        "tr_blogger_bullish": item.get("bloggerBullishSentiment"),
        "tr_blogger_sector_avg": item.get("bloggerSectorAvg"),
        "tr_news_bullish": item.get("newsSentimentsBullishPercent"),
        "tr_news_bearish": item.get("newsSentimentsBearishPercent"),
        "tr_consensus_num": _CONSENSUS.get(item.get("analystConsensus"), np.nan),
        "tr_hedge_trend_num": _TREND.get(item.get("hedgeFundTrend"), np.nan),
        "tr_insider_trend_num": _TREND.get(item.get("insiderTrend"), np.nan),
        "tr_news_sent_num": _SENTIMENT.get(item.get("newsSentiment"), np.nan),
        "tr_blogger_cons_num": _SENTIMENT.get(item.get("bloggerConsensus"), np.nan),
        "tr_investor_sent_num": _SENTIMENT.get(item.get("investorSentiment"), np.nan),
        "tr_sma_num": _SMA.get(item.get("sma"), np.nan),
        "tr_analyst_consensus": item.get("analystConsensus"),
        "tr_hedge_trend": item.get("hedgeFundTrend"),
        "tr_insider_trend": item.get("insiderTrend"),
        "tr_news_sentiment": item.get("newsSentiment"),
        "tr_sma": item.get("sma"),
    }

def fetch_tipranks(tickers: list) -> pd.DataFrame:
    results = {}
    chunks = [tickers[i:i + CFG["batch_size_tr"]] for i in range(0, len(tickers), CFG["batch_size_tr"])]
    for chunk in tqdm(chunks, desc="TipRanks batches"):
        try:
            resp = requests.get(TR_URL, params={"tickers": ",".join(chunk)}, headers=TR_HEADERS, timeout=15)
            if resp.status_code == 200:
                for item in resp.json():
                    t = item.get("ticker", "")
                    if t:
                        results[t] = _parse_tipranks(item)
        except Exception as e:
            print(f" ⚠️ TipRanks error: {e}")
        time.sleep(CFG["sleep_tr"])
    if not results:
        print(" ⚠️ TipRanks returned no data.")
        return pd.DataFrame(columns=["ticker"])
    tr_df = pd.DataFrame.from_dict(results, orient="index").reset_index()
    tr_df.rename(columns={"index": "ticker"}, inplace=True)
    print(f" ✅ TipRanks: {tr_df['tr_smart_score'].notna().sum()}/{len(tickers)} SmartScores")
    return tr_df

# ====================== S&P 500 TICKERS ======================
WIKI_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

def get_sp500_tickers() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        resp = requests.get(url, headers=WIKI_HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")
        table = soup.find("table", {"id": "constituents"}) or soup.find("table", {"class": "wikitable"})
        rows = []
        for tr in table.find_all("tr")[1:]:
            cols = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cols) >= 4:
                rows.append({"ticker": cols[0].replace(".", "-"), "name": cols[1], "sector": cols[2], "industry": cols[3]})
        if rows:
            print(f"✅ Loaded {len(rows)} tickers (BeautifulSoup)")
            return pd.DataFrame(rows)
    except Exception as e:
        print(f" ⚠️ Strategy 1 failed: {e}")
    try:
        tables = pd.read_html(url, attrs={"id": "constituents"}) or pd.read_html(url)
        raw = tables[0]
        raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]
        df = pd.DataFrame({
            "ticker": raw.iloc[:,0].str.replace(".", "-", regex=False),
            "name": raw.iloc[:,1],
            "sector": raw.iloc[:,2],
            "industry": raw.iloc[:,3],
        })
        print(f"✅ Loaded {len(df)} tickers (read_html)")
        return df
    except Exception as e:
        print(f" ⚠️ Strategy 2 failed: {e}")
    raise RuntimeError("❌ Failed to fetch S&P 500 tickers")

# ====================== YAHOO + MOMENTUM ======================
FUNDAMENTAL_FIELDS = [ ... ]  # אותו רשימה כמו בגרסה שלך

def fetch_yf_parallel(tickers): ...  # אותו קוד

def fetch_price_multi(tickers): ... 

def add_price_momentum(df, tickers): ... 

# ====================== COMPUTED METRICS (מלא) ======================
def _safe(val, default=np.nan):
    if val is None or pd.isna(val): return default
    try: return float(val)
    except: return default

# העתק כאן את כל compute_piotroski, compute_altman, compute_roic, compute_fcf_metrics,
# compute_earnings_quality, compute_pt_upside, compute_tr_pt_upside, sector_percentile,
# build_pillar_scores, compute_composite, build_sector_thresholds, compute_valuation_score,
# compute_coverage, add_sector_context – בדיוק כמו בגרסה v5.1 שלך (הם לא השתנו)

# ====================== CACHING (מתוקן) ======================
_SECTOR_THRESHOLDS = {}

def load_cache():
    global _SECTOR_THRESHOLDS
    if not os.path.exists(CACHE_FILE):
        return None
    try:
        with open(CACHE_FILE, "rb") as f:
            payload = pickle.load(f)
        if len(payload) == 3:
            data, saved_thresholds, ts = payload
            _SECTOR_THRESHOLDS = saved_thresholds
        else:
            data, ts = payload
            _SECTOR_THRESHOLDS = build_sector_thresholds(data)
        age = datetime.now() - ts
        if age < timedelta(hours=CFG["cache_hours"]):
            print(f"✅ Cache loaded ({int(age.total_seconds()//60)} min old)")
            return data
    except Exception as e:
        print(f" ⚠️ Cache read error: {e}")
    return None

def save_cache(df):
    try:
        with open(CACHE_FILE, "wb") as f:
            pickle.dump((df, _SECTOR_THRESHOLDS, datetime.now()), f)
        print("💾 Cache saved")
    except Exception as e:
        print(f" ⚠️ Cache save error: {e}")

# ====================== EXCEL + PLOTS (מלא) ======================
# העתק כאן את style_and_export, _format_sheet, plot_all, _plot_radar, _print_summary – בדיוק כמו בגרסה v5.1 שלך

# ====================== MAIN PIPELINE (מתוקן) ======================
def run_pipeline(use_cache: bool = True):
    global _SECTOR_THRESHOLDS
    print("=" * 70)
    print(f" S&P 500 ADVANCED RANKING v5.2 – {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    cached = load_cache() if use_cache else None
    if cached is not None:
        df = cached.copy()
        print("Using cache...")
    else:
        # כל ה-fetch, compute, filters, pillars, scores – בדיוק כמו בגרסה שלך
        universe = get_sp500_tickers()
        tickers = universe["ticker"].tolist()
        yf_data = fetch_yf_parallel(tickers)
        fund_df = pd.DataFrame.from_dict(yf_data, orient="index").reset_index()
        fund_df.rename(columns={"index": "ticker"}, inplace=True)
        df = universe.merge(fund_df, on="ticker", how="left")
        tr_df = fetch_tipranks(tickers)
        if not tr_df.empty:
            df = df.merge(tr_df, on="ticker", how="left")
        # ... (המשך כל ה-compute, liquidity, coverage, thresholds, pillars, composite, valuation, rank, sector context)
        # (הקוד הזה זהה לגמרי לגרסה v5.1 שלך – העתק אותו לכאן)
        _SECTOR_THRESHOLDS = build_sector_thresholds(df)
        save_cache(df)

    _print_summary(df)
    print("🎨 Generating charts...")
    plot_all(df)
    print("💾 Exporting Excel...")
    style_and_export(df, CFG["output_file"])

    # Copy charts to artifacts
    for f in ["top30_composite.png", "sector_scores.png", "valuation_vs_quality.png",
              "smartscore_dist.png", "momentum_decomp.png", "top5_radar.png"]:
        if os.path.exists(f):
            shutil.copy(f, f"artifacts/{f}")

    print("✅ Pipeline complete! Artifacts ready.")
    return df

# ====================== RUN ======================
if __name__ == "__main__":
    run_pipeline(use_cache=True)
