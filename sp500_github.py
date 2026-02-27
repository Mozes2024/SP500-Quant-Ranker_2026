# ============================================================
#  S&P 500 ADVANCED RANKING SYSTEM  v5.3 – GitHub Edition
#  Thin launcher — all logic lives in ranker/ package
# ============================================================
#
# This file exists for backward compatibility:
#   - GitHub Actions workflow runs `python sp500_github.py`
#   - Tests can import: `from sp500_github import compute_roic, CFG`
#
# All actual code is in ranker/ modules.
# ============================================================

import subprocess, sys

subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "yfinance", "pandas", "numpy", "openpyxl==3.1.2",
    "requests", "beautifulsoup4", "matplotlib", "seaborn",
    "tqdm", "scipy", "-q",
])

import os, warnings
import pandas as pd
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.float_format", "{:.2f}".format)
os.makedirs("artifacts", exist_ok=True)

# ── Re-export everything from ranker/ for backward compatibility ──
from ranker.config import (
    CFG, CFG_WEIGHTS_NO_TR, PILLAR_MAP, CACHE_FILE,
    FUNDAMENTAL_FIELDS, CORE_METRIC_COLS,
    EXPORT_COLS, FRIENDLY_NAMES, PCT_COLS_DECIMAL, PCT_COLS_FRACTION, ALL_PCT_COLS,
    GLOBAL_THRESHOLDS, _SECTOR_THRESHOLDS, build_sector_thresholds,
    _TR_AVAILABLE,
)
from ranker.utils import _safe, _coverage
from ranker.data_tickers import get_sp500_tickers
from ranker.data_tipranks import fetch_tipranks, _parse_tipranks
from ranker.data_yahoo import (
    _get_one, _probe_earnings_revisions,
    fetch_yf_parallel, fetch_price_multi, fetch_spy_returns,
    add_price_momentum,
)
from ranker.metrics import (
    compute_piotroski, compute_altman, compute_roic,
    compute_fcf_metrics, compute_earnings_quality,
    compute_pt_upside, compute_tr_pt_upside,
    compute_earnings_revision_score,
)
from ranker.pillars import build_pillar_scores, sector_percentile
from ranker.composite import (
    compute_composite, compute_valuation_score,
    compute_coverage, add_sector_context,
)
from ranker.cache import load_cache, save_cache
from ranker.export_excel import style_and_export
from ranker.export_json import export_json
from ranker.charts import plot_all
from ranker.breakout import merge_breakout_signals
from ranker.summary import _print_summary
from ranker.pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline(use_cache=True)
