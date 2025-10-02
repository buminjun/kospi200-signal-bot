# datafeed.py
import time
import math
import logging
from datetime import datetime
from typing import Optional, Tuple

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from pykrx import stock as krx

# ---------- 설정 ----------
_YF_SLEEP_SEC = 0.5         # 티커 간 딜레이(레이트 리밋 회피)
_MAX_RETRY    = 3           # 소스별 재시도 횟수
_BACKOFF_BASE = 1.2         # 1.2^n 백오프

# 모듈 전역 캐시(동일 런 내 중복요청 방지)
_CACHE = {}

# 야후 세션(UA 지정: 드물게 UA 없는 요청이 차단되는 이슈 회피)
_YF_SESSION = requests.Session()
_YF_SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
})
yf.utils.get_yf_session = lambda: _YF_SESSION  # yfinance 내부 세션 교체


def _norm(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """여러 소스 포맷을 공통 포맷으로 정규화: index=DatetimeIndex, cols=open/high/low/close/volume"""
    if df is None or df.empty:
        return None

    # KRX/야후 컬럼 케이스 구분
    cols = {c.lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            if n in cols: return cols[n]
        return None

    o = pick("open", "시가")
    h = pick("high", "고가")
    l = pick("low", "저가")
    c = pick("close", "종가")
    v = pick("volume", "거래량", "거래대금")  # 거래대금이 올 때도 있음

    need = [o, h, l, c]
    if any(x is None for x in need):
        return None

    out = pd.DataFrame({
        "open":  df[o].astype(float),
        "high":  df[h].astype(float),
        "low":   df[l].astype(float),
        "close": df[c].astype(float),
    }, index=pd.to_datetime(df.index))

    if v is not None:
        out["volume"] = pd.to_numeric(df[v], errors="coerce").fillna(0).astype(np.int64)
    else:
        out["volume"] = 0

    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out = out.dropna(subset=["open","high","low","close"])
    return out


def _retry_sleep(attempt: int):
    time.sleep((_BACKOFF_BASE ** attempt))


def _fetch_krx(code: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """pykrx 1순위. 코드: '005930' 같은 6자리"""
    for i in range(_MAX_RETRY):
        try:
            df = krx.get_market_ohlcv_by_date(start.replace("-", ""), end.replace("-", ""), code)
            if df is not None and not df.empty:
                return _norm(df)
        except Exception as e:
            logging.warning(f"[pykrx] {code} try={i+1}/{_MAX_RETRY} err={e}")
            _retry_sleep(i)
    return None


def _fetch_yf(code: str, start: str, end: str) -> Optional[pd.DataFrame]:
    """야후 2순위. KOSPI→.KS, 실패시 .KQ. 최근 YF가 비정상 응답을 줄 수 있어 재시도/슬립."""
    tickers = [f"{code}.KS", f"{code}.KQ"]
    for tkr in tickers:
        for i in range(_MAX_RETRY):
            try:
                # 야후는 start/end 문자열 또는 datetime 지원
                df = yf.download(tkr, start=start, end=end, progress=False, threads=False)
                if df is not None and not df.empty:
                    time.sleep(_YF_SLEEP_SEC)  # rate-limit 회피
                    return _norm(df)
            except Exception as e:
                logging.warning(f"[yfinance] {tkr} try={i+1}/{_MAX_RETRY} err={e}")
                _retry_sleep(i)
    return None


def get_ohlcv(code: str, start: str, end: str, min_rows: int = 120) -> Optional[pd.DataFrame]:
    """
    단일 티커 일봉 OHLCV 로딩.
    우선순위: pykrx → yfinance(.KS/.KQ). 둘 다 실패하면 None.
    min_rows 미만이면 품질 불량으로 간주해 None 반환.
    """
    key = (code, start, end)
    if key in _CACHE:
        return _CACHE[key]

    # 1) pykrx
    df = _fetch_krx(code, start, end)
    if (df is None or len(df) < min_rows):
        # 2) yfinance
        df = _fetch_yf(code, start, end)

    # 품질 체크
    if df is None or len(df) < min_rows:
        _CACHE[key] = None
        return None

    _CACHE[key] = df
    return df
