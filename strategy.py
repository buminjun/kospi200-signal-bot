# strategy.py
import pandas as pd
import numpy as np

# =========================
# 지표 유틸
# =========================
def _sma(s, w):
    return s.rolling(w).mean()

def _atr(df, period=14):
    # df: columns = open, high, low, close, volume (소문자)
    h_l  = df["high"] - df["low"]
    h_pc = (df["high"] - df["close"].shift()).abs()
    l_pc = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_rs(close_series, bench_close, window=60):
    """
    상대강도 RS = (종목 N일 수익률) / (벤치마크 N일 수익률)
    """
    base = close_series / close_series.shift(window)
    bench = bench_close.reindex(close_series.index)
    bench = bench / bench.shift(window)
    return base / bench

# =========================
# 메인 지표 계산
# =========================
def compute_indicators(df, lookback=120, kospi_close=None, rs_window=60, hhv_window=30):
    """
    df: 종목 일봉 DataFrame (columns: open, high, low, close, volume)
    kospi_close: KOSPI 종가(pd.Series, DatetimeIndex)
    """
    out = df.copy()
    # 기본 지표
    out["SMA20"] = _sma(out["close"], 20)
    out["SMA60"] = _sma(out["close"], 60)
    out["ATR14"] = _atr(out, 14)

    # 전일까지의 HHV (돌파 기준: 당일 종가 >= 전일까지의 최고가 * (1+buffer))
    out["HHV30"] = out["high"].rolling(hhv_window, min_periods=hhv_window).max().shift(1)

    # 상대강도 (옵션)
    if kospi_close is not None and isinstance(kospi_close, pd.Series):
        out["RS"] = compute_rs(out["close"], kospi_close, window=rs_window)
    else:
        out["RS"] = np.nan

    return out

# =========================
# 시그널
# =========================
def entry_signal(row, buffer=0.0, require_ma_trend=True, rs_min=None):
    """
    매수: HHV30 돌파(+버퍼) && (SMA20>=SMA60 if require_ma_trend) && (RS>=rs_min if 지정)
    """
    req_cols = ["HHV30", "SMA20", "SMA60"]
    if any(pd.isna(row.get(c)) for c in req_cols):
        return False

    ok_break = row["close"] >= row["HHV30"] * (1.0 + float(buffer))
    ok_ma    = (row["SMA20"] >= row["SMA60"]) if require_ma_trend else True
    ok_rs    = True
    if rs_min is not None:
        rs_val = row.get("RS")
        ok_rs = (pd.notna(rs_val) and (rs_val >= float(rs_min)))

    return bool(ok_break and ok_ma and ok_rs)

def exit_signal(price_now, entry_price, atr_entry, sma20_now, use_ma=True, stop_atr_multiple=1.5):
    """
    매도: ATR 손절선 or SMA20 하향이탈(옵션)
    """
    cond_stop = False
    if atr_entry is not None and atr_entry > 0 and entry_price is not None:
        cond_stop = price_now <= (entry_price - stop_atr_multiple * atr_entry)
    cond_ma = (price_now < sma20_now) if (use_ma and sma20_now is not None) else False
    return bool(cond_stop or cond_ma)

# =========================
# 포지션 사이징 (리스크 기반)
# =========================
def position_size(equity, risk, atr_val):
    """
    atr_val: ATR14 값(원화)
    risk: 계좌 대비 1회 거래 리스크 비율 (예: 0.01 = 1%)
    """
    try:
        if atr_val is None or atr_val <= 0:
            return 0
        risk_amt = float(equity) * float(risk)
        shares = int(max(risk_amt / atr_val, 0))
        return shares
    except Exception:
        return 0
