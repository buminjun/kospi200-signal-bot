# strategy.py
import pandas as pd
import numpy as np

# =========================
# 기본 유틸
# =========================
def _sma(s, w):
    return s.rolling(w).mean()

def _atr(df, period=14):
    h_l  = df["high"] - df["low"]
    h_pc = (df["high"] - df["close"].shift()).abs()
    l_pc = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def _weekly(df):
    # 주봉: 금요일 기준
    o = df["open"].resample("W-FRI").first()
    h = df["high"].resample("W-FRI").max()
    l = df["low" ].resample("W-FRI").min()
    c = df["close"].resample("W-FRI").last()
    v = df["volume"].resample("W-FRI").sum()
    w = pd.DataFrame({"open":o,"high":h,"low":l,"close":c,"volume":v}).dropna()
    return w

# =========================
# 상대강도(RS)
# =========================
def compute_rs(close_series, bench_close, window=120):
    """
    RS = (종목 N일 수익률) / (벤치마크 N일 수익률)
    """
    if bench_close is None:
        return pd.Series(index=close_series.index, dtype=float)
    base  = close_series / close_series.shift(window)
    bench = (bench_close.reindex(close_series.index) /
             bench_close.reindex(close_series.index).shift(window))
    with np.errstate(divide='ignore', invalid='ignore'):
        rs = base / bench
    return rs

# =========================
# 지표/특징 계산
# =========================
def compute_indicators(df,
                       lookback=300,
                       bench_close=None,
                       rs_window=120,
                       hhv_window=30):
    """
    df columns(lower): open, high, low, close, volume (DatetimeIndex, tz-naive)
    """
    out = df.copy()

    # MAs / ATR
    out["SMA10"]  = _sma(out["close"], 10)
    out["SMA20"]  = _sma(out["close"], 20)
    out["SMA60"]  = _sma(out["close"], 60)
    out["SMA150"] = _sma(out["close"], 150)
    out["SMA200"] = _sma(out["close"], 200)
    out["ATR14"]  = _atr(out, 14)

    # HHV/LLV
    out["HHV30"]  = out["high"].rolling(hhv_window, min_periods=hhv_window).max().shift(1)
    out["LLV252"] = out["low"].rolling(252, min_periods=200).min()  # ~52주 저가

    # RS
    out["RS"] = compute_rs(out["close"], bench_close, window=rs_window)

    # 주봉 파생(규칙 5/6용)
    w = _weekly(out)
    w["chg"] = w["close"].diff()
    w["vol_chg"] = w["volume"].diff()
    w["is_up"] = w["chg"] > 0
    w["is_dn"] = w["chg"] < 0
    w["vol_up_when_up"] = (w["is_up"] & (w["volume"] > w["volume"].shift()))
    w["vol_dn_when_dn"] = (w["is_dn"] & (w["volume"] < w["volume"].shift()))

    # 최근 12주 요약 비율/카운트 (티커 전체 동일기간 비교 위함)
    lastN = 12
    wN = w.tail(lastN)
    up_cnt  = int(wN["is_up"].sum())
    dn_cnt  = int(wN["is_dn"].sum())
    up_vol_ok = int(wN["vol_up_when_up"].sum())
    dn_vol_ok = int(wN["vol_dn_when_dn"].sum())
    out.loc[wN.index[-1] if len(wN) else out.index[-1], "W_UP_CNT"] = up_cnt
    out.loc[wN.index[-1] if len(wN) else out.index[-1], "W_DN_CNT"] = dn_cnt
    out.loc[wN.index[-1] if len(wN) else out.index[-1], "W_UP_VOL_OK"] = up_vol_ok
    out.loc[wN.index[-1] if len(wN) else out.index[-1], "W_DN_VOL_OK"] = dn_vol_ok

    # 횡보 탐지용 (규칙8)
    # 20일 표준편차/종가 < 3% 이면 '횡보'로 간주
    out["RANGE_TIGHT"] = (out["close"].rolling(20).std() / out["close"]).fillna(1.0) < 0.03
    # 장대양봉: (종가-시가) >= 1.2*ATR14 & 거래량이 20일 평균 대비 1.5배
    out["VOL_MA20"] = out["volume"].rolling(20).mean()
    body = out["close"] - out["open"]
    out["BIG_BULL"] = (body >= (1.2 * out["ATR14"])) & (out["volume"] >= 1.5 * out["VOL_MA20"])

    return out

# =========================
# 규칙 평가
# =========================
def _slope_up(series, lookback=20):
    if series.isna().iloc[-1] or series.isna().iloc[-lookback:].any():
        return False
    return bool(series.iloc[-1] > series.iloc[-lookback])

def _higher_high_low(df, win=5):
    """
    최근 2개 '연속' 구간의 고점/저점이 각각 이전 구간보다 상승인지 확인.
    """
    if len(df) < win * 3:
        return False
    seg2 = df.iloc[-3*win:-2*win]  # 과거
    seg1 = df.iloc[-2*win:-win]    # 직전
    seg0 = df.iloc[-win:]          # 현재

    def hi_lo(seg):
        return seg["high"].max(), seg["low"].min()

    hi2, lo2 = hi_lo(seg2)
    hi1, lo1 = hi_lo(seg1)
    hi0, lo0 = hi_lo(seg0)

    # 연속 상승: seg1 > seg2 그리고 seg0 > seg1
    return (hi1 > hi2 and lo1 > lo2) and (hi0 > hi1 and lo0 > lo1)


def entry_signal_7rules(row, df_full, strict_25=0.25):
    """
    7개 필수:
    1) 종가 > 150MA 또는 200MA (가능하면 둘 다 위)
    2) 150MA > 200MA
    3) 200MA 상승 기울기
    4) 고점/저점 상승(최근 구간)
    5) 주봉 상승시 거래량 증가
    6) 거래량 수반 상승주봉 > 거래량 수반 하락주봉
    7) 52주 저가대비 +25% 이상
    """
    c = row["close"]; s150 = row["SMA150"]; s200 = row["SMA200"]
    cond1 = False
    if not pd.isna(s150) and not pd.isna(s200):
        cond1 = (c > s150) and (c > s200)
    elif not pd.isna(s200):
        cond1 = (c > s200)
    elif not pd.isna(s150):
        cond1 = (c > s150)

    cond2 = (row["SMA150"] > row["SMA200"]) if (not pd.isna(row["SMA150"]) and not pd.isna(row["SMA200"])) else False
    cond3 = _slope_up(df_full["SMA200"], lookback=20)
    cond4 = _higher_high_low(df_full, win=5)

    # 최근 주봉 요약(12주)
    up_cnt     = int(df_full["W_UP_CNT"].dropna().iloc[-1]) if "W_UP_CNT" in df_full.columns and not df_full["W_UP_CNT"].dropna().empty else 0
    dn_cnt     = int(df_full["W_DN_CNT"].dropna().iloc[-1]) if "W_DN_CNT" in df_full.columns and not df_full["W_DN_CNT"].dropna().empty else 0
    up_vol_ok  = int(df_full["W_UP_VOL_OK"].dropna().iloc[-1]) if "W_UP_VOL_OK" in df_full.columns and not df_full["W_UP_VOL_OK"].dropna().empty else 0
    dn_vol_ok  = int(df_full["W_DN_VOL_OK"].dropna().iloc[-1]) if "W_DN_VOL_OK" in df_full.columns and not df_full["W_DN_VOL_OK"].dropna().empty else 0

    cond5 = (up_vol_ok >= int(0.6 * max(up_cnt, 1)))  # 상승주 중 60% 이상이 거래량 증가 동반
    cond6 = (up_vol_ok > dn_vol_ok)

    cond7 = False
    if not pd.isna(row.get("LLV252", np.nan)):
        cond7 = (row["close"] >= (row["LLV252"] * (1.0 + strict_25)))

    checks = [cond1,cond2,cond3,cond4,cond5,cond6,cond7]
    return all(checks)

def strong_buy_signal_8(row, df_full):
    """
    8) 횡보 후 첫 장대 양봉 (+거래량 급증)
    """
    tight = bool(row.get("RANGE_TIGHT", False))
    big   = bool(row.get("BIG_BULL", False))
    return tight and big

# =========================
# 매도 시그널 (실전용)
# =========================
def exit_signal(price_now, sma10_now, entry_price, drop_pct=0.05):
    """
    매도: ① 10일선 하향 이탈 or ② 진입가 대비 -5% 하락
    """
    cond_ma   = (price_now < sma10_now) if (sma10_now is not None and not pd.isna(sma10_now)) else False
    cond_drop = (entry_price is not None and price_now <= entry_price * (1.0 - float(drop_pct)))
    return bool(cond_ma or cond_drop)
