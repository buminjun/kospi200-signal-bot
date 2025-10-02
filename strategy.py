# strategy.py
import pandas as pd
import numpy as np

# =========================
# 기본 지표 유틸
# =========================
def _sma(s, w):
    return s.rolling(w).mean()

def _atr(df, period=14):
    h_l  = df["high"] - df["low"]
    h_pc = (df["high"] - df["close"].shift()).abs()
    l_pc = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# =========================
# 지표 계산
# =========================
def compute_indicators(df):
    """
    df: 일봉 데이터프레임 (open, high, low, close, volume)
    """
    out = df.copy()

    # 이동평균선
    out["SMA20"]  = _sma(out["close"], 20)
    out["SMA60"]  = _sma(out["close"], 60)
    out["SMA150"] = _sma(out["close"], 150)
    out["SMA200"] = _sma(out["close"], 200)

    # ATR
    out["ATR14"]  = _atr(out, 14)

    # 고점/저점 추세 확인용
    out["HigherHigh"] = out["high"] > out["high"].shift(1)
    out["HigherLow"]  = out["low"]  > out["low"].shift(1)

    # 52주 신저가
    out["Low52W"] = out["low"].rolling(252, min_periods=252).min()

    return out

# =========================
# 조건 체크
# =========================
def check_rules(df):
    """
    마지막 row 기준으로 7가지 조건 충족 여부 확인
    """
    last = df.iloc[-1]

    cond1 = last["close"] > last["SMA150"] and last["close"] > last["SMA200"]
    cond2 = last["SMA150"] > last["SMA200"]
    cond3 = last["SMA200"] > last["SMA200"].shift(20)  # 200일선 상승 추세
    cond4 = last["HigherHigh"] and last["HigherLow"]    # 고점/저점 상승
    cond5 = (df["close"].iloc[-1] > df["close"].iloc[-2]) and (df["volume"].iloc[-1] > df["volume"].iloc[-2])
    cond6 = (df["volume"].rolling(20).apply(lambda x: (x > x.mean()).sum()).iloc[-1]) > 10  # 상승봉 거래량 > 하락봉 거래량
    cond7 = last["close"] >= last["Low52W"] * 1.25     # 52주 저가 대비 +25% 이상

    rules = [cond1, cond2, cond3, cond4, cond5, cond6, cond7]
    return all(rules), rules

# =========================
# 추가: 횡보 후 첫 장대양봉 (강력매수)
# =========================
def check_strong_buy(df, lookback=20, body_ratio=1.5):
    """
    최근 lookback 기간 동안 횡보하다가 첫 장대양봉이 나오는지 체크
    body_ratio: 당일 캔들 몸통이 직전 평균 몸통 대비 몇 배 큰지
    """
    recent = df.iloc[-lookback:]
    avg_body = (recent["close"] - recent["open"]).abs().mean()

    today_body = df["close"].iloc[-1] - df["open"].iloc[-1]
    today_vol  = df["volume"].iloc[-1]

    cond_body = today_body > avg_body * body_ratio and today_body > 0
    cond_vol  = today_vol > recent["volume"].mean() * 1.5

    return cond_body and cond_vol

