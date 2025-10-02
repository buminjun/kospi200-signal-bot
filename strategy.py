# strategy.py
import pandas as pd
import numpy as np

# =========================
# 유틸 함수
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
def compute_indicators(df, lookback=250):
    """
    df: 일봉 데이터프레임 (columns: open, high, low, close, volume)
    """
    out = df.copy()

    # 이동평균선
    out["SMA10"]  = _sma(out["close"], 10)
    out["SMA20"]  = _sma(out["close"], 20)
    out["SMA60"]  = _sma(out["close"], 60)
    out["SMA150"] = _sma(out["close"], 150)
    out["SMA200"] = _sma(out["close"], 200)

    # ATR
    out["ATR14"] = _atr(out, 14)

    # 52주 저가 (252거래일 ≈ 1년)
    out["Low52W"] = out["low"].rolling(252, min_periods=252).min()

    # 고점/저점 추세 확인용
    out["HHV20"] = out["high"].rolling(20).max()
    out["LLV20"] = out["low"].rolling(20).min()

    # 주봉 변환 (거래량 규칙 체크용)
    df_w = df.resample("W").agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
    df_w["Up"]   = df_w["close"] > df_w["close"].shift()
    df_w["VolUp"] = (df_w["Up"]) & (df_w["volume"] > df_w["volume"].shift())
    df_w["VolDn"] = (~df_w["Up"]) & (df_w["volume"] > df_w["volume"].shift())
    out["W_VolUp"] = df_w["VolUp"].reindex(out.index).fillna(False)
    out["W_VolDn"] = df_w["VolDn"].reindex(out.index).fillna(False)

    return out

# =========================
# 매수 신호
# =========================
def entry_signal(row, df=None):
    """
    7개 규칙 전부 충족해야 매수
    """

    # 규칙 1: 주가가 150일 또는 200일 이동평균선 위
    rule1 = (row["close"] > row["SMA150"]) or (row["close"] > row["SMA200"])

    # 규칙 2: 150일 이평선 > 200일 이평선
    rule2 = row["SMA150"] > row["SMA200"]

    # 규칙 3: 200일 이평선이 확실한 상승세
    rule3 = row["SMA200"] > row["SMA200"].shift(20) if isinstance(row, pd.Series) else False

    # 규칙 4: 고점과 저점이 연이어 높아진다
    rule4 = (row["high"] >= row["HHV20"].shift(1)) and (row["low"] >= row["LLV20"].shift(1))

    # 규칙 5: 주봉 상승 시 거래량 증가
    rule5 = bool(row.get("W_VolUp", False))

    # 규칙 6: 상승 주봉 거래량 > 하락 주봉 거래량
    rule6 = (row.get("W_VolUp", False)) and not (row.get("W_VolDn", False))

    # 규칙 7: 52주 신저가보다 최소 25% 이상 상승
    if pd.notna(row["Low52W"]) and row["Low52W"] > 0:
        rule7 = (row["close"] >= row["Low52W"] * 1.25)
    else:
        rule7 = False

    return all([rule1, rule2, rule3, rule4, rule5, rule6, rule7])

# =========================
# 강력 매수 (8번 규칙)
# =========================
def strong_entry_signal(df):
    """
    횡보 후 첫 장대 양봉
    df: DataFrame (최근 구간 확인)
    """
    if len(df) < 5:
        return False
    recent = df.iloc[-1]
    prev = df.iloc[-5:-1]

    cond_range = (prev["high"].max() - prev["low"].min()) / prev["low"].min() < 0.05
    cond_candle = (recent["close"] > recent["open"] * 1.05) and (recent["close"] > prev["high"].max())

    return bool(cond_range and cond_candle)

# =========================
# 매도 신호
# =========================
def exit_signal(row, entry_price=None):
    """
    매도 조건:
    ① 10일선 하락이탈
    ② 진입가 대비 -5% 손실
    """
    cond1 = row["close"] < row["SMA10"]
    cond2 = False
    if entry_price:
        cond2 = row["close"] <= entry_price * 0.95
    return cond1 or cond2

# =========================
# 포지션 사이징
# =========================
def position_size(equity, risk, atr_val):
    try:
        if atr_val is None or atr_val <= 0:
            return 0
        risk_amt = float(equity) * float(risk)
        shares = int(max(risk_amt / atr_val, 0))
        return shares
    except Exception:
        return 0


