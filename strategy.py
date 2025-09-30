import pandas as pd

# -----------------------------
# 매수 신호 체크
# -----------------------------
def check_entry_signal(df, cfg, rs_value, avg_volume, today_volume):
    """
    매수 조건:
    - HHV30 돌파
    - SMA20 ≥ SMA60
    - RS(Window=120) ≥ cfg['filters']['rs_min']
    - 거래량 ≥ 20일 평균 * volume_factor
    """
    hhv_win = cfg["entry"]["hhv_window"]
    buffer   = cfg["entry"]["buffer"]

    last_close = df["close"].iloc[-1]
    hhv = df["high"].rolling(hhv_win).max().iloc[-2]  # 직전까지의 HHV
    sma20 = df["close"].rolling(20).mean().iloc[-1]
    sma60 = df["close"].rolling(60).mean().iloc[-1]

    cond_breakout = last_close > hhv * (1 + buffer)
    cond_ma = sma20 >= sma60
    cond_rs = rs_value >= cfg["filters"]["rs_min"]
    cond_vol = today_volume >= avg_volume * cfg["filters"]["volume_factor"]

    if cond_breakout and cond_ma and cond_rs and cond_vol:
        return True
    return False


# -----------------------------
# 매도 신호 체크
# -----------------------------
def check_exit_signal(pos_row, df, cfg):
    """
    매도 조건:
    - ATR 기반 손절
    - SMA20 하향 이탈
    - 타임스탑 (예: 5일 이상 보유 시 청산)
    """
    stop_atr_multiple = cfg["exit"]["stop_atr_multiple"]
    ma_exit = cfg["exit"]["ma_exit"]
    time_stop = cfg["exit"]["time_stop"]

    entry_price = pos_row["entry_price"]
    entry_date  = pd.to_datetime(pos_row["entry_date"])

    last_row = df.iloc[-1]
    close = last_row["close"]
    atr   = last_row["ATR14"]

    # 1) ATR 손절
    if atr > 0 and close < entry_price - stop_atr_multiple * atr:
        return True, "ATR 손절"

    # 2) SMA20 하향 이탈
    if ma_exit:
        sma20 = df["close"].rolling(20).mean().iloc[-1]
        if close < sma20:
            return True, "SMA20 하향 이탈"

    # 3) 타임스탑 (보유일수 >= time_stop)
    today = pd.to_datetime(df.index[-1])
    if (today - entry_date).days >= time_stop:
        return True, f"타임스탑 {time_stop}일"

    return False, ""

