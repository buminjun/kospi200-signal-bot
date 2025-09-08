import pandas as pd
import numpy as np

def sma(series, window):
    return series.rolling(window).mean()

def atr(df: pd.DataFrame, period=14):
    # df: columns = [open, high, low, close, volume]
    h_l = df['high'] - df['low']
    h_pc = (df['high'] - df['close'].shift()).abs()
    l_pc = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_indicators(df: pd.DataFrame, lookback=120):
    # 입력: KRX 일별 데이터 (index: 날짜, columns: open, high, low, close, volume)
    df = df.copy().tail(lookback+60)   # 여유 확보
    df['SMA20'] = sma(df['close'], 20)
    df['SMA60'] = sma(df['close'], 60)
    df['ATR14'] = atr(df, 14)
    df['HHV30'] = df['high'].rolling(30).max()
    return df

def entry_signal(row, buffer=0.0, require_ma_trend=True):
    cond_breakout = row['close'] >= row['HHV30'] * (1.0 + buffer)
    cond_ma = (row['SMA20'] >= row['SMA60']) if require_ma_trend else True
    return bool(cond_breakout and cond_ma)

def exit_signal(current_price, pos_entry, atr_entry, sma20_now, ma_exit=True, stop_mult=1.5):
    cond_stop = current_price <= (pos_entry - stop_mult * atr_entry)
    cond_ma = (current_price < sma20_now) if (ma_exit and pd.notna(sma20_now)) else False
    return bool(cond_stop or cond_ma)

def position_size(equity, risk, atr_val):
    # 1ATR 손실이 risk*equity를 넘지 않게
    if atr_val is None or atr_val <= 0:
        return 0
    risk_cap = equity * risk
    shares = int(risk_cap / atr_val)
    return max(shares, 0)
