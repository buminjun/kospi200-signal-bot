# scanner.py
import os
import sys
import pandas as pd
import numpy as np
import requests
import pytz
from datetime import datetime, timedelta
import yfinance as yf

# 보조 데이터소스
import FinanceDataReader as fdr
from pykrx import stock

from strategy import (
    compute_indicators,
    entry_signal_7rules,
    strong_buy_signal_8,
    exit_signal,
)

KST = pytz.timezone("Asia/Seoul")

# =========================
# 시간 유틸
# =========================
def now_kst():
    return datetime.now(tz=KST)

def inside_market_hours(cfg, ts=None):
    ts = ts or now_kst()
    wd = ts.weekday()  # 0=Mon
    if wd >= 5:
        return False
    start = cfg.get("market_hours", {}).get("start_kst", "09:00")
    end   = cfg.get("market_hours", {}).get("end_kst",   "15:30")
    s_h, s_m = map(int, start.split(":"))
    e_h, e_m = map(int, end.split(":"))
    t = ts.time()
    return (t >= datetime(ts.year,ts.month,ts.day,s_h,s_m,tzinfo=KST).time() and
            t <= datetime(ts.year,ts.month,ts.day,e_h,e_m,tzinfo=KST).time())

def should_send_summary(ts, every_min):
    return every_min > 0 and (ts.minute % every_min) == 0

# =========================
# 알림
# =========================
def send_telegram(msg, token, chat_id):
    if not token or not chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, data={"chat_id": chat_id, "text": msg})
    except Exception as e:
        print(f"[TG] error: {e}")

def send_ntfy(msg, url):
    if not url:
        return
    try:
        requests.post(url, data=msg.encode("utf-8"))
    except Exception as e:
        print(f"[NTFY] error: {e}")

def _notify(msg, use_tg, use_ntfy, token_env, chat_env, ntfy_env):
    print(msg)
    tg_token = os.getenv(token_env, "")
    tg_chat  = os.getenv(chat_env , "")
    ntfy_url = os.getenv(ntfy_env , "")
    if use_tg:
        send_telegram(msg, tg_token, tg_chat)
    if use_ntfy:
        send_ntfy(msg, ntfy_url)

# =========================
# CSV IO
# =========================
def load_positions(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=["code","name","entry_date","entry_price","shares"])
    return pd.read_csv(path, encoding="utf-8-sig")

def save_positions(df, path):
    df.to_csv(path, index=False, encoding="utf-8-sig")

def load_universe(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    if "code" not in df.columns:
        raise KeyError("universe.csv must have 'code' column")
    if "name" not in df.columns:
        df["name"] = df["code"]
    return df[["code","name"]]

# =========================
# 데이터 취득
# =========================
def fetch_yf_single(code, start_dt, end_dt):
    try:
        ticker = code if code.endswith((".KS",".KQ")) else f"{code}.KS"
        df = yf.download(
            ticker,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            progress=False,
            threads=False
        )
        if df is None or df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        df = df.rename(columns={
            "Open": "open","High": "high","Low": "low",
            "Close": "close","Adj Close": "close","Volume": "volume"
        })
        return df[["open","high","low","close","volume"]]
    except Exception as e:
        print(f"[yfinance] {code} fail: {e}")
        return None

def fetch_fdr(code, start_dt, end_dt):
    try:
        df = fdr.DataReader(code, start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"))
        if df is None or df.empty: return None
        df.index = pd.to_datetime(df.index)
        return df.rename(columns={"시가":"open","고가":"high","저가":"low","종가":"close","거래량":"volume"})[["open","high","low","close","volume"]]
    except: return None

def fetch_pykrx(code, start_dt, end_dt):
    try:
        s = start_dt.strftime("%Y%m%d"); e = end_dt.strftime("%Y%m%d")
        df = stock.get_market_ohlcv_by_date(s,e,code)
        if df is None or df.empty: return None
        df.index = pd.to_datetime(df.index)
        return df.rename(columns={"시가":"open","고가":"high","저가":"low","종가":"close","거래량":"volume"})[["open","high","low","close","volume"]]
    except: return None

def fetch_daily_df(code, start_dt, end_dt):
    df = fetch_yf_single(code, start_dt, end_dt)
    src = "yfinance"
    if df is None or df.empty:
        df = fetch_fdr(code, start_dt, end_dt)
        src = "FDR"
    if df is None or df.empty:
        df = fetch_pykrx(code, start_dt, end_dt)
        src = "pykrx"
    if df is None or df.empty:
        print(f"[DATA] {code} empty")
        return None, None
    return df, src

# =========================
# 포맷팅
# =========================
def fmt_price(x): return f"{x:,.0f}"

def format_buy_msg(ts, code, name, price, kind="매수"):
    return f"✅ {kind} 신호\n{name}({code}) @ {fmt_price(price)}\n{ts.strftime('%Y-%m-%d %H:%M:%S')} KST"

def format_sell_msg(ts, code, name, price, reason):
    return f"🔻 매도 ({reason})\n{name}({code}) @ {fmt_price(price)}\n{ts.strftime('%Y-%m-%d %H:%M:%S')} KST"

# =========================
# 스캔
# =========================
def scan_once(cfg):
    ts = now_kst()
    start_dt = ts - timedelta(days=420)
    end_dt = ts

    uni = load_universe(cfg["universe_csv"])
    pos = load_positions(cfg["positions_csv"])

    use_tg = cfg["telegram"]["enabled"]
    use_ntfy = cfg["ntfy"]["enabled"]
    market_open = inside_market_hours(cfg, ts)

    buy_cands, strong_cands, sell_cands, failed = [], [], [], []

    for code, name in uni[["code","name"]].itertuples(index=False):
        df, src = fetch_daily_df(code, start_dt, end_dt)
        if df is None: 
            failed.append(code); continue
        ind = compute_indicators(df)
        row = ind.iloc[-1]

        if entry_signal_7rules(row, ind):
            buy_cands.append((code,name,float(row["close"])))
        elif strong_buy_signal_8(row, ind):
            strong_cands.append((code,name,float(row["close"])))

    # 매수 실행
    if market_open and (buy_cands or strong_cands):
        queue = strong_cands + buy_cands
        for code,name,price in queue:
            msg = format_buy_msg(ts,code,name,price,kind="강력매수" if (code,name,price) in strong_cands else "매수")
            _notify(msg,use_tg,use_ntfy,cfg["telegram"]["token_env"],cfg["telegram"]["chat_id_env"],cfg["ntfy"]["url_env"])

    # 매도 체크
    for r in pos.to_dict("records"):
        code,name,entry_price = str(r["code"]).zfill(6),r["name"],float(r.get("entry_price",0) or 0)
        df,_ = fetch_daily_df(code,start_dt,end_dt)
        if df is None or len(df)<20: continue
        ind = compute_indicators(df)
        last = ind.iloc[-1]; price_now = float(last["close"]); sma10 = last.get("SMA10",np.nan)
        reason = None
        if price_now < sma10: reason="10SMA 이탈"
        if price_now <= entry_price*0.95: reason="진입가 -5%"
        if reason:
            msg = format_sell_msg(ts,code,name,price_now,reason)
            _notify(msg,use_tg,use_ntfy,cfg["telegram"]["token_env"],cfg["telegram"]["chat_id_env"],cfg["ntfy"]["url_env"])
            pos = pos[pos["code"]!=code]

    save_positions(pos, cfg["positions_csv"])

    # 요약
    every = int(cfg.get("notifications",{}).get("summary_every_min",60))
    if should_send_summary(ts,every):
        summary = f"📬 요약\n대상 {len(uni)}개, 실패 {len(failed)}개\n매수 {len(buy_cands)}개 / 강력매수 {len(strong_cands)}개 / 매도 {len(sell_cands)}개"
        _notify(summary,use_tg,use_ntfy,cfg["telegram"]["token_env"],cfg["telegram"]["chat_id_env"],cfg["ntfy"]["url_env"])

# =========================
# main
# =========================
import yaml
def load_config():
    with open("config.yaml","r",encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg=load_config()
    scan_once(cfg)

if __name__=="__main__":
    main()
