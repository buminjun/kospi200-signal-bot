import os
import sys
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pytz
import yaml
import requests
import yfinance as yf
import requests_cache

from strategy import (
    compute_indicators,
    entry_signal_7rules,
    strong_buy_signal_8,
    exit_signal,
)

KST = pytz.timezone("Asia/Seoul")

# =========================
# yfinance 세션 설정 (User-Agent 흉내 + 캐시)
# =========================
session = requests_cache.CachedSession('yfinance.cache')
session.headers['User-Agent'] = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/91.0.4472.124 Safari/537.36"
)

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

def should_send_summary(ts, every_min, jitter=False):
    if every_min <= 0:
        return False
    if jitter:
        return (ts.minute % every_min) in (0,1)
    return (ts.minute % every_min) == 0

# =========================
# 알림
# =========================
def send_telegram(msg, token, chat_id):
    if not token or not chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, data={"chat_id": chat_id, "text": msg})
        if resp.status_code != 200:
            print(f"[TG] HTTP {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"[TG] error: {e}")

def send_ntfy(msg, url):
    if not url:
        return
    try:
        resp = requests.post(url, data=msg.encode("utf-8"))
        if resp.status_code >= 300:
            print(f"[NTFY] HTTP {resp.status_code}: {resp.text}")
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
    try:
        if not os.path.exists(path):
            return pd.DataFrame(columns=["code","name","entry_date","entry_price","shares"])
        df = pd.read_csv(path, encoding="utf-8-sig")
        for c in ["code","name","entry_date","entry_price","shares"]:
            if c not in df.columns:
                df[c] = np.nan
        df["code"] = df["code"].astype(str).str.zfill(6)
        return df
    except Exception as e:
        print(f"[positions] load error: {e}")
        return pd.DataFrame(columns=["code","name","entry_date","entry_price","shares"])

def save_positions(df, path):
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"[positions] save error: {e}")

def load_universe(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "code" in df.columns:
        df["code"] = df["code"].astype(str).str.zfill(6)
    elif "종목코드" in df.columns:
        df["code"] = df["종목코드"].astype(str).str.zfill(6)
    else:
        raise KeyError("CSV에 'code' 또는 '종목코드' 컬럼이 없습니다")

    if "name" not in df.columns and "종목명" in df.columns:
        df["name"] = df["종목명"]

    return df[["code","name"]].dropna().drop_duplicates()

# =========================
# 데이터 가져오기 (yfinance 전용)
# =========================
def fetch_yf(code, start_dt, end_dt, market="KS"):
    ticker = f"{code}.{market}"
    try:
        df = yf.download(
            ticker,
            start=start_dt.strftime("%Y-%m-%d"),
            end=end_dt.strftime("%Y-%m-%d"),
            session=session,
            progress=False
        )
        if df is None or df.empty:
            print(f"[yfinance] {ticker} → empty")
            return None
        df = df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "close",
            "Volume": "volume"
        })
        df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except Exception as e:
        print(f"[yfinance] {ticker} fail: {e}")
        return None

def fetch_daily_df(code, start_dt, end_dt):
    today = datetime.now().strftime("%Y%m%d")
    csv_path = f"ohlcv_{today}.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, encoding="utf-8-sig")
        df = df[df["code"] == code]
        if not df.empty:
            df = df.rename(columns={
                "date": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume"
            })
            df = df.set_index("Date")
            df.index = pd.to_datetime(df.index)
            return df, "local_csv"
    # fallback → yfinance
    return fetch_yf(code, start_dt, end_dt)


def fetch_benchmark(start_dt, end_dt):
    try:
        df = yf.download("^KS11",
                         start=start_dt.strftime("%Y-%m-%d"),
                         end=end_dt.strftime("%Y-%m-%d"),
                         session=session,
                         progress=False)
        if df is None or df.empty:
            return None
        s = df["Close"]
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    except Exception as e:
        print(f"[yfinance] KS11 fail: {e}")
    return None

# =========================
# 포맷팅
# =========================
def fmt_price(x): 
    try: return f"{float(x):,.0f}"
    except: return str(x)

def format_buy_msg(ts, code, name, price, kind="매수"):
    return (f"✅ {kind} 신호\n"
            f"{name}({code}) @ {fmt_price(price)}\n"
            f"{ts.strftime('%Y-%m-%d %H:%M:%S')} KST")

def format_strong_msg(ts, code, name, price):
    return (f"🚀 강력매수(규칙8 단독)\n"
            f"{name}({code}) @ {fmt_price(price)}\n"
            f"{ts.strftime('%Y-%m-%d %H:%M:%S')} KST")

def format_sell_msg(ts, code, name, price, reason):
    return (f"🔻 매도 신호 ({reason})\n"
            f"{name}({code}) @ {fmt_price(price)}\n"
            f"{ts.strftime('%Y-%m-%d %H:%M:%S')} KST")

# =========================
# 스캔 메인
# =========================
def scan_once(cfg):
    ts = now_kst()
    start_dt = ts - timedelta(days=max(lookback_padding(cfg), 420))
    end_dt   = ts

    uni = load_universe(cfg["universe_csv"])
    pos = load_positions(cfg["positions_csv"])

    bench = fetch_benchmark(start_dt, end_dt)

    use_tg   = cfg["telegram"]["enabled"]
    use_ntfy = cfg["ntfy"]["enabled"]
    force_summary = os.getenv("FORCE_SUMMARY", "0") == "1"
    market_open = inside_market_hours(cfg, ts)

    buy_cands = []
    strong_cands = []
    sell_cands = []
    failed = []

    for code, name in uni[["code","name"]].itertuples(index=False):
        df, src = fetch_daily_df(code, start_dt, end_dt)
        if df is None or df.empty:
            failed.append(code)
            continue

        ind = compute_indicators(
            df,
            lookback=cfg.get("lookback", 120),
            bench_close=bench,
            rs_window=cfg.get("filters",{}).get("rs_window", 120),
            hhv_window=cfg.get("entry",{}).get("hhv_window", 30),
        )
        row = ind.iloc[-1]

        ok7 = entry_signal_7rules(row, ind, strict_25=0.25)
        ok8 = strong_buy_signal_8(row, ind)

        if ok7:
            buy_cands.append((code, name, float(row["close"])))
        elif ok8:
            strong_cands.append((code, name, float(row["close"])))

        time.sleep(1)  # 종목별 조회 사이 지연 (차단 방지)

    if market_open and (buy_cands or strong_cands):
        queue = strong_cands + buy_cands
        cur = len(pos)
        cap = max(cfg["max_positions"] - cur, 0)
        for code, name, price in queue[:cap]:
            kind = "강력매수" if (code, name, price) in strong_cands else "매수"
            _notify(format_buy_msg(ts, code, name, price, kind=kind),
                    use_tg, use_ntfy,
                    cfg["telegram"]["token_env"], cfg["telegram"]["chat_id_env"], cfg["ntfy"]["url_env"])
            new_row = {
                "code": code, "name": name,
                "entry_date": ts.strftime("%Y-%m-%d"),
                "entry_price": float(price),
                "shares": 0
            }
            pos = pd.concat([pos, pd.DataFrame([new_row])], ignore_index=True)

    time_stop_days = int(cfg.get("exit",{}).get("time_stop_days", 5))
    for r in pos.to_dict("records"):
        code = str(r["code"]).zfill(6); name = r["name"]
        entry_price = float(r.get("entry_price", 0) or 0)
        entry_date  = r.get("entry_date", None)
        if not entry_price or not entry_date:
            continue
        df, src = fetch_daily_df(code, start_dt, end_dt)
        if df is None or df.empty or len(df) < 20:
            continue
        ind = compute_indicators(df)
        last = ind.iloc[-1]
        price_now = float(last["close"])
        sma10_now = float(last["SMA10"]) if not pd.isna(last["SMA10"]) else None

        reason = None
        if exit_signal(price_now, sma10_now, entry_price, drop_pct=0.05):
            if sma10_now is not None and price_now < sma10_now:
                reason = "10SMA 이탈"
            if price_now <= entry_price * 0.95:
                reason = "진입가 -5%"

        if reason is None and time_stop_days > 0:
            try:
                d0 = datetime.strptime(entry_date, "%Y-%m-%d").date()
                if (ts.date() - d0).days >= time_stop_days:
                    reason = f"보유 {time_stop_days}일 경과"
            except:
                pass

        if reason:
            sell_cands.append((code, name, price_now, reason))

    if market_open and sell_cands:
        closed = []
        for code, name, price_now, reason in sell_cands:
            _notify(format_sell_msg(ts, code, name, price_now, reason),
                    use_tg, use_ntfy,
                    cfg["telegram"]["token_env"], cfg["telegram"]["chat_id_env"], cfg["ntfy"]["url_env"])
            closed.append(code)
        if closed:
            pos = pos[~pos["code"].isin(closed)]

    save_positions(pos, cfg["positions_csv"])

    every = int(cfg.get("notifications",{}).get("summary_every_min", 60))
    if (market_open and should_send_summary(ts, every)) or (force_summary and (ts.minute % every == 0)):
        summary = (f"📬 요약\n"
                   f"대상: {len(uni)}개 / 실패: {len(failed)}개\n"
                   f"매수: {len(buy_cands)}개 / 강력매수: {len(strong_cands)}개 / 매도: {len(sell_cands)}개\n"
                   f"{ts.strftime('%Y-%m-%d %H:%M:%S')} KST")
        _notify(summary, use_tg, use_ntfy,
                cfg["telegram"]["token_env"], cfg["telegram"]["chat_id_env"], cfg["ntfy"]["url_env"])

    if failed:
        print(f"[DATA] failed tickers ({len(failed)}): {failed[:10]}{' ...' if len(failed)>10 else ''}")

def lookback_padding(cfg):
    lb = int(cfg.get("lookback", 120))
    return max(lb + 220, 320)

# =========================
# main
# =========================
def load_config():
    p = "config.yaml"
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_config()
    scan_once(cfg)

if __name__ == "__main__":
    main()






