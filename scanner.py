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
# =========================
# CSV IO (포지션/유니버스)
# =========================
def load_positions(path: str):
    """보유 포지션 CSV를 읽어옵니다. 없으면 빈 프레임 반환."""
    try:
        import pandas as pd
        import numpy as np
        if not os.path.exists(path):
            return pd.DataFrame(columns=["code", "name", "entry_date", "entry_price", "shares"])
        df = pd.read_csv(path, encoding="utf-8-sig")
        # 필요한 컬럼 보정
        for c in ["code", "name", "entry_date", "entry_price", "shares"]:
            if c not in df.columns:
                df[c] = np.nan
        # 코드는 항상 6자리 문자열
        df["code"] = df["code"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6)
        # 숫자형 보정
        if "entry_price" in df.columns:
            df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
        if "shares" in df.columns:
            df["shares"] = pd.to_numeric(df["shares"], errors="coerce").fillna(0).astype(int)
        return df[["code", "name", "entry_date", "entry_price", "shares"]]
    except Exception as e:
        print(f"[positions] load error: {e}")
        return pd.DataFrame(columns=["code", "name", "entry_date", "entry_price", "shares"])

def save_positions(df, path: str):
    """보유 포지션 CSV 저장."""
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"[positions] save error: {e}")

def load_universe(path: str):
    """
    유니버스 CSV를 읽어 (code, name) 반환.
    - code/종목코드/티커/ticker/symbol 중 아무거나 허용
    - name/종목명 없으면 code를 name으로 사용
    - code는 항상 6자리 문자열로 변환
    """
    import pandas as pd
    df = pd.read_csv(path, encoding="utf-8-sig")
    # 헤더 정리
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    # code 컬럼 찾기
    code_col = None
    for cand in ["code", "종목코드", "티커", "ticker", "symbol"]:
        if cand in df.columns:
            code_col = cand
            break
    if code_col is None:
        raise KeyError("universe.csv must have 'code' column")

    # name 컬럼 찾기 (없으면 code 재사용)
    name_col = None
    for cand in ["name", "종목명", "이름"]:
        if cand in df.columns:
            name_col = cand
            break
    if name_col is None:
        name_col = code_col
        df[name_col] = df[code_col]

    # 값 정리: 문자열화, 앞자리 0 보존
    out = pd.DataFrame({
        "code": df[code_col].astype(str).str.replace(r"\D", "", regex=True).str.zfill(6),
        "name": df[name_col].astype(str).str.strip()
    })
    out = out.dropna().drop_duplicates()
    return out


# =========================
# 데이터 취득
# =========================
def fetch_yf(code, start_dt, end_dt):
    """야후 → 티커 변환 후 일봉 OHLCV"""
    try:
        # 코드가 6자리만 있으면 .KS 붙여줌
        t = code if code.endswith((".KS", ".KQ")) else f"{code}.KS"
        df = yf.download(
            tickers=t,
            start=start_dt.strftime("%Y-%m-%d"),
            end=(end_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,   # 배당/액면조정 반영
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            return None
        # yfinance 컬럼 정규화
        cols = {c.lower(): c for c in df.columns}
        out = pd.DataFrame(index=pd.to_datetime(df.index))
        out["open"]   = df[ [k for k in df.columns if "open"   in k.lower()][0] ]
        out["high"]   = df[ [k for k in df.columns if "high"   in k.lower()][0] ]
        out["low"]    = df[ [k for k in df.columns if "low"    in k.lower()][0] ]
        out["close"]  = df[ [k for k in df.columns if "close"  in k.lower()][0] ]
        out["volume"] = df[ [k for k in df.columns if "volume" in k.lower()][0] ].fillna(0)
        return out.sort_index()
    except Exception as e:
        print(f"[yfinance] {code} fail: {e}")
        return None

def fetch_pykrx(code, start_dt, end_dt):
    """pykrx → 네이버 백엔드 사용"""
    try:
        s = start_dt.strftime("%Y%m%d"); e = end_dt.strftime("%Y%m%d")
        df = stock.get_market_ohlcv_by_date(s, e, code)
        if df is None or df.empty:
            return None
        df = df.rename(columns={"시가":"open","고가":"high","저가":"low","종가":"close","거래량":"volume"})
        df.index = pd.to_datetime(df.index)
        return df[["open","high","low","close","volume"]].sort_index()
    except Exception as e:
        print(f"[pykrx] {code} fail: {e}")
        return None

def fetch_daily_df(code, start_dt, end_dt):
    """1순위: yfinance → 실패 시 pykrx"""
    df = fetch_yf(code, start_dt, end_dt)
    src = "yfinance"
    if df is None or df.empty:
        df = fetch_pykrx(code, start_dt, end_dt)
        src = "pykrx"
    if df is None or df.empty:
        print(f"[DATA] {code} → empty from both sources")
        return None, None
    return df, src

def fetch_benchmark(start_dt, end_dt):
    """벤치마크(KOSPI) 시총지수: 야후 ^KS11 → 실패 시 pykrx index 1001"""
    # 1) yfinance
    try:
        k = yf.download(
            "^KS11",
            start=start_dt.strftime("%Y-%m-%d"),
            end=(end_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if k is not None and not k.empty:
            close = k[ [c for c in k.columns if "close" in c.lower()][0] ]
            s = pd.Series(close.values, index=pd.to_datetime(close.index))
            return s.sort_index()
    except Exception as e:
        print(f"[yfinance] ^KS11 fail: {e}")

    # 2) pykrx (KOSPI: 1001)
    try:
        s = stock.get_index_ohlcv_by_date(
            start_dt.strftime("%Y%m%d"), end_dt.strftime("%Y%m%d"), "1001"
        )["종가"]
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    except Exception as e:
        print(f"[pykrx] KOSPI fail: {e}")

    return None

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



