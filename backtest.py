# backtest.py
import os, argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import yaml

# ===== 설정 로드 =====
def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_universe(csv_path):
    for enc in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            df = pd.read_csv(csv_path, dtype=str, sep=None, engine="python", encoding=enc)
            break
        except Exception:
            continue
    df.columns = [str(c).lstrip("\ufeff").strip().lower() for c in df.columns]
    code_col = next((c for c in ["code","종목코드","티커","ticker","symbol","코드"] if c in df.columns), None)
    name_col = next((c for c in ["name","종목명","이름","명","company"] if c in df.columns), None)
    if code_col is None:
        raise RuntimeError("CSV에서 code/종목코드 컬럼을 찾지 못함")
    if name_col is None:
        df["__name__"] = df[code_col]
        name_col = "__name__"

    def to6(x):
        s = "".join(ch for ch in str(x) if ch.isdigit())
        return s.zfill(6)

    df["code"] = df[code_col].map(to6)
    df["name"] = df[name_col].astype(str).str.strip()
    df = df.dropna(subset=["code"])
    df = df[df["code"].str.len()==6].drop_duplicates("code")
    return df[["code","name"]]

# ===== 지표 =====
def sma(s, w): return s.rolling(w).mean()

def atr(df, period=14):
    h_l  = df["High"] - df["Low"]
    h_pc = (df["High"] - df["Close"].shift()).abs()
    l_pc = (df["Low"]  - df["Close"].shift()).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_ind(ydf):
    # yfinance 컬럼명(Open/High/Low/Close/Volume)
    df = ydf.copy()
    out = pd.DataFrame(index=df.index)
    out["Open"]  = df["Open"]
    out["High"]  = df["High"]
    out["Low"]   = df["Low"]
    out["Close"] = df["Close"]
    out["SMA20"] = sma(out["Close"], 20)
    out["SMA60"] = sma(out["Close"], 60)
    out["ATR14"] = atr(out, 14)
    out["HHV30"] = out["High"].rolling(30).max()
    return out

# ===== 신호 =====
def entry_ok(r, buffer, require_ma):
    if pd.isna(r["HHV30"]) or pd.isna(r["SMA20"]) or pd.isna(r["SMA60"]): 
        return False
    cond_break = r["Close"] >= r["HHV30"] * (1.0 + buffer)
    cond_ma    = (r["SMA20"] >= r["SMA60"]) if require_ma else True
    return bool(cond_break and cond_ma)

def exit_hit(price, entry, atr_entry, sma20, use_ma=True, stop_mult=1.5):
    cond_stop = price <= (entry - (stop_mult * atr_entry)) if (atr_entry and atr_entry>0) else False
    cond_ma   = (price < sma20) if (use_ma and pd.notna(sma20)) else False
    return bool(cond_stop or cond_ma)

# ===== 데이터 다운로드 (yfinance) =====
def fetch_ydf(code, start_dt, end_dt):
    """
    .KS 우선, 실패 시 .KQ. 둘 다 실패하면 빈 DF 반환.
    yfinance가 'possibly delisted; No timezone found' 을 던져도 그냥 스킵.
    """
    # 세션/리트라이 약간의 안정화
    import requests
    from requests.adapters import HTTPAdapter, Retry
    sess = requests.Session()
    retries = Retry(total=3, backoff_factor=0.7, status_forcelist=[429, 500, 502, 503, 504])
    sess.mount("https://", HTTPAdapter(max_retries=retries))
    sess.mount("http://",  HTTPAdapter(max_retries=retries))

    for suffix in (".KS", ".KQ"):
        t = f"{code}{suffix}"
        try:
            yft = yf.Ticker(t, session=sess)
            # 날짜 범위 직접 지정(최근 4년 받아서 이후 3년만 사용)
            ydf = yft.history(start=start_dt, end=end_dt, interval="1d", auto_adjust=False)
            if ydf is not None and not ydf.empty and {"Open","High","Low","Close","Volume"}.issubset(ydf.columns):
                return ydf
        except Exception as e:
            print(f"[yfinance] {t} 실패 → {e}")
            continue
    return pd.DataFrame()

# ===== 백테스트 =====
def backtest(cfg, years=3):
    uni = load_universe(cfg["universe_csv"])

    end = datetime.now().date()
    start_dl = end - timedelta(days=int(365*(years+1)))  # 다운로드는 +1y 여유
    start_bt = end - timedelta(days=int(365*years))      # 실제 백테스트 시작

    trades = []
    equity = 1.0
    equity_curve = []

    # 파라미터
    buf = float(cfg["entry"]["buffer"])
    req = bool(cfg["entry"]["require_ma_trend"])
    use_ma_exit = bool(cfg["exit"]["ma_exit"])
    stop_mult   = float(cfg["exit"]["stop_atr_multiple"])

    n_fail = 0
    n_ok   = 0

    for _, row in uni.iterrows():
        code, name = row["code"], row["name"]
        ydf = fetch_ydf(code, start_dl, end)
        if ydf.empty or len(ydf) < 100:
            n_fail += 1
            continue

        # 백테스트 구간만 사용
        ydf = ydf[ydf.index.date >= start_bt]
        if ydf.empty or len(ydf) < 70:
            n_fail += 1
            continue

        df = compute_ind(ydf)
        in_pos = False
        entry_price = atr_entry = None
        entry_date  = None

        # 다음날 시가 체결 가정
        for i in range(60, len(df)-1):
            today = df.iloc[i]
            nxt   = df.iloc[i+1]

            # 진입
            if not in_pos and entry_ok(today, buf, req):
                if pd.notna(nxt["Open"]):
                    entry_price = float(nxt["Open"])
                    atr_entry   = float(today["ATR14"]) if pd.notna(today["ATR14"]) else None
                    entry_date  = df.index[i+1].date()
                    in_pos = True
                continue

            # 청산
            if in_pos:
                price_now = float(nxt["Open"]) if pd.notna(nxt["Open"]) else float(today["Close"])
                sma20_now = float(today["SMA20"]) if pd.notna(today["SMA20"]) else None
                if exit_hit(price_now, entry_price, atr_entry, sma20_now, use_ma=use_ma_exit, stop_mult=stop_mult):
                    ret = (price_now/entry_price - 1.0)
                    trades.append({
                        "code": code, "name": name,
                        "entry_date": str(entry_date), "entry": entry_price,
                        "exit_date": str(df.index[i+1].date()), "exit": price_now,
                        "ret": ret
                    })
                    equity *= (1.0 + ret)
                    in_pos = False
                    entry_price = atr_entry = entry_date = None

            equity_curve.append({"date": str(df.index[i+1].date()), "equity": equity})

        n_ok += 1

    tr = pd.DataFrame(trades)
    stats = {}
    if not tr.empty:
        wins = tr[tr["ret"] > 0]
        loss = tr[tr["ret"] <= 0]
        stats["n_trades"] = len(tr)
        stats["win_rate"] = float(len(wins)/len(tr))*100.0
        stats["avg_win"]  = float(wins["ret"].mean()*100.0) if not wins.empty else 0.0
        stats["avg_loss"] = float(loss["ret"].mean()*100.0) if not loss.empty else 0.0
        if not wins.empty and not loss.empty and loss["ret"].mean()!=0:
            stats["payoff_ratio"] = float(abs(wins["ret"].mean()/loss["ret"].mean()))
        else:
            stats["payoff_ratio"] = None
        p = stats["win_rate"]/100.0
        aw = wins["ret"].mean() if not wins.empty else 0.0
        al = abs(loss["ret"].mean()) if not loss.empty else 0.0
        stats["expectancy_%"] = float((p*aw - (1-p)*al)*100.0)
    else:
        stats = {"n_trades":0,"win_rate":0.0,"avg_win":0.0,"avg_loss":0.0,"payoff_ratio":None,"expectancy_%":0.0}

    eq = pd.DataFrame(equity_curve).drop_duplicates("date", keep="last")

    # 메타 정보
    meta = {
        "universe_total": int(len(uni)),
        "download_ok": int(n_ok),
        "download_fail": int(n_fail),
        "start": str(start_bt),
        "end":   str(end),
    }
    return tr, pd.DataFrame([stats]), eq, meta

# ===== 텔레그램 =====
def send_telegram(text):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat  = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        print("[TG] token/chat missing → skip")
        return
    import requests
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat, "text": text, "parse_mode":"HTML", "disable_web_page_preview": True}
    try:
        requests.post(url, json=payload, timeout=20)
    except Exception as e:
        print("[TG]", e)

def format_float(x, digits=2, na="N/A"):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return na
    try:
        return f"{x:.{digits}f}"
    except Exception:
        return na

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tr, st, eq, meta = backtest(cfg, years=args.years)

    # 저장
    os.makedirs("bt_out", exist_ok=True)
    tr.to_csv("bt_out/trades.csv", index=False, encoding="utf-8")
    st.to_csv("bt_out/stats.csv", index=False, encoding="utf-8")
    eq.to_csv("bt_out/equity_curve.csv", index=False, encoding="utf-8")
    pd.DataFrame([meta]).to_csv("bt_out/meta.csv", index=False, encoding="utf-8")

    # 요약 텔레그램 (None/NaN 안전 포맷)
    s = st.iloc[0].to_dict()
    payoff_txt = format_float(s.get("payoff_ratio"), 2, "N/A")
    msg = (
        f"📊 KOSPI200 백테스트 ({args.years}년)\n"
        f"기간: {meta['start']} ~ {meta['end']}\n"
        f"커버: {meta['download_ok']}/{meta['universe_total']} (실패 {meta['download_fail']})\n"
        f"트레이드 수: {int(s['n_trades'])}\n"
        f"승률: {format_float(s['win_rate'], 1)}%\n"
        f"평균 수익: {format_float(s['avg_win'])}% / 평균 손실: {format_float(s['avg_loss'])}%\n"
        f"손익비(Payoff): {payoff_txt}\n"
        f"기대값/트레이드: {format_float(s['expectancy_%'])}%"
    )
    send_telegram(msg)

    print("\n=== BACKTEST SUMMARY ===\n", st)
    print("\n=== META ===\n", meta)

if __name__ == "__main__":
    main()

