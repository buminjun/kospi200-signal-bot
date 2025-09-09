# backtest.py
import os, time, argparse
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yaml

# 1) FDR 우선
import FinanceDataReader as fdr
# 2) yfinance 폴백(+백오프)
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter, Retry


# =========================
# 설정/공통 유틸
# =========================
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

def format_float(x, digits=2, na="N/A"):
    if x is None:
        return na
    try:
        xf = float(x)
        if np.isnan(xf) or np.isinf(xf):
            return na
        return f"{xf:.{digits}f}"
    except Exception:
        return na

def safe_price(x):
    try:
        v = float(x)
        if np.isnan(v) or np.isinf(v) or v <= 0:
            return None
        return v
    except Exception:
        return None


# =========================
# 지표/신호
# =========================
def sma(s, w): return s.rolling(w).mean()

def atr(df, period=14):
    h_l  = df["High"] - df["Low"]
    h_pc = (df["High"] - df["Close"].shift()).abs()
    l_pc = (df["Low"]  - df["Close"].shift()).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_ind(ydf):
    # 컬럼명: Open/High/Low/Close/Volume
    df = ydf.copy()
    out = pd.DataFrame(index=df.index)
    out["Open"]  = df["Open"]
    out["High"]  = df["High"]
    out["Low"]   = df["Low"]
    out["Close"] = df["Close"]
    out["SMA20"] = sma(out["Close"], 20)
    out["SMA60"] = sma(out["Close"], 60)
    out["ATR14"] = atr(out, 14)
    out["HHV30"] = out["High"].rolling(30, min_periods=30).max().shift(1)
    return out

def entry_ok(r, buffer, require_ma):
    if pd.isna(r["HHV30"]) or pd.isna(r["SMA20"]) or pd.isna(r["SMA60"]):
        return False
    cond_break = r["Close"] >= r["HHV30"] * (1.0 + buffer)
    cond_ma    = (r["SMA20"] >= r["SMA60"]) if require_ma else True
    return bool(cond_break and cond_ma)

def exit_hit(price, entry, atr_entry, sma20, use_ma=True, stop_mult=1.5):
    cond_stop = price <= (entry - (stop_mult * atr_entry)) if (atr_entry and atr_entry>0) else False
    cond_ma   = (price < sma20) if (use_ma and sma20 is not None) else False
    return bool(cond_stop or cond_ma)


# =========================
# 데이터: FDR → yfinance 폴백
# =========================
def _yf_history_with_backoff(ticker, start_dt, end_dt, tries=4, base_sleep=2.0):
    sess = requests.Session()
    retries = Retry(total=3, backoff_factor=0.7, status_forcelist=[429, 500, 502, 503, 504])
    sess.mount("https://", HTTPAdapter(max_retries=retries))
    sess.mount("http://",  HTTPAdapter(max_retries=retries))
    yft = yf.Ticker(ticker, session=sess)
    for k in range(tries):
        try:
            df = yft.history(start=start_dt, end=end_dt, interval="1d", auto_adjust=False)
            if df is not None and not df.empty and {"Open","High","Low","Close","Volume"}.issubset(df.columns):
                return df
        except Exception as e:
            wait = base_sleep * (2 ** k)
            print(f"[yfinance] {ticker} 실패 → {e} ; {wait:.1f}s 대기 후 재시도")
        time.sleep(base_sleep * (2 ** k))
    return pd.DataFrame()

def fetch_kr_df(code, start_dt, end_dt):
    """
    1) FinanceDataReader: 심볼 '005930' 그대로 사용
    2) 실패 시 yfinance: .KS → .KQ 순서, 백오프 재시도
    반환: Open/High/Low/Close/Volume 컬럼
    """
    # FDR
    try:
        fdf = fdr.DataReader(code, start_dt, end_dt)
        if fdf is not None and not fdf.empty:
            cols = ["Open","High","Low","Close","Volume"]
            if set(cols).issubset(fdf.columns):
                return fdf[cols]
    except Exception as e:
        print(f"[FDR] {code} 실패 → {e}")

    # yfinance 폴백
    for suffix in (".KS", ".KQ"):
        t = f"{code}{suffix}"
        ydf = _yf_history_with_backoff(t, start_dt, end_dt, tries=4, base_sleep=2.0)
        if ydf is not None and not ydf.empty:
            return ydf
    return pd.DataFrame()


# =========================
# 백테스트 본체
# =========================
def backtest(cfg, years=3):
    uni = load_universe(cfg["universe_csv"])

    end = datetime.now().date()
    start_dl = end - timedelta(days=int(365*(years+1)))  # 다운로드 여유
    start_bt = end - timedelta(days=int(365*years))      # 실제 테스트 시작

    trades = []
    equity = 1.0
    equity_curve = []

    # 파라미터
    buf = float(cfg["entry"]["buffer"])
    req = bool(cfg["entry"]["require_ma_trend"])
    use_ma_exit = bool(cfg["exit"]["ma_exit"])
    stop_mult   = float(cfg["exit"]["stop_atr_multiple"])

    n_fail = n_ok = 0
    bad_entry = bad_exit = 0

    for _, row in uni.iterrows():
        code, name = row["code"], row["name"]
        ydf = fetch_kr_df(code, start_dl, end)
        if ydf.empty or len(ydf) < 100:
            n_fail += 1
            continue

        # 백테스트 구간만
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
                ep = safe_price(nxt.get("Open"))
                if ep is None:
                    ep = safe_price(nxt.get("Close"))  # 대체
                if ep is None:
                    bad_entry += 1
                    continue
                entry_price = ep
                atr_entry   = float(today["ATR14"]) if pd.notna(today["ATR14"]) else None
                entry_date  = df.index[i+1].date()
                in_pos = True
                continue

            # 청산
            if in_pos:
                pn = safe_price(nxt.get("Open"))
                if pn is None:
                    pn = safe_price(today.get("Close"))
                if pn is None or entry_price is None or entry_price <= 0:
                    bad_exit += 1
                else:
                    sma20_now = float(today["SMA20"]) if pd.notna(today["SMA20"]) else None
                    if exit_hit(pn, entry_price, atr_entry, sma20_now, use_ma=use_ma_exit, stop_mult=stop_mult):
                        ret = (pn / entry_price - 1.0)
                        trades.append({
                            "code": code, "name": name,
                            "entry_date": str(entry_date), "entry": entry_price,
                            "exit_date": str(df.index[i+1].date()), "exit": pn,
                            "ret": ret
                        })
                        equity *= (1.0 + ret)
                        in_pos = False
                        entry_price = atr_entry = entry_date = None

            equity_curve.append({"date": str(df.index[i+1].date()), "equity": equity})

        n_ok += 1

    tr = pd.DataFrame(trades)
    if not tr.empty:
        wins = tr[tr["ret"] > 0]
        loss = tr[tr["ret"] <= 0]
        win_rate = float(len(wins)/len(tr))*100.0
        avg_win  = float(wins["ret"].mean()*100.0) if not wins.empty else 0.0
        avg_loss = float(loss["ret"].mean()*100.0) if not loss.empty else 0.0
        payoff   = float(abs(wins["ret"].mean()/loss["ret"].mean())) if (not wins.empty and not loss.empty and loss["ret"].mean()!=0) else None
        p = win_rate/100.0
        aw = wins["ret"].mean() if not wins.empty else 0.0
        al = abs(loss["ret"].mean()) if not loss.empty else 0.0
        expectancy = float((p*aw - (1-p)*al)*100.0)
    else:
        win_rate = avg_win = avg_loss = expectancy = 0.0
        payoff = None

    st = pd.DataFrame([{
        "n_trades": len(tr),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": payoff,
        "expectancy_%": expectancy
    }])

    eq = pd.DataFrame(equity_curve).drop_duplicates("date", keep="last")

    meta = {
        "universe_total": int(len(uni)),
        "download_ok": int(n_ok),
        "download_fail": int(n_fail),
        "bad_entry": int(bad_entry),
        "bad_exit": int(bad_exit),
        "start": str(start_bt),
        "end":   str(end),
    }
    return tr, st, eq, meta


# =========================
# 텔레그램
# =========================
def send_telegram(text):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat  = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat:
        print("[TG] token/chat missing → skip")
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat, "text": text, "parse_mode":"HTML", "disable_web_page_preview": True}
    try:
        requests.post(url, json=payload, timeout=20)
    except Exception as e:
        print("[TG]", e)


# =========================
# 엔트리포인트
# =========================
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

    # 텔레그램 요약 (안전 포맷)
    s = st.iloc[0].to_dict()
    msg = (
        f"📊 KOSPI200 백테스트 ({args.years}년)\n"
        f"기간: {meta['start']} ~ {meta['end']}\n"
        f"커버: {meta['download_ok']}/{meta['universe_total']} (실패 {meta['download_fail']})\n"
        f"데이터 이상: 진입{meta['bad_entry']}, 청산{meta['bad_exit']}\n"
        f"트레이드 수: {int(s['n_trades'])}\n"
        f"승률: {format_float(s['win_rate'], 1)}%\n"
        f"평균 수익: {format_float(s['avg_win'])}% / 평균 손실: {format_float(s['avg_loss'])}%\n"
        f"손익비(Payoff): {format_float(s.get('payoff_ratio'), 2)}\n"
        f"기대값/트레이드: {format_float(s['expectancy_%'])}%"
    )
    send_telegram(msg)

    print("\n=== BACKTEST SUMMARY ===\n", st)
    print("\n=== META ===\n", meta)

if __name__ == "__main__":
    main()
