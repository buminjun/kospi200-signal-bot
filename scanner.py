# scanner.py
import os
import time
import yaml
import argparse
import pandas as pd
from datetime import datetime, timedelta
from pytz import timezone

# 데이터 소스
from pykrx import stock
import yfinance as yf
import FinanceDataReader as fdr
import requests
from requests.adapters import HTTPAdapter, Retry

# 전략
from strategy import compute_indicators, entry_signal, exit_signal, position_size

KST = timezone("Asia/Seoul")

# -----------------------------
# 알림 (텔레그램/ntfy)
# -----------------------------
def _notify(text, use_tg, use_ntfy, token_env, chat_id_env, ntfy_env):
    sent = False
    if use_tg:
        try:
            token = os.getenv(token_env, "")
            chat_id = os.getenv(chat_id_env, "")
            if token and chat_id:
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML", "disable_web_page_preview": True}
                requests.post(url, json=payload, timeout=15)
                sent = True
        except Exception as e:
            print(f"[TG] {e}")
    if use_ntfy:
        try:
            url = os.getenv(ntfy_env, "")
            if url:
                requests.post(url, data=text.encode("utf-8"), timeout=10)
                sent = True
        except Exception as e:
            print(f"[NTFY] {e}")
    if not sent:
        print(text)

# -----------------------------
# 유틸
# -----------------------------
def now_kst():
    return datetime.now(KST)

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_universe(csv_path):
    # 인코딩/구분자 유연 처리
    last_err = None
    for enc in ["utf-8-sig", "utf-8", "cp949"]:
        try:
            df = pd.read_csv(csv_path, dtype=str, sep=None, engine="python", encoding=enc)
            break
        except Exception as e:
            last_err = e
            continue
    if "df" not in locals():
        raise last_err

    # 공백 정리
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.strip()

    # 헤더 정규화
    def norm_col(c): return str(c).lstrip("\ufeff").strip().lower()
    df.columns = [norm_col(c) for c in df.columns]

    code_aliases = {"code", "종목코드", "티커", "ticker", "symbol", "코드"}
    name_aliases = {"name", "종목명", "이름", "명", "company"}
    code_col = next((c for c in df.columns if c in code_aliases), None)
    name_col = next((c for c in df.columns if c in name_aliases), None)

    if code_col is None and df.shape[1] == 1:
        the_col = df.columns[0]
        if df[the_col].str.contains(",").any():
            tmp = df[the_col].str.split(",", n=1, expand=True)
            if tmp.shape[1] == 2:
                df["code"] = tmp[0]; df["name"] = tmp[1]
                code_col, name_col = "code", "name"

    if code_col is None:
        print(f"[load_universe] CSV columns detected: {list(df.columns)}")
        raise KeyError("CSV에서 종목코드 컬럼을 찾지 못했습니다. (code/종목코드/티커/ticker/symbol/코드 중 하나)")

    if name_col is None:
        df["__name__"] = df[code_col]; name_col = "__name__"

    def to_6(s):
        if pd.isna(s): return None
        s = "".join(ch for ch in str(s) if ch.isdigit())
        return s.zfill(6)

    df["code"] = df[code_col].map(to_6)
    df["name"] = df[name_col].astype(str).str.strip()
    df = df.dropna(subset=["code"])
    df = df[df["code"].str.len() == 6]
    df = df.drop_duplicates(subset=["code"], keep="first")
    return df[["code", "name"]]

def load_positions(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=["code","name","entry_date","entry_price","atr_entry","shares"])
    df = pd.read_csv(path, dtype={"code": str})
    df["code"] = df["code"].str.zfill(6)
    return df

def save_positions(df, path):
    df.to_csv(path, index=False, encoding="utf-8")

def inside_market_hours(cfg):
    t = now_kst().time()
    s = datetime.strptime(cfg["market_hours"]["start_kst"], "%H:%M").time()
    e = datetime.strptime(cfg["market_hours"]["end_kst"], "%H:%M").time()
    return (t >= s) and (t <= e)

def should_send_summary(ts, every_min=60):
    try:
        every = int(every_min)
        if every <= 0: every = 60
    except Exception:
        every = 60
    return (ts.minute % every) == 0

# -----------------------------
# 데이터 소스
# -----------------------------
def fetch_daily_df(code, start, end):
    sess = requests.Session()
    retries = Retry(total=3, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
    sess.mount("http://", HTTPAdapter(max_retries=retries))
    sess.mount("https://", HTTPAdapter(max_retries=retries))
    sess.headers.update({"User-Agent": "Mozilla/5.0"})

    # 1) pykrx
    try:
        df = stock.get_market_ohlcv_by_date(start, end, code)
        if df is not None and not df.empty:
            df = df.rename(columns={"시가":"open","고가":"high","저가":"low","종가":"close","거래량":"volume"})
            df.index = pd.to_datetime(df.index)
            df = df[["open","high","low","close","volume"]].astype(float)
            return df
    except Exception as e:
        print(f"[pykrx] {code} 조회 실패 → {e}")

    # 2) yfinance 폴백
    for suffix in (".KS", ".KQ"):
        try:
            yft = yf.Ticker(f"{code}{suffix}", session=sess)
            ydf = yft.history(period="18mo", interval="1d", auto_adjust=False)
            if ydf is not None and not ydf.empty:
                ydf = ydf.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
                ydf.index = pd.to_datetime(ydf.index)
                ydf = ydf[["open","high","low","close","volume"]].astype(float)
                return ydf
        except Exception as e:
            print(f"[yfinance] {code}{suffix} 조회 실패 → {e}")
    return pd.DataFrame()

def fetch_kospi_close(start, end):
    try:
        kospi = fdr.DataReader("KS11", start, end)
        if kospi is not None and not kospi.empty and "Close" in kospi.columns:
            s = kospi["Close"].copy()
            s.index = pd.to_datetime(s.index)
            return s
    except Exception as e:
        print(f"[FDR] KS11 실패 → {e}")
    try:
        y = yf.Ticker("^KS11").history(start=start, end=end, interval="1d", auto_adjust=False)
        if y is not None and not y.empty and "Close" in y.columns:
            s = y["Close"].copy()
            s.index = pd.to_datetime(s.index)
            return s
    except Exception as e:
        print(f"[yfinance] ^KS11 실패 → {e}")
    return None

# -----------------------------
# 메시지 포맷
# -----------------------------
def format_buy_msg(ts, row, code, name, shares):
    rs_txt = f"{row['RS']:.2f}" if pd.notna(row.get("RS")) else "N/A"
    return (
        f"🟢 <b>매수 신호</b>\n"
        f"종목: {name}({code})\n"
        f"시간: {ts.strftime('%Y-%m-%d %H:%M:%S')} KST\n"
        f"종가: {row['close']:.0f} / HHV30: {row['HHV30']:.0f}\n"
        f"SMA20: {row['SMA20']:.0f} ≥ SMA60: {row['SMA60']:.0f}\n"
        f"ATR14: {row['ATR14']:.0f} / RS: {rs_txt}\n"
        f"수량(리스크 기반): {shares}주"
    )

def format_sell_msg(ts, code, name, price, reason):
    return (
        f"🔴 <b>매도 신호</b>\n"
        f"종목: {name}({code})\n"
        f"시간: {ts.strftime('%Y-%m-%d %H:%M:%S')} KST\n"
        f"현재가: {price:.0f}\n"
        f"사유: {reason}"
    )

# -----------------------------
# 스캔 본체
# -----------------------------
def scan_once(cfg):
    uni = load_universe(cfg["universe_csv"])
    pos = load_positions(cfg["positions_csv"])

    start_dt = (now_kst() - timedelta(days=400)).date()
    end_dt   = now_kst().date()
    start_krx = start_dt.strftime("%Y%m%d")
    end_krx   = end_dt.strftime("%Y%m%d")
    start_iso = start_dt.strftime("%Y-%m-%d")
    end_iso   = end_dt.strftime("%Y-%m-%d")

    rs_win = int(cfg.get("filters", {}).get("rs_window", 60))
    rs_min = float(cfg.get("filters", {}).get("rs_min", 1.0))
    hhv_win = int(cfg.get("entry", {}).get("hhv_window", 30))
    summary_every = int(cfg.get("notifications", {}).get("summary_every_min", 60))

    kospi_close = fetch_kospi_close(start_iso, end_iso)

    buy_candidates = []
    sell_candidates = []
    near_candidates = []

    for _, r in uni.iterrows():
        code, name = r["code"], r["name"]
        df = fetch_daily_df(code, start_krx, end_krx)
        if df.empty or len(df) < max(70, rs_win + 5):
            continue

        ind = compute_indicators(
            df, lookback=cfg["lookback"],
            kospi_close=kospi_close, rs_window=rs_win, hhv_window=hhv_win
        )
        last = ind.iloc[-1]

        if entry_signal(
            last,
            buffer=cfg["entry"]["buffer"],
            require_ma_trend=cfg["entry"]["require_ma_trend"],
            rs_min=rs_min
        ):
            if not (pos["code"] == code).any():
                atr_val = float(last["ATR14"]) if pd.notna(last["ATR14"]) else None
                shares = position_size(cfg["equity"], cfg["risk"], atr_val)
                if shares > 0:
                    buy_candidates.append((code, name, last, shares))

        if (pos["code"] == code).any():
            p = pos.loc[pos["code"] == code].iloc[0]
            entry_price = float(p["entry_price"])
            atr_entry   = float(p["atr_entry"])
            price_now   = float(last["close"])
            sma20_now   = float(last["SMA20"]) if pd.notna(last["SMA20"]) else None
            if exit_signal(price_now, entry_price, atr_entry, sma20_now,
                           use_ma=cfg["exit"]["ma_exit"],
                           stop_atr_multiple=cfg["exit"]["stop_atr_multiple"]):
                reason = []
                if cfg["exit"]["ma_exit"] and sma20_now is not None and price_now < sma20_now:
                    reason.append("SMA20 하향이탈")
                sell_candidates.append((code, name, price_now, " + ".join(reason) if reason else "규칙 충족"))

        if pd.notna(last.get("HHV30")) and last["HHV30"] > 0:
            dist = (float(last["HHV30"]) - float(last["close"])) / float(last["HHV30"])
            if 0 <= dist <= float(cfg.get("watchlist", {}).get("near_hhv30_pct", 0.01)):
                near_candidates.append((code, name, float(dist)))

    ts = now_kst()
    market_open = inside_market_hours(cfg)
    use_tg   = cfg["telegram"]["enabled"]
    use_ntfy = cfg["ntfy"]["enabled"]
    force_summary = os.getenv("FORCE_SUMMARY", "0") == "1"

    # --- 매수/매도: 장중에만 ---
    if market_open and buy_candidates:
        current_n = len(pos)
        capacity = max(cfg["max_positions"] - current_n, 0)
        for code, name, last, shares in buy_candidates[:capacity]:
            msg = format_buy_msg(ts, last, code, name, shares)
            _notify(msg, use_tg, use_ntfy,
                    cfg["telegram"]["token_env"], cfg["telegram"]["chat_id_env"], cfg["ntfy"]["url_env"])
            new_row = {
                "code": code, "name": name,
                "entry_date": ts.strftime("%Y-%m-%d"),
                "entry_price": float(last["close"]),
                "atr_entry": float(last["ATR14"]) if pd.notna(last["ATR14"]) else 0.0,
                "shares": int(shares)
            }
            pos = pd.concat([pos, pd.DataFrame([new_row])], ignore_index=True)

    if market_open and sell_candidates:
        closed_codes = []
        for code, name, price_now, reason in sell_candidates:
            msg = format_sell_msg(ts, code, name, price_now, reason)
            _notify(msg, use_tg, use_ntfy,
                    cfg["telegram"]["token_env"], cfg["telegram"]["chat_id_env"], cfg["ntfy"]["url_env"])
            closed_codes.append(code)
        if closed_codes:
            pos = pos[~pos["code"].isin(closed_codes)]

    save_positions(pos, cfg["positions_csv"])

    # --- 요약: 장중 + 지정주기 ---
    if market_open and should_send_summary(ts, summary_every)) or force_summary:
        summary = (f"📬 스캔 요약\n"
                   f"대상: {len(uni)}개\n"
                   f"매수 신호: {len(buy_candidates)}개\n"
                   f"매도 신호: {len(sell_candidates)}개\n"
                   f"RS(window={rs_win}, min={rs_min}) / HHV={hhv_win}\n"
                   f"시각: {ts.strftime('%Y-%m-%d %H:%M:%S')} KST")
        _notify(summary, use_tg, use_ntfy,
                cfg["telegram"]["token_env"], cfg["telegram"]["chat_id_env"], cfg["ntfy"]["url_env"])

    # --- HHV30 근접 Top10: 장중 + 지정주기 ---
    if market_open and near_candidates and should_send_summary(ts, summary_every)) or force_summary:
        near_candidates.sort(key=lambda x: x[2])
        top = near_candidates[:10]
        pct_txt = f"{int(float(cfg.get('watchlist', {}).get('near_hhv30_pct', 0.01))*100)}%"
        lines = [f"🔎 HHV30 근접 Top {len(top)} (임계 {pct_txt})"]
        for c, n, d in top:
            lines.append(f"- {n}({c}) • 거리 {d*100:.2f}%")
        _notify("\n".join(lines), use_tg, use_ntfy,
                cfg["telegram"]["token_env"], cfg["telegram"]["chat_id_env"], cfg["ntfy"]["url_env"])

# -----------------------------
# 엔트리 포인트
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="KOSPI200 Signal Scanner (HHV + RS filter)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--mode", choices=["eod", "loop"], default="eod",
                    help="eod=장마감 후 1회, loop=장중 주기 스캔")
    ap.add_argument("--interval", type=int, default=300, help="loop 모드 주기(초)")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.mode == "eod":
        scan_once(cfg)
    else:
        print("[LOOP] 시작. 장중 시간에만 동작합니다.")
        try:
            while True:
                if inside_market_hours(cfg):
                    scan_once(cfg)
                else:
                    print("[LOOP] 장시간 외. 대기…")
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n[LOOP] 사용자 요청으로 종료합니다.")

if __name__ == "__main__":
    main()









