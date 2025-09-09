# scanner.py
import os
import time
import yaml
import argparse
import pandas as pd
from datetime import datetime, timedelta
from pytz import timezone
from pykrx import stock

# 전략 함수는 별도 파일(strategy.py)에서 가져옵니다.
from strategy import compute_indicators, entry_signal, exit_signal, position_size

KST = timezone("Asia/Seoul")

# -----------------------------
# 알림 (텔레그램/ntfy) - 토큰/URL은 환경변수(Secrets)에서 읽음
# -----------------------------
def _notify(text, use_tg, use_ntfy, token_env, chat_id_env, ntfy_env):
    """
    텔레그램/ntfy 로 보낼 수 없을 때는 콘솔로만 출력.
    """
    sent = False
    if use_tg:
        try:
            import requests
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
            import requests
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
    """
    유연한 CSV 로더:
    - 인코딩: utf-8-sig → utf-8 → cp949
    - 구분자 자동 추정(sep=None, engine='python')
    - 헤더 정규화 및 alias 매핑
    """
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

    # 값 공백 제거 (applymap 경고 회피)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    # 컬럼명 정규화(BOM 제거, 소문자)
    def norm_col(c):
        return str(c).lstrip("\ufeff").strip().lower()
    df.columns = [norm_col(c) for c in df.columns]

    code_aliases = {"code", "종목코드", "티커", "ticker", "symbol", "코드"}
    name_aliases = {"name", "종목명", "이름", "명", "company"}

    code_col = next((c for c in df.columns if c in code_aliases), None)
    name_col = next((c for c in df.columns if c in name_aliases), None)

    # 한 컬럼에 "005930,삼성전자" 형태일 때 분리
    if code_col is None and df.shape[1] == 1:
        the_col = df.columns[0]
        if df[the_col].str.contains(",").any():
            tmp = df[the_col].str.split(",", n=1, expand=True)
            if tmp.shape[1] == 2:
                df["code"] = tmp[0]
                df["name"] = tmp[1]
                code_col, name_col = "code", "name"

    if code_col is None:
        print(f"[load_universe] CSV columns detected: {list(df.columns)}")
        raise KeyError("CSV에서 종목코드 컬럼을 찾지 못했습니다. (code/종목코드/티커/ticker/symbol/코드 중 하나)")

    if name_col is None:
        df["__name__"] = df[code_col]
        name_col = "__name__"

    def to_6(s):
        if pd.isna(s):
            return None
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

# -----------------------------
# 데이터 소스: pykrx → yfinance 폴백
# -----------------------------
def fetch_daily_df(code, start, end):
    """
    1) pykrx(Naver) 시도
    2) 실패/차단 시 yfinance 폴백
    - 프록시: HTTP(S)_PROXY 환경변수 자동 감지
    - yfinance는 period=18mo로 요청 (날짜 파싱 이슈 완화)
    """
    import requests
    from requests.adapters import HTTPAdapter, Retry
    import yfinance as yf

    proxies = {}
    for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"]:
        v = os.getenv(k)
        if v:
            if "http" in k.lower():
                proxies["http"] = v
            if "https" in k.lower():
                proxies["https"] = v

    sess = requests.Session()
    if proxies:
        sess.proxies.update(proxies)
    sess.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    retries = Retry(total=3, backoff_factor=1.0, status_forcelist=[429, 500, 502, 503, 504])
    sess.mount("http://", HTTPAdapter(max_retries=retries))
    sess.mount("https://", HTTPAdapter(max_retries=retries))

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
    for ticker in (f"{code}.KS", f"{code}.KQ"):
        try:
            yft = yf.Ticker(ticker, session=sess)
            ydf = yft.history(period="18mo", interval="1d", auto_adjust=False)
            if ydf is not None and not ydf.empty:
                ydf = ydf.rename(columns={
                    "Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"
                })
                ydf.index = pd.to_datetime(ydf.index)
                ydf = ydf[["open","high","low","close","volume"]].astype(float)
                return ydf
        except Exception as e:
            print(f"[yfinance] {ticker} 조회 실패 → {e}")

    return pd.DataFrame()

# -----------------------------
# 메시지 포맷
# -----------------------------
def format_buy_msg(ts, row, code, name, shares):
    return (
        f"🟢 <b>매수 신호</b>\n"
        f"종목: {name}({code})\n"
        f"시간: {ts.strftime('%Y-%m-%d %H:%M:%S')} KST\n"
        f"종가: {row['close']:.0f} / HHV30: {row['HHV30']:.0f}\n"
        f"SMA20: {row['SMA20']:.0f} ≥ SMA60: {row['SMA60']:.0f}\n"
        f"ATR14: {row['ATR14']:.0f}\n"
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
    start = (now_kst() - timedelta(days=400)).strftime("%Y%m%d")
    end   = now_kst().strftime("%Y%m%d")

    buy_candidates = []
    sell_candidates = []
    near_candidates = []   # HHV30 근접 후보

    # 근접 임계치(예: 0.01 = 1%)
    try:
        near_pct = float(cfg.get("watchlist", {}).get("near_hhv30_pct", 0.01))
    except Exception:
        near_pct = 0.01

    for _, r in uni.iterrows():
        code, name = r["code"], r["name"]
        df = fetch_daily_df(code, start, end)
        if df.empty or len(df) < 70:
            continue

        df = compute_indicators(df, lookback=cfg["lookback"])
        last = df.iloc[-1]

        # ===== 매수 신호 =====
        if entry_signal(
            last,
            buffer=cfg["entry"]["buffer"],
            require_ma_trend=cfg["entry"]["require_ma_trend"]
        ):
            if not (pos["code"] == code).any():  # 미보유만
                atr_val = float(last["ATR14"]) if pd.notna(last["ATR14"]) else None
                shares = position_size(cfg["equity"], cfg["risk"], atr_val)
                if shares > 0:
                    buy_candidates.append((code, name, last, shares))

        # ===== 매도 신호 =====
        if (pos["code"] == code).any():
            p = pos.loc[pos["code"] == code].iloc[0]
            entry_price = float(p["entry_price"])
            atr_entry   = float(p["atr_entry"])
            price_now   = float(last["close"])
            reason_ma = (price_now < float(last["SMA20"])) if cfg["exit"]["ma_exit"] else False
            reason_sl = price_now <= (entry_price - cfg["exit"]["stop_atr_multiple"] * atr_entry)
            if reason_ma or reason_sl:
                reason = []
                if reason_ma: reason.append("SMA20 하향이탈")
                if reason_sl: reason.append(f"ATR {cfg['exit']['stop_atr_multiple']}배 손절")
                sell_candidates.append((code, name, price_now, " + ".join(reason)))

        # ===== HHV30 근접 후보 =====
        if pd.notna(last.get("HHV30")) and last["HHV30"] > 0:
            dist = (float(last["HHV30"]) - float(last["close"])) / float(last["HHV30"])
            if 0 <= dist <= near_pct:
                near_candidates.append((code, name, float(dist)))

    # ===== 알림 & 포지션 갱신 =====
    use_tg   = cfg["telegram"]["enabled"]
    use_ntfy = cfg["ntfy"]["enabled"]
    ts = now_kst()

    # 매수
    if buy_candidates:
        current_n = len(pos)
        capacity = max(cfg["max_positions"] - current_n, 0)
        for code, name, last, shares in buy_candidates[:capacity]:
            msg = format_buy_msg(ts, last, code, name, shares)
            _notify(msg, use_tg, use_ntfy,
                    cfg["telegram"]["token_env"], cfg["telegram"]["chat_id_env"], cfg["ntfy"]["url_env"])
            # 포지션 기록
            new_row = {
                "code": code, "name": name,
                "entry_date": ts.strftime("%Y-%m-%d"),
                "entry_price": float(last["close"]),
                "atr_entry": float(last["ATR14"]),
                "shares": int(shares)
            }
            pos = pd.concat([pos, pd.DataFrame([new_row])], ignore_index=True)

    # 매도
    closed_codes = []
    for code, name, price_now, reason in sell_candidates:
        msg = format_sell_msg(ts, code, name, price_now, reason)
        _notify(msg, use_tg, use_ntfy,
                cfg["telegram"]["token_env"], cfg["telegram"]["chat_id_env"], cfg["ntfy"]["url_env"])
        closed_codes.append(code)
    if closed_codes:
        pos = pos[~pos["code"].isin(closed_codes)]

    save_positions(pos, cfg["positions_csv"])

    # --- 하루 요약 알림(신호 없어도 보냄) ---
    summary = (f"📬 EOD 스캔 완료\n"
               f"대상: {len(uni)}개\n"
               f"매수 신호: {len(buy_candidates)}개\n"
               f"매도 신호: {len(sell_candidates)}개\n"
               f"시각: {ts.strftime('%Y-%m-%d %H:%M:%S')} KST")
    _notify(summary, use_tg, use_ntfy,
            cfg["telegram"]["token_env"], cfg["telegram"]["chat_id_env"], cfg["ntfy"]["url_env"])

    # --- HHV30 근접 후보 알림 (Top 10) ---
    if near_candidates:
        near_candidates.sort(key=lambda x: x[2])  # dist 오름차순(가까운 순)
        top = near_candidates[:10]
        # near_pct(0.01) → 1%
        pct_txt = f"{int(near_pct * 100)}%"
        lines = [f"🔎 HHV30 근접 Top {len(top)} (임계 {pct_txt})"]
        for c, n, d in top:
            lines.append(f"- {n}({c}) • 거리 {d*100:.2f}%")
        _notify("\n".join(lines), use_tg, use_ntfy,
                cfg["telegram"]["token_env"], cfg["telegram"]["chat_id_env"], cfg["ntfy"]["url_env"])

# -----------------------------
# 엔트리 포인트
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="KOSPI200 Signal Bot (+ HHV30 근접 후보)")
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


