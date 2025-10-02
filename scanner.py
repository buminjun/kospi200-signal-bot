# -*- coding: utf-8 -*-
import os, sys, time, random, datetime as dt
import pandas as pd
import numpy as np
import yfinance as yf
import requests

# --- Optional: FinanceDataReader, PyKRX (폴백용)
try:
    import FinanceDataReader as fdr
except Exception:
    fdr = None

try:
    from pykrx import stock as krx
except Exception:
    krx = None


# ================================
# 텔레그램 유틸
# ================================
def send_telegram(msg: str) -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        print("[telegram] token/chat_id 미설정 → 콘솔로만 출력")
        print(msg)
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, json={"chat_id": chat_id, "text": msg})
        if resp.status_code != 200:
            print(f"[telegram] 실패 {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"[telegram] 예외: {e}")


# ================================
# 공통/도움 함수
# ================================
def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance/FDR/KRX 결과를 open/high/low/close/volume 컬럼으로 표준화"""
    colmap = {
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume",
        "시가": "open", "고가": "high", "저가": "low", "종가": "close", "거래량": "volume",
    }
    out = df.rename(columns=colmap).copy()
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in out.columns:
            raise ValueError(f"Missing column: {c}")
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    out = out[["open", "high", "low", "close", "volume"]]
    # 결측 제거
    out = out[~out["close"].isna()]
    return out


def _sleep_backoff(attempt: int):
    sec = (2 ** (attempt - 1)) + random.uniform(0, 0.5)
    time.sleep(min(sec, 3.0))  # CI에서 과도한 대기 방지


def fetch_daily_df(code: str, start, end) -> pd.DataFrame | None:
    """yfinance → FDR → KRX 순서로 시도"""
    start = pd.to_datetime(start).strftime("%Y-%m-%d")
    end = pd.to_datetime(end).strftime("%Y-%m-%d")

    # 1) yfinance
    last_err = None
    for suffix in [".KS", ".KQ"]:
        ticker = f"{code}{suffix}"
        for k in range(1, 3):  # 2회 재시도
            try:
                df = yf.download(ticker, start=start, end=end, interval="1d",
                                 auto_adjust=False, progress=False, threads=False, timeout=15)
                if df is not None and not df.empty:
                    return _normalize_ohlcv(df)
                last_err = RuntimeError("yfinance empty")
            except Exception as e:
                last_err = e
            _sleep_backoff(k)
    print(f"[yfinance] {code} 실패 → {last_err}")

    # 2) FDR
    if fdr is not None:
        for k in range(1, 3):
            try:
                df = fdr.DataReader(code, start, end)
                if df is not None and not df.empty:
                    return _normalize_ohlcv(df)
                last_err = RuntimeError("FDR empty")
            except Exception as e:
                last_err = e
            _sleep_backoff(k)
        print(f"[FDR] {code} 실패 → {last_err}")

    # 3) KRX
    if krx is not None:
        s = pd.to_datetime(start).strftime("%Y%m%d")
        e = pd.to_datetime(end).strftime("%Y%m%d")
        for k in range(1, 3):
            try:
                df = krx.get_market_ohlcv_by_date(s, e, code)
                if df is not None and not df.empty:
                    return _normalize_ohlcv(df)
                last_err = RuntimeError("KRX empty")
            except Exception as e:
                last_err = e
            _sleep_backoff(k)
        print(f"[KRX] {code} 실패 → {last_err}")

    return None


def load_universe(csv_path: str) -> list[tuple[str, str]]:
    """코스피200 CSV 로드 (code/name 컬럼 자동 인식)"""
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    cols = {c.replace("\ufeff", "").strip().lower(): c for c in df.columns}
    code_col = None
    name_col = None
    for cand in ["code", "종목코드", "ticker", "symbol"]:
        if cand in cols:
            code_col = cols[cand]
            break
    for cand in ["name", "종목명"]:
        if cand in cols:
            name_col = cols[cand]
            break
    if not code_col:
        raise KeyError("CSV에 종목코드 컬럼이 없습니다. (code/종목코드/ticker/symbol)")
    if not name_col:
        # name 없으면 코드=이름으로 사용
        df["_name"] = df[code_col]
        name_col = "_name"

    codes = df[code_col].astype(str).str.zfill(6)
    names = df[name_col].astype(str)
    return list(zip(codes, names))


def load_positions(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame(columns=["code", "name", "entry_date", "entry_price", "shares"])
    df = pd.read_csv(path)
    if "entry_price" in df.columns:
        df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce")
    return df


def save_positions(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


# ================================
# 지표/시그널
# ================================
def sma(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w).mean()


def rules_1_to_7(df: pd.DataFrame) -> bool:
    """
    1) 종가 > SMA150, 2) SMA150 > SMA200, 3) SMA200 상승(최근 20일 기준)
    4) 고점/저점 상승(최근 40→최근 20 비교), 5) 주봉 상승시 거래량 증가/하락시 감소,
    6) 거래량 수반한 상승주봉 수 > 하락주봉 수, 7) 52주 저점 대비 +25% 이상
    """
    if len(df) < 250:
        return False
    close = df["close"]
    high = df["high"]
    low = df["low"]
    vol = df["volume"]

    sma150 = sma(close, 150)
    sma200 = sma(close, 200)

    # 1,2
    if not (close.iloc[-1] > sma150.iloc[-1] > sma200.iloc[-1]):
        return False

    # 3: SMA200 상승세 (20일 전 대비 상승)
    if not (sma200.iloc[-1] > sma200.iloc[-20]):
        return False

    # 4: 최근 20일 high/low가 그 이전 20일 대비 상향
    if len(df) < 60:
        return False
    hi_recent = high.iloc[-20:].max()
    lo_recent = low.iloc[-20:].min()
    hi_prev = high.iloc[-40:-20].max()
    lo_prev = low.iloc[-40:-20].min()
    if not (hi_recent > hi_prev and lo_recent > lo_prev):
        return False

    # 5,6: 주봉 가격/거래량
    w_close = close.resample("W").last()
    w_vol = vol.resample("W").sum()
    if len(w_close) < 10:
        return False
    w_ret = w_close.diff()
    # 5: 상승주(>0)에서 거래량 증가(>0), 하락주(<0)에서 거래량 감소(<0) 경향
    up_weeks = (w_ret > 0) & (w_vol.diff() > 0)
    down_weeks = (w_ret < 0) & (w_vol.diff() > 0)  # 하락인데 거래량 증가 → 약함
    # 6: 거래량 수반한 상승주 수 > 하락주 수
    if up_weeks.sum() <= down_weeks.sum():
        return False

    # 7: 52주 신저점 대비 +25% 이상
    low52 = close.iloc[-252:].min()
    if close.iloc[-1] < 1.25 * low52:
        return False

    return True


def rule_8_strong_buy(df: pd.DataFrame) -> bool:
    """
    횡보 후 첫 장대 양봉:
    - 최근 20일 박스(rng < 1.10), 당일 양봉 +5% 이상
    """
    if len(df) < 30:
        return False
    rng = df["close"].iloc[-20:].max() / df["close"].iloc[-20:].min()
    last_gain = (df["close"].iloc[-1] - df["open"].iloc[-1]) / df["open"].iloc[-1]
    return (rng < 1.10) and (last_gain > 0.05)


def sell_signal(price_now: float, entry_price: float | None, sma10_now: float | None) -> tuple[bool, str]:
    """
    매도 조건:
      - 10일선 하락 이탈 (종가 < SMA10)
      - 진입가 대비 -5% 이상 하락
    """
    cond_ma10 = (sma10_now is not None) and (price_now < float(sma10_now))
    cond_drop = (entry_price is not None) and (price_now <= float(entry_price) * 0.95)
    if cond_ma10:
        return True, "10일선 하향 이탈"
    if cond_drop:
        return True, "-5% 하락"
    return False, ""


# ================================
# 스캐너 본체
# ================================
def run_scan(
    universe_csv: str = "kospi200.csv",
    positions_csv: str = "positions.csv",
    lookback_days: int = 300,
    always_send_summary: bool = True
):
    ts = dt.datetime.now(dt.timezone.utc).astimezone(dt.timezone(dt.timedelta(hours=9)))  # KST
    start = (ts - dt.timedelta(days=lookback_days)).date()
    end = ts.date()

    uni = load_universe(universe_csv)
    pos = load_positions(positions_csv)

    buy_list = []       # 7개 규칙 충족
    strong_list = []    # 8번만 충족(강력매수)
    sell_list = []      # 보유종목 매도 후보
    ok, fail = 0, 0

    for code, name in uni:
        df = fetch_daily_df(code, start, end)
        if df is None or df.empty:
            fail += 1
            continue
        ok += 1

        # SMA10 계산(매도용)
        df["SMA10"] = sma(df["close"], 10)

        # ---- 매수 신호
        r7 = rules_1_to_7(df)
        r8 = rule_8_strong_buy(df)
        if r7:
            buy_list.append((code, name))
        elif (not r7) and r8:
            strong_list.append((code, name))

        # ---- 매도 신호 (보유 종목만 평가)
        if not pos.empty and (code in set(pos["code"].astype(str))):
            price_now = float(df["close"].iloc[-1])
            sma10_now = float(df["SMA10"].iloc[-1]) if not np.isnan(df["SMA10"].iloc[-1]) else None
            # 해당 보유 종목 평균 단가(여러 행이 있을 수 있어 평균 사용)
            entry_price = pd.to_numeric(pos.loc[pos["code"].astype(str) == code, "entry_price"], errors="coerce").mean()
            do_sell, reason = sell_signal(price_now, entry_price, sma10_now)
            if do_sell:
                sell_list.append((code, name, price_now, reason))

    # ---- 요약 알림
    if always_send_summary:
        summary = (
            "📬 스캔 요약 (7필수/8강력/매도)\n"
            f"대상: {ok + fail} (성공 {ok}, 실패 {fail})\n"
            f"매수: {len(buy_list)}개, 강력매수: {len(strong_list)}개, 매도: {len(sell_list)}개\n"
            f"시각: {ts.strftime('%Y-%m-%d %H:%M:%S')} KST"
        )
        print(summary)
        send_telegram(summary)

    # ---- 개별 알림
    for code, name in buy_list:
        msg = f"📈 매수신호(7조건 충족)\n{name}({code})"
        print(msg)
        send_telegram(msg)

    for code, name in strong_list:
        msg = f"🚀 강력매수(8번 단독 충족: 횡보 후 첫 장대양봉)\n{name}({code})"
        print(msg)
        send_telegram(msg)

    for code, name, price_now, reason in sell_list:
        msg = (f"🔻 매도신호 ({reason})\n"
               f"{name}({code}) • 현재가: {price_now:,.0f}")
        print(msg)
        send_telegram(msg)


# ================================
# 진입 포지션 기록(선택)
#  - 사용자가 실제 매수 체결 시, 아래 함수를 수동 호출해
#    positions.csv에 기록해두면 매도 신호 판정에 사용됩니다.
# ================================
def record_entry(positions_csv: str, code: str, name: str, entry_price: float, shares: int = 0):
    df = load_positions(positions_csv)
    new_row = {
        "code": str(code),
        "name": name,
        "entry_date": dt.date.today().strftime("%Y-%m-%d"),
        "entry_price": float(entry_price),
        "shares": int(shares),
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    save_positions(df, positions_csv)
    print(f"[positions] 추가: {new_row}")


# ================================
# 실행부
# ================================
if __name__ == "__main__":
    # 기본값: 같은 폴더의 kospi200.csv / positions.csv 사용
    uni_csv = os.getenv("UNIVERSE_CSV", "kospi200.csv")
    pos_csv = os.getenv("POSITIONS_CSV", "positions.csv")

    # 요약은 이 스크립트를 **시간마다** 실행하면 매 실행마다 1회 전송됨
    run_scan(
        universe_csv=uni_csv,
        positions_csv=pos_csv,
        lookback_days=300,
        always_send_summary=True
    )

















