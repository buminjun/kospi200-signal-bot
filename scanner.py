# scanner.py
import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from strategy import compute_indicators, check_rules, check_strong_buy

# =========================
# 유틸 함수
# =========================
def now_kst():
    return datetime.utcnow() + timedelta(hours=9)

def load_universe(path="kospi200.csv"):
    df = pd.read_csv(path, dtype=str)
    if "종목코드" in df.columns:
        df["code"] = df["종목코드"].str.zfill(6)
    elif "code" not in df.columns:
        raise KeyError("CSV에 '종목코드' 또는 'code' 컬럼이 필요합니다.")
    return df

def fetch_price(code, years=2):
    """
    yfinance에서 한국 종목 데이터 다운로드
    """
    ticker = f"{code}.KS"
    try:
        df = yf.download(ticker, period=f"{years}y")
        df = df.rename(columns=str.lower)
        return df
    except Exception as e:
        print(f"[Error] {code}: {e}")
        return pd.DataFrame()

# =========================
# 알림 포맷
# =========================
def format_buy_msg(ts, code, name, strong=False):
    if strong:
        return f"🚀 강력매수 신호 [{code} {name}] @ {ts.strftime('%Y-%m-%d %H:%M')}"
    else:
        return f"📈 매수 신호 [{code} {name}] @ {ts.strftime('%Y-%m-%d %H:%M')}"

# =========================
# 메인 스캐너
# =========================
def scan(cfg):
    ts = now_kst()
    uni = load_universe(cfg["universe_csv"])

    buy_signals = []
    strong_signals = []

    for _, row in uni.iterrows():
        code = row["code"]
        name = row.get("종목명", code)

        df = fetch_price(code, years=2)
        if df.empty or len(df) < 250:
            continue

        ind = compute_indicators(df)

        # --- 7개 규칙 체크
        ok, rules = check_rules(ind)
        if ok:
            buy_signals.append((code, name))
            continue

        # --- 강력매수 체크 (장대양봉)
        if check_strong_buy(ind):
            strong_signals.append((code, name))

    # =========================
    # 결과 알림
    # =========================
    if buy_signals:
        for code, name in buy_signals:
            print(format_buy_msg(ts, code, name, strong=False))

    if strong_signals:
        for code, name in strong_signals:
            print(format_buy_msg(ts, code, name, strong=True))

# =========================
# 실행
# =========================
if __name__ == "__main__":
    cfg = {
        "universe_csv": "kospi200.csv"
    }
    scan(cfg)















