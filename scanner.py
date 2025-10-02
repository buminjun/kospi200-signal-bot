# scanner.py
import os
import sys
import json
import time
import pandas as pd
import yfinance as yf
import requests
from datetime import datetime, timedelta
from strategy import compute_indicators, check_rules, check_strong_buy

# =============== 기본 유틸 ===============
def now_kst():
    return datetime.utcnow() + timedelta(hours=9)

def load_universe(path="kospi200.csv"):
    # code / 종목코드 중 하나 존재하면 사용
    df = pd.read_csv(path, dtype=str)
    cols = [c.strip() for c in df.columns]
    df.columns = cols
    if "code" in cols:
        df["code"] = df["code"].str.zfill(6)
        df["name"] = df.get("name", df.get("종목명", df["code"]))
    elif "종목코드" in cols:
        df["code"] = df["종목코드"].str.zfill(6)
        df["name"] = df.get("종목명", df["code"])
    else:
        raise KeyError("CSV에 'code' 또는 '종목코드' 컬럼이 필요합니다.")
    return df[["code","name"]]

def fetch_price(code, years=2):
    """
    yfinance로 한국 종목 일봉 다운로드 (코스피 .KS / 코스닥 .KQ 자동 시도)
    """
    for suffix in [".KS", ".KQ"]:
        ticker = f"{code}{suffix}"
        try:
            df = yf.download(ticker, period=f"{years}y", auto_adjust=False, progress=False)
            if not df.empty:
                df = df.rename(columns=str.lower)
                # DatetimeIndex 보장
                df.index = pd.to_datetime(df.index)
                return df
        except Exception as e:
            print(f"[WARN] {code}{suffix} 다운로드 실패: {e}", file=sys.stderr)
            time.sleep(0.3)
    return pd.DataFrame()

# =============== 텔레그램 알림 ===============
def send_telegram(msg, token_env="TELEGRAM_BOT_TOKEN", chat_env="TELEGRAM_CHAT_ID"):
    token = os.getenv(token_env)
    chat_id = os.getenv(chat_env)
    if not token or not chat_id:
        print(f"[TG-SKIP] 환경변수 미설정: {token_env}/{chat_env}. 메시지:\n{msg}")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": msg,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            print(f"[TG-ERR] {r.status_code} {r.text}")
            return False
        return True
    except Exception as e:
        print(f"[TG-EXC] {e}")
        return False

def notify(msg, cfg):
    if cfg.get("telegram", {}).get("enabled", True):
        send_telegram(
            msg,
            cfg["telegram"].get("token_env", "TELEGRAM_BOT_TOKEN"),
            cfg["telegram"].get("chat_id_env", "TELEGRAM_CHAT_ID"),
        )
    else:
        print(msg)

# =============== 메시지 포맷 ===============
def fmt_buy(ts, code, name):
    return f"📈 <b>매수 신호</b> — {name} ({code})\n⏱ {ts.strftime('%Y-%m-%d %H:%M')} KST\n조건: 7개 규칙 전부 충족"

def fmt_strong(ts, code, name):
    return f"🚀 <b>강력 매수 신호</b> — {name} ({code})\n⏱ {ts.strftime('%Y-%m-%d %H:%M')} KST\n조건: 횡보 후 첫 장대양봉(8번만 단독 충족)"

# =============== 스캔 메인 ===============
def scan(cfg):
    ts = now_kst()
    uni = load_universe(cfg["universe_csv"])

    buy_signals = []
    strong_signals = []

    for _, r in uni.iterrows():
        code, name = r["code"], r["name"]

        df = fetch_price(code, years=2)
        if df.empty or len(df) < 252:
            continue

        ind = compute_indicators(df)

        # 7개 규칙 (전부 충족 시 True)
        ok7, _rules = check_rules(ind)

        if ok7:
            buy_signals.append((code, name))
        else:
            # 8번(횡보 후 첫 장대양봉)만 충족 시 강력매수
            if check_strong_buy(ind):
                strong_signals.append((code, name))

    # === 알림 ===
    for code, name in buy_signals:
        notify(fmt_buy(ts, code, name), cfg)

    for code, name in strong_signals:
        notify(fmt_strong(ts, code, name), cfg)

    # 콘솔에도 요약 출력
    print(json.dumps({
        "ts": ts.isoformat(),
        "buy_count": len(buy_signals),
        "strong_count": len(strong_signals)
    }, ensure_ascii=False))

# =============== 실행 진입 ===============
if __name__ == "__main__":
    # config.yaml 지원 (없으면 기본값)
    cfg = {
        "universe_csv": "kospi200.csv",
        "telegram": {
            "enabled": True,
            "token_env": "TELEGRAM_BOT_TOKEN",
            "chat_id_env": "TELEGRAM_CHAT_ID",
        }
    }
    # 외부 config.yaml 있으면 덮어쓰기
    if os.path.exists("config.yaml"):
        import yaml
        with open("config.yaml", "r", encoding="utf-8") as f:
            cfg_file = yaml.safe_load(f) or {}
        # 얕은 병합
        for k, v in cfg_file.items():
            if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                cfg[k].update(v)
            else:
                cfg[k] = v

    scan(cfg)
















