import os
import pandas as pd
import datetime as dt
import yfinance as yf
from strategy import compute_indicators, entry_signal, exit_signal

# =========================
# 포지션 로드/저장 유틸
# =========================
def load_positions(path: str) -> pd.DataFrame:
    """현재 보유 포지션을 CSV에서 불러오기"""
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if df.empty:
                return pd.DataFrame(columns=["code","name","entry_date","entry_price","atr_entry","shares"])
            return df
        except Exception as e:
            print(f"[positions] 로드 실패: {e}")
            return pd.DataFrame(columns=["code","name","entry_date","entry_price","atr_entry","shares"])
    else:
        return pd.DataFrame(columns=["code","name","entry_date","entry_price","atr_entry","shares"])

def save_positions(df: pd.DataFrame, path: str):
    """보유 포지션을 CSV에 저장"""
    try:
        df.to_csv(path, index=False)
    except Exception as e:
        print(f"[positions] 저장 실패: {e}")

# =========================
# 보조 함수
# =========================
def now_kst():
    return dt.datetime.utcnow() + dt.timedelta(hours=9)

def inside_market_hours(cfg):
    ts = now_kst().time()
    start = dt.datetime.strptime(cfg["market_hours"]["start_kst"], "%H:%M").time()
    end   = dt.datetime.strptime(cfg["market_hours"]["end_kst"], "%H:%M").time()
    return start <= ts <= end

def should_send_summary(ts, every_min=60):
    """요약 알림 주기 체크 (기본: 매시 정각)"""
    return (ts.minute % every_min == 0)

# =========================
# 알림 (텔레그램/ntfy)
# =========================
import requests

def _notify(msg, use_tg, use_ntfy, tg_token, tg_chat_id, ntfy_url):
    print(msg)  # 콘솔에도 출력
    if use_tg and tg_token and tg_chat_id:
        try:
            url = f"https://api.telegram.org/bot{tg_token}/sendMessage"
            requests.post(url, data={"chat_id": tg_chat_id, "text": msg})
        except Exception as e:
            print(f"[telegram] 실패: {e}")
    if use_ntfy and ntfy_url:
        try:
            requests.post(ntfy_url, data=msg.encode("utf-8"))
        except Exception as e:
            print(f"[ntfy] 실패: {e}")

# =========================
# 메인 스캔 함수
# =========================
def scan_once(cfg):
    ts = now_kst()
    market_open = inside_market_hours(cfg)

    # 보유 포지션 불러오기
    pos = load_positions(cfg["positions_csv"])

    # ✅ 여기서 universe_csv 불러오기
    try:
        uni = pd.read_csv(cfg["universe_csv"])
    except Exception as e:
        print(f"[universe] 로드 실패: {e}")
        return

    buy_candidates = []
    sell_candidates = []

    for _, row in uni.iterrows():
        code = str(row["종목코드"]).zfill(6)
        name = row["종목명"]

        try:
            ticker = yf.Ticker(f"{code}.KS")
            df = ticker.history(period="6mo")
            if df.empty: 
                continue

            df = df.rename(columns={"Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
            ind = compute_indicators(df, lookback=cfg["lookback"])

            last = ind.iloc[-1]

            # 매수 조건 체크
            if entry_signal(last,
                            buffer=cfg["entry"]["buffer"],
                            require_ma_trend=cfg["entry"]["require_ma_trend"],
                            rs_min=cfg["filters"]["rs_min"]):
                buy_candidates.append((code, name, last, 10))  # shares는 간단히 10개 예시

            # 매도 조건 체크 (보유중인 경우만)
            if code in pos["code"].values:
                entry_price = pos.loc[pos["code"]==code,"entry_price"].values[0]
                atr_entry   = pos.loc[pos["code"]==code,"atr_entry"].values[0]
                shares      = pos.loc[pos["code"]==code,"shares"].values[0]

                if exit_signal(last["close"], entry_price, atr_entry, last["SMA20"], 
                               use_ma=cfg["exit"]["ma_exit"], stop_atr_multiple=cfg["exit"]["stop_atr_multiple"]):
                    sell_candidates.append((code, name, last["close"], "EXIT"))

        except Exception as e:
            print(f"[{code}] 조회 실패: {e}")

    # ===== 알림 처리 =====
    use_tg   = cfg["telegram"]["enabled"]
    use_ntfy = cfg["ntfy"]["enabled"]

    # 매수 신호
    for code, name, last, shares in buy_candidates:
        msg = f"📈 매수 신호: {name}({code}) @ {last['close']:.2f}"
        _notify(msg, use_tg, use_ntfy, cfg["telegram"]["token_env"], cfg["telegram"]["chat_id_env"], cfg["ntfy"]["url_env"])
        new_row = {
            "code": code, "name": name,
            "entry_date": ts.strftime("%Y-%m-%d"),
            "entry_price": float(last["close"]),
            "atr_entry": float(last["ATR14"]),
            "shares": shares
        }
        pos = pd.concat([pos, pd.DataFrame([new_row])], ignore_index=True)

    # 매도 신호
    for code, name, price_now, reason in sell_candidates:
        msg = f"📉 매도 신호: {name}({code}) @ {price_now:.2f} ({reason})"
        _notify(msg, use_tg, use_ntfy, cfg["telegram"]["token_env"], cfg["telegram"]["chat_id_env"], cfg["ntfy"]["url_env"])
        pos = pos[pos["code"] != code]

    # 저장
    save_positions(pos, cfg["positions_csv"])

    # 요약
    if market_open and should_send_summary(ts, cfg["notifications"]["summary_every_min"]):
        summary = (f"📬 스캔 요약\n"
                   f"대상: {len(uni)}개\n"
                   f"매수 신호: {len(buy_candidates)}개\n"
                   f"매도 신호: {len(sell_candidates)}개\n"
                   f"시각: {ts.strftime('%Y-%m-%d %H:%M:%S')} KST")
        _notify(summary, use_tg, use_ntfy, cfg["telegram"]["token_env"], cfg["telegram"]["chat_id_env"], cfg["ntfy"]["url_env"])

# =========================
# main
# =========================
import yaml, argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="live", help="live/eod")
    args = parser.parse_args()

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.mode == "eod":
        scan_once(cfg)
    else:
        # 실시간 모드라면 반복 실행
        while True:
            scan_once(cfg)
            import time; time.sleep(300)

if __name__ == "__main__":
    main()





















