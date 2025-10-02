import os
import pandas as pd
from datetime import datetime, timedelta
from strategy import entry_signal, compute_indicators, _weekly_from_daily
from utils import _notify, load_universe, fetch_daily_df, save_positions

# ---------------------------------------------------
# 주요 실행 루프
# ---------------------------------------------------
def scan_once(cfg):
    ts = datetime.now()  # 현재 시각
    uni = load_universe(cfg["universe_csv"])  # 종목 universe 로드

    buy_candidates = []
    strong_buy_candidates = []
    sell_candidates = []
    pos = pd.read_csv(cfg["positions_csv"]) if os.path.exists(cfg["positions_csv"]) else pd.DataFrame(columns=["code","name"])

    for _, row in uni.iterrows():
        code = str(row["code"])
        name = row["name"]

        # 일봉 데이터 가져오기
        end = ts.strftime("%Y%m%d")
        start = (ts - timedelta(days=365*2)).strftime("%Y%m%d")
        df = fetch_daily_df(code, start, end)
        if df is None or df.empty:
            continue

        # 주봉 변환
        wdf = _weekly_from_daily(df)

        # 지표 계산
        df = compute_indicators(df)

        # 진입 신호 판정
        sig = entry_signal(df, weekly_df=wdf)

        if sig == "buy":
            buy_candidates.append((code, name, df))
        elif sig == "strong_buy":
            strong_buy_candidates.append((code, name, df))

        # (추가: 매도조건 로직 있으면 여기서 sell_candidates 채움)

    # ---------------------------------------------------
    # 알림 처리
    # ---------------------------------------------------
    if buy_candidates:
        for code, name, df in buy_candidates:
            msg = f"📈 매수 신호: {name} ({code}) - 7개 규칙 충족"
            _notify(msg, cfg["telegram"]["enabled"], cfg["ntfy"]["enabled"],
                    cfg["telegram"]["token_env"], cfg["telegram"]["chat_id_env"], cfg["ntfy"]["url_env"])

    if strong_buy_candidates:
        for code, name, df in strong_buy_candidates:
            msg = f"🚀 강력 매수 신호: {name} ({code}) - 횡보 후 첫 장대양봉"
            _notify(msg, cfg["telegram"]["enabled"], cfg["ntfy"]["enabled"],
                    cfg["telegram"]["token_env"], cfg["telegram"]["chat_id_env"], cfg["ntfy"]["url_env"])

    # (매도 알림도 필요시 추가)
    if sell_candidates:
        for code, name, price_now, reason in sell_candidates:
            msg = f"⚠️ 매도 신호: {name} ({code}) - {reason}"
            _notify(msg, cfg["telegram"]["enabled"], cfg["ntfy"]["enabled"],
                    cfg["telegram"]["token_env"], cfg["telegram"]["chat_id_env"], cfg["ntfy"]["url_env"])

    # 포지션 저장
    save_positions(pos, cfg["positions_csv"])


def main():
    import yaml
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    scan_once(cfg)


if __name__ == "__main__":
    main()
