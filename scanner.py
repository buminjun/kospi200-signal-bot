# scanner.py
# -*- coding: utf-8 -*-
import os
import sys
import time
import json
import math
import pytz
import yaml
import requests
import warnings
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from pykrx import stock as krx

warnings.filterwarnings("ignore", category=FutureWarning)

# =========================================================
# 유틸: 시간/알림
# =========================================================
KST = pytz.timezone("Asia/Seoul")

def now_kst():
    return datetime.now(KST)

def inside_market_hours(cfg):
    ts = now_kst()
    st = cfg.get("market_hours", {}).get("start_kst", "09:00")
    en = cfg.get("market_hours", {}).get("end_kst", "15:30")
    sh, sm = [int(x) for x in st.split(":")]
    eh, em = [int(x) for x in en.split(":")]
    sdt = ts.replace(hour=sh, minute=sm, second=0, microsecond=0)
    edt = ts.replace(hour=eh, minute=em, second=0, microsecond=0)
    return sdt <= ts <= edt

def _env(name, default=""):
    v = os.getenv(name, default)
    return v if v is not None else default

def send_telegram(text, token_env, chat_id_env):
    token = _env(token_env)
    chat_id = _env(chat_id_env)
    if not token or not chat_id:
        return False
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        resp = requests.post(url, data={"chat_id": chat_id, "text": text})
        return resp.ok
    except Exception:
        return False

def send_ntfy(text, url_env):
    url = _env(url_env)
    if not url:
        return False
    try:
        resp = requests.post(url, data=text.encode("utf-8"))
        return resp.ok
    except Exception:
        return False

def _notify(text, use_tg, use_ntfy, token_env, chat_id_env, ntfy_url_env):
    ok = False
    if use_tg:
        ok |= send_telegram(text, token_env, chat_id_env)
    if use_ntfy:
        ok |= send_ntfy(text, ntfy_url_env)
    return ok

# =========================================================
# 유틸: 데이터 로딩 (pykrx)
# =========================================================
def _yyyymmdd(d):
    return d.strftime("%Y%m%d")

def _to_ohlcv_lower(df):
    out = pd.DataFrame({
        "open":  df["시가"].astype(float),
        "high":  df["고가"].astype(float),
        "low":   df["저가"].astype(float),
        "close": df["종가"].astype(float),
        "volume":df["거래량"].astype(float)
    })
    out.index = pd.to_datetime(out.index)
    out = out.sort_index()
    return out

def fetch_daily_df(code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame | None:
    from pykrx import stock
    s = start.strftime("%Y%m%d")
    e = end.strftime("%Y%m%d")
    last_err = None
    for i in range(3):  # 최대 3회 재시도
        try:
            df = stock.get_market_ohlcv_by_date(s, e, code)
            if df is None or df.empty:
                last_err = RuntimeError("empty")
                time.sleep(0.5)
                continue
            df = df.rename(columns={
                "시가":"open", "고가":"high", "저가":"low",
                "종가":"close", "거래량":"volume"
            })
            df.index = pd.to_datetime(df.index)
            # 숫자형 강제
            for c in ["open","high","low","close","volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["open","high","low","close"])
            if df.empty:
                last_err = RuntimeError("nan-only")
                time.sleep(0.5)
                continue
            return df
        except Exception as e:
            last_err = e
            time.sleep(0.8)
    print(f"[pykrx] {code} 조회 실패 → {last_err}")
    return None


# --- [PATCH] KOSPI 벤치마크 종가 (RS용) ---
def fetch_kospi_close(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series | None:
    from pykrx import stock
    s = start.strftime("%Y%m%d")
    e = end.strftime("%Y%m%d")
    try:
        # KOSPI 지수코드 = '1001'
        idx = stock.get_index_ohlcv_by_date(s, e, "1001")
        if idx is None or idx.empty:
            return None
        idx.index = pd.to_datetime(idx.index)
        close = idx["종가"].astype(float)
        close.name = "KOSPI"
        return close
    except Exception as e:
        print(f"[pykrx] KOSPI 지수 조회 실패 → {e}")
        return None

# =========================================================
# 유틸: CSV 유니버스/포지션
# =========================================================
def load_universe(csv_path: str) -> pd.DataFrame:
    import pandas as _pd
    # 다양한 인코딩 시도
    for enc in ["utf-8-sig", "cp949", "utf-8"]:
        try:
            df = _pd.read_csv(csv_path, sep=None, engine="python", encoding=enc)
            break
        except Exception:
            continue
    else:
        # 전부 실패 시 탭 구분자로 시도
        df = _pd.read_csv(csv_path, sep="\t", engine="python")

    # 공백정리 (applymap 경고 회피: DataFrame.map 사용)
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    cols = [c.strip() for c in df.columns]
    df.columns = cols

    # 종목코드 칼럼 탐색
    cand = None
    for c in ["code", "종목코드", "티커", "ticker", "symbol"]:
        if c in df.columns:
            cand = c
            break
    if cand is None:
        # 첫번째 열을 코드로 가정 (탭/스페이스로 붙여넣은 경우 대비)
        cand = df.columns[0]

    # 6자리만 추출
    codes = (
        df[cand].astype(str)
        .str.extract(r"(\d{6})", expand=False)
        .dropna()
        .str.zfill(6)
    )
    out = _pd.DataFrame({"code": codes})
    # 이름(있으면)도 함께
    name_col = None
    for c in ["name", "종목명"]:
        if c in df.columns:
            name_col = c
            break
    if name_col:
        out["name"] = df[name_col].astype(str)
    else:
        out["name"] = ""

    # 중복/결측 제거
    out = out.dropna().drop_duplicates(subset=["code"]).reset_index(drop=True)
    return out


# =========================================================
# 지표/규칙 계산
# =========================================================
def sma(s, w):
    return s.rolling(w, min_periods=w).mean()

def atr(df, period=14):
    h_l  = df["high"] - df["low"]
    h_pc = (df["high"] - df["close"].shift()).abs()
    l_pc = (df["low"]  - df["close"].shift()).abs()
    tr = pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()

def weekly_frame(df):
    """주봉(금요일 기준) OHLCV"""
    w = pd.DataFrame()
    w["open"] = df["open"].resample("W-FRI").first()
    w["high"] = df["high"].resample("W-FRI").max()
    w["low"]  = df["low"].resample("W-FRI").min()
    w["close"]= df["close"].resample("W-FRI").last()
    w["volume"] = df["volume"].resample("W-FRI").sum()
    w = w.dropna()
    return w

def trend_higher_high_low(df, look=20):
    """
    최근 2구간(각각 look일) 고점/저점 비교: HH/HL 모두 상승이면 True
    """
    if len(df) < look*3:
        return False
    seg1 = df.iloc[-look:]
    seg2 = df.iloc[-2*look:-look]
    hh1 = seg1["high"].max()
    ll1 = seg1["low"].min()
    hh2 = seg2["high"].max()
    ll2 = seg2["low"].min()
    return (hh1 > hh2) and (ll1 > ll2)

def strong_bull_after_consolidation(df, base_window=20, width=0.10, body_pct=0.02):
    """
    8번: 횡보 후 첫 장대양봉
    - 횡보: 최근 base_window 동안 (최고-최저)/최저 <= width (예: 10%)
    - 오늘 장대양봉: close>open, (close-open)/open >= body_pct(>=2%), 그리고 직전 최고가 돌파
    """
    if len(df) < base_window + 1:
        return False
    recent = df.iloc[-base_window:]
    range_pct = (recent["high"].max() - recent["low"].min()) / max(1e-9, recent["low"].min())
    if range_pct > width:
        return False
    today = df.iloc[-1]
    prev  = df.iloc[:-1]
    if any(pd.isna(today[["open","close","high"]])) or prev.empty:
        return False
    is_bull = today["close"] > today["open"]
    big_body = (today["close"] - today["open"]) / today["open"] >= body_pct
    breakout = today["close"] >= prev["high"].rolling(20, min_periods=1).max().iloc[-1]
    near_high = today["close"] >= today["high"] * 0.98
    return bool(is_bull and big_body and breakout and near_high)

def compute_indicators(df):
    out = df.copy()
    out["SMA10"]  = sma(out["close"], 10)
    out["SMA20"]  = sma(out["close"], 20)
    out["SMA50"]  = sma(out["close"], 50)
    out["SMA150"] = sma(out["close"], 150)
    out["SMA200"] = sma(out["close"], 200)
    out["ATR14"]  = atr(out, 14)
    out["LOW_52W"]= out["low"].rolling(252, min_periods=252).min()
    return out

def check_7_rules(df):
    """
    7가지 필수 규칙 모두 True면 (True, detail)
    detail: dict로 각 규칙 True/False 포함
    규칙:
      1) 종가가 150일선 또는 200일선 위
      2) 150일선 > 200일선
      3) 200일선 상승(오늘 SMA200 > 5영업일 전 SMA200)
      4) 고점과 저점이 연이어 높아짐(최근 2구간 HH/HL 상승)
      5) 주봉 상승 시 거래량 증가, 하락 시 감소 경향
      6) 거래량 수반한 상승 주봉이 하락 주봉보다 많음(최근 12주)
      7) 52주 신저가 대비 주가 +25% 이상
    """
    detail = {i: False for i in range(1, 8)}
    if len(df) < 260:
        return False, detail

    last = df.iloc[-1]
    # 1)
    cond1 = ((last["close"] > last["SMA150"]) or (last["close"] > last["SMA200"]))
    # 2)
    cond2 = (last["SMA150"] > last["SMA200"])
    # 3)
    if df["SMA200"].notna().sum() < 210:
        cond3 = False
    else:
        s = df["SMA200"].dropna()
        cond3 = (s.iloc[-1] > s.iloc[-5]) if len(s) >= 6 else False
    # 4)
    cond4 = trend_higher_high_low(df, look=20)

    # 5) & 6): 주봉
    w = weekly_frame(df)
    cond5 = False
    cond6 = False
    if len(w) >= 13:
        w = w.copy()
        w["dir"] = np.sign(w["close"].diff()).fillna(0)  # +1 up week, -1 down week
        w["vol_chg"] = w["volume"].diff().fillna(0)
        # 최근 12주 체크
        ww = w.iloc[-12:]
        # 5) 상승주( dir>0 )에서 vol_chg>0 비중이 높고, 하락주( dir<0 )에서 vol_chg<0 비중이 높다
        ups = ww[ww["dir"] > 0]
        dns = ww[ww["dir"] < 0]
        up_ok = (len(ups) == 0) or ((ups["vol_chg"] > 0).mean() >= 0.6)  # 60% 이상
        dn_ok = (len(dns) == 0) or ((dns["vol_chg"] < 0).mean() >= 0.6)
        cond5 = bool(up_ok and dn_ok)
        # 6) 거래량 수반한 상승주봉 수 > 하락주봉 수 (상승주봉 중 vol_chg>0 개수 vs 하락주봉 중 vol_chg>0(=나쁨) 개수 비교)
        ups_with_vol = int(((ups["vol_chg"] > 0).sum()) if len(ups) else 0)
        dns_with_vol = int(((dns["vol_chg"] > 0).sum()) if len(dns) else 0)
        # 상승주봉에서 거래량 동반(+) 수가, 하락주봉에서 거래량 동반(+) 수보다 많다 → 상승 쪽이 우세
        cond6 = (ups_with_vol > dns_with_vol)

    # 7)
    cond7 = (last["LOW_52W"] > 0) and (last["close"] >= last["LOW_52W"] * 1.25)

    for i, c in enumerate([cond1, cond2, cond3, cond4, cond5, cond6, cond7], start=1):
        detail[i] = bool(c)

    ok_all = all(detail.values())
    return ok_all, detail

def exit_signal(price_now, entry_price, ma10_now):
    """
    매도: 10일선 하락 이탈 OR -5% 하락
    """
    if any(v is None or math.isnan(v) for v in [price_now, entry_price]) or entry_price <= 0:
        return False, None
    cond_ma10 = (ma10_now is not None and not math.isnan(ma10_now) and price_now < ma10_now)
    cond_5pct = (price_now <= entry_price * 0.95)
    reason = None
    if cond_ma10:
        reason = "MA10 하향 이탈"
    if cond_5pct:
        reason = "−5% 손절"
    return bool(cond_ma10 or cond_5pct), reason

# =========================================================
# 메시지 포맷
# =========================================================
def fmt_detail(detail):
    txt = []
    for i in range(1, 8):
        txt.append(f"R{i}:{'✔' if detail.get(i) else '✖'}")
    return " ".join(txt)

def format_buy_msg(ts, code, name, price, kind="매수신호"):
    kst = ts.strftime("%Y-%m-%d %H:%M:%S")
    return f"🟢 {kind}\n{name}({code}) @ {price:,.0f}\n시각: {kst} KST"

def format_sell_msg(ts, code, name, price, reason):
    kst = ts.strftime("%Y-%m-%d %H:%M:%S")
    return f"🔴 매도신호\n{name}({code}) @ {price:,.0f}\n사유: {reason}\n시각: {kst} KST"

def format_summary(ts, uni_n, buy_n, strong_n):
    kst = ts.strftime("%Y-%m-%d %H:%M:%S")
    return (f"📬 스캔 요약\n대상: {uni_n}개\n매수 신호: {buy_n}개\n강력매수: {strong_n}개\n시각: {kst} KST")

def format_pos_summary(ts, pos_df):
    kst = ts.strftime("%Y-%m-%d %H:%M:%S")
    if pos_df is None or pos_df.empty:
        return f"📑 포지션 요약\n보유: 0\n시각: {kst} KST"
    lines = [f"📑 포지션 요약 (보유 {len(pos_df)}개) – {kst} KST"]
    for _, r in pos_df.iterrows():
        lines.append(f"- {r.get('name','?')}({r['code']}) @ {float(r['entry_price']):,.0f} x {int(r['shares'])}")
    return "\n".join(lines)

# 요약 주기 판단
_last_summary_minute = {"general": None, "position": None}
def should_send_summary(ts, every_min, kind="general"):
    if every_min is None or every_min <= 0:
        return False
    last_min = _last_summary_minute.get(kind)
    cur_slot = (ts.minute // every_min)
    key = (ts.hour, cur_slot)
    if key != last_min:
        _last_summary_minute[kind] = key
        return True
    return False

# =========================================================
# 스캐너 메인 로직
# =========================================================
def scan_once(cfg):
    ts = now_kst()

    uni = load_universe(cfg["universe_csv"])
    pos = load_positions(cfg["positions_csv"])

    # 데이터 기간: 규칙 계산 + 52주 최저 + 여유
    lookback_days = max(260, int(cfg.get("lookback", 260))) + 30
    start = (ts - timedelta(days=lookback_days)).date()
    end   = ts.date()

    buy_candidates = []
    strong_candidates = []

    for _, row in uni.iterrows():
        code = row["code"]
        name = row.get("name", code)

        try:
            df = fetch_daily_df(code, start, end)
            if df is None or df.empty:
                continue
            df = compute_indicators(df)
            if df["SMA200"].isna().all():
                continue

            # 7개 규칙
            ok7, detail = check_7_rules(df)
            last = df.iloc[-1]

            # 8번 별도(횡보 후 첫 장대양봉)
            strong8 = strong_bull_after_consolidation(df, base_window=20, width=0.10, body_pct=0.02)

            # 현재가
            price_now = float(last["close"])

            if ok7:
                buy_candidates.append((code, name, price_now, detail))
            elif strong8:
                # 8번만 충족 시 → 강력매수
                strong_candidates.append((code, name, price_now, {"8": True}))

        except Exception as e:
            # 종목 단위 예외는 그냥 스킵
            print(f"[WARN] {code} {name} 처리 실패 → {e}")

    # 알림 채널
    use_tg   = cfg.get("telegram", {}).get("enabled", False)
    use_ntfy = cfg.get("ntfy", {}).get("enabled", False)

    # 포지션 로직(장중에만)
    if inside_market_hours(cfg):
        # 매도 먼저
        closed_codes = []
        if not pos.empty:
            # 최신 가격/지표 다시 가져와 매도 판단
            for i, r in pos.iterrows():
                pcode = r["code"]
                pname = r["name"]
                entry_price = float(r["entry_price"])
                try:
                    dfp = fetch_daily_df(pcode, start, end)
                    if dfp is None or dfp.empty:
                        continue
                    dfp = compute_indicators(dfp)
                    lastp = dfp.iloc[-1]
                    ma10  = float(lastp["SMA10"]) if pd.notna(lastp["SMA10"]) else None
                    price_now = float(lastp["close"])
                    go_exit, reason = exit_signal(price_now, entry_price, ma10)
                    if go_exit:
                        msg = format_sell_msg(ts, pcode, pname, price_now, reason)
                        _notify(msg, use_tg, use_ntfy,
                                cfg["telegram"]["token_env"], cfg["telegram"]["chat_id_env"], cfg["ntfy"]["url_env"])
                        closed_codes.append(pcode)
                except Exception as e:
                    print(f"[WARN] 매도체크 실패 {pcode} → {e}")
        if closed_codes:
            pos = pos[~pos["code"].isin(closed_codes)]

        # 매수 (7개 규칙 충족 우선, 자리 남으면 강력매수도)
        capacity = max(cfg.get("max_positions", 5) - len(pos), 0)

        if capacity > 0 and buy_candidates:
            for code, name, price_now, detail in buy_candidates[:capacity]:
                msg = format_buy_msg(ts, code, name, price_now, kind="매수신호(7규칙 충족)")
                msg += "\n" + fmt_detail(detail)
                _notify(msg, use_tg, use_ntfy,
                        cfg["telegram"]["token_env"], cfg["telegram"]["chat_id_env"], cfg["ntfy"]["url_env"])
                new_row = {
                    "code": code, "name": name,
                    "entry_date": ts.strftime("%Y-%m-%d"),
                    "entry_price": float(price_now),
                    "shares": 0  # 수동 체결이니 수량은 0으로 기록(원하면 계산해 넣어도 됨)
                }
                pos = pd.concat([pos, pd.DataFrame([new_row])], ignore_index=True)
                capacity -= 1
                if capacity <= 0:
                    break

        # 자리 남아있으면 강력매수도 일부 채택(옵션)
        if capacity > 0 and strong_candidates:
            for code, name, price_now, _ in strong_candidates[:capacity]:
                msg = format_buy_msg(ts, code, name, price_now, kind="⚡강력매수(8번 충족)")
                _notify(msg, use_tg, use_ntfy,
                        cfg["telegram"]["token_env"], cfg["telegram"]["chat_id_env"], cfg["ntfy"]["url_env"])
                new_row = {
                    "code": code, "name": name,
                    "entry_date": ts.strftime("%Y-%m-%d"),
                    "entry_price": float(price_now),
                    "shares": 0
                }
                pos = pd.concat([pos, pd.DataFrame([new_row])], ignore_index=True)

    # 포지션 저장
    save_positions(pos, cfg["positions_csv"])

    # --- 요약 알림 (장중 매시/또는 FORCE_SUMMARY=1) ---
    summary_every = int(cfg.get("notifications", {}).get("summary_every_min", 60))
    force_summary = os.getenv("FORCE_SUMMARY", "0") == "1"

    if (inside_market_hours(cfg) and should_send_summary(ts, summary_every, kind="general")) or force_summary:
        summary = format_summary(ts, len(uni), len(buy_candidates), len(strong_candidates))
        _notify(summary, use_tg, use_ntfy,
                cfg["telegram"]["token_env"], cfg["telegram"]["chat_id_env"], cfg["ntfy"]["url_env"])

    if (inside_market_hours(cfg) and should_send_summary(ts, summary_every, kind="position")) or force_summary:
        psum = format_pos_summary(ts, pos)
        _notify(psum, use_tg, use_ntfy,
                cfg["telegram"]["token_env"], cfg["telegram"]["chat_id_env"], cfg["ntfy"]["url_env"])

    # 로그 (액션 로그에서 확인 편의)
    print(json.dumps({
        "ts": ts.isoformat(),
        "universe": len(uni),
        "buy_7rules": len(buy_candidates),
        "strong_8only": len(strong_candidates),
        "positions_after": 0 if pos is None else len(pos)
    }, ensure_ascii=False))

# =========================================================
# 진입점
# =========================================================
def main():
    cfg_path = "config.yaml"
    if not os.path.exists(cfg_path):
        print("config.yaml 파일이 없습니다.", file=sys.stderr)
        sys.exit(1)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    mode = "eod"
    if len(sys.argv) >= 3 and sys.argv[1] in ("--mode","-m"):
        mode = sys.argv[2].strip().lower()

    if mode == "eod":
        scan_once(cfg)
    elif mode == "loop":
        print("[LOOP] 시작. 장중 시간에만 동작합니다.")
        while True:
            try:
                if inside_market_hours(cfg):
                    scan_once(cfg)
                else:
                    print("[LOOP] 장 외 시간.")
                time.sleep(60)  # 1분 간격
            except KeyboardInterrupt:
                print("중지됨.")
                break
            except Exception as e:
                print(f"[ERR] 루프 오류: {e}")
                time.sleep(10)
    else:
        scan_once(cfg)

if __name__ == "__main__":
    main()



















