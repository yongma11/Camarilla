"""
카마릴라 R4 돌파매매 마스터 (Camarilla Breakout Master)
- SOXL 카마릴라 R4 변형 돌파매매 전략의 일일 주문표 + 백테스트 Streamlit 앱
- 동파법 마스터 v9.8 의 구조(GitHub 저장소 기반 상태 영속화, 3탭 UI)를 참고하여 제작

전략 로직 (검증된 B안 기본값):
  저항선(R) = 전일종가 + (전일고가-전일저가) * 1.1 * coef   [coef=0.70]
  - 장중 고가가 R을 돌파하면 매수 (시가가 이미 R 위라면 시가에 매수)
  - 다음 거래일 시가에 전량 매도 (MOO)
  - 직전 20일 변동성이 과거 분포 상위 (1-vol_filter_pct) 구간이면 그날은 매매 휴식
"""

from __future__ import annotations

import io
import math
from datetime import datetime, date

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from github import Github
    HAS_PYGITHUB = True
except Exception:
    HAS_PYGITHUB = False

# ----------------------------------------------------------------------------
# 설정값 (검증된 B안 = 기본값)
# ----------------------------------------------------------------------------
TICKER = "SOXL"
DEFAULTS = dict(
    total_capital=10_000_000.0,
    coef=0.70,
    fraction=1.0,
    use_vol_filter=True,
    vol_filter_pct=0.80,
    fee_rate=0.0005,      # 편도 0.05%
    slippage_pct=0.0010,  # 편도 0.10%
)

HOLDINGS_PATH = "camarilla_holdings.csv"
JOURNAL_PATH = "camarilla_journal.csv"
EQUITY_PATH = "camarilla_equity.csv"
SETTINGS_PATH = "camarilla_settings.csv"

st.set_page_config(page_title="카마릴라 R4 돌파매매 마스터", layout="wide", page_icon="📈")

# ----------------------------------------------------------------------------
# GitHub 영속화 (secrets에 GH_TOKEN / GH_REPO 가 없으면 세션 임시 저장으로 자동 폴백)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_repo():
    if not HAS_PYGITHUB:
        return None
    try:
        token = st.secrets["GH_TOKEN"]
        repo_name = st.secrets["GH_REPO"]
    except Exception:
        return None
    try:
        g = Github(token)
        return g.get_repo(repo_name)
    except Exception as e:
        st.sidebar.error(f"GitHub 연결 실패: {e}")
        return None


def load_csv(path: str, default_df: pd.DataFrame, state_key: str) -> pd.DataFrame:
    if state_key in st.session_state:
        return st.session_state[state_key]
    repo = get_repo()
    if repo is not None:
        try:
            content = repo.get_contents(path)
            df = pd.read_csv(io.BytesIO(content.decoded_content))
            st.session_state[state_key] = df
            return df
        except Exception:
            pass
    st.session_state[state_key] = default_df.copy()
    return st.session_state[state_key]


def save_csv(path: str, df: pd.DataFrame, message: str, state_key: str):
    st.session_state[state_key] = df.copy()
    repo = get_repo()
    if repo is None:
        return  # 세션 임시 저장만 (GitHub 미연동)
    csv_str = df.to_csv(index=False)
    try:
        content = repo.get_contents(path)
        repo.update_file(path, message, csv_str, content.sha)
    except Exception:
        try:
            repo.create_file(path, message, csv_str)
        except Exception as e:
            st.error(f"GitHub 저장 실패: {e}")


def is_github_connected() -> bool:
    return get_repo() is not None


DEFAULT_HOLDINGS = pd.DataFrame([{
    "in_position": False, "entry_date": "", "entry_price": 0.0, "qty": 0.0
}])
DEFAULT_JOURNAL = pd.DataFrame(columns=[
    "date", "action", "price", "qty", "amount", "fee", "pnl", "equity_after", "note"
])
DEFAULT_EQUITY = pd.DataFrame(columns=["date", "equity", "note"])
DEFAULT_SETTINGS = pd.DataFrame([DEFAULTS])


def get_holdings():
    return load_csv(HOLDINGS_PATH, DEFAULT_HOLDINGS, "holdings_df")


def get_journal():
    return load_csv(JOURNAL_PATH, DEFAULT_JOURNAL, "journal_df")


def get_equity_log():
    return load_csv(EQUITY_PATH, DEFAULT_EQUITY, "equity_df")


def get_settings():
    df = load_csv(SETTINGS_PATH, DEFAULT_SETTINGS, "settings_df")
    row = df.iloc[0].to_dict()
    merged = dict(DEFAULTS)
    merged.update({k: row[k] for k in row if k in merged})
    merged["use_vol_filter"] = bool(merged["use_vol_filter"]) if not isinstance(merged["use_vol_filter"], str) \
        else merged["use_vol_filter"].strip().lower() in ("true", "1", "yes")
    return merged


def save_settings(new_settings: dict):
    save_csv(SETTINGS_PATH, pd.DataFrame([new_settings]), "update settings", "settings_df")


# ----------------------------------------------------------------------------
# 데이터 로딩
# ----------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="SOXL 시세 데이터 불러오는 중...")
def get_price_history(period: str = "max") -> pd.DataFrame:
    raw = yf.download(TICKER, period=period, auto_adjust=True, progress=False)
    if raw is None or len(raw) == 0:
        return pd.DataFrame(columns=["date", "O", "H", "L", "C"])
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    raw = raw.rename(columns={"Date": "date", "Open": "O", "High": "H", "Low": "L", "Close": "C"})
    raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)
    df = raw[["date", "O", "H", "L", "C"]].dropna().sort_values("date").reset_index(drop=True)
    return df


def get_live_price() -> float | None:
    try:
        fi = yf.Ticker(TICKER).fast_info
        p = fi.get("lastPrice") or fi.get("last_price")
        return float(p) if p else None
    except Exception:
        return None


# ----------------------------------------------------------------------------
# 전략 핵심 로직
# ----------------------------------------------------------------------------
def compute_signal_frame(df: pd.DataFrame, coef: float) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    O, H, L, C = out["O"].values, out["H"].values, out["L"].values, out["C"].values
    n = len(out)
    prevH = np.roll(H, 1); prevH[0] = np.nan
    prevL = np.roll(L, 1); prevL[0] = np.nan
    prevC = np.roll(C, 1); prevC[0] = np.nan
    resistance = prevC + (prevH - prevL) * 1.1 * coef

    ret_cc = np.zeros(n)
    ret_cc[1:] = C[1:] / C[:-1] - 1
    vol20 = pd.Series(ret_cc).rolling(20).std().values
    vol_shift = np.roll(vol20, 1)
    vol_shift[0] = np.nan
    vol_rank = pd.Series(vol_shift).expanding(min_periods=60).rank(pct=True).values

    out["resistance"] = resistance
    out["vol20"] = vol20
    out["vol_rank"] = vol_rank
    return out


def run_backtest(df: pd.DataFrame, coef: float, fraction: float = 1.0,
                  vol_filter_pct: float | None = None,
                  fee_rate: float = 0.0, slippage_pct: float = 0.0,
                  start_date=None, end_date=None):
    sig = compute_signal_frame(df, coef)
    n = len(sig)
    O, H = sig["O"].values, sig["H"].values
    resistance = sig["resistance"].values
    vol_rank = sig["vol_rank"].values
    dates = pd.to_datetime(sig["date"].values)
    nextO = np.roll(O, -1); nextO[-1] = np.nan

    signal = (H >= resistance) & ~np.isnan(resistance)
    if vol_filter_pct is not None:
        vol_ok = (vol_rank <= vol_filter_pct) & ~np.isnan(vol_rank)
        signal = signal & vol_ok

    mask = np.ones(n, dtype=bool)
    if start_date is not None:
        mask &= dates >= pd.Timestamp(start_date)
    if end_date is not None:
        mask &= dates <= pd.Timestamp(end_date)

    entry_raw = np.where(O >= resistance, O, resistance)
    entry_eff = entry_raw * (1 + slippage_pct)
    exit_eff = nextO * (1 - slippage_pct)
    gross_ret = exit_eff / entry_eff - 1
    trade_ret = gross_ret - fee_rate * 2

    equity = np.empty(n)
    eq, peak, mdd = 1.0, 1.0, 0.0
    underwater, max_underwater = 0, 0
    trade_dates, trade_rets = [], []
    for i in range(n):
        if mask[i] and signal[i] and not np.isnan(trade_ret[i]):
            r = trade_ret[i] * fraction
            eq *= (1 + r)
            trade_dates.append(dates[i])
            trade_rets.append(trade_ret[i])
        equity[i] = eq
        if eq > peak:
            peak = eq
            underwater = 0
        else:
            underwater += 1
            max_underwater = max(max_underwater, underwater)
        mdd = min(mdd, eq / peak - 1)

    eq_series = pd.Series(equity, index=dates)
    trades = np.array(trade_rets)
    valid_dates = dates[mask]
    years = (valid_dates[-1] - valid_dates[0]).days / 365.25 if len(valid_dates) >= 2 else np.nan
    final_eq = eq
    cagr = final_eq ** (1 / years) - 1 if years and years > 0 and final_eq > 0 else np.nan
    calmar = cagr / abs(mdd) if mdd < 0 and not np.isnan(cagr) else np.nan
    win_rate = (trades > 0).mean() if len(trades) > 0 else np.nan
    avg_win = trades[trades > 0].mean() if (trades > 0).any() else np.nan
    avg_loss = trades[trades <= 0].mean() if (trades <= 0).any() else np.nan

    stats = dict(
        n_trades=len(trades), win_rate=win_rate, avg_win=avg_win, avg_loss=avg_loss,
        cagr=cagr, mdd=mdd, calmar=calmar, max_underwater=max_underwater,
        final_equity=final_eq, years=years,
    )
    trade_log = pd.DataFrame({"date": trade_dates, "trade_ret": trade_rets})
    return stats, eq_series, trade_log, sig


def get_today_order_info(df: pd.DataFrame, coef: float, vol_filter_pct: float | None):
    """가장 마지막으로 완성된 거래일 데이터를 기준으로 '다음 거래일' 매수 기준가/매매가능 여부를 계산"""
    sig = compute_signal_frame(df, coef)
    last = sig.iloc[-1]
    resistance_next = last["C"] + (last["H"] - last["L"]) * 1.1 * coef

    # vol_rank for the NEXT trading day uses vol20 computed through `last` (today),
    # i.e. the expanding percentile rank including the most recent completed bar
    ret_cc = np.zeros(len(sig))
    C = sig["C"].values
    ret_cc[1:] = C[1:] / C[:-1] - 1
    vol20 = pd.Series(ret_cc).rolling(20).std()
    vol_rank_series = vol20.expanding(min_periods=60).rank(pct=True)
    vol_rank_next_val = vol_rank_series.iloc[-1]
    vol_ok = True if vol_filter_pct is None else (
        (not np.isnan(vol_rank_next_val)) and vol_rank_next_val <= vol_filter_pct
    )
    return dict(
        basis_date=last["date"],
        resistance_next=float(resistance_next),
        vol_rank_next=float(vol_rank_next_val) if not np.isnan(vol_rank_next_val) else None,
        vol_ok=bool(vol_ok),
        last_close=float(last["C"]),
        last_high=float(last["H"]),
        last_low=float(last["L"]),
    )


# ----------------------------------------------------------------------------
# UI 헬퍼
# ----------------------------------------------------------------------------
def fmt_pct(x):
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:.2f}%"


def fmt_money(x):
    return f"${x:,.2f}"


# ============================================================================
# 사이드바: 전략 파라미터 설정
# ============================================================================
st.sidebar.title("⚙️ 전략 설정")
if is_github_connected():
    st.sidebar.success("GitHub 저장소 연동됨 — 보유상태/거래일지가 영구 저장됩니다.")
else:
    st.sidebar.warning(
        "GitHub 미연동: secrets에 GH_TOKEN / GH_REPO 를 설정하면 "
        "보유상태/거래일지가 영구 저장됩니다. 지금은 이 브라우저 세션에서만 임시 유지됩니다."
    )

settings = get_settings()

with st.sidebar.form("settings_form"):
    total_capital = st.number_input("총 투자자금($ 또는 원)", min_value=0.0,
                                     value=float(settings["total_capital"]), step=100000.0)
    coef = st.slider("저항선 계수 (coef, R4=0.55, 검증된 추천=0.70)", 0.10, 1.00,
                      float(settings["coef"]), 0.01)
    fraction = st.slider("매수 비중 (자산 대비, B안=100%)", 0.10, 1.00,
                          float(settings["fraction"]), 0.05)
    use_vol_filter = st.checkbox("변동성 필터 사용 (추천)", value=bool(settings["use_vol_filter"]))
    vol_filter_pct = st.slider("변동성 필터 임계값 (상위 N% 회피, B안=80%)", 0.50, 0.99,
                                float(settings["vol_filter_pct"]), 0.01)
    fee_rate = st.number_input("편도 수수료율 (%)", min_value=0.0, value=float(settings["fee_rate"]) * 100,
                                step=0.01, format="%.3f") / 100
    slippage_pct = st.number_input("편도 슬리피지 (%)", min_value=0.0, value=float(settings["slippage_pct"]) * 100,
                                    step=0.01, format="%.3f") / 100
    submitted = st.form_submit_button("설정 저장")
    if submitted:
        new_settings = dict(total_capital=total_capital, coef=coef, fraction=fraction,
                             use_vol_filter=use_vol_filter, vol_filter_pct=vol_filter_pct,
                             fee_rate=fee_rate, slippage_pct=slippage_pct)
        save_settings(new_settings)
        st.sidebar.success("저장 완료")

st.sidebar.caption(
    "검증된 추천값(B안): coef=0.70, 비중=100%, 변동성필터 상위20% 회피. "
    "MDD를 -50%대에서 -25%대로 낮추면서 CAGR은 대부분 보존하는 조합으로, "
    "4구간 분할/워크포워드 검증을 거쳤습니다."
)

eff_vol_filter_pct = vol_filter_pct if use_vol_filter else None

# ============================================================================
# 메인: 3개 탭
# ============================================================================
st.title("📈 카마릴라 R4 돌파매매 마스터 — SOXL")

tab1, tab2, tab3 = st.tabs(["🗓️ 오늘의 주문표", "🧪 백테스트", "📖 전략 로직"])

# ----------------------------------------------------------------------------
# TAB 1: 오늘의 주문표
# ----------------------------------------------------------------------------
with tab1:
    price_df = get_price_history()
    if len(price_df) < 30:
        st.error("시세 데이터를 불러오지 못했습니다. 잠시 후 다시 시도해주세요.")
    else:
        info = get_today_order_info(price_df, coef, eff_vol_filter_pct)
        holdings = get_holdings()
        row0 = holdings.iloc[0]
        in_position = bool(row0["in_position"]) if not isinstance(row0["in_position"], str) \
            else row0["in_position"].strip().lower() in ("true", "1", "yes")

        live_price = get_live_price()

        st.caption(
            f"기준일(마지막 완성 거래일): **{pd.Timestamp(info['basis_date']).date()}** "
            f"(종가 {fmt_money(info['last_close'])}, 고가 {fmt_money(info['last_high'])}, 저가 {fmt_money(info['last_low'])})"
            + (f"  ·  실시간가(참고용): {fmt_money(live_price)}" if live_price else "")
        )

        col1, col2 = st.columns(2)

        # ---- 보유 현황 / 매도 주문 ----
        with col1:
            st.subheader("① 보유 현황 / 매도 주문")
            if in_position:
                st.info(
                    f"**보유중** — 진입일 {row0['entry_date']}, 진입가 {fmt_money(float(row0['entry_price']))}, "
                    f"수량 {row0['qty']}"
                )
                st.markdown("👉 **다음 거래일 시가(MOO)에 전량 매도**하세요.")
                with st.form("sell_form"):
                    sell_date = st.date_input("체결일", value=date.today())
                    sell_price = st.number_input("체결가(시가)", min_value=0.0, value=float(live_price or 0.0))
                    sell_qty = st.number_input("체결수량", min_value=0.0, value=float(row0["qty"]))
                    sell_submit = st.form_submit_button("매도 체결 등록")
                    if sell_submit:
                        amount = sell_price * sell_qty
                        fee = amount * fee_rate
                        entry_amount = float(row0["entry_price"]) * sell_qty
                        pnl = amount - entry_amount - fee
                        journal = get_journal()
                        new_row = pd.DataFrame([{
                            "date": sell_date, "action": "SELL", "price": sell_price, "qty": sell_qty,
                            "amount": amount, "fee": fee, "pnl": pnl, "equity_after": np.nan,
                            "note": f"entry {row0['entry_date']}@{row0['entry_price']}"
                        }])
                        journal = pd.concat([journal, new_row], ignore_index=True)
                        save_csv(JOURNAL_PATH, journal, "sell fill", "journal_df")
                        new_holdings = pd.DataFrame([{
                            "in_position": False, "entry_date": "", "entry_price": 0.0, "qty": 0.0
                        }])
                        save_csv(HOLDINGS_PATH, new_holdings, "close position", "holdings_df")
                        eqlog = get_equity_log()
                        last_eq = float(eqlog["equity"].iloc[-1]) if len(eqlog) else float(total_capital)
                        new_eq = last_eq + pnl
                        eqlog = pd.concat([eqlog, pd.DataFrame(
                            [{"date": sell_date, "equity": new_eq, "note": "sell"}])], ignore_index=True)
                        save_csv(EQUITY_PATH, eqlog, "equity update", "equity_df")
                        st.success(f"매도 체결 등록 완료 (손익 {fmt_money(pnl)}).")
                        st.rerun()
            else:
                st.success("현재 보유 포지션 없음.")

        # ---- 매수 신호 / 관망 ----
        with col2:
            st.subheader("② 다음 매수 신호")
            st.metric("매수 기준가(저항선)", fmt_money(info["resistance_next"]))
            if info["vol_rank_next"] is not None:
                st.caption(f"변동성 백분위: {info['vol_rank_next']*100:.1f}% (필터 임계값 {eff_vol_filter_pct*100 if eff_vol_filter_pct else '미사용'}%)")

            if in_position:
                st.markdown("⏸ 이미 포지션 보유중이므로 신규 매수는 진행하지 않습니다.")
            elif not info["vol_ok"]:
                st.warning("🛑 변동성 필터 발동 — **오늘은 매매 휴식**을 권장합니다.")
            else:
                breakout_now = live_price is not None and live_price >= info["resistance_next"]
                if breakout_now:
                    st.markdown(f"🚀 **이미 기준가 돌파!** 현재가 {fmt_money(live_price)} ≥ 기준가 {fmt_money(info['resistance_next'])} → 매수 진행")
                else:
                    st.markdown(f"👀 **{fmt_money(info['resistance_next'])} 돌파 시 매수** (또는 시가가 이 가격 위로 갭상승 시 시가 매수)")

                suggested_qty = math.floor((total_capital * fraction) / info["resistance_next"]) if info["resistance_next"] > 0 else 0
                st.caption(f"제안 수량 (자산 {fmt_money(total_capital)} × 비중 {fraction*100:.0f}% 기준): 약 {suggested_qty}주")

                with st.form("buy_form"):
                    buy_date = st.date_input("체결일", value=date.today(), key="buy_date")
                    buy_price = st.number_input("체결가", min_value=0.0,
                                                 value=float(live_price or info["resistance_next"]))
                    buy_qty = st.number_input("체결수량", min_value=0.0, value=float(suggested_qty))
                    buy_submit = st.form_submit_button("매수 체결 등록")
                    if buy_submit:
                        amount = buy_price * buy_qty
                        fee = amount * fee_rate
                        journal = get_journal()
                        new_row = pd.DataFrame([{
                            "date": buy_date, "action": "BUY", "price": buy_price, "qty": buy_qty,
                            "amount": amount, "fee": fee, "pnl": np.nan, "equity_after": np.nan,
                            "note": "breakout entry"
                        }])
                        journal = pd.concat([journal, new_row], ignore_index=True)
                        save_csv(JOURNAL_PATH, journal, "buy fill", "journal_df")
                        new_holdings = pd.DataFrame([{
                            "in_position": True, "entry_date": str(buy_date),
                            "entry_price": buy_price, "qty": buy_qty
                        }])
                        save_csv(HOLDINGS_PATH, new_holdings, "open position", "holdings_df")
                        st.success("매수 체결 등록 완료. 다음 거래일에 시가 매도 안내가 표시됩니다.")
                        st.rerun()

        st.divider()
        st.subheader("거래일지")
        journal = get_journal()
        if len(journal):
            st.dataframe(journal.sort_values("date", ascending=False), use_container_width=True)
        else:
            st.caption("아직 기록된 거래가 없습니다.")

# ----------------------------------------------------------------------------
# TAB 2: 백테스트
# ----------------------------------------------------------------------------
with tab2:
    st.subheader("파라미터 백테스트")
    price_df = get_price_history()
    if len(price_df) < 60:
        st.error("시세 데이터를 불러오지 못했습니다.")
    else:
        min_d, max_d = price_df["date"].min().date(), price_df["date"].max().date()
        c1, c2, c3 = st.columns(3)
        with c1:
            bt_coef = st.slider("저항선 계수", 0.10, 1.00, float(coef), 0.01, key="bt_coef")
            bt_fraction = st.slider("매수 비중", 0.10, 1.00, float(fraction), 0.05, key="bt_fraction")
        with c2:
            bt_use_vol = st.checkbox("변동성 필터 사용", value=use_vol_filter, key="bt_use_vol")
            bt_vol_pct = st.slider("변동성 필터 임계값", 0.50, 0.99, float(vol_filter_pct), 0.01, key="bt_vol_pct")
        with c3:
            bt_fee = st.number_input("편도 수수료(%)", min_value=0.0, value=fee_rate * 100, step=0.01,
                                      format="%.3f", key="bt_fee") / 100
            bt_slip = st.number_input("편도 슬리피지(%)", min_value=0.0, value=slippage_pct * 100, step=0.01,
                                       format="%.3f", key="bt_slip") / 100

        d1, d2 = st.columns(2)
        with d1:
            bt_start = st.date_input("시작일", value=min_d, min_value=min_d, max_value=max_d, key="bt_start")
        with d2:
            bt_end = st.date_input("종료일", value=max_d, min_value=min_d, max_value=max_d, key="bt_end")

        if st.button("백테스트 실행", type="primary"):
            stats, eq_series, trade_log, sig = run_backtest(
                price_df, bt_coef, bt_fraction,
                bt_vol_pct if bt_use_vol else None,
                bt_fee, bt_slip, bt_start, bt_end,
            )
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("CAGR", fmt_pct(stats["cagr"]))
            m2.metric("MDD", fmt_pct(stats["mdd"]))
            m3.metric("Calmar", f"{stats['calmar']:.2f}" if not np.isnan(stats["calmar"]) else "-")
            m4.metric("승률", fmt_pct(stats["win_rate"]))
            m5.metric("거래수", f"{stats['n_trades']}")

            n1, n2, n3 = st.columns(3)
            n1.metric("평균 수익 거래", fmt_pct(stats["avg_win"]))
            n2.metric("평균 손실 거래", fmt_pct(stats["avg_loss"]))
            n3.metric("최장 미갱신일수", f"{stats['max_underwater']}일")

            dd_series = eq_series / eq_series.cummax() - 1
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3],
                                 vertical_spacing=0.05, subplot_titles=("자산곡선 (로그스케일, 시작=1.0)", "낙폭(Drawdown)"))
            fig.add_trace(go.Scatter(x=eq_series.index, y=eq_series.values, name="Equity", line=dict(color="#1f77b4")), row=1, col=1)
            fig.update_yaxes(type="log", row=1, col=1)
            fig.add_trace(go.Scatter(x=dd_series.index, y=dd_series.values * 100, name="Drawdown(%)",
                                      fill="tozeroy", line=dict(color="#d62728")), row=2, col=1)
            fig.update_layout(height=560, showlegend=False, margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

            yearly = eq_series.resample("YE").last()
            yearly_ret = yearly.pct_change()
            if len(yearly_ret):
                yearly_ret.iloc[0] = yearly.iloc[0] / 1.0 - 1
            yearly_df = pd.DataFrame({
                "연도": yearly.index.year, "연말자산(시작=1.0)": yearly.values.round(3),
                "연간수익률": (yearly_ret.values * 100).round(2)
            })
            st.subheader("연도별 성과")
            st.dataframe(yearly_df, use_container_width=True, hide_index=True)

            st.subheader("거래 로그")
            if len(trade_log):
                tl = trade_log.copy()
                tl["trade_ret_%"] = (tl["trade_ret"] * 100).round(3)
                st.dataframe(tl[["date", "trade_ret_%"]].sort_values("date", ascending=False),
                             use_container_width=True, height=300)
                csv_bytes = trade_log.to_csv(index=False).encode("utf-8")
                st.download_button("거래 로그 다운로드 (CSV)", csv_bytes, "camarilla_trade_log.csv", "text/csv")
            else:
                st.caption("해당 기간/조건에서 발생한 거래가 없습니다.")

# ----------------------------------------------------------------------------
# TAB 3: 전략 로직 설명
# ----------------------------------------------------------------------------
with tab3:
    st.markdown(
"""
### 전략 개요

**카마릴라 R4 변형 돌파매매**: 전일 OHLC로 계산한 저항선을 장중에 돌파하면 매수하고, 다음 거래일
시가(MOO)에 전량 매도하는 1일 스윙 돌파전략입니다.

```
저항선 = 전일종가 + (전일고가 - 전일저가) × 1.1 × coef
```

- `coef = 0.55` → 정통 카마릴라 4차 저항선(R4)
- `coef = 0.70` → 검증을 통해 추천하는 값 (약한 가짜 돌파를 더 걸러냄)
- 장중 고가가 저항선을 돌파하면 그 가격(또는 이미 돌파된 채 시작한 시가)에 매수
- 익일 시가에 전량 매도

**장점(원 아이디어 제안자 설명)**: 기존 윌리엄스 변동성 돌파매매는 당일 시가가 나와야 매수가를 계산할 수
있지만, 이 전략은 저항선이 **전일 종가 시점에 이미 확정**되므로 전날 밤에 다음날 주문 계획을 세울 수
있어 수동/디스크리션 트레이더에게 유리합니다.

### 변동성 필터 (강건성 검증을 통해 추가)

직전 20일 실현변동성(종가 기준 일별수익률의 표준편차)이 그동안의 분포에서 상위 N%(기본 20%)에 해당하는
"극단적 변동성 구간"이면, 신호가 떠도 그날은 매매를 쉽니다. 단순히 매수 비중을 줄이는 것보다 MDD를
낮추는 데 훨씬 효과적이었습니다.

### 검증된 파라미터 (2010-2026 SOXL 데이터, 4구간 분할 + 70/30 워크포워드 검증)

| 구분 | coef | 비중 | 변동성필터 | CAGR | MDD | Calmar |
|---|---|---|---|---|---|---|
| 원본 (R4, 필터없음) | 0.55 | 100% | 없음 | 82.3% | -50.9% | 1.62 |
| A. 보수적 | 0.70 | 70% | 상위 20% 회피 | 36.8% | **-18.0%** | 2.05 |
| **B. 균형 (기본값)** | **0.70** | **100%** | **상위 20% 회피** | **54.3%** | **-25.3%** | **2.15** |

저항선을 살짝 높이고(coef 0.55→0.70) 변동성 필터를 추가하는 것만으로 MDD를 50%대에서 18~25%대로
낮추면서 CAGR은 대부분 보존했습니다. 70/30 워크포워드(미래 구간) 검증에서도 우위가 유지되어, 과적합이
아닌 강건한 개선임을 확인했습니다.

### 남는 한계 (반드시 인지할 것)

- **단일 종목·단일 사이클 의존**: SOXL(반도체 3배 레버리지)의 2010~2026년 데이터에만 기반합니다. 다른
  레버리지 ETF나 다른 시대에 동일하게 통할지는 검증되지 않았습니다.
- **슬리피지/수수료**: 백테스트 탭에서 옵션으로 반영할 수 있으나, 돌파매매 특성상 실제 체결가는
  저항선보다 불리하게 체결될 가능성이 높습니다. 슬리피지 가정을 낙관적으로 두면 실제 성과보다
  좋게 나올 수 있습니다.
- **변동성 필터 임계값(80%) 자체도 추정값**: 75~85% 구간에서 자산이 커지면 재점검이 필요합니다.
- **유동성 제약 미반영**: 투자 규모가 커지면 SOXL 일일 거래량 한도에 부딪힐 수 있습니다.
- 이 앱은 **주문 정보를 안내하고 체결 기록을 보관**하는 도구이며, 자동매매를 수행하지 않습니다. 실제
  주문은 사용자가 직접 거래소/증권사 앱에서 실행해야 합니다.
"""
    )
