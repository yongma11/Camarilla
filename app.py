# 파일명: soxs_camarilla_test.py
# 목적: "SOXL 카마릴라 R4 돌파 신호가 없는 날, SOXS 자체 R4 돌파로 보완매매" 가설 검증용
#       독립 테스트 앱. app_v10_combo.py(운영용)는 전혀 건드리지 않으며,
#       GitHub/Google Sheets/settings.json 연동도 하지 않는 순수 백테스트 도구입니다.
#
# 가설:
#   SOXL과 SOXS는 거의 반비례 관계 (3배 레버리지 vs 3배 인버스).
#   - SOXL이 강하게 상승 → SOXL 자체 R4 저항선 돌파 → (기존 카마릴라 오버레이) SOXL 매수
#   - SOXL이 강하게 하락 → 같은 날 SOXS는 강하게 상승 → SOXS 자체 R4 저항선 돌파가
#     SOXS 차트에서 잡힐 것 → 그 신호로 SOXS를 매수하면, SOXL 신호가 없는 날에도
#     돌파매매 기회를 추가로 포착할 수 있지 않을까?
#
# 검증 방식: 동파법(슬롯/매수/매도 본 로직)은 손대지 않고 그대로 두고, 카마릴라 오버레이만
#   3가지 시나리오로 비교한다.
#     A) off       : 오버레이 없음 (동파법 단독, 베이스라인)
#     B) soxl       : 동파법 + SOXL 단독 카마릴라 오버레이 (기존 v10.0과 동일 로직)
#     C) dual       : 동파법 + 카마릴라(SOXL 우선, SOXL 신호 없는 날 SOXS 자체 R4 확인)
#   세 시나리오 모두 동파법 real_cash를 공유하는 단일 계좌 모델이며, 카마릴라 손익이
#   동파법의 RESET_CYCLE 가상자본 계산에는 섞이지 않는다 (회계 분리, 기존 설계와 동일).
#
# 주의: 이 샌드박스에는 인터넷 접속이 없어 yfinance 데이터를 직접 받아 실행해볼 수
#   없습니다. 문법/로직은 합성 데이터로 검증했고, 실제 결과 확인은 사용자 환경(인터넷
#   되는 곳)에서 `streamlit run soxs_camarilla_test.py` 로 실행해야 합니다.

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

st.set_page_config(page_title="🧪 SOXS 보완매매 가설 테스트", page_icon="🧪", layout="wide")

# ============================================================
# 동파법 상수 (app_v10_combo.py와 동일 — 본 로직 비교를 위해 독립 복제)
# ============================================================
PARAMS = {
    'Safe':    {'buy': 3.0, 'sell': 0.5, 'time': 28, 'desc': '🛡️ 방어 (Safe)'},
    'Offense': {'buy': 3.0, 'sell': 4.0, 'time': 7,  'desc': '⚔️ 공세 (Offense)'},
}
LOCAL_PARAMS = {
    'Safe':    {'buy': 0.03, 'sell': 1.005, 'time': 28},
    'Offense': {'buy': 0.03, 'sell': 1.04,  'time': 7},
}
MAX_SLOTS_SAFE    = 7
MAX_SLOTS_OFFENSE = 6
RESET_CYCLE       = 12
def max_slots_for(mode):
    return MAX_SLOTS_OFFENSE if mode == 'Offense' else MAX_SLOTS_SAFE

QS_MA_WINDOW     = 30
QS_LOW_THRESH    = 0.75
QS_HIGH_THRESH   = 1.25
QS_LOW_MULT      = 3.0
QS_HIGH_MULT     = 0.90
QS_HIGH_SLOT_CAP = 4
LOSS_STREAK_N    = 3
LOSS_STREAK_MUL  = 0.8
LOSS_STREAK_WIN  = 5

def loss_streak_mul(recent_outcomes):
    if len(recent_outcomes) >= LOSS_STREAK_N and \
       all(not x for x in recent_outcomes[-LOSS_STREAK_N:]):
        return LOSS_STREAK_MUL
    return 1.0

def calc_mode_series(df_qqq):
    if df_qqq is None:
        return None, None
    last_friday = df_qqq.index[df_qqq.index.dayofweek == 4].max()
    df_qqq_clean = df_qqq[df_qqq.index <= last_friday]
    qqq_weekly = df_qqq_clean.resample('W-FRI').last()
    delta      = qqq_weekly.diff()
    up         = delta.clip(lower=0)
    down       = -1 * delta.clip(upper=0)
    ema_up     = up.ewm(com=13, adjust=False).mean()
    ema_down   = down.ewm(com=13, adjust=False).mean()
    rs         = ema_up / ema_down
    rsi_series = 100 - (100 / (1 + rs))
    modes        = []
    current_mode = 'Safe'
    for i in range(len(rsi_series)):
        if i < 2:
            modes.append(current_mode); continue
        rsi_t1 = rsi_series.iloc[i - 1]
        rsi_t2 = rsi_series.iloc[i - 2]
        if np.isnan(rsi_t1) or np.isnan(rsi_t2):
            modes.append(current_mode); continue
        safe    = ((rsi_t2 > 65) and (rsi_t2 > rsi_t1)) or \
                  ((40 < rsi_t2 < 50) and (rsi_t2 > rsi_t1)) or \
                  ((rsi_t1 < 50) and (rsi_t2 > 50))
        offense = ((rsi_t2 < 35) and (rsi_t2 < rsi_t1)) or \
                  ((50 < rsi_t2 < 60) and (rsi_t2 < rsi_t1)) or \
                  ((rsi_t1 > 50) and (rsi_t2 < 50))
        if safe:      current_mode = 'Safe'
        elif offense: current_mode = 'Offense'
        modes.append(current_mode)
    weekly_mode = pd.Series(modes, index=qqq_weekly.index)
    daily_mode = weekly_mode.resample('D').ffill()
    daily_rsi  = rsi_series.resample('D').ffill()
    full_end = df_qqq.index[-1]
    if daily_mode.index[-1] < full_end:
        full_idx   = pd.date_range(daily_mode.index[0], full_end, freq='D')
        daily_mode = daily_mode.reindex(full_idx).ffill()
        daily_rsi  = daily_rsi.reindex(full_idx).ffill()
    return daily_mode, daily_rsi

def calc_qs_strength(df, window=QS_MA_WINDOW):
    ratio = df['SOXL'] / df['QQQ']
    ma    = ratio.rolling(window).mean()
    return (ratio / ma).rename('QS')

def qs_label_and_mul(qs_val):
    if qs_val < QS_LOW_THRESH:
        return f"🔥 과매도 ({qs_val:.3f}) → 슬롯 ×{QS_LOW_MULT}", QS_LOW_MULT, "low"
    elif qs_val > QS_HIGH_THRESH:
        return f"❄️ 과매수 ({qs_val:.3f}) → 슬롯 ×{QS_HIGH_MULT}", QS_HIGH_MULT, "high"
    else:
        return f"✅ 중립 ({qs_val:.3f}) → 슬롯 ×1.0", 1.0, "mid"

# ============================================================
# 데이터 로딩: QQQ(모드용) + SOXL(OHLC+배당) + SOXS(OHLC)
# ============================================================
@st.cache_data(ttl=600)
def get_data_dual():
    """QQQ Close / SOXL O,H,L,C,Div / SOXS O,H,L,C 를 한 DataFrame으로 합친다.
    app_v10_combo.py의 get_data_final()과 동일한 패턴(auto_adjust=False, 3회 재시도)."""
    for attempt in range(3):
        try:
            start_date   = '2010-01-01'
            end_date_str = (datetime.utcnow() + timedelta(days=2)).strftime('%Y-%m-%d')
            df_qqq  = yf.download("QQQ",  start=start_date, end=end_date_str,
                                   progress=False, auto_adjust=False, actions=True)
            df_soxl = yf.download("SOXL", start=start_date, end=end_date_str,
                                   progress=False, auto_adjust=False, actions=True)
            df_soxs = yf.download("SOXS", start=start_date, end=end_date_str,
                                   progress=False, auto_adjust=False, actions=False)
            if df_qqq.empty or df_soxl.empty or df_soxs.empty:
                time.sleep(1); continue

            def col(frame, field, ticker):
                is_multi = isinstance(frame.columns, pd.MultiIndex)
                return frame[field][ticker] if is_multi else frame[field]

            qqq_close  = col(df_qqq,  'Close', 'QQQ')
            soxl_close = col(df_soxl, 'Close', 'SOXL')
            soxl_open  = col(df_soxl, 'Open',  'SOXL')
            soxl_high  = col(df_soxl, 'High',  'SOXL')
            soxl_low   = col(df_soxl, 'Low',   'SOXL')
            try:
                soxl_div = col(df_soxl, 'Dividends', 'SOXL').fillna(0).astype(float)
            except (KeyError, AttributeError):
                soxl_div = pd.Series(0.0, index=soxl_close.index)
            soxs_close = col(df_soxs, 'Close', 'SOXS')
            soxs_open  = col(df_soxs, 'Open',  'SOXS')
            soxs_high  = col(df_soxs, 'High',  'SOXS')
            soxs_low   = col(df_soxs, 'Low',   'SOXS')

            df = pd.DataFrame({
                'QQQ': qqq_close,
                'SOXL': soxl_close, 'SOXL_O': soxl_open, 'SOXL_H': soxl_high, 'SOXL_L': soxl_low,
                'SOXL_Div': soxl_div,
                'SOXS': soxs_close, 'SOXS_O': soxs_open, 'SOXS_H': soxs_high, 'SOXS_L': soxs_low,
            })
            df = df.sort_index().dropna(subset=['QQQ', 'SOXL', 'SOXS'])
            for c in ['QQQ', 'SOXL', 'SOXL_O', 'SOXL_H', 'SOXL_L', 'SOXS', 'SOXS_O', 'SOXS_H', 'SOXS_L']:
                df[c] = df[c].ffill().bfill()
            df['SOXL_Div'] = df['SOXL_Div'].fillna(0)
            df.index = df.index.tz_localize(None)
            return df
        except Exception:
            time.sleep(1)
    return None

# ============================================================
# 카마릴라 R4 신호 (종목 무관 — O/H/L/C 컬럼을 받아서 계산)
# ============================================================
def compute_camarilla_signal_generic(o, h, l, c, coef=0.70, vol_filter_pct=0.80):
    """resistance[d] = 전일종가 + 전일고저폭 * 1.1 * coef
    signal[d]     = 오늘 고가 >= 저항선 (+ 변동성 백분위 필터)
    entry_raw[d]  = 오늘 시가 >= 저항선이면 시가, 아니면 저항선가
    nextO[d]      = 다음날 시가 (청산가). 마지막 행은 NaN."""
    O = o.values; H = h.values; L = l.values; C = c.values
    n = len(O)
    prevH = np.roll(H, 1); prevH[0] = np.nan
    prevL = np.roll(L, 1); prevL[0] = np.nan
    prevC = np.roll(C, 1); prevC[0] = np.nan
    resistance = prevC + (prevH - prevL) * 1.1 * coef
    ret_cc = np.zeros(n)
    ret_cc[1:] = C[1:] / C[:-1] - 1
    vol20 = pd.Series(ret_cc).rolling(20).std().values
    vol_shift = np.roll(vol20, 1); vol_shift[0] = np.nan
    vol_rank = pd.Series(vol_shift).expanding(min_periods=60).rank(pct=True).values
    nextO = np.roll(O, -1); nextO[-1] = np.nan
    signal = (H >= resistance) & ~np.isnan(resistance)
    if vol_filter_pct is not None:
        vol_ok = (vol_rank <= vol_filter_pct) & ~np.isnan(vol_rank)
        signal = signal & vol_ok
    entry_raw = np.where(O >= resistance, O, resistance)
    return pd.DataFrame({
        'signal': signal, 'entry_raw': entry_raw, 'nextO': nextO,
        'resistance': resistance, 'vol_rank': vol_rank,
    }, index=o.index)

# ============================================================
# 통합 백테스트: 동파법(불변) + 카마릴라 오버레이(3가지 시나리오)
# ============================================================
def run_dual_backtest(df, start_date, end_date, init_cap, overlay_mode='off',
                       overlay_fraction=0.70, cama_coef=0.70, cama_vol_filter_pct=0.80,
                       cama_fee_rate=0.0005, cama_slippage_pct=0.0010,
                       include_fees=False, buy_fee_rate=0.00015, sell_fee_rate=0.0001706):
    """overlay_mode:
       'off'  → 동파법 단독 (카마릴라 미사용, 베이스라인)
       'soxl' → 동파법 + SOXL 단독 카마릴라 오버레이 (기존 v10.0과 동일)
       'dual' → 동파법 + 카마릴라(SOXL 우선 확인 → 없으면 SOXS 자체 R4 확인)
       동파법 로직 자체는 세 모드에서 완전히 동일한 코드로 실행된다(차이는 ①③ 카마릴라
       스텝뿐). 단, real_cash를 공유하므로 카마릴라 손익이 동파법의 다음 매수 가용현금에
       영향을 줄 수 있다 (기존 v10.0 아키텍처와 동일한 의도된 결합)."""
    if df is None:
        return None, None, None

    mode_daily, _ = calc_mode_series(df['QQQ'])
    qs_daily      = calc_qs_strength(df)
    use_overlay   = overlay_mode in ('soxl', 'dual')

    concat_list = [df['SOXL'], df['SOXL_Div'], mode_daily, qs_daily]
    col_names   = ['Price', 'Div', 'Mode', 'QS']
    if use_overlay:
        sig_soxl = compute_camarilla_signal_generic(
            df['SOXL_O'], df['SOXL_H'], df['SOXL_L'], df['SOXL'],
            coef=cama_coef, vol_filter_pct=cama_vol_filter_pct)
        concat_list += [sig_soxl['signal'], sig_soxl['entry_raw'], sig_soxl['nextO']]
        col_names   += ['SoxlSig', 'SoxlEntry', 'SoxlNextO']
        if overlay_mode == 'dual':
            sig_soxs = compute_camarilla_signal_generic(
                df['SOXS_O'], df['SOXS_H'], df['SOXS_L'], df['SOXS'],
                coef=cama_coef, vol_filter_pct=cama_vol_filter_pct)
            concat_list += [sig_soxs['signal'], sig_soxs['entry_raw'], sig_soxs['nextO']]
            col_names   += ['SoxsSig', 'SoxsEntry', 'SoxsNextO']

    sim_df = pd.concat(concat_list, axis=1)
    sim_df.columns = col_names
    sim_df['Div'] = sim_df['Div'].fillna(0)
    sim_df = sim_df.dropna(subset=['Price', 'Mode', 'QS'])
    mask = (sim_df.index >= pd.to_datetime(start_date)) & (sim_df.index <= pd.to_datetime(end_date))
    sim_df = sim_df[mask]
    if sim_df.empty:
        return None, None, None
    sim_df['Prev_Price'] = sim_df['Price'].shift(1)
    sim_df['Prev_QS']    = sim_df['QS'].shift(1)
    sim_df['Prev_Mode']  = sim_df['Mode'].shift(1)
    sim_df = sim_df.dropna(subset=['Prev_Price', 'Prev_QS', 'Prev_Mode'])

    real_cash    = init_cap
    cum_profit   = 0.0
    cum_loss     = 0.0
    cum_dividends = 0.0
    slots        = []
    equity_curve = []
    cama_log     = []
    gross_profit = 0.0
    gross_loss   = 0.0
    cycle_days   = 0
    slot_sizes   = {'Safe': init_cap / MAX_SLOTS_SAFE, 'Offense': init_cap / MAX_SLOTS_OFFENSE}
    recent_slot_outcomes = []
    total_buy_fees  = 0.0
    total_sell_fees = 0.0

    cama_position = None  # {'invest_amt','entry_date','entry_raw','next_open','instrument'}
    cama_stats = {
        'SOXL': {'count': 0, 'win': 0, 'pnl': 0.0},
        'SOXS': {'count': 0, 'win': 0, 'pnl': 0.0},
    }

    for date, row in sim_df.iterrows():
        price      = row['Price']
        div_amt    = float(row.get('Div', 0) or 0)
        prev_price = row['Prev_Price']
        prev_qs    = row['Prev_QS']
        prev_mode  = row['Prev_Mode']

        # ── ① 전날 열어둔 카마릴라 포지션을 '오늘 시가'로 청산 (종목 무관, 동일 규칙) ──
        if cama_position is not None:
            exit_eff  = cama_position['next_open'] * (1 - cama_slippage_pct)
            entry_eff = cama_position['entry_raw'] * (1 + cama_slippage_pct)
            if not np.isnan(exit_eff) and entry_eff > 0:
                trade_ret = (exit_eff / entry_eff - 1) - cama_fee_rate * 2
            else:
                trade_ret = 0.0
            proceeds = cama_position['invest_amt'] * (1 + trade_ret)
            pnl      = proceeds - cama_position['invest_amt']
            real_cash += proceeds
            inst = cama_position['instrument']
            cama_stats[inst]['pnl']   += pnl
            cama_stats[inst]['count'] += 1
            if pnl > 0:
                cama_stats[inst]['win'] += 1
            cama_log.append({
                "진입일": cama_position['entry_date'].date(), "청산일": date.date(),
                "종목": inst, "투입금": f"${cama_position['invest_amt']:,.0f}",
                "수익률": f"{trade_ret*100:+.2f}%", "손익": f"${pnl:,.2f}",
            })
            cama_position = None

        # ── 배당 cash 주입 (동파법 보유 SOXL 주식 기준, 원본과 동일) ──
        if div_amt > 0:
            held_shares = sum(s['shares'] for s in slots)
            if held_shares > 0:
                div_cash = div_amt * held_shares
                real_cash += div_cash
                cum_dividends += div_cash

        ls_mul = loss_streak_mul(recent_slot_outcomes)
        _, qs_mul, _ = qs_label_and_mul(prev_qs)
        effective_max = QS_HIGH_SLOT_CAP if prev_qs > QS_HIGH_THRESH else max_slots_for(prev_mode)

        # ── 동파법 매도 (원본 그대로) ──
        sold_idx = []
        for i in range(len(slots) - 1, -1, -1):
            s = slots[i]
            s['days'] += 1
            rule = LOCAL_PARAMS.get(s['birth_mode'], LOCAL_PARAMS['Safe'])
            if (price >= s['buy_price'] * rule['sell']) or (s['days'] >= rule['time']):
                gross_rev = s['shares'] * price
                sell_fee  = gross_rev * sell_fee_rate if include_fees else 0.0
                net_rev   = gross_rev - sell_fee
                cost_basis = s.get('cost_basis', s['shares'] * s['buy_price'])
                prof = net_rev - cost_basis
                real_cash += net_rev
                total_sell_fees += sell_fee
                if prof > 0:
                    cum_profit += prof; gross_profit += prof
                    recent_slot_outcomes.append(True)
                else:
                    cum_loss += abs(prof); gross_loss += abs(prof)
                    recent_slot_outcomes.append(False)
                recent_slot_outcomes = recent_slot_outcomes[-LOSS_STREAK_WIN:]
                sold_idx.append(i)
        for i in sold_idx:
            del slots[i]

        # ── 동파법 매수 (원본 그대로) ──
        curr_rule = LOCAL_PARAMS.get(prev_mode, LOCAL_PARAMS['Safe'])
        cur_slot_size = slot_sizes[prev_mode]
        loc_price = prev_price * (1 + curr_rule['buy'])
        if price <= loc_price and len(slots) < effective_max:
            amt = min(real_cash, cur_slot_size * qs_mul * ls_mul)
            shares = int(amt / loc_price)
            if shares > 0:
                invested  = shares * price
                buy_fee   = invested * buy_fee_rate if include_fees else 0.0
                cost_basis = invested + buy_fee
                real_cash -= cost_basis
                total_buy_fees += buy_fee
                slots.append({'buy_price': price, 'shares': shares, 'days': 0,
                              'birth_mode': prev_mode, 'cost_basis': cost_basis})

        # ── ③ 카마릴라 신규 진입: SOXL 신호 우선 확인 → (dual 모드) 없으면 SOXS 확인 ──
        if use_overlay and cama_position is None:
            opened = False
            if bool(row.get('SoxlSig', False)):
                entry_raw = row['SoxlEntry']; next_open = row['SoxlNextO']
                if entry_raw > 0 and not np.isnan(entry_raw) and not np.isnan(next_open):
                    leftover   = max(0.0, real_cash)
                    invest_amt = overlay_fraction * leftover
                    if invest_amt > 1e-6:
                        real_cash -= invest_amt
                        cama_position = {'invest_amt': invest_amt, 'entry_date': date,
                                          'entry_raw': entry_raw, 'next_open': next_open,
                                          'instrument': 'SOXL'}
                        opened = True
            if (not opened) and overlay_mode == 'dual' and bool(row.get('SoxsSig', False)):
                entry_raw = row['SoxsEntry']; next_open = row['SoxsNextO']
                if entry_raw > 0 and not np.isnan(entry_raw) and not np.isnan(next_open):
                    leftover   = max(0.0, real_cash)
                    invest_amt = overlay_fraction * leftover
                    if invest_amt > 1e-6:
                        real_cash -= invest_amt
                        cama_position = {'invest_amt': invest_amt, 'entry_date': date,
                                          'entry_raw': entry_raw, 'next_open': next_open,
                                          'instrument': 'SOXS'}

        dongpa_equity  = sum(s['shares'] * price for s in slots)
        cama_equity    = cama_position['invest_amt'] if cama_position is not None else 0.0
        current_equity = real_cash + dongpa_equity + cama_equity
        equity_curve.append({'Date': date, 'Equity': current_equity})

        cycle_days += 1
        if cycle_days >= RESET_CYCLE:
            # 카마릴라 손익은 가상자본 계산에 포함하지 않음 (회계 분리, 기존 설계와 동일)
            virtual = init_cap + (cum_profit * 0.7) - (cum_loss * 0.6) + cum_dividends * 0.7
            if virtual < 1000:
                virtual = 1000
            slot_sizes['Safe']    = virtual / MAX_SLOTS_SAFE
            slot_sizes['Offense'] = virtual / MAX_SLOTS_OFFENSE
            cycle_days = 0

    res_df  = pd.DataFrame(equity_curve).set_index('Date')
    df_cama = pd.DataFrame(cama_log) if cama_log else pd.DataFrame()
    metrics = {'overlay_mode': overlay_mode}
    if not res_df.empty:
        total_ret = (res_df['Equity'].iloc[-1] / init_cap) - 1
        days = (res_df.index[-1] - res_df.index[0]).days
        cagr = (1 + total_ret) ** (365 / days) - 1 if days > 0 else 0
        peak = res_df['Equity'].cummax()
        mdd  = ((res_df['Equity'] - peak) / peak).min()
        metrics.update({
            'final_equity': res_df['Equity'].iloc[-1], 'cagr': cagr, 'mdd': mdd,
            'calmar': (cagr / abs(mdd)) if mdd < 0 else np.nan,
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else 99.9,
            'total_dividends': cum_dividends,
            'total_fees': total_buy_fees + total_sell_fees,
            'soxl_trades': cama_stats['SOXL']['count'],
            'soxl_win_rate': (cama_stats['SOXL']['win'] / cama_stats['SOXL']['count']) if cama_stats['SOXL']['count'] > 0 else np.nan,
            'soxl_pnl': cama_stats['SOXL']['pnl'],
            'soxs_trades': cama_stats['SOXS']['count'],
            'soxs_win_rate': (cama_stats['SOXS']['win'] / cama_stats['SOXS']['count']) if cama_stats['SOXS']['count'] > 0 else np.nan,
            'soxs_pnl': cama_stats['SOXS']['pnl'],
            'cama_total_pnl': cama_stats['SOXL']['pnl'] + cama_stats['SOXS']['pnl'],
            'cama_total_trades': cama_stats['SOXL']['count'] + cama_stats['SOXS']['count'],
        })
    else:
        metrics.update({'final_equity': np.nan, 'cagr': np.nan, 'mdd': np.nan, 'calmar': np.nan})
    return res_df, metrics, df_cama

# ============================================================
# Streamlit UI
# ============================================================
def main():
    st.title("🧪 SOXS 보완매매 가설 테스트 (독립 검증용, app_v10_combo.py와 무관)")
    st.caption(
        "동파법 본 로직은 그대로 두고, 카마릴라 오버레이만 'SOXL 신호 없는 날 SOXS 자체 "
        "R4 돌파로 보완매매' 방식으로 확장했을 때 효과가 있는지 비교합니다. "
        "결과가 좋으면 이후 app_v10_combo.py에 정식으로 포팅합니다."
    )

    with st.spinner("데이터 로딩 중 (QQQ/SOXL/SOXS)..."):
        df = get_data_dual()
    if df is None:
        st.error("⚠️ 데이터를 불러오지 못했습니다. 인터넷 연결과 yfinance 상태를 확인해주세요.")
        st.stop()

    c1, c2, c3 = st.columns(3)
    init_cap = c1.number_input("백테스트 초기 자본 ($)", value=10000.0, step=1000.0)
    start_d  = c2.date_input("검증 시작일", value=datetime(2018, 1, 1), min_value=datetime(2010, 1, 1))
    end_d    = c3.date_input("검증 종료일", value=df.index[-1].date(), min_value=datetime(2010, 1, 1))

    st.markdown("#### 🧨 카마릴라 오버레이 공통 파라미터 (3가지 시나리오에 동일 적용)")
    o1, o2, o3 = st.columns(3)
    overlay_fraction = o1.slider("투입 비중 (남은 현금 중 %)", 0.0, 1.0, 0.70, 0.05)
    cama_coef        = o2.slider("R4 저항선 계수 (coef)", 0.10, 2.00, 0.70, 0.05)
    use_vol_filter   = o3.checkbox("변동성 백분위 필터 사용", value=True)
    cama_vol_filter_pct = None
    if use_vol_filter:
        cama_vol_filter_pct = st.slider("변동성 백분위 상한", 0.0, 1.0, 0.80, 0.05)
    f1, f2, f3 = st.columns(3)
    cama_fee_rate      = f1.number_input("카마릴라 왕복 수수료율 (편도)", value=0.0005, step=0.0001, format="%.4f")
    cama_slippage_pct  = f2.number_input("카마릴라 슬리피지율 (편도)", value=0.0010, step=0.0001, format="%.4f")
    include_fees       = f3.checkbox("동파법 매매 수수료 적용", value=False)

    if st.button("🚀 3가지 시나리오 비교 실행 (off / SOXL단독 / SOXL+SOXS dual)", type="primary"):
        with st.spinner("백테스트 실행 중..."):
            results = {}
            for mode, label in [('off', 'A) 오버레이 없음'), ('soxl', 'B) SOXL 단독'), ('dual', 'C) SOXL+SOXS dual')]:
                res_df, metrics, df_cama = run_dual_backtest(
                    df, start_d, end_d, init_cap, overlay_mode=mode,
                    overlay_fraction=overlay_fraction, cama_coef=cama_coef,
                    cama_vol_filter_pct=cama_vol_filter_pct, cama_fee_rate=cama_fee_rate,
                    cama_slippage_pct=cama_slippage_pct, include_fees=include_fees)
                results[mode] = {'label': label, 'res_df': res_df, 'metrics': metrics, 'cama_log': df_cama}

        if any(results[m]['res_df'] is None for m in results):
            st.error("데이터 부족 — 기간을 조정해보세요.")
        else:
            st.markdown("### 📊 시나리오 비교")
            comp_rows = []
            for mode in ['off', 'soxl', 'dual']:
                m = results[mode]['metrics']
                comp_rows.append({
                    "시나리오": results[mode]['label'],
                    "최종자산": f"${m['final_equity']:,.0f}",
                    "CAGR": f"{m['cagr']*100:.2f}%",
                    "MDD": f"{m['mdd']*100:.2f}%",
                    "Calmar": f"{m['calmar']:.2f}" if not np.isnan(m['calmar']) else "─",
                    "카마릴라 거래수": m.get('cama_total_trades', 0),
                    "└ SOXL 거래/승률/손익": (
                        f"{m.get('soxl_trades',0)}건 / "
                        f"{(m.get('soxl_win_rate', np.nan)*100):.1f}%" if m.get('soxl_trades',0) > 0 else "0건"
                    ) + (f" / ${m.get('soxl_pnl',0):,.0f}" if m.get('soxl_trades',0) > 0 else ""),
                    "└ SOXS 거래/승률/손익": (
                        f"{m.get('soxs_trades',0)}건 / "
                        f"{(m.get('soxs_win_rate', np.nan)*100):.1f}%" if m.get('soxs_trades',0) > 0 else "0건"
                    ) + (f" / ${m.get('soxs_pnl',0):,.0f}" if m.get('soxs_trades',0) > 0 else ""),
                })
            st.dataframe(pd.DataFrame(comp_rows).set_index("시나리오"), use_container_width=True)

            st.markdown("### 📈 자산 성장 곡선 비교")
            fig, ax = plt.subplots(figsize=(12, 5))
            colors = {'off': 'gray', 'soxl': '#1a73e8', 'dual': '#e65100'}
            for mode in ['off', 'soxl', 'dual']:
                rdf = results[mode]['res_df']
                ax.plot(rdf.index, rdf['Equity'], label=results[mode]['label'],
                        color=colors[mode], linewidth=1.6)
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.yaxis.set_major_formatter(mtick.StrMethodFormatter('${x:,.0f}'))
            ax.set_title("동파법 + 카마릴라 오버레이 시나리오별 자산 곡선")
            st.pyplot(fig)

            dual_cama_log = results['dual']['cama_log']
            if not dual_cama_log.empty:
                with st.expander(f"🧨 dual 시나리오 카마릴라 거래 로그 ({len(dual_cama_log)}건)", expanded=False):
                    soxs_n = (dual_cama_log['종목'] == 'SOXS').sum()
                    soxl_n = (dual_cama_log['종목'] == 'SOXL').sum()
                    st.caption(f"SOXL {soxl_n}건 / SOXS {soxs_n}건")
                    st.dataframe(dual_cama_log, use_container_width=True, hide_index=True)

            st.markdown(
                "💡 **읽는 법**: dual 시나리오의 최종자산/CAGR/MDD가 soxl 단독보다 낫고, "
                "SOXS 레그 자체의 승률/손익도 양호하다면 가설이 유효합니다. 반대로 SOXS 레그가 "
                "손실 위주이거나 SOXL 레그의 기회를 깎아먹는다면(레버리지 ETF의 변동성 감쇄 때문에 "
                "SOXS의 '되돌림'이 SOXL 하락폭만큼 깨끗하게 대응하지 않을 수 있음) 기각하는 게 맞습니다."
            )


if __name__ == "__main__":
    main()
