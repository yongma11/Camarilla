"""
동파법 마스터 v9.8 핵심 로직 — streamlit/yfinance/github 의존성 제거한 순수 파이썬 재현.
원본 소스(app.py, v9.8)에서 다음 함수/상수를 '검증된 그대로' 옮김:
  - LOCAL_PARAMS, MAX_SLOTS_SAFE/OFFENSE, RESET_CYCLE, max_slots_for
  - QS_MA_WINDOW/QS_LOW_THRESH/QS_HIGH_THRESH/QS_LOW_MULT/QS_HIGH_MULT/QS_HIGH_SLOT_CAP
  - LOSS_STREAK_N/MUL/WIN, loss_streak_mul
  - calc_mode_series, calc_qs_strength, qs_label_and_mul
  - run_backtest_fixed  (입력 df 형식만 로컬 CSV 기반으로 교체, 로직 본문은 1바이트도 안 건드림)

데이터 입력은 get_data_final() (yfinance) 대신 load_local_data() (로컬 CSV) 사용.
"""
import pandas as pd
import numpy as np

# ── 원본 상수 (app.py v9.8 그대로) ─────────────────────────────────
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

LOSS_STREAK_N   = 3
LOSS_STREAK_MUL = 0.8
LOSS_STREAK_WIN = 5

def loss_streak_mul(recent_outcomes):
    if len(recent_outcomes) >= LOSS_STREAK_N and \
       all(not x for x in recent_outcomes[-LOSS_STREAK_N:]):
        return LOSS_STREAK_MUL
    return 1.0


# ── 데이터 로딩 (로컬 CSV, yfinance 대체) ──────────────────────────
def load_local_data(camarilla_dir):
    """qqq_close_full.csv / soxl_ohlc_clean.csv / soxl_dividends.csv 를 읽어
    run_backtest_fixed 가 기대하는 형태의 df (columns: QQQ, SOXL, SOXL_Div,
    그리고 오버레이용 SOXL_O/H/L) 를 만든다."""
    qqq = pd.read_csv(f"{camarilla_dir}/qqq_close_full.csv", header=None,
                       names=["date", "QQQ"], parse_dates=["date"])
    soxl = pd.read_csv(f"{camarilla_dir}/soxl_ohlc_clean.csv", parse_dates=["date"])
    soxl = soxl.rename(columns={"O": "SOXL_O", "H": "SOXL_H", "L": "SOXL_L", "C": "SOXL"})
    soxl = soxl[["date", "SOXL_O", "SOXL_H", "SOXL_L", "SOXL"]]
    div = pd.read_csv(f"{camarilla_dir}/soxl_dividends.csv", parse_dates=["date"])
    div = div.rename(columns={"amount": "SOXL_Div"})

    df = pd.merge(qqq, soxl, on="date", how="inner")
    df = pd.merge(df, div, on="date", how="left")
    df["SOXL_Div"] = df["SOXL_Div"].fillna(0.0)
    df = df.sort_values("date").set_index("date")
    df.index.name = None
    return df


# ── 데이터 로딩 (yfinance, Streamlit Cloud 배포용) ──────────────────
def _yf_download_with_fallback(ticker, period="max", auto_adjust=False, actions=False):
    """기존 app.py의 get_price_history() 와 동일한 2단계 폴백 패턴."""
    import yfinance as yf
    raw = None
    try:
        raw = yf.download(ticker, period=period, auto_adjust=auto_adjust,
                           actions=actions, progress=False, threads=False)
        if raw is None or len(raw) == 0:
            raw = None
    except Exception:
        raw = None
    if raw is None:
        raw = yf.Ticker(ticker).history(period=period, auto_adjust=auto_adjust, actions=actions)
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"{ticker} 시세를 가져오지 못했습니다 (yfinance).")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    raw = raw.rename(columns={"Date": "date"})
    raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)
    return raw


def fetch_yf_data(period="max"):
    """qqq_close_full.csv / soxl_ohlc_clean.csv / soxl_dividends.csv 를 대체.
    SOXL은 auto_adjust=False + actions=True 로 받아 (1) 분할조정은 반영되지만
    (2) 배당으로 가격이 조정되지는 않은 원본 OHLC와, 배당액(Dividends 컬럼)을
    동시에 얻는다 — run_backtest_fixed가 배당을 현금으로 별도 가산하는 로직과
    이중계산되지 않도록 하기 위함 (auto_adjust=True를 쓰면 배당이 이미 가격에
    반영돼 있어 이중계산이 됨)."""
    qqq_raw = _yf_download_with_fallback("QQQ", period=period, auto_adjust=False, actions=False)
    qqq = qqq_raw.rename(columns={"Close": "QQQ"})[["date", "QQQ"]].dropna()

    soxl_raw = _yf_download_with_fallback("SOXL", period=period, auto_adjust=False, actions=True)
    soxl = soxl_raw.rename(columns={
        "Open": "SOXL_O", "High": "SOXL_H", "Low": "SOXL_L", "Close": "SOXL",
        "Dividends": "SOXL_Div",
    })
    keep = ["date", "SOXL_O", "SOXL_H", "SOXL_L", "SOXL"]
    if "SOXL_Div" in soxl.columns:
        keep.append("SOXL_Div")
    soxl = soxl[keep].dropna(subset=["SOXL_O", "SOXL_H", "SOXL_L", "SOXL"])
    if "SOXL_Div" not in soxl.columns:
        soxl["SOXL_Div"] = 0.0

    df = pd.merge(qqq, soxl, on="date", how="inner")
    df["SOXL_Div"] = df["SOXL_Div"].fillna(0.0)
    df = df.sort_values("date").set_index("date")
    df.index.name = None
    return df


# ── 원본 함수 (app.py v9.8 그대로, 1바이트도 수정 없음) ─────────────
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
        if safe:    current_mode = 'Safe'
        elif offense: current_mode = 'Offense'
        modes.append(current_mode)
    weekly_mode = pd.Series(modes, index=qqq_weekly.index)
    daily_mode = weekly_mode.resample('D').ffill()
    daily_rsi  = rsi_series.resample('D').ffill()
    full_end = df_qqq.index[-1]
    if daily_mode.index[-1] < full_end:
        full_idx  = pd.date_range(daily_mode.index[0], full_end, freq='D')
        daily_mode = daily_mode.reindex(full_idx).ffill()
        daily_rsi  = daily_rsi.reindex(full_idx).ffill()
    return daily_mode, daily_rsi

def calc_qs_strength(df, window=QS_MA_WINDOW):
    ratio = df['SOXL'] / df['QQQ']
    ma    = ratio.rolling(window).mean()
    return (ratio / ma).rename('QS')

def qs_label_and_mul(qs_val):
    if qs_val < QS_LOW_THRESH:
        return f"과매도 ({qs_val:.3f}) -> 슬롯 x{QS_LOW_MULT}", QS_LOW_MULT, "qs-low"
    elif qs_val > QS_HIGH_THRESH:
        return f"과매수 ({qs_val:.3f}) -> 슬롯 x{QS_HIGH_MULT}", QS_HIGH_MULT, "qs-high"
    else:
        return f"중립 ({qs_val:.3f}) -> 슬롯 x1.0", 1.0, "qs-mid"


def run_backtest_fixed(df, start_date, end_date, init_cap,
                       include_fees=False, include_tax=False,
                       buy_fee_rate=0.00015, sell_fee_rate=0.0001706,
                       tax_deduction_usd=1786.0, tax_rate=0.22,
                       tax_strategy='A', custom_schedule=None):
    # ★ v9.8 (정리): A = 5월 일괄 (default), B = 커스텀 (직접 월 선택)
    TAX_SCHEDULES = {
        'A': [(1.00, (5,1), (5,31))],
    }
    if tax_strategy == 'B' and custom_schedule is not None and len(custom_schedule) > 0:
        tax_tranches_def = custom_schedule
    else:
        tax_tranches_def = TAX_SCHEDULES.get('A', [(1.00, (5,1), (5,31))])
    if df is None:
        return None, None, None, None
    mode_daily, rsi_daily = calc_mode_series(df['QQQ'])
    qs_daily = calc_qs_strength(df)
    sim_df = pd.concat([df['SOXL'], df['SOXL_Div'], mode_daily, rsi_daily, qs_daily], axis=1)
    sim_df.columns = ['Price', 'Div', 'Mode', 'RSI', 'QS']
    sim_df['Div'] = sim_df['Div'].fillna(0)
    sim_df = sim_df.dropna(subset=['Price','Mode','QS'])
    mask   = (sim_df.index >= pd.to_datetime(start_date)) & \
             (sim_df.index <= pd.to_datetime(end_date))
    sim_df = sim_df[mask]
    if sim_df.empty:
        return None, None, None, None
    sim_df['Prev_Price'] = sim_df['Price'].shift(1)
    sim_df['Prev_QS']    = sim_df['QS'].shift(1)
    sim_df['Prev_Mode']  = sim_df['Mode'].shift(1)
    sim_df = sim_df.dropna(subset=['Prev_Price','Prev_QS','Prev_Mode'])
    real_cash    = init_cap
    cum_profit   = 0.0
    cum_loss     = 0.0
    cum_dividends = 0.0
    slots        = []
    equity_curve = []
    debug_logs   = []
    gross_profit = 0.0
    gross_loss   = 0.0
    cycle_days        = 0
    slot_sizes = {
        'Safe':    init_cap / MAX_SLOTS_SAFE,
        'Offense': init_cap / MAX_SLOTS_OFFENSE,
    }
    recent_slot_outcomes = []
    total_buy_fees    = 0.0
    total_sell_fees   = 0.0
    annual_realized   = 0.0
    total_tax_paid    = 0.0
    last_year_seen    = None
    tax_log           = []
    yearly_realized_log = {}
    yearly_fee_log      = {}
    yearly_tax_log      = {}
    yearly_div_log      = {}
    pending_tranches    = []
    forced_count        = 0
    negative_cash_days  = 0
    for date, row in sim_df.iterrows():
        price      = row['Price']
        div_amt    = float(row.get('Div', 0) or 0)
        mode       = row['Mode']
        rsi_val    = row['RSI']
        prev_price = row['Prev_Price']
        prev_qs    = row['Prev_QS']
        prev_mode  = row['Prev_Mode']
        cur_year = date.year
        if div_amt > 0:
            held_shares = sum(s['shares'] for s in slots)
            if held_shares > 0:
                div_cash = div_amt * held_shares
                real_cash += div_cash
                cum_dividends += div_cash
                yearly_div_log[cur_year] = yearly_div_log.get(cur_year, 0.0) + div_cash
        if last_year_seen is not None and cur_year != last_year_seen:
            yearly_realized_log[last_year_seen] = annual_realized
            if include_tax:
                annual_tax = max(0.0, annual_realized - tax_deduction_usd) * tax_rate
                if annual_tax > 0:
                    for entry in tax_tranches_def:
                        if len(entry) == 4:
                            frac, (em, ed), (fm, fd), yoff = entry
                        else:
                            frac, (em, ed), (fm, fd) = entry
                            yoff = 0
                        if yoff == -1:
                            tax_due = annual_tax * frac
                            actual = min(tax_due, max(0.0, real_cash))
                            if actual > 0:
                                real_cash -= actual
                                total_tax_paid += actual
                                yearly_tax_log[last_year_seen] = yearly_tax_log.get(last_year_seen, 0.0) + actual
                                tax_log.append((date, actual, 'dec_anticipated'))
                            remaining = tax_due - actual
                            if remaining > 1e-6:
                                pending_tranches.append({
                                    'amount':   remaining,
                                    'earliest': pd.Timestamp(year=cur_year, month=1, day=1),
                                    'force':    pd.Timestamp(year=cur_year, month=1, day=31),
                                    'paid':     0.0,
                                    'year':     last_year_seen,
                                })
                        else:
                            pending_tranches.append({
                                'amount':   annual_tax * frac,
                                'earliest': pd.Timestamp(year=cur_year, month=em, day=ed),
                                'force':    pd.Timestamp(year=cur_year, month=fm, day=fd),
                                'paid':     0.0,
                                'year':     last_year_seen,
                            })
            annual_realized = 0.0
        last_year_seen = cur_year
        ls_mul = loss_streak_mul(recent_slot_outcomes)
        if prev_qs < QS_LOW_THRESH:    qs_label = 'Low'
        elif prev_qs > QS_HIGH_THRESH: qs_label = 'High'
        else:                          qs_label = 'Normal'
        ls_label = 'ON' if ls_mul < 1.0 else '-'
        _, qs_mul, _ = qs_label_and_mul(prev_qs)
        effective_max = QS_HIGH_SLOT_CAP if prev_qs > QS_HIGH_THRESH else max_slots_for(prev_mode)
        sold_idx = []
        sold_qty_total = 0
        sold_pnl_total = 0.0
        for i in range(len(slots) - 1, -1, -1):
            s    = slots[i]
            s['days'] += 1
            rule = LOCAL_PARAMS.get(s['birth_mode'], LOCAL_PARAMS['Safe'])
            if (price >= s['buy_price'] * rule['sell']) or (s['days'] >= rule['time']):
                gross_rev = s['shares'] * price
                sell_fee  = gross_rev * sell_fee_rate if include_fees else 0.0
                net_rev   = gross_rev - sell_fee
                cost_basis = s.get('cost_basis', s['shares'] * s['buy_price'])
                prof       = net_rev - cost_basis
                real_cash += net_rev
                total_sell_fees += sell_fee
                yearly_fee_log[cur_year] = yearly_fee_log.get(cur_year, 0.0) + sell_fee
                annual_realized += prof
                if prof > 0:
                    cum_profit += prof;  gross_profit += prof
                    recent_slot_outcomes.append(True)
                else:
                    cum_loss   += abs(prof); gross_loss += abs(prof)
                    recent_slot_outcomes.append(False)
                recent_slot_outcomes = recent_slot_outcomes[-LOSS_STREAK_WIN:]
                sold_idx.append(i)
                sold_qty_total += s['shares']
                sold_pnl_total += prof
        for i in sold_idx:
            del slots[i]
        if sold_qty_total > 0:
            allocated_cap = sum(s['shares'] * price for s in slots)
            total_asset = real_cash + allocated_cap
            balance_qty = sum(s['shares'] for s in slots)
            debug_logs.append({
                "날짜": date.date(), "Action": "매도",
                "적용 모드": prev_mode,
                "QS신호": f"{qs_label} ({prev_qs:.3f})",
                "LS가드": ls_label, "최대슬롯": effective_max,
                "종가": f"${price:.2f}",
                "수량": f"{-sold_qty_total:+,d}",
                "실현손익": f"${sold_pnl_total:,.2f}",
                "Balance_Qty": f"{balance_qty:,d}",
                "Total_Cash": f"${real_cash:,.0f}",
                "Allocated_Cap": f"${allocated_cap:,.0f}",
                "Total_Asset": f"${total_asset:,.0f}",
                "Return_Pct": f"{(total_asset/init_cap - 1)*100:+.2f}%",
            })
        curr_rule     = LOCAL_PARAMS.get(prev_mode, LOCAL_PARAMS['Safe'])
        cur_slot_size = slot_sizes[prev_mode]
        loc_price     = prev_price * (1 + curr_rule['buy'])
        if price <= loc_price and len(slots) < effective_max:
            amt    = min(real_cash, cur_slot_size * qs_mul * ls_mul)
            shares = int(amt / loc_price)
            if shares > 0:
                invested = shares * price
                buy_fee  = invested * buy_fee_rate if include_fees else 0.0
                cost_basis = invested + buy_fee
                real_cash -= cost_basis
                total_buy_fees += buy_fee
                yearly_fee_log[cur_year] = yearly_fee_log.get(cur_year, 0.0) + buy_fee
                slots.append({'buy_price': price, 'shares': shares, 'days': 0,
                              'birth_mode': prev_mode, 'cost_basis': cost_basis})
                allocated_cap = sum(s['shares'] * price for s in slots)
                total_asset = real_cash + allocated_cap
                balance_qty = sum(s['shares'] for s in slots)
                debug_logs.append({
                    "날짜": date.date(), "Action": "매수",
                    "적용 모드": prev_mode,
                    "QS신호": f"{qs_label} ({prev_qs:.3f})",
                    "LS가드": ls_label, "최대슬롯": effective_max,
                    "종가": f"${price:.2f}",
                    "수량": f"+{shares:,d}",
                    "실현손익": "$0.00",
                    "Balance_Qty": f"{balance_qty:,d}",
                    "Total_Cash": f"${real_cash:,.0f}",
                    "Allocated_Cap": f"${allocated_cap:,.0f}",
                    "Total_Asset": f"${total_asset:,.0f}",
                    "Return_Pct": f"{(total_asset/init_cap - 1)*100:+.2f}%",
                })
        current_equity = real_cash + sum(s['shares'] * price for s in slots)
        equity_curve.append({'Date': date, 'Equity': current_equity})
        cycle_days += 1
        is_cycle_end = (cycle_days >= RESET_CYCLE)
        for tranche in pending_tranches:
            remaining = tranche['amount'] - tranche['paid']
            if remaining <= 1e-6: continue
            past_earliest = date >= tranche['earliest']
            past_force    = date >= tranche['force']
            if past_force:
                actual = remaining
                real_cash -= actual
                total_tax_paid += actual
                tranche['paid'] += actual
                yearly_tax_log[cur_year] = yearly_tax_log.get(cur_year, 0.0) + actual
                tax_log.append((date, actual, 'force'))
                forced_count += 1
            elif past_earliest and is_cycle_end:
                actual = min(remaining, max(0.0, real_cash))
                if actual > 0:
                    real_cash -= actual
                    total_tax_paid += actual
                    tranche['paid'] += actual
                    yearly_tax_log[cur_year] = yearly_tax_log.get(cur_year, 0.0) + actual
                    tax_log.append((date, actual, 'cycle'))
        pending_tranches = [t for t in pending_tranches if (t['amount'] - t['paid']) > 1e-6]
        if real_cash < 0:
            negative_cash_days += 1
        if is_cycle_end:
            virtual = init_cap + (cum_profit * 0.7) - (cum_loss * 0.6) - total_tax_paid + cum_dividends * 0.7
            if virtual < 1000: virtual = 1000
            slot_sizes['Safe']    = virtual / MAX_SLOTS_SAFE
            slot_sizes['Offense'] = virtual / MAX_SLOTS_OFFENSE
            cycle_days = 0
    if last_year_seen is not None:
        yearly_realized_log.setdefault(last_year_seen, annual_realized)
    pending_unrealized = max(0.0, annual_realized - tax_deduction_usd) * tax_rate if include_tax else 0.0
    pending_unfunded   = sum(t['amount'] - t['paid'] for t in pending_tranches)
    pending_tax_at_end = pending_unrealized + pending_unfunded
    res_df   = pd.DataFrame(equity_curve).set_index('Date')
    if debug_logs:
        df_debug = pd.DataFrame(debug_logs)
        df_debug = df_debug.reset_index(drop=True)
    else:
        df_debug = pd.DataFrame()
    if not res_df.empty:
        res_df['Returns'] = res_df['Equity'].pct_change()
        downside_returns  = res_df.loc[res_df['Returns'] < 0, 'Returns']
        downside_std      = downside_returns.std() * np.sqrt(252)
        total_ret         = (res_df['Equity'].iloc[-1] / init_cap) - 1
        days              = (res_df.index[-1] - res_df.index[0]).days
        cagr              = (1 + total_ret) ** (365 / days) - 1 if days > 0 else 0
        sortino           = cagr / downside_std if downside_std > 0 else 0
        metrics           = {
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else 99.9,
            'sortino': sortino,
            'total_buy_fees':  total_buy_fees,
            'total_sell_fees': total_sell_fees,
            'total_fees':      total_buy_fees + total_sell_fees,
            'total_tax_paid':  total_tax_paid,
            'tax_pending_end': pending_tax_at_end,
            'tax_log':         tax_log,
            'include_fees':    include_fees,
            'include_tax':     include_tax,
            'tax_strategy':    tax_strategy,
            'forced_count':    forced_count,
            'negative_cash_days': negative_cash_days,
            'total_dividends':    cum_dividends,
        }
    else:
        metrics = {'profit_factor': 0, 'sortino': 0,
                   'total_buy_fees': 0, 'total_sell_fees': 0, 'total_fees': 0,
                   'total_tax_paid': 0, 'tax_pending_end': 0, 'tax_log': [],
                   'include_fees': include_fees, 'include_tax': include_tax,
                   'tax_strategy': tax_strategy, 'forced_count': 0, 'negative_cash_days': 0,
                   'total_dividends': 0}
    yearly_stats = []
    def calc_mdd(series):
        peak = series.cummax()
        return ((series - peak) / peak).min()
    prev_equity = init_cap
    for yr in res_df.index.year.unique():
        df_yr      = res_df[res_df.index.year == yr]
        end_equity = df_yr['Equity'].iloc[-1]
        yr_return  = (end_equity - prev_equity) / prev_equity
        yr_mdd     = calc_mdd(df_yr['Equity'])
        yearly_stats.append({
            "연도": yr,
            "수익률": yr_return,
            "MDD": yr_mdd,
            "기말자산": end_equity,
            "수수료": yearly_fee_log.get(yr, 0.0),
            "양도세": yearly_tax_log.get(yr, 0.0),
            "실현손익": yearly_realized_log.get(yr, 0.0),
            "배당": yearly_div_log.get(yr, 0.0),
        })
        prev_equity = end_equity
    return res_df, metrics, pd.DataFrame(yearly_stats).set_index("연도"), df_debug
