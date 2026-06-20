"""
동파법 마스터 v9.8 + 카마릴라 R4 돌파매매 오버레이 결합 백테스트.

아이디어:
  매일 동파법이 "오늘 매수에 쓸 돈"을 먼저 다 쓰게 한 뒤, 남은 현금(real_cash)의
  overlay_fraction(기본 70%) 만큼을 카마릴라 R4 변형 돌파전략에 투입한다.
  카마릴라 포지션은 그 다음날 시가에 무조건 청산되어 현금이 회수되고,
  회수된 현금은 그 다음날 동파법의 매수 자금으로 다시 쓰일 수 있다 (자본 재활용).

핵심 설계 포인트 (간섭 방지):
  - 매일 루프 순서: ① 전날 열어둔 카마릴라 포지션을 '오늘 시가'로 청산 → 현금 회수
                    ② 동파법의 매도/매수 스텝을 원본 로직 그대로 실행 (real_cash 사용)
                    ③ 동파법이 다 쓰고 남은 real_cash 의 overlay_fraction 만큼만
                       오늘 신호가 있으면 카마릴라에 투입 (단일 슬롯)
  - 따라서 동파법은 항상 '자기 몫'을 100% 먼저 가져간 �다음에만 오버레이가 남은 돈을
    건드리므로, 오버레이가 동파법의 당일 매수 실행을 줄이는 일은 구조적으로 없다.
  - RESET_CYCLE 가상자본 재계산은 동파법 자신의 cum_profit/cum_loss/배당만 사용
    (카마릴라 손익은 섞지 않음 — 두 전략의 회계를 분리).
  - overlay_enabled=False 로 두면 ①③ 스텝이 통째로 스킵되어 run_backtest_fixed 와
    완전히 동일한 루프가 된다 (검증: verify_overlay_off.py).
"""
import numpy as np
import pandas as pd

from dongpa_core import (
    load_local_data,
    calc_mode_series, calc_qs_strength, qs_label_and_mul, loss_streak_mul,
    LOCAL_PARAMS, MAX_SLOTS_SAFE, MAX_SLOTS_OFFENSE, RESET_CYCLE, max_slots_for,
    QS_LOW_THRESH, QS_HIGH_THRESH, QS_HIGH_SLOT_CAP, LOSS_STREAK_WIN,
)


def compute_camarilla_signal(df, coef=0.70, vol_filter_pct=0.80):
    """verify_core.py 의 compute_signal_frame / 진입-청산 로직을 그대로 가져와
    df(SOXL_O/H/L/SOXL) 기준으로 신호·진입가·청산가·트레이드수익률을 계산한다."""
    O = df['SOXL_O'].values
    H = df['SOXL_H'].values
    L = df['SOXL_L'].values
    C = df['SOXL'].values
    n = len(df)
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
        'signal': signal,
        'entry_raw': entry_raw,
        'nextO': nextO,
    }, index=df.index)


def run_combined_backtest(df, start_date, end_date, init_cap,
                           overlay_enabled=True, overlay_fraction=0.70,
                           coef=0.70, vol_filter_pct=0.80,
                           cama_fee_rate=0.0005, cama_slippage_pct=0.0010,
                           include_fees=False, include_tax=False,
                           cama_include_tax=False,
                           buy_fee_rate=0.00015, sell_fee_rate=0.0001706,
                           tax_deduction_usd=1786.0, tax_rate=0.22,
                           tax_strategy='A', custom_schedule=None):
    """run_backtest_fixed 와 동일한 동파법 루프 + 카마릴라 오버레이.
    overlay_enabled=False 이면 오버레이 관련 두 스텝(①③)이 완전히 스킵되어
    run_backtest_fixed 와 동일한 결과를 낸다.

    include_tax: 동파법 자신의 실현손익에 양도세를 매길지 여부.
    cama_include_tax: 카마릴라 실현손익도 (동파법과 합산해서) 같은 해외주식 양도세
        풀에 넣을지 여부. 둘 다 False면 기존(세금 없음)과 완전히 동일.
        둘 중 하나만 켜면 그 쪽 손익만 과세 대상이 되고, 둘 다 켜면 실제
        세법처럼 두 전략의 손익을 합산해서 연 1회 과세한다."""
    TAX_SCHEDULES = {'A': [(1.00, (5, 1), (5, 31))]}
    if tax_strategy == 'B' and custom_schedule is not None and len(custom_schedule) > 0:
        tax_tranches_def = custom_schedule
    else:
        tax_tranches_def = TAX_SCHEDULES.get('A', [(1.00, (5, 1), (5, 31))])
    if df is None:
        return None, None, None, None, None

    mode_daily, rsi_daily = calc_mode_series(df['QQQ'])
    qs_daily = calc_qs_strength(df)
    cama_sig = compute_camarilla_signal(df, coef=coef, vol_filter_pct=vol_filter_pct)

    sim_df = pd.concat([df['SOXL'], df['SOXL_Div'], mode_daily, rsi_daily, qs_daily,
                         cama_sig['signal'], cama_sig['entry_raw'], cama_sig['nextO']], axis=1)
    sim_df.columns = ['Price', 'Div', 'Mode', 'RSI', 'QS', 'CamaSignal', 'CamaEntryRaw', 'CamaNextO']
    sim_df['Div'] = sim_df['Div'].fillna(0)
    sim_df = sim_df.dropna(subset=['Price', 'Mode', 'QS'])
    mask = (sim_df.index >= pd.to_datetime(start_date)) & (sim_df.index <= pd.to_datetime(end_date))
    sim_df = sim_df[mask]
    if sim_df.empty:
        return None, None, None, None, None
    sim_df['Prev_Price'] = sim_df['Price'].shift(1)
    sim_df['Prev_QS'] = sim_df['QS'].shift(1)
    sim_df['Prev_Mode'] = sim_df['Mode'].shift(1)
    sim_df = sim_df.dropna(subset=['Prev_Price', 'Prev_QS', 'Prev_Mode'])

    real_cash = init_cap
    cum_profit = 0.0
    cum_loss = 0.0
    cum_dividends = 0.0
    slots = []
    equity_curve = []
    debug_logs = []
    cama_log = []
    gross_profit = 0.0
    gross_loss = 0.0
    cycle_days = 0
    slot_sizes = {'Safe': init_cap / MAX_SLOTS_SAFE, 'Offense': init_cap / MAX_SLOTS_OFFENSE}
    recent_slot_outcomes = []
    total_buy_fees = 0.0
    total_sell_fees = 0.0
    annual_realized = 0.0
    annual_realized_tax = 0.0
    annual_dongpa_tax_base = 0.0  # annual_realized_tax 중 '동파법 귀속분'만 (대칭 보정용)
    total_tax_paid = 0.0
    total_tax_paid_for_virtual = 0.0  # 가상자본 계산에 쓰는 '동파법 귀속' 세금만 누적
    last_year_seen = None
    tax_log = []
    yearly_realized_log = {}
    yearly_fee_log = {}
    yearly_tax_log = {}
    yearly_div_log = {}
    yearly_cama_pnl_log = {}
    pending_tranches = []
    forced_count = 0
    negative_cash_days = 0

    cama_position = None     # {'invest_amt', 'entry_date', 'entry_raw', 'next_open'}
    cama_trade_count = 0
    cama_win_count = 0
    cama_total_pnl = 0.0

    for date, row in sim_df.iterrows():
        price = row['Price']
        div_amt = float(row.get('Div', 0) or 0)
        prev_price = row['Prev_Price']
        prev_qs = row['Prev_QS']
        prev_mode = row['Prev_Mode']
        cur_year = date.year
        cama_pnl_for_tax = 0.0

        # ── ① 전날 열어둔 카마릴라 포지션을 '오늘 시가'로 청산 ──────────
        if overlay_enabled and cama_position is not None:
            exit_eff = cama_position['next_open'] * (1 - cama_slippage_pct)
            entry_eff = cama_position['entry_raw'] * (1 + cama_slippage_pct)
            if not np.isnan(exit_eff) and entry_eff > 0:
                gross_ret = exit_eff / entry_eff - 1
                trade_ret = gross_ret - cama_fee_rate * 2
            else:
                trade_ret = 0.0
            proceeds = cama_position['invest_amt'] * (1 + trade_ret)
            pnl = proceeds - cama_position['invest_amt']
            real_cash += proceeds
            cama_total_pnl += pnl
            cama_pnl_for_tax = pnl
            yearly_cama_pnl_log[cur_year] = yearly_cama_pnl_log.get(cur_year, 0.0) + pnl
            cama_trade_count += 1
            if pnl > 0:
                cama_win_count += 1
            cama_log.append({
                "진입일": cama_position['entry_date'].date(), "청산일": date.date(),
                "투입금": f"${cama_position['invest_amt']:,.0f}",
                "수익률": f"{trade_ret*100:+.2f}%", "손익": f"${pnl:,.2f}",
            })
            cama_position = None

        # ── ② 동파법 본 로직 (run_backtest_fixed 와 동일) ──────────────
        if div_amt > 0:
            held_shares = sum(s['shares'] for s in slots)
            if held_shares > 0:
                div_cash = div_amt * held_shares
                real_cash += div_cash
                cum_dividends += div_cash
                yearly_div_log[cur_year] = yearly_div_log.get(cur_year, 0.0) + div_cash

        if last_year_seen is not None and cur_year != last_year_seen:
            yearly_realized_log[last_year_seen] = annual_realized
            if include_tax or cama_include_tax:
                annual_tax = max(0.0, annual_realized_tax - tax_deduction_usd) * tax_rate
                # 합산 세액 중 '동파법 귀속분' 비율. cama_include_tax=False 면 항상 1.0
                # (annual_dongpa_tax_base == annual_realized_tax 이므로 기존 동작과 100% 동일).
                # 둘 다 켜져 있을 때만 의미가 생기며, 음수 비대칭(한쪽 손실)으로 비율이
                # [0,1] 밖으로 튀는 극단적 케이스를 막기 위해 클램프한다.
                if annual_tax > 0 and annual_realized_tax != 0:
                    dongpa_share = annual_dongpa_tax_base / annual_realized_tax
                    dongpa_share = min(1.0, max(0.0, dongpa_share))
                else:
                    dongpa_share = 1.0
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
                                total_tax_paid_for_virtual += actual * dongpa_share
                                yearly_tax_log[last_year_seen] = yearly_tax_log.get(last_year_seen, 0.0) + actual
                                tax_log.append((date, actual, 'dec_anticipated'))
                            remaining = tax_due - actual
                            if remaining > 1e-6:
                                pending_tranches.append({
                                    'amount': remaining,
                                    'earliest': pd.Timestamp(year=cur_year, month=1, day=1),
                                    'force': pd.Timestamp(year=cur_year, month=1, day=31),
                                    'paid': 0.0, 'year': last_year_seen, 'dongpa_share': dongpa_share,
                                })
                        else:
                            pending_tranches.append({
                                'amount': annual_tax * frac,
                                'earliest': pd.Timestamp(year=cur_year, month=em, day=ed),
                                'force': pd.Timestamp(year=cur_year, month=fm, day=fd),
                                'paid': 0.0, 'year': last_year_seen, 'dongpa_share': dongpa_share,
                            })
            annual_realized = 0.0
            annual_realized_tax = 0.0
            annual_dongpa_tax_base = 0.0
        last_year_seen = cur_year
        if cama_include_tax:
            # 카마릴라 청산손익도 동파법과 같은 해외주식 양도세 풀에 합산
            # (위 ①에서 계산한 pnl을 연도 롤오버 처리가 끝난 뒤에 더해야
            #  새해 첫날 거래분이 직전 연도 세금 계산에 섞이지 않는다)
            annual_realized_tax += cama_pnl_for_tax

        ls_mul = loss_streak_mul(recent_slot_outcomes)
        if prev_qs < QS_LOW_THRESH:
            qs_label = 'Low'
        elif prev_qs > QS_HIGH_THRESH:
            qs_label = 'High'
        else:
            qs_label = 'Normal'
        ls_label = 'ON' if ls_mul < 1.0 else '-'
        _, qs_mul, _ = qs_label_and_mul(prev_qs)
        effective_max = QS_HIGH_SLOT_CAP if prev_qs > QS_HIGH_THRESH else max_slots_for(prev_mode)

        sold_idx = []
        sold_qty_total = 0
        sold_pnl_total = 0.0
        for i in range(len(slots) - 1, -1, -1):
            s = slots[i]
            s['days'] += 1
            rule = LOCAL_PARAMS.get(s['birth_mode'], LOCAL_PARAMS['Safe'])
            if (price >= s['buy_price'] * rule['sell']) or (s['days'] >= rule['time']):
                gross_rev = s['shares'] * price
                sell_fee = gross_rev * sell_fee_rate if include_fees else 0.0
                net_rev = gross_rev - sell_fee
                cost_basis = s.get('cost_basis', s['shares'] * s['buy_price'])
                prof = net_rev - cost_basis
                real_cash += net_rev
                total_sell_fees += sell_fee
                yearly_fee_log[cur_year] = yearly_fee_log.get(cur_year, 0.0) + sell_fee
                annual_realized += prof
                if include_tax:
                    annual_realized_tax += prof
                    annual_dongpa_tax_base += prof
                if prof > 0:
                    cum_profit += prof; gross_profit += prof
                    recent_slot_outcomes.append(True)
                else:
                    cum_loss += abs(prof); gross_loss += abs(prof)
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
                "날짜": date.date(), "Action": "매도", "적용 모드": prev_mode,
                "QS신호": f"{qs_label} ({prev_qs:.3f})", "LS가드": ls_label, "최대슬롯": effective_max,
                "종가": f"${price:.2f}", "수량": f"{-sold_qty_total:+,d}",
                "실현손익": f"${sold_pnl_total:,.2f}", "Balance_Qty": f"{balance_qty:,d}",
                "Total_Cash": f"${real_cash:,.0f}", "Allocated_Cap": f"${allocated_cap:,.0f}",
                "Total_Asset": f"${total_asset:,.0f}", "Return_Pct": f"{(total_asset/init_cap-1)*100:+.2f}%",
            })

        curr_rule = LOCAL_PARAMS.get(prev_mode, LOCAL_PARAMS['Safe'])
        cur_slot_size = slot_sizes[prev_mode]
        loc_price = prev_price * (1 + curr_rule['buy'])
        if price <= loc_price and len(slots) < effective_max:
            amt = min(real_cash, cur_slot_size * qs_mul * ls_mul)
            shares = int(amt / loc_price)
            if shares > 0:
                invested = shares * price
                buy_fee = invested * buy_fee_rate if include_fees else 0.0
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
                    "날짜": date.date(), "Action": "매수", "적용 모드": prev_mode,
                    "QS신호": f"{qs_label} ({prev_qs:.3f})", "LS가드": ls_label, "최대슬롯": effective_max,
                    "종가": f"${price:.2f}", "수량": f"+{shares:,d}",
                    "실현손익": "$0.00", "Balance_Qty": f"{balance_qty:,d}",
                    "Total_Cash": f"${real_cash:,.0f}", "Allocated_Cap": f"${allocated_cap:,.0f}",
                    "Total_Asset": f"${total_asset:,.0f}", "Return_Pct": f"{(total_asset/init_cap-1)*100:+.2f}%",
                })

        # ── ③ 동파법이 쓰고 남은 현금의 overlay_fraction 만큼 카마릴라 신규 진입 ──
        if overlay_enabled and row['CamaSignal'] and cama_position is None:
            next_open = row['CamaNextO']
            entry_raw = row['CamaEntryRaw']
            if not np.isnan(next_open) and entry_raw > 0:
                leftover = max(0.0, real_cash)
                invest_amt = overlay_fraction * leftover
                if invest_amt > 1e-6:
                    real_cash -= invest_amt
                    cama_position = {
                        'invest_amt': invest_amt, 'entry_date': date,
                        'entry_raw': entry_raw, 'next_open': next_open,
                    }

        dongpa_equity = sum(s['shares'] * price for s in slots)
        cama_equity = cama_position['invest_amt'] if cama_position is not None else 0.0
        current_equity = real_cash + dongpa_equity + cama_equity
        equity_curve.append({'Date': date, 'Equity': current_equity,
                              'DongpaEquity': real_cash + dongpa_equity,
                              'CamaEquity': cama_equity})

        cycle_days += 1
        is_cycle_end = (cycle_days >= RESET_CYCLE)
        for tranche in pending_tranches:
            remaining = tranche['amount'] - tranche['paid']
            if remaining <= 1e-6:
                continue
            past_earliest = date >= tranche['earliest']
            past_force = date >= tranche['force']
            if past_force:
                actual = remaining
                real_cash -= actual
                total_tax_paid += actual
                total_tax_paid_for_virtual += actual * tranche.get('dongpa_share', 1.0)
                tranche['paid'] += actual
                yearly_tax_log[cur_year] = yearly_tax_log.get(cur_year, 0.0) + actual
                tax_log.append((date, actual, 'force'))
                forced_count += 1
            elif past_earliest and is_cycle_end:
                actual = min(remaining, max(0.0, real_cash))
                if actual > 0:
                    real_cash -= actual
                    total_tax_paid += actual
                    total_tax_paid_for_virtual += actual * tranche.get('dongpa_share', 1.0)
                    tranche['paid'] += actual
                    yearly_tax_log[cur_year] = yearly_tax_log.get(cur_year, 0.0) + actual
                    tax_log.append((date, actual, 'cycle'))
        pending_tranches = [t for t in pending_tranches if (t['amount'] - t['paid']) > 1e-6]
        if real_cash < 0:
            negative_cash_days += 1
        if is_cycle_end:
            # ★ 동파법 자신의 cum_profit/cum_loss/배당만 사용 (카마릴라 손익 미포함).
            #   세금은 total_tax_paid_for_virtual(= 합산세액 중 동파법 귀속분만)을 써서,
            #   카마릴라가 낸 세금 때문에 동파법 슬롯 크기가 부당하게 깎이지 않게 한다.
            #   (cama_include_tax=False 면 dongpa_share가 항상 1.0이라 기존 동작과 동일)
            virtual = init_cap + (cum_profit * 0.7) - (cum_loss * 0.6) - total_tax_paid_for_virtual + cum_dividends * 0.7
            if virtual < 1000:
                virtual = 1000
            slot_sizes['Safe'] = virtual / MAX_SLOTS_SAFE
            slot_sizes['Offense'] = virtual / MAX_SLOTS_OFFENSE
            cycle_days = 0

    if last_year_seen is not None:
        yearly_realized_log.setdefault(last_year_seen, annual_realized)
    pending_unrealized = (max(0.0, annual_realized_tax - tax_deduction_usd) * tax_rate
                          if (include_tax or cama_include_tax) else 0.0)
    pending_unfunded = sum(t['amount'] - t['paid'] for t in pending_tranches)
    pending_tax_at_end = pending_unrealized + pending_unfunded

    res_df = pd.DataFrame(equity_curve).set_index('Date')
    df_debug = pd.DataFrame(debug_logs).reset_index(drop=True) if debug_logs else pd.DataFrame()
    df_cama = pd.DataFrame(cama_log) if cama_log else pd.DataFrame()

    if not res_df.empty:
        res_df['Returns'] = res_df['Equity'].pct_change()
        downside_returns = res_df.loc[res_df['Returns'] < 0, 'Returns']
        downside_std = downside_returns.std() * np.sqrt(252)
        total_ret = (res_df['Equity'].iloc[-1] / init_cap) - 1
        days = (res_df.index[-1] - res_df.index[0]).days
        cagr = (1 + total_ret) ** (365 / days) - 1 if days > 0 else 0
        sortino = cagr / downside_std if downside_std > 0 else 0
        peak = res_df['Equity'].cummax()
        mdd = ((res_df['Equity'] - peak) / peak).min()
        metrics = {
            'cagr': cagr, 'mdd': mdd, 'calmar': (cagr / abs(mdd)) if mdd < 0 else np.nan,
            'final_equity': res_df['Equity'].iloc[-1],
            'profit_factor': gross_profit / gross_loss if gross_loss > 0 else 99.9,
            'sortino': sortino,
            'total_buy_fees': total_buy_fees, 'total_sell_fees': total_sell_fees,
            'total_fees': total_buy_fees + total_sell_fees,
            'total_tax_paid': total_tax_paid, 'tax_pending_end': pending_tax_at_end,
            'tax_log': tax_log, 'include_fees': include_fees, 'include_tax': include_tax,
            'cama_include_tax': cama_include_tax,
            'tax_strategy': tax_strategy, 'forced_count': forced_count,
            'negative_cash_days': negative_cash_days, 'total_dividends': cum_dividends,
            'cama_trade_count': cama_trade_count,
            'cama_win_rate': (cama_win_count / cama_trade_count) if cama_trade_count > 0 else np.nan,
            'cama_total_pnl': cama_total_pnl,
            'overlay_enabled': overlay_enabled, 'overlay_fraction': overlay_fraction,
        }
    else:
        metrics = {'cagr': np.nan, 'mdd': np.nan, 'calmar': np.nan, 'final_equity': np.nan}

    yearly_stats = []
    def calc_mdd(series):
        peak = series.cummax()
        return ((series - peak) / peak).min()
    prev_equity = init_cap
    for yr in res_df.index.year.unique():
        df_yr = res_df[res_df.index.year == yr]
        end_equity = df_yr['Equity'].iloc[-1]
        yr_return = (end_equity - prev_equity) / prev_equity
        yr_mdd = calc_mdd(df_yr['Equity'])
        yearly_stats.append({
            "연도": yr, "수익률": yr_return, "MDD": yr_mdd, "기말자산": end_equity,
            "수수료": yearly_fee_log.get(yr, 0.0), "양도세": yearly_tax_log.get(yr, 0.0),
            "실현손익": yearly_realized_log.get(yr, 0.0), "배당": yearly_div_log.get(yr, 0.0),
            "카마릴라손익": yearly_cama_pnl_log.get(yr, 0.0),
        })
        prev_equity = end_equity

    return res_df, metrics, pd.DataFrame(yearly_stats).set_index("연도"), df_debug, df_cama


if __name__ == "__main__":
    df = load_local_data(".")
    START, END, INIT_CAP = "2012-01-01", "2026-06-18", 10000

    res_off, m_off, yr_off, dbg_off, cama_off = run_combined_backtest(
        df, START, END, INIT_CAP, overlay_enabled=False)
    res_on, m_on, yr_on, dbg_on, cama_on = run_combined_backtest(
        df, START, END, INIT_CAP, overlay_enabled=True, overlay_fraction=0.70)

    print("=== 동파법 단독 (overlay OFF) ===")
    print(f"최종자산: ${m_off['final_equity']:,.0f}  CAGR: {m_off['cagr']*100:.2f}%  MDD: {m_off['mdd']*100:.2f}%  Calmar: {m_off['calmar']:.2f}")
    print("\n=== 동파법 + 카마릴라 오버레이 (overlay ON, 70%) ===")
    print(f"최종자산: ${m_on['final_equity']:,.0f}  CAGR: {m_on['cagr']*100:.2f}%  MDD: {m_on['mdd']*100:.2f}%  Calmar: {m_on['calmar']:.2f}")
    print(f"카마릴라 거래수: {m_on['cama_trade_count']}  승률: {m_on['cama_win_rate']*100:.1f}%  카마릴라 누적손익: ${m_on['cama_total_pnl']:,.0f}")
