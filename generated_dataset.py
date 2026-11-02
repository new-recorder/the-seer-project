# generate_dataset.py
import numpy as np
import pandas as pd
from scipy.stats import t
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Set random seed once at the top
np.random.seed(42)

def run_simulation():
    # 1. Calendar setup
    start_target = pd.Timestamp('2006-01-02')
    warmup_start = start_target - pd.tseries.offsets.BusinessDay(150)
    total_periods = 150 + 5200 + 5
    full_dates = pd.bdate_range(start=warmup_start, periods=total_periods)
    N = len(full_dates)
    
    # Map dates for lookup
    date_to_idx = {d: i for i, d in enumerate(full_dates)}
    
    # Brent Anchors
    brent_anchors_raw = {
        '2006-01-02': 62.0, '2006-07-14': 78.0, '2006-10-15': 60.0, '2007-01-18': 54.0, '2007-07-15': 76.0,
        '2007-12-31': 92.0, '2008-03-15': 105.0, '2008-07-03': 142.0, '2008-09-15': 95.0, '2008-12-24': 40.0,
        '2009-03-01': 45.0, '2009-06-15': 70.0, '2009-12-01': 77.0, '2010-04-15': 87.0, '2010-06-01': 73.0,
        '2010-12-31': 94.0, '2011-04-29': 124.0, '2011-10-03': 103.0, '2012-03-15': 125.0, '2012-06-25': 90.0,
        '2012-09-14': 115.0, '2013-04-15': 98.0, '2013-08-27': 116.0, '2013-12-31': 110.0, '2014-06-20': 114.0,
        '2014-09-15': 97.0, '2014-12-16': 60.0, '2015-01-28': 48.0, '2015-06-10': 65.0, '2015-08-24': 43.0,
        '2015-12-31': 37.0, '2016-01-20': 28.0, '2016-06-08': 50.0, '2016-10-15': 52.0, '2017-01-31': 56.0,
        '2017-06-21': 46.0, '2017-12-29': 66.0, '2018-05-15': 78.0, '2018-10-03': 85.0, '2018-12-24': 51.0,
        '2019-04-25': 74.0, '2019-08-15': 58.0, '2019-12-31': 66.0, '2020-03-02': 51.0, '2020-03-31': 25.0,
        '2020-04-21': 19.0, '2020-05-15': 30.0, '2020-09-15': 41.0, '2020-12-31': 51.0, '2021-03-10': 69.0,
        '2021-07-05': 76.0, '2021-10-25': 85.0, '2021-12-01': 70.0, '2021-12-31': 78.0, '2022-03-08': 125.0,
        '2022-04-11': 100.0, '2022-06-08': 123.0, '2022-09-26': 84.0, '2022-12-31': 82.0, '2023-03-20': 72.0,
        '2023-04-12': 85.0, '2023-06-26': 73.0, '2023-09-27': 95.0, '2023-12-13': 75.0, '2024-04-12': 91.0,
        '2024-09-10': 70.0, '2024-12-31': 74.0, '2025-01-15': 82.0, '2025-04-09': 62.0, '2025-06-19': 78.0,
        '2025-09-15': 67.0, '2025-12-31': 61.0
    }
    
    # Differential Anchors (Bonny Light premium to Brent)
    diff_anchors_raw = {
        '2006-01': 1.8, '2008-01': 2.5, '2011-06': 3.2, '2013-12': 2.5, '2014-12': 1.2,
        '2016-01': 0.8, '2017-12': 1.8, '2019-12': 1.8, '2020-03': 0.3, '2020-05': -0.5,
        '2020-09': 1.0, '2021-12': 2.0, '2022-06': 3.5, '2023-01': 2.0, '2025-12': 1.2
    }
    
    # Shock Windows
    shock_windows_raw = [
        ('2008-06-15', '2008-07-20'), ('2008-09-15', '2009-02-28'),
        ('2011-02-20', '2011-04-30'), ('2014-11-15', '2015-02-15'),
        ('2015-08-15', '2016-02-15'), ('2018-10-15', '2018-12-31'),
        ('2019-09-14', '2019-10-10'), ('2020-03-01', '2020-05-31'),
        ('2022-02-24', '2022-06-30'), ('2025-04-02', '2025-05-15'),
        ('2025-06-12', '2025-06-30')
    ]
    
    # Nigerian Disruptions
    ng_disruptions_raw = [
        ('2006-02', '2006-06', 0.20), ('2008-06', '2009-09', 0.40),
        ('2012-10', '2012-11', 0.15), ('2016-02', '2016-09', 0.55),
        ('2021-06', '2022-12', 0.45), ('2022-06', '2022-10', 0.35),
        ('2024-02', '2024-04', 0.15)
    ]
    
    is_shock_day = np.zeros(N, dtype=bool)
    for start_s, end_s in shock_windows_raw:
        s_dt = pd.Timestamp(start_s)
        e_dt = pd.Timestamp(end_s)
        for i, d in enumerate(full_dates):
            if s_dt <= d <= e_dt:
                is_shock_day[i] = True
                
    # Interpolation helper on business-day step grid (prevents Monday gap artifacts)
    def interp_bday(series, full_dates):
        series = series.sort_index()
        pos = np.searchsorted(full_dates.values, series.index.values)
        keep = ~pd.Series(pos).duplicated(keep='last').values
        return np.interp(np.arange(len(full_dates)), pos[keep], series.values[keep])
        
    brent_anchor_series = pd.Series(index=pd.to_datetime(list(brent_anchors_raw.keys())), data=list(brent_anchors_raw.values()))
    A_t = interp_bday(np.log(brent_anchor_series), full_dates)
    
    diff_anchor_series = pd.Series(index=pd.to_datetime([k + '-01' for k in diff_anchors_raw.keys()]), data=list(diff_anchors_raw.values()))
    d_anchor_t = interp_bday(diff_anchor_series, full_dates)
    
    # Independent Driver 1: DXY daily returns
    dxy_scale_base = 0.0045 / np.sqrt(5.0 / 3.0)
    dxy_returns = np.zeros(N)
    dxy_h = np.ones(N) * 1.0
    z_dxy = t.rvs(df=5.0, size=N) / np.sqrt(5.0 / 3.0)
    for i in range(N):
        curr_std = dxy_scale_base * (1.35 if is_shock_day[i] else 1.0) * np.sqrt(dxy_h[i])
        dxy_returns[i] = z_dxy[i] * curr_std
        if i < N - 1:
            dxy_h[i+1] = np.clip(0.02 + 0.10 * (z_dxy[i]**2) * dxy_h[i] + 0.88 * dxy_h[i], 0.25, 6.0)
            
    dxy_ret_series = pd.Series(dxy_returns)
    dxy_logret_5d = dxy_ret_series.rolling(window=5, min_periods=1).sum().values
    
    # Independent Driver 2: EIA inventory surprise (Wednesday)
    inv_z_raw = np.zeros(N)
    wed_indices = [i for i, d in enumerate(full_dates) if d.weekday() == 2]
    inv_shocks = t.rvs(df=5.0, size=len(wed_indices)) / np.sqrt(5.0 / 3.0) * 3.2
    
    weekly_inv_series = pd.Series(index=full_dates, data=np.nan)
    for idx_w, w_pos in enumerate(wed_indices):
        weekly_inv_series.iloc[w_pos] = inv_shocks[idx_w]
        
    weekly_nonan = weekly_inv_series.dropna()
    rolling_mean_w = weekly_nonan.rolling(window=52, min_periods=20).mean()
    rolling_std_w = weekly_nonan.rolling(window=52, min_periods=20).std()
    rolling_z_w = (weekly_nonan - rolling_mean_w) / rolling_std_w
    
    inv_z_series = pd.Series(index=full_dates, data=0.0)
    for d_w in weekly_nonan.index:
        pos = date_to_idx[d_w]
        val_z = rolling_z_w.loc[d_w]
        if pd.notna(val_z):
            inv_z_series.iloc[pos] = val_z
            
    inv_z_filled = inv_z_series.replace(0.0, np.nan).ffill().fillna(0.0).values
    
    # Independent Driver 3: Nigerian output (Monthly 2003-01 to 2025-12)
    months_range = pd.period_range(start='2003-01', end='2025-12', freq='M')
    M_len = len(months_range)
    ng_base = 2.15 + np.cumsum(np.random.normal(0, 0.005, M_len))
    ng_base = np.clip(ng_base, 2.05, 2.25)
    
    ng_noise = np.zeros(M_len)
    for m in range(1, M_len):
        ng_noise[m] = 0.6 * ng_noise[m-1] + np.random.normal(0, 0.025)
        
    disruption_depth_m = np.zeros(M_len)
    for start_m, end_m, depth in ng_disruptions_raw:
        s_per = pd.Period(start_m, freq='M')
        e_per = pd.Period(end_m, freq='M')
        for m_idx, m_p in enumerate(months_range):
            if s_per <= m_p <= e_per:
                disruption_depth_m[m_idx] += depth
                
    minor_outage_months = ['2007-04', '2010-08', '2013-02', '2017-09', '2019-05', '2023-08']
    for mo in minor_outage_months:
        p_mo = pd.Period(mo, freq='M')
        for m_idx, m_p in enumerate(months_range):
            if m_p == p_mo:
                disruption_depth_m[m_idx] += 0.12
                
    ng_output_m = ng_base + ng_noise - disruption_depth_m
    ng_output_m = np.clip(ng_output_m, 1.0, 2.5)
    
    monthly_pub_data = pd.Series(index=months_range, data=ng_output_m)
    ng_z_m = (monthly_pub_data - monthly_pub_data.rolling(window=24, min_periods=12).mean()) / monthly_pub_data.rolling(window=24, min_periods=12).std()
    ng_z_m = ng_z_m.clip(-5.0, 5.0).fillna(0.0)
    
    ng_output_z_daily = np.zeros(N)
    current_z = 0.0
    for m_idx, m_p in enumerate(months_range):
        pub_date = m_p.to_timestamp(how='end') + pd.Timedelta(days=15)
        z_val = ng_z_m.iloc[m_idx]
        for i, d in enumerate(full_dates):
            if m_idx == 0 and d < pub_date:
                ng_output_z_daily[i] = z_val
            elif i > 0 and full_dates[i-1] < pub_date <= d:
                current_z = z_val
            if d >= pub_date:
                ng_output_z_daily[i] = current_z
                
    disruption_severity_m = np.clip(disruption_depth_m / 0.8, 0.0, 1.0)
    ng_severity_daily = pd.Series(disruption_severity_m, index=months_range) \
                          .reindex(full_dates.to_period('M')).ffill().values

    # Regime Markov Chain simulation (0 calm, 1 up, 2 down, 3 shock)
    P_base = np.array([
        [0.965, 0.015, 0.015, 0.005],
        [0.025, 0.960, 0.005, 0.010],
        [0.025, 0.005, 0.960, 0.010],
        [0.040, 0.020, 0.020, 0.920]
    ])
    
    regime = np.zeros(N, dtype=int)
    regime[0] = 0
    
    regime_vols = [0.0125, 0.014, 0.016, 0.033]
    regime_drifts = [0.0, 0.0003, -0.0003, 0.0]
    
    z_brent_raw = t.rvs(df=4.5, size=N) / np.sqrt(4.5 / 2.5)
    h_brent = np.ones(N) * 1.0
    x_brent = np.zeros(N)
    brent_prices = np.zeros(N)
    bl_prices = np.zeros(N)
    bl_diffs = np.zeros(N)
    
    A_diff = np.gradient(A_t)
    
    for i in range(1, N):
        curr_r = regime[i-1]
        row = P_base[curr_r].copy()
        
        shock_wt = 20.0 if is_shock_day[i] else 0.6
        row[3] *= shock_wt
        
        slope = A_diff[i]
        if slope > 0.002:
            row[1] *= 2.5
        elif slope < -0.002:
            row[2] *= 2.5
        else:
            row[0] *= 1.8
            
        row /= row.sum()
        regime[i] = np.random.choice([0, 1, 2, 3], p=row)
        
        r_reg = regime[i]
        vol_reg = regime_vols[r_reg]
        drift_reg = regime_drifts[r_reg]
        
        inv_effect = 0.0
        z_inv_val = inv_z_filled[i]
        day_w = full_dates[i].weekday()
        if day_w == 2:
            inv_effect = -0.004 * z_inv_val
        elif day_w in [3, 4] or day_w in [0, 1]:
            inv_effect = -0.0005 * z_inv_val
            
        ng_lift = 0.0003 * ng_severity_daily[i]
        dxy_impact = -1.6 * dxy_returns[i]
        
        z_t = z_brent_raw[i]
        sqrt_h = np.sqrt(h_brent[i-1])
        
        dow_i = full_dates[i].weekday()
        dow_scale = 1.12 if dow_i == 2 else (1.05 if dow_i == 0 else 1.0)
        
        r_t = drift_reg + vol_reg * sqrt_h * z_t * dow_scale + dxy_impact + inv_effect + ng_lift
        
        h_brent[i] = np.clip(0.02 + 0.10 * (z_brent_raw[i-1]**2) * h_brent[i-1] + 0.88 * h_brent[i-1], 0.25, 6.0)
        x_brent[i] = np.clip(0.96 * x_brent[i-1] + r_t, -0.09, 0.09)
        
        log_b = A_t[i] + x_brent[i]
        b_val = np.exp(log_b)
        b_val = np.clip(b_val, 15.5, 144.0)
        brent_prices[i] = round(b_val, 2)
        
    brent_prices[0] = round(np.clip(np.exp(A_t[0]), 15.5, 144.0), 2)
    
    d_t = np.zeros(N)
    d_t[0] = d_anchor_t[0]
    
    z_diff = t.rvs(df=5.0, size=N)
    for i in range(1, N):
        target_diff = d_anchor_t[i] + 2.5 * ng_severity_daily[i]
        noise_std = 0.20 * (1.8 if is_shock_day[i] else 1.0)
        noise_d = z_diff[i] * noise_std
        
        d_prev = d_t[i-1]
        d_new = d_prev + 0.05 * (target_diff - d_prev) + noise_d
        
        b_curr = brent_prices[i]
        d_new = np.clip(d_new, max(-3.0, 15.0 - b_curr), min(6.0, 150.0 - b_curr))
        d_t[i] = d_new
        
    bl_raw = brent_prices + d_t
    bl_raw = np.clip(bl_raw, 15.0, 150.0)
    bl_prices = np.round(bl_raw, 2)
    
    bl_brent_diff_arr = np.round(bl_prices - brent_prices, 2)
    
    df_full = pd.DataFrame(index=full_dates)
    df_full['month'] = full_dates.month
    df_full['month_sin'] = np.sin(2 * np.pi * df_full['month'] / 12.0)
    df_full['month_cos'] = np.cos(2 * np.pi * df_full['month'] / 12.0)
    df_full['day_of_week'] = full_dates.dayofweek.astype(float)
    df_full['bl_price'] = bl_prices
    
    df_full['bl_logret_1d'] = np.log(df_full['bl_price'] / df_full['bl_price'].shift(1)).fillna(0.0)
    df_full['bl_vol_20d'] = df_full['bl_logret_1d'].rolling(window=20, min_periods=1).std().fillna(0.0)
    
    sma50 = df_full['bl_price'].rolling(window=50, min_periods=1).mean()
    df_full['bl_price_to_sma50'] = (df_full['bl_price'] / sma50).fillna(1.0)
    
    df_full['bl_brent_diff'] = bl_brent_diff_arr
    
    brent_series = pd.Series(brent_prices, index=full_dates)
    brent_logret_5d = np.log(brent_series / brent_series.shift(5)).fillna(0.0)
    df_full['brent_logret_5d'] = brent_logret_5d
    
    df_full['dxy_logret_5d'] = dxy_logret_5d
    df_full['ng_output_zscore'] = ng_output_z_daily
    df_full['us_inventory_surprise_z'] = inv_z_filled
    
    df_full['target_bl_price_t_plus_5d'] = df_full['bl_price'].shift(-5).bfill()
    
    start_idx = date_to_idx[pd.Timestamp('2006-01-02')]
    df_trimmed = df_full.iloc[start_idx : start_idx + 5200].copy()
    
    cols_to_keep = [
        'month_sin', 'month_cos', 'day_of_week', 'bl_price', 'bl_logret_1d',
        'bl_vol_20d', 'bl_price_to_sma50', 'bl_brent_diff', 'brent_logret_5d',
        'dxy_logret_5d', 'ng_output_zscore', 'us_inventory_surprise_z',
        'target_bl_price_t_plus_5d'
    ]
    df_final = df_trimmed[cols_to_keep].astype(np.float64)
    
    df_date_index = pd.DataFrame({
        'row_id': np.arange(5200),
        'date': full_dates[start_idx : start_idx + 5200].strftime('%Y-%m-%d')
    })
    
    return df_final, df_date_index, full_dates[start_idx : start_idx + 5200], brent_prices[start_idx : start_idx + 5200], regime[start_idx : start_idx + 5200]

df_data, df_dates, date_series, brent_series_trim, regime_trim = run_simulation()

df_data.to_csv('bonny_light_model_ready.csv', index=False)
df_dates.to_csv('bonny_light_date_index.csv', index=False)

report_lines = []
def log(msg=''):
    print(msg)
    report_lines.append(str(msg))

log("=== BONNY LIGHT COMMODITY MODEL VALIDATION REPORT ===")

assert df_data.shape == (5200, 13), f"Shape mismatch: {df_data.shape}"
assert list(df_data.columns) == [
    'month_sin', 'month_cos', 'day_of_week', 'bl_price', 'bl_logret_1d',
    'bl_vol_20d', 'bl_price_to_sma50', 'bl_brent_diff', 'brent_logret_5d',
    'dxy_logret_5d', 'ng_output_zscore', 'us_inventory_surprise_z',
    'target_bl_price_t_plus_5d'
], "Column order mismatch"

for col in df_data.columns:
    assert df_data[col].dtype == np.float64, f"Column {col} dtype is {df_data[col].dtype}"

assert df_data.isna().sum().sum() == 0, "Found NaN values"
assert np.isinf(df_data.values).sum() == 0, "Found Inf values"

first_date = date_series[0]
last_date = date_series[-1]
log(f"First date: {first_date.strftime('%Y-%m-%d')}, Last date: {last_date.strftime('%Y-%m-%d')}")
assert first_date == pd.Timestamp('2006-01-02'), f"First date mismatch: {first_date}"
assert pd.Timestamp('2025-11-20') <= last_date <= pd.Timestamp('2025-12-12'), f"Last date out of range: {last_date}"

for d in date_series:
    assert d.weekday() < 5, f"Non-weekday found: {d}"
assert len(date_series.unique()) == 5200, "Duplicate dates found"

dow_computed = date_series.dayofweek.astype(np.float64)
assert np.allclose(df_data['day_of_week'].values, dow_computed), "day_of_week mismatch"

assert df_data['bl_price'].min() >= 15.0 and df_data['bl_price'].max() <= 150.0, "bl_price bounds violated"
assert df_data['bl_brent_diff'].min() >= -3.0 and df_data['bl_brent_diff'].max() <= 6.0, "bl_brent_diff bounds violated"

assert np.allclose(df_data['target_bl_price_t_plus_5d'].values[:-5], df_data['bl_price'].values[5:]), "Target shift mismatch"

brent_clip_pct = np.mean((brent_series_trim <= 15.501) | (brent_series_trim >= 143.999)) * 100.0
log(f"Brent at clip bounds percentage: {brent_clip_pct:.2f}%")
assert brent_clip_pct < 1.0, f"Too many Brent clip hits: {brent_clip_pct:.2f}%"

distinct_ng = df_data['ng_output_zscore'].nunique()
log(f"Distinct ng_output_zscore values: {distinct_ng}")
assert distinct_ng <= 280, f"Too many distinct ng output values: {distinct_ng}"

inv_diff = df_data['us_inventory_surprise_z'].diff().fillna(0.0)
change_indices = np.where(inv_diff != 0.0)[0]
for idx in change_indices:
    d_w = date_series[idx]
    assert d_w.weekday() == 2, f"Inventory surprise changed on non-Wednesday: {d_w}"
log("Inventory surprise change check passed (changes only on Wednesdays).")

df_with_date = df_data.copy()
df_with_date['date'] = date_series
df_with_date['year'] = date_series.year

p2008 = df_with_date[df_with_date['year'] == 2008]['bl_price']
m2008_max = p2008.max()
log(f"2008 BL max: {m2008_max}")
assert 125 <= m2008_max <= 150, f"2008 max out of range: {m2008_max}"

p2008_dec = p2008[df_with_date['date'].dt.month == 12]
m2008_dec_min = p2008_dec.min()
log(f"2008 Dec BL min: {m2008_dec_min}")
assert m2008_dec_min < 55, f"2008 Dec min too high: {m2008_dec_min}"

for yr in [2011, 2012, 2013, 2014]:
    yr_mean = df_with_date[df_with_date['year'] == yr]['bl_price'].mean()
    log(f"{yr} BL mean: {yr_mean:.2f}")
    assert 85 <= yr_mean <= 125, f"{yr} mean out of range: {yr_mean}"

m2016_min = df_with_date[df_with_date['year'] == 2016]['bl_price'].min()
log(f"2016 BL min: {m2016_min}")
assert m2016_min < 40, f"2016 min too high: {m2016_min}"

m2020_min = df_with_date[df_with_date['year'] == 2020]['bl_price'].min()
log(f"2020 BL min: {m2020_min}")
assert m2020_min < 30, f"2020 min too high: {m2020_min}"

m2022_max = df_with_date[df_with_date['year'] == 2022]['bl_price'].max()
log(f"2022 BL max: {m2022_max}")
assert 105 <= m2022_max <= 140, f"2022 max out of range: {m2022_max}"

for yr in [2023, 2024, 2025]:
    yr_mean = df_with_date[df_with_date['year'] == yr]['bl_price'].mean()
    log(f"{yr} BL mean: {yr_mean:.2f}")
    assert 60 <= yr_mean <= 95, f"{yr} mean out of range: {yr_mean}"

brent_ret_series = np.log(brent_series_trim[1:] / brent_series_trim[:-1])
bl_ret_series = df_data['bl_logret_1d'].values[1:]
corr_bl_brent = np.corrcoef(bl_ret_series, brent_ret_series)[0, 1]
log(f"Correlation BL vs Brent daily returns: {corr_bl_brent:.4f}")
assert 0.95 <= corr_bl_brent <= 0.995, f"BL vs Brent return correlation out of range: {corr_bl_brent}"

corr_brent5d_dxy5d = np.corrcoef(df_data['brent_logret_5d'].values, df_data['dxy_logret_5d'].values)[0, 1]
log(f"Correlation Brent 5d vs DXY 5d: {corr_brent5d_dxy5d:.4f}")
assert -0.35 <= corr_brent5d_dxy5d <= -0.20, f"Brent 5d vs DXY 5d corr out of range: {corr_brent5d_dxy5d}"

corr_inv_brent5d = np.corrcoef(df_data['us_inventory_surprise_z'].values, df_data['brent_logret_5d'].values)[0, 1]
log(f"Correlation Inventory z vs Brent 5d return: {corr_inv_brent5d:.4f}")
assert -0.15 <= corr_inv_brent5d <= -0.05, f"Inventory z vs Brent 5d corr out of range: {corr_inv_brent5d}"

dxy_5d_series = df_data['dxy_logret_5d'].values
ac1_dxy = np.corrcoef(dxy_5d_series[1:], dxy_5d_series[:-1])[0, 1]
log(f"Lag-1 autocorrelation of dxy_logret_5d: {ac1_dxy:.4f}")
assert 0.6 <= ac1_dxy <= 0.9, f"DXY 5d lag-1 autocorr out of range: {ac1_dxy}"

predictor_cols = [c for c in df_data.columns if c != 'target_bl_price_t_plus_5d']
X_vif = sm.add_constant(df_data[predictor_cols])
vifs = [variance_inflation_factor(X_vif.values, i) for i in range(1, X_vif.shape[1])]
for c_name, vif_val in zip(predictor_cols, vifs):
    log(f"VIF for {c_name}: {vif_val:.2f}")
    assert vif_val < 5.0, f"VIF for {c_name} exceeds 5.0: {vif_val:.2f}"

log("\n=== STATISTICAL CHECKS & REPORTING ===")
log(df_data.describe().T.to_string())

yearly_summary = df_with_date.groupby('year')['bl_price'].agg(['min', 'mean', 'max'])
log("\nYearly Bonny Light min / mean / max:")
log(yearly_summary.to_string())

log("\nFull Correlation Matrix:")
log(df_data.corr().round(4).to_string())

from statsmodels.tsa.stattools import adfuller
adf_res = adfuller(df_data['bl_logret_1d'].dropna())
log(f"\nADF test on bl_logret_1d: ADF statistic = {adf_res[0]:.4f}, p-value = {adf_res[1]:.4e}")

sq_ret = df_data['bl_logret_1d'] ** 2
ac_sq_1 = sq_ret.autocorr(lag=1)
ac_sq_5 = sq_ret.autocorr(lag=5)
ac_sq_10 = sq_ret.autocorr(lag=10)
log(f"Squared returns autocorr - Lag 1: {ac_sq_1:.4f}, Lag 5: {ac_sq_5:.4f}, Lag 10: {ac_sq_10:.4f}")

kurt_val = df_data['bl_logret_1d'].kurtosis()
log(f"Excess kurtosis of bl_logret_1d: {kurt_val:.4f}")

df_with_regime = df_data.copy()
df_with_regime['regime'] = regime_trim
regime_names = {0: 'Calm', 1: 'Uptrend', 2: 'Downtrend', 3: 'Shock'}
for r_id, r_name in regime_names.items():
    sub = df_with_regime[df_with_regime['regime'] == r_id]
    count_r = len(sub)
    std_r = sub['bl_logret_1d'].std()
    log(f"Regime {r_name} (ID {r_id}): Days = {count_r}, Return Std = {std_r:.4f}")

for dow_val, dow_name in [(0, 'Monday'), (1, 'Tuesday'), (2, 'Wednesday'), (3, 'Thursday'), (4, 'Friday')]:
    sub_dow = df_with_regime[df_with_regime['day_of_week'] == dow_val]
    std_dow = sub_dow['bl_logret_1d'].std()
    log(f"Day of week {dow_name}: Return Std = {std_dow:.4f}")

train_df = df_data.iloc[0:3640]
val_df = df_data.iloc[3645:4420]
test_df = df_data.iloc[4425:5200]

train_dates = date_series[0:3640]
val_dates = date_series[3645:4420]
test_dates = date_series[4425:5200]

log(f"\nSplits Row Counts & Date Ranges:")
log(f"Train: rows 0-3639 ({len(train_df)} rows), {train_dates[0].strftime('%Y-%m-%d')} to {train_dates[-1].strftime('%Y-%m-%d')}")
log(f"Validation: rows 3645-4419 ({len(val_df)} rows), {val_dates[0].strftime('%Y-%m-%d')} to {val_dates[-1].strftime('%Y-%m-%d')}")
log(f"Test: rows 4425-5199 ({len(test_df)} rows), {test_dates[0].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')}")

def count_phases(regime_arr, price_arr):
    log_p = np.log(price_arr)
    chg_60 = np.zeros(len(price_arr))
    chg_60[60:] = log_p[60:] - log_p[:-60]
    
    bull_count = 0
    bear_count = 0
    in_bull = False
    in_bear = False
    length_bull = 0
    length_bear = 0
    
    for val in chg_60:
        if val > 0.15:
            length_bull += 1
            if length_bull >= 5 and not in_bull:
                in_bull = True
                bull_count += 1
        else:
            length_bull = 0
            in_bull = False
            
        if val < -0.15:
            length_bear += 1
            if length_bear >= 5 and not in_bear:
                in_bear = True
                bear_count += 1
        else:
            length_bear = 0
            in_bear = False
            
    shock_count = 0
    in_shock = False
    length_shock = 0
    for r in regime_arr:
        if r == 3:
            length_shock += 1
            if length_shock >= 5 and not in_shock:
                in_shock = True
                shock_count += 1
        else:
            length_shock = 0
            in_shock = False
            
    return bull_count, bear_count, shock_count

for split_name, s_idx, e_idx in [('Train', 0, 3640), ('Validation', 3645, 4420), ('Test', 4425, 5200)]:
    r_sub = regime_trim[s_idx:e_idx]
    p_sub = df_data['bl_price'].values[s_idx:e_idx]
    bull, bear, shock = count_phases(r_sub, p_sub)
    log(f"{split_name} phases -> Bull: {bull}, Bear: {bear}, Shock episodes: {shock}")

train_y = np.log(train_df['target_bl_price_t_plus_5d'] / train_df['bl_price'])
val_y = np.log(val_df['target_bl_price_t_plus_5d'] / val_df['bl_price'])
test_y = np.log(test_df['target_bl_price_t_plus_5d'] / test_df['bl_price'])

feat_cols = [c for c in train_df.columns if c not in ['bl_price', 'target_bl_price_t_plus_5d']]
X_train = train_df[feat_cols]
X_val = val_df[feat_cols]
X_test = test_df[feat_cols]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

ridge = Ridge(alpha=200.0)
ridge.fit(X_train_scaled, train_y)

def evaluate_preds(y_true, y_pred):
    dir_acc = np.mean(np.sign(y_pred) == np.sign(y_true)) * 100.0
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot)
    return dir_acc, r2

val_pred = ridge.predict(X_val_scaled)
test_pred = ridge.predict(X_test_scaled)

val_dir, val_r2 = evaluate_preds(val_y.values, val_pred)
test_dir, test_r2 = evaluate_preds(test_y.values, test_pred)

log(f"\nValidation Ridge - Dir Acc: {val_dir:.2f}%, R2: {val_r2:.4f}")
log(f"Test Ridge - Dir Acc: {test_dir:.2f}%, R2: {test_r2:.4f}")

test_dir_flag = "PASSED" if 50.0 <= test_dir <= 56.0 else "FLAGGED"
test_r2_flag = "PASSED" if -0.05 <= test_r2 <= 0.05 else "FLAGGED"
log(f"Test Directional Accuracy Test ({test_dir:.2f}%): {test_dir_flag}")
log(f"Test R2 Test ({test_r2:.4f}): {test_r2_flag}")

with open('validation_report.txt', 'w') as f:
    f.write('\n'.join(report_lines))

log("\nAll files successfully generated and validated.")