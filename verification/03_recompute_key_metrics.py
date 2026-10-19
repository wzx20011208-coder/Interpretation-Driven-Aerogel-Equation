# -*- coding: utf-8 -*-
"""
审稿修订_模型改动：按原流水线同口径重建数据并重算关键指标。
只读原数据/结果 JSON，所有新结果输出到本文件夹。
不修改任何源代码或原始结果文件。
"""
import sys, io, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

FEATURES = ['温度', '湿度', '冻融次数', '裂缝长度', '含水率']
TARGET = '导热系数'

def load_data():
    df_deg = pd.read_excel(os.path.join(DATA_DIR, 'degradation_dataset.xlsx'), sheet_name='degradation_data')
    df_deg.columns = ['样本', '温度', '湿度', '冻融次数', '裂缝长度', '含水率', '导热系数']
    df_deg = df_deg[FEATURES + [TARGET]].copy()
    df_deg['数据来源'] = '劣化'
    df_deg['sample_weight'] = 1.0

    df_intact_summary = pd.read_excel(
        os.path.join(DATA_DIR, 'intact_conditions_summary.xlsx'),
        sheet_name='summary')
    df_intact = pd.DataFrame({
        '温度': df_intact_summary['Temperature_C'],
        '湿度': df_intact_summary['RH_pct'],
        '冻融次数': 0,
        '裂缝长度': 0.0,
        '含水率': 0,
        '导热系数': df_intact_summary['Thermal_Conductivity_Mean'],
        '数据来源': '完好',
        'sample_weight': df_intact_summary['N_Readings'].astype(float),
    })
    return pd.concat([df_deg, df_intact], ignore_index=True), df_deg

def build_X(df):
    T = df['温度'].values.astype(float)
    RH = df['湿度'].values.astype(float)
    FT = df['冻融次数'].values.astype(float)
    Crack = df['裂缝长度'].values.astype(float)
    Water = df['含水率'].values.astype(float)
    sFT, sCrack = np.sqrt(FT), np.sqrt(Crack)
    cols = {
        'T': T, 'RH': RH, '√FT': sFT, '√Crack': sCrack, 'Water': Water,
        'T²': T**2, 'RH²': RH**2, 'Water²': Water**2,
        'T×√Crack': T * sCrack, 'RH×Water': RH * Water,
        '√FT×√Crack': sFT * sCrack, 'RH×√Crack': RH * sCrack,
    }
    order = ['T', 'RH', '√FT', '√Crack', 'Water', 'T²', 'RH²', 'Water²',
             'T×√Crack', 'RH×Water', '√FT×√Crack', 'RH×√Crack']
    X = np.column_stack([cols[n] for n in order])
    return X, cols, order

def eq_predict(coefs, T, RH, FT, Crack, Water):
    sFT, sCrack = np.sqrt(FT), np.sqrt(Crack)
    return (coefs['const'] + coefs['T']*T + coefs['T²']*T**2 + coefs['RH²']*RH**2
            + coefs['√FT']*sFT + coefs['√Crack']*sCrack + coefs['Water']*Water
            + coefs['Water²']*Water**2 + coefs['T×√Crack']*T*sCrack
            + coefs['RH×Water']*RH*Water + coefs['√FT×√Crack']*sFT*sCrack
            + coefs['RH×√Crack']*RH*sCrack)

df_all, df_deg = load_data()
y = df_all[TARGET].values.astype(float)
X, cols, order = build_X(df_all)
Xc = sm.add_constant(X)
model = sm.OLS(y, Xc).fit()

coef_names = ['const'] + order
coefs = dict(zip(coef_names, model.params))
resid = y - model.predict(Xc)
rss = float(np.sum(resid**2))
n = len(y)

results = {}
results['data'] = {
    'n_degraded': len(df_deg), 'n_intact': len(df_all) - len(df_deg), 'n_total': n,
    'crack_mm_levels': sorted(df_deg['裂缝长度'].unique().tolist()),
    'water_levels': sorted(df_deg['含水率'].unique().tolist()),
    'temp_levels': sorted(df_deg['温度'].unique().tolist()),
}

# ---- 1. 最终方程重算：R² / RMSE / AIC / BIC / VIF ----
def metrics(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    r2 = 1 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean((y_true - y_pred)**2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return r2, rmse, mae, mape

r2, rmse, mae, mape = metrics(y, model.predict(Xc))
results['final11'] = {
    'train_r2': float(model.rsquared),
    'adj_r2': float(model.rsquared_adj),
    'train_rmse': rmse, 'train_mae': mae, 'train_mape': mape,
    'rss': rss, 'rss_over_n': rss / n,
    'aic_statsmodels': float(model.aic),
    'bic_statsmodels': float(model.bic),
    'bic_manual': float(n * np.log(rss / n) + model.df_model * np.log(n)),
    'coefs': {k: float(v) for k, v in coefs.items()},
    'pvalues': {k: float(v) for k, v in zip(coef_names, model.pvalues)},
}

vif = {}
Xv = X
for i, nm in enumerate(order):
    vif[nm] = float(variance_inflation_factor(Xv, i))
results['final11']['vif'] = vif

# ---- 2. 低温裂缝：方程 vs 实测 ----
intact_map = {}
for _, r in df_all[df_all['数据来源'] == '完好'].iterrows():
    intact_map[(int(r['温度']), int(r['湿度']))] = float(r['导热系数'])

pure_crack = df_deg[(df_deg['冻融次数'] == 0) & (df_deg['含水率'] == 0) & (df_deg['裂缝长度'] > 0)]
lt_rows = []
for _, r in pure_crack.iterrows():
    T, RH, cr = int(r['温度']), int(r['湿度']), float(r['裂缝长度'])
    lam_m = float(r['导热系数'])
    lam0 = intact_map[(T, RH)]
    eff_m = lam_m - lam0
    lam_eq0 = eq_predict(coefs, T, RH, 0, 0, 0)
    lam_eq = eq_predict(coefs, T, RH, 0, cr, 0)
    eff_eq = lam_eq - lam_eq0
    lt_rows.append({'T': T, 'RH': RH, 'Crack_mm': cr, 'lambda_measured': lam_m,
                    'lambda_intact_mean': lam0, 'measured_effect': eff_m,
                    'equation_effect': eff_eq,
                    'sign_match': 'same' if np.sign(eff_m) == np.sign(eff_eq) else 'conflict',
                    'equation_positive': bool(eff_eq > 0), 'measured_positive': bool(eff_m > 0)})
results['lowT_crack'] = lt_rows

lowT = [r for r in lt_rows if r['T'] <= 20]
results['lowT_crack_summary'] = {
    'n_rows_lowT': len(lowT),
    'eq_positive_lowT': sum(1 for r in lowT if r['equation_positive']),
    'measured_positive_lowT': sum(1 for r in lowT if r['measured_positive']),
    'conflicts': sum(1 for r in lowT if r['sign_match'] == 'conflict'),
}

# 符号翻转温度（净系数=0）
net_crack_zero = {}
for RH in (30, 60, 90):
    # net coef (per sqrt-mm) = b√Crack + bT×√Crack*T + bRH×√Crack*RH
    b_s = coefs['√Crack']; b_t = coefs['T×√Crack']; b_rh = coefs['RH×√Crack']
    T_zero = -(b_s + b_rh * RH) / b_t
    net_crack_zero[RH] = float(T_zero)
results['crack_sign_flip_temperature'] = net_crack_zero

# 38% 捕获率：逐点均值口径
overall_m = float(np.mean([r['measured_effect'] for r in lt_rows]))
overall_eq = float(np.mean([r['equation_effect'] for r in lt_rows]))
results['crack_capture_38'] = {
    'measured_mean_effect': overall_m,
    'equation_mean_effect': overall_eq,
    'capture_pct': overall_eq / overall_m * 100,
}
by_rh = {}
for RH in (30, 60, 90):
    sub = [r for r in lt_rows if r['RH'] == RH]
    mm = float(np.mean([r['measured_effect'] for r in sub]))
    ee = float(np.mean([r['equation_effect'] for r in sub]))
    by_rh[RH] = {'measured_mean': mm, 'equation_mean': ee, 'capture_pct': ee / mm * 100 if mm else None}
results['crack_capture_by_rh'] = by_rh

# ---- 3. FPI 复现 ----
fpi_csv = os.path.join(DATA_DIR, 'FPI_CellMatched_detail.csv')
df_fpi = pd.read_csv(fpi_csv)
ft_max, cr_max, wt_max = 24, 10, 56
fpi_rows = []
for _, row in df_fpi.iterrows():
    T, RH = int(row['Temperature_C']), int(row['RH_pct'])
    l0 = eq_predict(coefs, T, RH, 0, 0, 0)
    d_ft = eq_predict(coefs, T, RH, ft_max, 0, 0) - l0
    d_cr = eq_predict(coefs, T, RH, 0, cr_max, 0) - l0
    d_wt = eq_predict(coefs, T, RH, 0, 0, wt_max) - l0
    L = l0 + d_ft + d_cr + d_wt
    C = eq_predict(coefs, T, RH, ft_max, cr_max, wt_max)
    fpi_eq = (C - L) / abs(L) * 100 if L != 0 else 0.0
    fpi_exp = float(row['fpi_pct'])
    fpi_rows.append({'T': T, 'RH': RH, 'fpi_exp': fpi_exp, 'fpi_eq': fpi_eq})
df_fpi2 = pd.DataFrame(fpi_rows)
results['fpi'] = {
    'n_cells': len(df_fpi2),
    'exp_mean': float(df_fpi2['fpi_exp'].mean()),
    'eq_mean': float(df_fpi2['fpi_eq'].mean()),
    'capture_mean_pct': float(df_fpi2['fpi_eq'].mean() / df_fpi2['fpi_exp'].mean() * 100),
    'pearson_r': float(np.corrcoef(df_fpi2['fpi_exp'], df_fpi2['fpi_eq'])[0, 1]),
    'spearman_r': float(pd.Series(df_fpi2['fpi_exp']).corr(pd.Series(df_fpi2['fpi_eq']), method='spearman')),
    'exp_min': float(df_fpi2['fpi_exp'].min()), 'exp_max': float(df_fpi2['fpi_exp'].max()),
    'eq_min': float(df_fpi2['fpi_eq'].min()), 'eq_max': float(df_fpi2['fpi_eq'].max()),
    'all_eq_below_exp': bool((df_fpi2['fpi_eq'] < df_fpi2['fpi_exp']).all()),
    'per_cell': fpi_rows,
}

# ---- 4. 含水率分段增量：实测 vs 方程 ----
water_sub = df_deg[(df_deg['冻融次数'] == 0) & (df_deg['裂缝长度'] == 0) & (df_deg['含水率'] > 0)]
wm = water_sub.groupby('含水率')['导热系数'].mean()
w0 = float(df_deg[(df_deg['冻融次数'] == 0) & (df_deg['裂缝长度'] == 0) & (df_deg['含水率'] == 0)]['导热系数'].mean())
meas_inc = {'0_to_23': float(wm.get(23, np.nan) - w0),
            '23_to_38': float(wm.get(38, np.nan) - wm.get(23, np.nan)),
            '38_to_56': float(wm.get(56, np.nan) - wm.get(38, np.nan))}
eq_inc = {}
for RH in (30, 60, 90):
    l0 = eq_predict(coefs, 25, RH, 0, 0, 0)
    l23 = eq_predict(coefs, 25, RH, 0, 0, 23)
    l38 = eq_predict(coefs, 25, RH, 0, 0, 38)
    l56 = eq_predict(coefs, 25, RH, 0, 0, 56)
    eq_inc[RH] = {'0_to_23': l23 - l0, '23_to_38': l38 - l23, '38_to_56': l56 - l38}
results['moisture_increments'] = {'measured_overall': meas_inc, 'equation_at_25C_by_RH': eq_inc}

# ---- 5. 完好区：方程 vs EN ISO 口径 ----
results['intact_baseline_note'] = (
    "完好27点为聚合均值(权重=样本数)，方程在全量351点拟合；完好态单独评估见 A4_intact_validation（mean_fit MAPE≈1.87%）")

out_path = os.path.join(OUT_DIR, 'recompute_key_metrics.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=1)

# ---- 打印摘要 ----
print("数据: 劣化%d + 完好%d = %d; 裂缝水平 %s (mm); 含水率水平 %s" % (
    len(df_deg), len(df_all) - len(df_deg), n,
    results['data']['crack_mm_levels'], results['data']['water_levels']))
print("\n[1] 最终11项重算: R²=%.4f RMSE=%.6f RSS=%.3e" % (r2, rmse, rss))
print("    statsmodels AIC=%.2f BIC=%.2f | 手动 BIC=%.2f | 存储值 AIC=-3280.14 BIC=-3236.49" % (
    model.aic, model.bic, results['final11']['bic_manual']))
print("    VIF>10 项数: %d / 11" % sum(1 for v in vif.values() if v > 10))
print("\n[2] 低温裂缝(5-20C) 共%d条: 方程正效应%d条, 实测正效应%d条, 符号冲突%d条" % (
    len(lowT), results['lowT_crack_summary']['eq_positive_lowT'],
    results['lowT_crack_summary']['measured_positive_lowT'],
    results['lowT_crack_summary']['conflicts']))
print("    符号翻转温度: " + ", ".join("RH=%d: %.1f°C" % (k, v) for k, v in net_crack_zero.items()))
print("    捕获率: 整体 %.1f%% | RH30 %.1f%% | RH60 %.1f%% | RH90 %.1f%%" % (
    results['crack_capture_38']['capture_pct'], by_rh[30]['capture_pct'],
    by_rh[60]['capture_pct'], by_rh[90]['capture_pct']))
print("\n[3] FPI: 实测均值 +%.2f%%, 方程均值 +%.2f%%, 均值捕获 %.1f%%, Pearson r=%.3f, 全部eq<exp=%s" % (
    results['fpi']['exp_mean'], results['fpi']['eq_mean'], results['fpi']['capture_mean_pct'],
    results['fpi']['pearson_r'], results['fpi']['all_eq_below_exp']))
print("\n[4] 含水率实测增量: " + ", ".join("%s=%.5f" % (k, v) for k, v in meas_inc.items()))
for RH, d in eq_inc.items():
    print("    方程(25C, RH=%d): " % RH + ", ".join("%s=%.5f" % (k, v) for k, v in d.items()))
print("\n结果已保存: %s" % out_path)
