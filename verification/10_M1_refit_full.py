# -*- coding: utf-8 -*-
"""
M1 修订方程完整重拟合（去掉 T×√Crack，10 项）。
仅读取原始数据与结果；所有新输出写到本文件夹 M1_refit/。
"""
import sys, io, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit
from sklearn.model_selection import KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "M1_refit")
os.makedirs(OUT, exist_ok=True)
FEATURES = ['温度', '湿度', '冻融次数', '裂缝长度', '含水率']
TARGET = '导热系数'
RANDOM_STATE = 42
KF = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

TERMS = ['T', 'T²', 'RH²', '√FT', '√Crack', 'Water', 'Water²',
         'RH×Water', '√FT×√Crack', 'RH×√Crack']

def load_data():
    df_deg = pd.read_excel(os.path.join(DATA_DIR, 'degradation_dataset.xlsx'), sheet_name='degradation_data')
    df_deg.columns = ['样本', '温度', '湿度', '冻融次数', '裂缝长度', '含水率', '导热系数']
    df_deg = df_deg[FEATURES + [TARGET]].copy()
    df_deg['数据来源'] = '劣化'
    df_deg['sample_weight'] = 1.0
    df_is = pd.read_excel(os.path.join(DATA_DIR, 'intact_conditions_summary.xlsx'),
                          sheet_name='summary')
    df_int = pd.DataFrame({
        '温度': df_is['Temperature_C'], '湿度': df_is['RH_pct'], '冻融次数': 0,
        '裂缝长度': 0.0, '含水率': 0, '导热系数': df_is['Thermal_Conductivity_Mean'],
        '数据来源': '完好', 'sample_weight': df_is['N_Readings'].astype(float)})
    df = pd.concat([df_deg, df_int], ignore_index=True)
    def scen(r):
        ft, cr, wt = r['冻融次数'], r['裂缝长度'], r['含水率']
        if ft == 0 and cr == 0 and wt == 0:
            return '完好'
        if ft > 0 and cr == 0 and wt == 0:
            return '纯冻融'
        if ft == 0 and cr > 0 and wt == 0:
            return '纯裂缝'
        if ft == 0 and cr == 0 and wt > 0:
            return '纯渗水'
        return '复合'
    df['场景'] = df.apply(scen, axis=1)
    return df, df_deg

def build_X(df, terms=None):
    terms = terms or TERMS
    T = df['温度'].values.astype(float)
    RH = df['湿度'].values.astype(float)
    FT = df['冻融次数'].values.astype(float)
    Cr = df['裂缝长度'].values.astype(float)
    W = df['含水率'].values.astype(float)
    sFT, sCr = np.sqrt(FT), np.sqrt(Cr)
    pool = {
        'T': T, 'T²': T**2, 'RH': RH, 'RH²': RH**2, 'FT': FT, 'Crack': Cr,
        '√FT': sFT, '√Crack': sCr,
        'Water': W, 'Water²': W**2, 'RH×Water': RH*W, '√FT×√Crack': sFT*sCr,
        'T×√Crack': T*sCr, 'RH×√Crack': RH*sCr,
        '√Crack×Water': sCr*W, 'T×RH': T*RH, 'T×√FT': T*sFT, 'T×Water': T*W,
        'RH×√FT': RH*sFT, '√FT×Water': sFT*W,
    }
    return np.column_stack([pool[t] for t in terms])

def metrics(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    return (1 - ss_res/ss_tot,
            float(np.sqrt(np.mean((y_true-y_pred)**2))),
            float(np.mean(np.abs(y_true-y_pred))),
            float(np.mean(np.abs((y_true-y_pred)/y_true))*100))

df, df_deg = load_data()
y = df[TARGET].values.astype(float)
X = build_X(df)
Xc = sm.add_constant(X)
m = sm.OLS(y, Xc).fit()
coefs = dict(zip(['const'] + TERMS, m.params))
res = {}

train_r2, train_rmse, train_mae, train_mape = metrics(y, m.predict(Xc))
res['final10'] = {
    'terms': TERMS, 'k': len(TERMS), 'n': len(y),
    'train_r2': float(m.rsquared), 'adj_r2': float(m.rsquared_adj),
    'train_rmse': train_rmse, 'train_mae': train_mae, 'train_mape': train_mape,
    'aic': float(m.aic), 'bic': float(m.bic),
    'coefs': {k: float(v) for k, v in coefs.items()},
    'pvalues': {k: float(v) for k, v in zip(['const'] + TERMS, m.pvalues)},
}

mh = sm.OLS(y, Xc).fit(cov_type='HC3')
res['final10']['hc3'] = {k: {'se_ols': float(m.bse[i]), 'se_hc3': float(mh.bse[i]),
                             'p_hc3': float(mh.pvalues[i])}
                         for i, k in enumerate(['const'] + TERMS)}
res['final10']['vif'] = {t: float(variance_inflation_factor(X, i)) for i, t in enumerate(TERMS)}
res['final10']['vif_gt10'] = sum(1 for v in res['final10']['vif'].values() if v > 10)

cv_r2, cv_rmse, cv_mae, cv_mape = [], [], [], []
for tr, te in KF.split(X):
    mm = sm.OLS(y[tr], sm.add_constant(X[tr])).fit()
    p = mm.predict(sm.add_constant(X[te]))
    r2_, rm, ma, mp = metrics(y[te], p)
    cv_r2.append(r2_); cv_rmse.append(rm); cv_mae.append(ma); cv_mape.append(mp)
res['final10']['cv'] = {'r2': float(np.mean(cv_r2)), 'r2_std': float(np.std(cv_r2)),
                        'rmse': float(np.mean(cv_rmse)), 'mae': float(np.mean(cv_mae)),
                        'mape': float(np.mean(cv_mape))}

print("[1] M1 最终10项: R²=%.4f RMSE=%.6f MAPE=%.2f%% | CV R²=%.4f±%.4f RMSE=%.6f MAPE=%.2f%%" % (
    res['final10']['train_r2'], train_rmse, train_mape,
    res['final10']['cv']['r2'], res['final10']['cv']['r2_std'],
    res['final10']['cv']['rmse'], res['final10']['cv']['mape']))
print("    AIC=%.1f BIC=%.1f | VIF>10: %d/10" % (m.aic, m.bic, res['final10']['vif_gt10']))

rng = np.random.default_rng(RANDOM_STATE)
strata = {s: df[df['场景'] == s].index.values for s in df['场景'].unique()}
boot_coefs = {k: [] for k in ['const'] + TERMS}
for _ in range(300):
    idx = np.concatenate([rng.choice(v, size=len(v), replace=True) for v in strata.values()])
    mm = sm.OLS(y[idx], sm.add_constant(X[idx])).fit()
    for i, k in enumerate(['const'] + TERMS):
        boot_coefs[k].append(float(mm.params[i]))
res['bootstrap'] = {'n': 300}
for k, v in boot_coefs.items():
    lo, hi = np.percentile(v, [2.5, 97.5])
    res['bootstrap'][k] = {'ci_lo': float(lo), 'ci_hi': float(hi),
                           'sign_stable': bool(np.all(np.sign(v) == np.sign(v[0])))}
res['bootstrap']['all_sign_stable'] = all(res['bootstrap'][k]['sign_stable'] for k in ['const'] + TERMS)
res['bootstrap']['all_ci_exclude_zero'] = all(
    (res['bootstrap'][k]['ci_lo'] > 0 or res['bootstrap'][k]['ci_hi'] < 0)
    for k in ['const'] + TERMS)
print("[2] bootstrap: 全部符号稳定=%s, CI全部不含零=%s" % (
    res['bootstrap']['all_sign_stable'], res['bootstrap']['all_ci_exclude_zero']))

bb = {'r2': 0.9821411506820473, 'rmse': 0.00046939869418030687, 'mape': 1.1009881453940942}
res['recovery'] = {
    'r2_ratio': res['final10']['train_r2'] / bb['r2'],
    'cv_r2_ratio': res['final10']['cv']['r2'] / bb['r2'],
    'rmse_ratio': res['final10']['cv']['rmse'] / bb['rmse'],
    'mape_ratio': res['final10']['cv']['mape'] / bb['mape'],
}
print("[3] 恢复率: 训练R²比=%.4f | CV-R²比=%.4f | RMSE比=%.2f MAPE比=%.2f" % (
    res['recovery']['r2_ratio'], res['recovery']['cv_r2_ratio'],
    res['recovery']['rmse_ratio'], res['recovery']['mape_ratio']))

df_fpi = pd.read_csv(os.path.join(DATA_DIR,
                                  'FPI_CellMatched_detail.csv'))
def eq_pred(T, RH, FT, Cr, W):
    sFT, sCr = np.sqrt(FT), np.sqrt(Cr)
    return (coefs['const'] + coefs['T']*T + coefs['T²']*T**2 + coefs['RH²']*RH**2
            + coefs['√FT']*sFT + coefs['√Crack']*sCr + coefs['Water']*W + coefs['Water²']*W**2
            + coefs['RH×Water']*RH*W + coefs['√FT×√Crack']*sFT*sCr + coefs['RH×√Crack']*RH*sCr)
fpi_cells = []
for _, r in df_fpi.iterrows():
    T, RH = int(r['Temperature_C']), int(r['RH_pct'])
    l0 = eq_pred(T, RH, 0, 0, 0)
    L = l0 + (eq_pred(T, RH, 24, 0, 0) - l0) + (eq_pred(T, RH, 0, 10, 0) - l0) + (eq_pred(T, RH, 0, 0, 56) - l0)
    C = eq_pred(T, RH, 24, 10, 56)
    fpi_eq = (C - L) / abs(L) * 100
    fpi_cells.append({'T': T, 'RH': RH, 'fpi_exp': float(r['fpi_pct']), 'fpi_eq': fpi_eq})
fdc = pd.DataFrame(fpi_cells)
res['fpi'] = {'exp_mean': float(fdc['fpi_exp'].mean()), 'eq_mean': float(fdc['fpi_eq'].mean()),
              'capture_pct': float(fdc['fpi_eq'].mean() / fdc['fpi_exp'].mean() * 100),
              'pearson_r': float(np.corrcoef(fdc['fpi_exp'], fdc['fpi_eq'])[0, 1]),
              'eq_max': float(fdc['fpi_eq'].max()), 'eq_min': float(fdc['fpi_eq'].min()),
              'all_below': bool((fdc['fpi_eq'] < fdc['fpi_exp']).all()), 'cells': fpi_cells}
print("[4] FPI: 实测均值 +%.2f%% | 方程均值 +%.2f%% | 捕获 %.1f%% | Pearson r=%.3f" % (
    res['fpi']['exp_mean'], res['fpi']['eq_mean'], res['fpi']['capture_pct'], res['fpi']['pearson_r']))

intact_map = {}
for _, r in df[df['数据来源'] == '完好'].iterrows():
    intact_map[(int(r['温度']), int(r['湿度']))] = float(r['导热系数'])
pure = df_deg[(df_deg['冻融次数'] == 0) & (df_deg['含水率'] == 0) & (df_deg['裂缝长度'] > 0)]
cells = []
for _, r in pure.iterrows():
    T, RH, CR = int(r['温度']), int(r['湿度']), float(r['裂缝长度'])
    eff_m = float(r['导热系数']) - intact_map[(T, RH)]
    eff_eq = (coefs['√Crack'] + coefs['RH×√Crack']*RH) * np.sqrt(CR)
    cells.append({'T': T, 'RH': RH, 'CR': CR, 'measured': eff_m, 'equation': eff_eq})
cc = pd.DataFrame(cells)
res['crack'] = {'n_cells': len(cc),
                'measured_mean': float(cc['measured'].mean()),
                'equation_mean': float(cc['equation'].mean()),
                'capture_pct': float(cc['equation'].mean() / cc['measured'].mean() * 100),
                'positive_cells': int((cc['equation'] > 0).sum()),
                'lowT_positive': int(((cc['T'] <= 20) & (cc['equation'] > 0)).sum()),
                'lowT_significant_conflicts': int(((cc['T'] <= 20)
                    & (np.sign(cc['measured']) != np.sign(cc['equation']))
                    & (np.abs(cc['measured']) > 0.0007)).sum())}
print("[5] 裂缝: 捕获 %.1f%% | 全域正效应格点 %d/%d | 低温正效应 %d | 低温显著冲突 %d" % (
    res['crack']['capture_pct'], res['crack']['positive_cells'], len(cc),
    res['crack']['lowT_positive'], res['crack']['lowT_significant_conflicts']))

wm = df_deg[(df_deg['冻融次数'] == 0) & (df_deg['裂缝长度'] == 0)].groupby('含水率')['导热系数'].mean()
w0 = float(df[df['数据来源'] == '完好']['导热系数'].mean())
res['moisture'] = {'measured': {'0_to_23': float(wm.get(23, np.nan) - w0),
                                '23_to_38': float(wm.get(38, np.nan) - wm.get(23, np.nan)),
                                '38_to_56': float(wm.get(56, np.nan) - wm.get(38, np.nan))}}
res['moisture']['equation'] = {}
for RH in (30, 60, 90):
    l0 = eq_pred(25, RH, 0, 0, 0)
    l23 = eq_pred(25, RH, 0, 0, 23)
    l38 = eq_pred(25, RH, 0, 0, 38)
    l56 = eq_pred(25, RH, 0, 0, 56)
    res['moisture']['equation'][RH] = {'0_to_23': l23-l0, '23_to_38': l38-l23, '38_to_56': l56-l38}
print("[6] 含水率实测: 0→23=%.5f 23→38=%.5f 38→56=%.5f | 方程(RH60): %.5f/%.5f/%.5f" % (
    res['moisture']['measured']['0_to_23'], res['moisture']['measured']['23_to_38'],
    res['moisture']['measured']['38_to_56'],
    res['moisture']['equation'][60]['0_to_23'],
    res['moisture']['equation'][60]['23_to_38'],
    res['moisture']['equation'][60]['38_to_56']))

intact27 = df[df['数据来源'] == '完好']
T27 = intact27['温度'].values.astype(float)
RH27 = intact27['湿度'].values.astype(float)
y27 = intact27[TARGET].values.astype(float)
def en_iso(x, lref, fT, fm):
    T, RH = x
    return lref * np.exp(fT * (T - 20)) * np.exp(fm * (RH - 50))
popt, _ = curve_fit(en_iso, (T27, RH27), y27, p0=[0.028, 0.005, 0.002], maxfev=100000)
p27 = en_iso((T27, RH27), *popt)
r2_27, rm_27, _, mp_27 = metrics(y27, p27)
deg_T = deg_y = None
deg_T = df_deg['温度'].values.astype(float)
deg_RH = df_deg['湿度'].values.astype(float)
deg_lam = df_deg[TARGET].values.astype(float)
p_ext = en_iso((deg_T, deg_RH), *popt)
r2_ext, rm_ext, _, mp_ext = metrics(deg_lam, p_ext)
def en_iso_damage(x, lref, fT, fm, kFT, kCR, kWT):
    T, RH, FT, Cr, W = x
    return lref * np.exp(fT*(T-20)) * np.exp(fm*(RH-50)) * (1 + kFT*FT + kCR*Cr + kWT*W)
XX = (df['温度'].values.astype(float), df['湿度'].values.astype(float),
      df['冻融次数'].values.astype(float), df['裂缝长度'].values.astype(float),
      df['含水率'].values.astype(float))
poptd, _ = curve_fit(en_iso_damage, XX, y, p0=[0.028, 0.005, 0.002, 0.001, 0.001, 0.001],
                     maxfev=200000)
pdmg = en_iso_damage(XX, *poptd)
r2_dmg, rm_dmg, _, mp_dmg = metrics(y, pdmg)
degX = (deg_T, deg_RH, df_deg['冻融次数'].values.astype(float),
        df_deg['裂缝长度'].values.astype(float), df_deg['含水率'].values.astype(float))
r2_dmg_deg, rm_dmg_deg, _, mp_dmg_deg = metrics(deg_lam, en_iso_damage(degX, *poptd))
deg_X = build_X(df_deg)
mdeg = sm.OLS(deg_lam, sm.add_constant(deg_X)).fit()
r2_deg, rm_deg, _, mp_deg = metrics(deg_lam, mdeg.predict(sm.add_constant(deg_X)))
res['en_iso'] = {
    'params': {'lambda_ref': float(popt[0]), 'fT': float(popt[1]), 'fm': float(popt[2])},
    'intact': {'r2': r2_27, 'rmse': rm_27, 'mape': mp_27},
    'extrap_degraded': {'r2': r2_ext, 'rmse': rm_ext, 'mape': mp_ext},
    'damage_full': {'r2': r2_dmg, 'rmse': rm_dmg, 'mape': mp_dmg,
                    'r2_degraded': r2_dmg_deg, 'rmse_degraded': rm_dmg_deg,
                    'mape_degraded': mp_dmg_deg},
    'm1_equation': {'degraded324_fitdomain': {'r2': r2_deg, 'rmse': rm_deg, 'mape': mp_deg},
                    'full351': {'r2': res['final10']['train_r2'], 'rmse': train_rmse,
                                'mape': train_mape},
                    'cv5': res['final10']['cv']},
}
print("[7] EN ISO: 完好R²=%.3f MAPE=%.2f%% | 劣化外推R²=%.3f RMSE=%.6f | "
      "损伤扩展 R²=%.3f RMSE=%.6f | M1(324) R²=%.4f RMSE=%.6f" % (
    r2_27, mp_27, r2_ext, rm_ext, r2_dmg, rm_dmg, r2_deg, rm_deg))

structs = {
    'linear5': ['T', 'RH', 'FT', 'Crack', 'Water'],
    'main8': ['T', 'RH', '√FT', '√Crack', 'Water', 'T²', 'RH²', 'Water²'],
    'full13': ['T', 'RH', '√FT', '√Crack', 'Water', 'T²', 'RH²', 'Water²',
               '√Crack×Water', 'RH×Water', 'T×√Crack', '√FT×√Crack', 'RH×√Crack'],
    'final10': TERMS,
}
res['complexity'] = {}
for nm, terms in structs.items():
    Xs = build_X(df, terms)
    ms = sm.OLS(y, sm.add_constant(Xs)).fit()
    r2s, rms = [], []
    for tr, te in KF.split(Xs):
        mm = sm.OLS(y[tr], sm.add_constant(Xs[tr])).fit()
        p = mm.predict(sm.add_constant(Xs[te]))
        r2s.append(metrics(y[te], p)[0])
        rms.append(metrics(y[te], p)[1])
    res['complexity'][nm] = {'k': len(terms)+1, 'train_r2': float(ms.rsquared),
                             'cv_r2': float(np.mean(r2s)), 'cv_rmse': float(np.mean(rms)),
                             'aic': float(ms.aic), 'bic': float(ms.bic)}
    print("[8] %-8s k=%2d R²=%.4f CV=%.4f AIC=%.1f BIC=%.1f" % (
        nm, len(terms)+1, ms.rsquared, np.mean(r2s), ms.aic, ms.bic))

names = list(structs.keys())
nested_r2, nested_rmse, sel = [], [], {k: 0 for k in names}
for tr, te in KF.split(X):
    best, best_bic = None, np.inf
    for nm in names:
        Xs_tr = build_X(df.iloc[tr], structs[nm])
        mm = sm.OLS(y[tr], sm.add_constant(Xs_tr)).fit()
        if mm.bic < best_bic:
            best, best_bic = nm, mm.bic
    sel[best] += 1
    Xs_tr = build_X(df.iloc[tr], structs[best])
    mmb = sm.OLS(y[tr], sm.add_constant(Xs_tr)).fit()
    p = mmb.predict(sm.add_constant(build_X(df.iloc[te], structs[best])))
    nested_r2.append(metrics(y[te], p)[0])
    nested_rmse.append(metrics(y[te], p)[1])
res['nested_cv'] = {'r2': float(np.mean(nested_r2)), 'rmse': float(np.mean(nested_rmse)),
                    'selection_counts': sel}
print("[9] 嵌套CV: R²=%.4f RMSE=%.6f 选择=%s" % (np.mean(nested_r2), np.mean(nested_rmse), sel))

with open(os.path.join(OUT, 'M1_refit_results.json'), 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print("\n已保存: M1_refit/M1_refit_results.json")
