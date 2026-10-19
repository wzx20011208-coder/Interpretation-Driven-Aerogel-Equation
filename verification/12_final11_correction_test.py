# -*- coding: utf-8 -*-
"""final11 + 低温高湿修正项候选对比（A+/边界修正方案）。"""
import sys, io, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
OUT = os.path.dirname(os.path.abspath(__file__))
KF = KFold(n_splits=5, shuffle=True, random_state=42)

df_deg = pd.read_excel(os.path.join(DATA_DIR, 'degradation_dataset.xlsx'), sheet_name='degradation_data')
df_deg.columns = ['样本', '温度', '湿度', '冻融次数', '裂缝长度', '含水率', '导热系数']
df_is = pd.read_excel(os.path.join(DATA_DIR, 'intact_conditions_summary.xlsx'),
                      sheet_name='summary')
df_int = pd.DataFrame({'温度': df_is['Temperature_C'], '湿度': df_is['RH_pct'], '冻融次数': 0,
                       '裂缝长度': 0.0, '含水率': 0, '导热系数': df_is['Thermal_Conductivity_Mean']})
df = pd.concat([df_deg, df_int], ignore_index=True)
y = df['导热系数'].values.astype(float)
T = df['温度'].values.astype(float); RH = df['湿度'].values.astype(float)
FT = df['冻融次数'].values.astype(float); Cr = df['裂缝长度'].values.astype(float)
W = df['含水率'].values.astype(float)
sFT, sCr = np.sqrt(FT), np.sqrt(Cr)

BASE = {'T': T, 'T²': T**2, 'RH²': RH**2, '√FT': sFT, '√Crack': sCr,
        'Water': W, 'Water²': W**2, 'RH×Water': RH*W, '√FT×√Crack': sFT*sCr,
        'T×√Crack': T*sCr, 'RH×√Crack': RH*sCr}
CORR = {
    'K0_无修正(final11)': {},
    'K1_(15-T)+×RH×√Crack': {'K1': np.maximum(15-T, 0) * RH * sCr},
    'K2_(15-T)+×√Crack': {'K2': np.maximum(15-T, 0) * sCr},
    'K3_(15-T)+×(RH-60)+×√Crack': {'K3': np.maximum(15-T, 0) * np.maximum(RH-60, 0) * sCr},
    'K4_(20-T)+×RH×√Crack': {'K4': np.maximum(20-T, 0) * RH * sCr},
    'K5_I(T≤15)×RH×√Crack': {'K5': (T <= 15).astype(float) * RH * sCr},
    'K6_(15-T)+×RH×√Crack + (15-T)+×√FT×√Crack': {
        'K6a': np.maximum(15-T, 0) * RH * sCr,
        'K6b': np.maximum(15-T, 0) * sFT * sCr},
    'K7_(15-T)+×(RH-30)+×√Crack': {'K7': np.maximum(15-T, 0) * np.maximum(RH-30, 0) * sCr},
    'K8_(15-T)+×√Crack + (15-T)+×RH×√Crack': {
        'K8a': np.maximum(15-T, 0) * sCr,
        'K8b': np.maximum(15-T, 0) * RH * sCr},
    'K9_(20-T)+×(RH-30)+×√Crack': {'K9': np.maximum(20-T, 0) * np.maximum(RH-30, 0) * sCr},
    'K10_I(T<20)×√Crack': {'K10': (T < 20).astype(float) * sCr},
    'K11_I(T<20)×√Crack + I(T<20)×RH×√Crack': {
        'K11a': (T < 20).astype(float) * sCr,
        'K11b': (T < 20).astype(float) * RH * sCr},
    'K12_(25-T)+×RH×√Crack': {'K12': np.maximum(25-T, 0) * RH * sCr},
    'K13_(15-T)+×RH×√Crack + (15-T)+×√Crack + (15-T)+×√FT×√Crack': {
        'K13a': np.maximum(15-T, 0) * RH * sCr,
        'K13b': np.maximum(15-T, 0) * sCr,
        'K13c': np.maximum(15-T, 0) * sFT * sCr},
}

# ---- 覆盖量化 ----
pure = df_deg[(df_deg['冻融次数'] == 0) & (df_deg['含水率'] == 0) & (df_deg['裂缝长度'] > 0)]
intact_map = {}
for _, r in df_int.iterrows():
    intact_map[(int(r['温度']), int(r['湿度']))] = float(r['导热系数'])
lowT = pure[pure['温度'] <= 20]
meas = [(float(r['导热系数']) - intact_map[(int(r['温度']), int(r['湿度']))]) for _, r in lowT.iterrows()]
print("覆盖量化: 纯裂缝低温(≤20°C)行数=%d/%d, 其中实测显著为负(|Δ|>0.0007)=%d, 接近零=%d, 为正=%d" % (
    len(lowT), len(pure), sum(1 for v in meas if v < -0.0007),
    sum(1 for v in meas if abs(v) <= 0.0007), sum(1 for v in meas if v > 0.0007)))
print("低温(≤20°C) 行占全部351的 %.1f%%" % (len(df[df['温度'] <= 20]) / len(df) * 100))

# ---- 修正项对比 ----
def metrics(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2); ss_tot = np.sum((y_true - y_true.mean())**2)
    return 1 - ss_res/ss_tot, float(np.sqrt(np.mean((y_true-y_pred)**2)))

def eval_model(terms, extra):
    X = np.column_stack([BASE[t] for t in terms] + [extra[k] for k in extra])
    Xc = sm.add_constant(X)
    m = sm.OLS(y, Xc).fit()
    coefs = dict(zip(['const'] + terms + list(extra.keys()), m.params))
    r2s, rms = [], []
    for tr, te in KF.split(X):
        mm = sm.OLS(y[tr], sm.add_constant(X[tr])).fit()
        p = mm.predict(sm.add_constant(X[te]))
        r2s.append(metrics(y[te], p)[0]); rms.append(metrics(y[te], p)[1])
    # 纯裂缝净效应（FT=0, W=0）
    def net(Tv, RHv, CRv):
        s = np.sqrt(CRv)
        acc = (coefs['√Crack'] + coefs['T×√Crack']*Tv + coefs['RH×√Crack']*RHv) * s
        for k in extra:
            if k == 'K1': acc += coefs['K1'] * max(15-Tv, 0) * RHv * s
            elif k == 'K2': acc += coefs['K2'] * max(15-Tv, 0) * s
            elif k == 'K3': acc += coefs['K3'] * max(15-Tv, 0) * max(RHv-60, 0) * s
            elif k == 'K4': acc += coefs['K4'] * max(20-Tv, 0) * RHv * s
            elif k == 'K5': acc += coefs['K5'] * (1 if Tv <= 15 else 0) * RHv * s
            elif k == 'K6a': acc += coefs['K6a'] * max(15-Tv, 0) * RHv * s
            elif k == 'K6b': pass
            elif k == 'K7': acc += coefs['K7'] * max(15-Tv, 0) * max(RHv-30, 0) * s
            elif k == 'K8a': acc += coefs['K8a'] * max(15-Tv, 0) * s
            elif k == 'K8b': acc += coefs['K8b'] * max(15-Tv, 0) * RHv * s
            elif k == 'K9': acc += coefs['K9'] * max(20-Tv, 0) * max(RHv-30, 0) * s
            elif k == 'K10': acc += coefs['K10'] * (1 if Tv < 20 else 0) * s
            elif k == 'K11a': acc += coefs['K11a'] * (1 if Tv < 20 else 0) * s
            elif k == 'K11b': acc += coefs['K11b'] * (1 if Tv < 20 else 0) * RHv * s
            elif k == 'K12': acc += coefs['K12'] * max(25-Tv, 0) * RHv * s
            elif k == 'K13a': acc += coefs['K13a'] * max(15-Tv, 0) * RHv * s
            elif k == 'K13b': acc += coefs['K13b'] * max(15-Tv, 0) * s
            elif k == 'K13c': pass
        return acc
    pos = 0; sig_conf = 0
    for Tv in (5,10,15,20,25,30,35,40,45):
        for RHv in (30,60,90):
            for CRv in (2.5,5.0,10.0):
                e = net(Tv, RHv, CRv)
                if e > 0: pos += 1
                if Tv <= 20:
                    me = float(pure[(pure['温度']==Tv)&(pure['湿度']==RHv)&(pure['裂缝长度']==CRv)]['导热系数'].iloc[0]) - intact_map[(Tv, RHv)]
                    if np.sign(me) != np.sign(e) and abs(me) > 0.0007:
                        sig_conf += 1
    return {'k': len(terms)+len(extra)+1, 'train_r2': float(m.rsquared),
            'cv_r2': float(np.mean(r2s)), 'cv_rmse': float(np.mean(rms)),
            'aic': float(m.aic), 'bic': float(m.bic),
            'pos81': pos, 'sig_conf_lowT': sig_conf,
            'coefs': {k: float(v) for k, v in coefs.items()},
            'pvalues': {k: float(v) for k, v in zip(['const']+terms+list(extra.keys()), m.pvalues)}}

terms = list(BASE.keys())
res = {}
print("\n%-28s %5s %8s %8s %8s %9s %8s %5s %5s" % (
    '模型', 'k', 'R2', 'CV-R2', 'RMSE', 'AIC', 'BIC', 'pos', 'conf'))
for nm, extra in CORR.items():
    r = eval_model(terms, extra)
    res[nm] = r
    print("%-28s %5d %8.4f %8.4f %8.6f %9.1f %8.1f %5d %5d" % (
        nm, r['k'], r['train_r2'], r['cv_r2'], r['cv_rmse'], r['aic'], r['bic'],
        r['pos81'], r['sig_conf_lowT']))

with open(os.path.join(OUT, 'final11_correction_test.json'), 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print("\nsaved: final11_correction_test.json")
