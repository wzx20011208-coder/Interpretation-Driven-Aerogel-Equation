# -*- coding: utf-8 -*-
"""FPI 统一口径重算：实测与方程均采用 27 格点水平均值口径。"""
import sys, io, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
OUT = os.path.dirname(os.path.abspath(__file__))

df_deg = pd.read_excel(os.path.join(DATA_DIR, 'degradation_dataset.xlsx'), sheet_name='degradation_data')
df_deg.columns = ['样本', '温度', '湿度', '冻融次数', '裂缝长度', '含水率', '导热系数']
df_is = pd.read_excel(os.path.join(DATA_DIR, 'intact_conditions_summary.xlsx'),
                      sheet_name='summary')
imap = {(int(r['Temperature_C']), int(r['RH_pct'])): float(r['Thermal_Conductivity_Mean']) for _, r in df_is.iterrows()}

coefs = {
    'const': 0.02338499648024806, 'T': 0.000221703151675485, 'T²': -8.980146749854794e-07,
    'RH²': 2.982675906299155e-07, '√FT': 0.0005083621695368741, '√Crack': 0.0005686850731791351,
    'Water': 0.00017241135567665282, 'Water²': -1.423018072952657e-06,
    'RH×Water': -2.9093013421483464e-07, '√FT×√Crack': 0.00016041957814100607,
    'T×√Crack': -2.1229691429828048e-05, 'RH×√Crack': -4.117454666459729e-06,
}
def eq_pred(T, RH, FT, CR, W):
    sFT, sCR = np.sqrt(np.maximum(FT, 0)), np.sqrt(np.maximum(CR, 0))
    return (coefs['const'] + coefs['T']*T + coefs['T²']*T**2 + coefs['RH²']*RH**2
            + coefs['√FT']*sFT + coefs['√Crack']*sCR + coefs['Water']*W + coefs['Water²']*W**2
            + coefs['RH×Water']*RH*W + coefs['√FT×√Crack']*sFT*sCR
            + coefs['T×√Crack']*T*sCR + coefs['RH×√Crack']*RH*sCR)

ft_only = df_deg[(df_deg['裂缝长度'] == 0) & (df_deg['含水率'] == 0) & (df_deg['冻融次数'] > 0)]
cr_only = df_deg[(df_deg['冻融次数'] == 0) & (df_deg['含水率'] == 0) & (df_deg['裂缝长度'] > 0)]
wt_only = df_deg[(df_deg['冻融次数'] == 0) & (df_deg['裂缝长度'] == 0) & (df_deg['含水率'] > 0)]
cp = df_deg[(df_deg['冻融次数'] > 0) & (df_deg['裂缝长度'] > 0) & (df_deg['含水率'] > 0)]

rows = []
for T in sorted(df_deg['温度'].unique()):
    for RH in sorted(df_deg['湿度'].unique()):
        T, RH = int(T), int(RH)
        l0 = imap[(T, RH)]
        y_ft = float(ft_only[(ft_only['温度'] == T) & (ft_only['湿度'] == RH)]['导热系数'].mean())
        y_cr = float(cr_only[(cr_only['温度'] == T) & (cr_only['湿度'] == RH)]['导热系数'].mean())
        y_wt = float(wt_only[(wt_only['温度'] == T) & (wt_only['湿度'] == RH)]['导热系数'].mean())
        y_cp = float(cp[(cp['温度'] == T) & (cp['湿度'] == RH)]['导热系数'].mean())
        # 方程同口径：水平均值
        y_ft_eq = float(np.mean([eq_pred(T, RH, f, 0, 0) for f in (18, 21, 24)]))
        y_cr_eq = float(np.mean([eq_pred(T, RH, 0, c, 0) for c in (2.5, 5.0, 10.0)]))
        y_wt_eq = float(np.mean([eq_pred(T, RH, 0, 0, w) for w in (23, 38, 56)]))
        # 复合三行：该格点对应的 L9 组合
        cp_rows = cp[(cp['温度'] == T) & (cp['湿度'] == RH)]
        y_cp_eq = float(np.mean([eq_pred(T, RH, r['冻融次数'], r['裂缝长度'], r['含水率']) for _, r in cp_rows.iterrows()]))
        # 实测
        d_ft, d_cr, d_wt = y_ft - l0, y_cr - l0, y_wt - l0
        L = l0 + d_ft + d_cr + d_wt
        fpi_exp = (y_cp - L) / abs(L) * 100
        # 方程
        d_ft_e, d_cr_e, d_wt_e = y_ft_eq - l0, y_cr_eq - l0, y_wt_eq - l0
        Le = l0 + d_ft_e + d_cr_e + d_wt_e
        fpi_eq = (y_cp_eq - Le) / abs(Le) * 100
        rows.append({'T': T, 'RH': RH, 'fpi_exp': fpi_exp, 'fpi_eq': fpi_eq})

df = pd.DataFrame(rows)
exp_m = df['fpi_exp'].mean()
eq_m = df['fpi_eq'].mean()
r_pearson = df['fpi_exp'].corr(df['fpi_eq'])
r_spearman = df['fpi_exp'].corr(df['fpi_eq'], method='spearman')
print("实测FPI均值: %.2f%%" % exp_m)
print("方程FPI均值(水平均值口径): %.2f%%" % eq_m)
print("捕获率: %.1f%%" % (eq_m / exp_m * 100))
print("Pearson r = %.3f | Spearman = %.3f" % (r_pearson, r_spearman))

# 旧口径对比：方程最大组合口径
old_eq = []
for T in sorted(df_deg['温度'].unique()):
    for RH in sorted(df_deg['湿度'].unique()):
        T, RH = int(T), int(RH)
        l0 = imap[(T, RH)]
        dft = eq_pred(T, RH, 24, 0, 0) - l0
        dcr = eq_pred(T, RH, 0, 10, 0) - l0
        dwt = eq_pred(T, RH, 0, 0, 56) - l0
        L = l0 + dft + dcr + dwt
        C = eq_pred(T, RH, 24, 10, 56)
        old_eq.append((C - L) / abs(L) * 100)
print("方程FPI均值(旧最大组合口径): %.2f%%" % np.mean(old_eq))

out = {'n': len(df), 'exp_mean': float(exp_m), 'eq_mean_levelmean': float(eq_m),
       'capture_pct': float(eq_m / exp_m * 100),
       'pearson_r': float(r_pearson), 'spearman_r': float(r_spearman),
       'eq_mean_old_maxcombo': float(np.mean(old_eq)),
       'per_cell': df.to_dict('records')}
with open(os.path.join(OUT, 'fpi_consistent.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print("saved: fpi_consistent.json")

# ---- 三阶项（线性三重）12项方程，同一水平均值口径 ----
coefs12 = dict(coefs)
coefs12['FT×Crack×Water'] = -2.202812168897476e-07
def eq_pred12(T, RH, FT, CR, W):
    sFT, sCR = np.sqrt(np.maximum(FT, 0)), np.sqrt(np.maximum(CR, 0))
    return (coefs12['const'] + coefs12['T']*T + coefs12['T²']*T**2 + coefs12['RH²']*RH**2
            + coefs12['√FT']*sFT + coefs12['√Crack']*sCR + coefs12['Water']*W + coefs12['Water²']*W**2
            + coefs12['RH×Water']*RH*W + coefs12['√FT×√Crack']*sFT*sCR
            + coefs12['T×√Crack']*T*sCR + coefs12['RH×√Crack']*RH*sCR
            + coefs12['FT×Crack×Water']*FT*CR*W)
eq12 = []
for T in sorted(df_deg['温度'].unique()):
    for RH in sorted(df_deg['湿度'].unique()):
        T, RH = int(T), int(RH)
        l0 = imap[(T, RH)]
        dft = float(np.mean([eq_pred12(T, RH, f, 0, 0) for f in (18, 21, 24)])) - l0
        dcr = float(np.mean([eq_pred12(T, RH, 0, c, 0) for c in (2.5, 5.0, 10.0)])) - l0
        dwt = float(np.mean([eq_pred12(T, RH, 0, 0, w) for w in (23, 38, 56)])) - l0
        L = l0 + dft + dcr + dwt
        cp_rows = cp[(cp['温度'] == T) & (cp['湿度'] == RH)]
        C = float(np.mean([eq_pred12(T, RH, r['冻融次数'], r['裂缝长度'], r['含水率']) for _, r in cp_rows.iterrows()]))
        eq12.append((C - L) / abs(L) * 100)
print("12项(线性三重)方程FPI均值(水平均值口径): %.2f%%" % np.mean(eq12))

# ---- 杠杆检验：逐格点剔除最大 FPI 格点 ----
idx_max = df['fpi_exp'].idxmax()
df2 = df.drop(index=idx_max)
capture_no_max = df2['fpi_eq'].mean() / df2['fpi_exp'].mean() * 100
print("剔除实测FPI最大格点(T=%s,RH=%s)后: 实测%.2f%% 方程%.2f%% 捕获%.1f%%" % (
    df.loc[idx_max, 'T'], df.loc[idx_max, 'RH'], df2['fpi_exp'].mean(), df2['fpi_eq'].mean(), capture_no_max))
