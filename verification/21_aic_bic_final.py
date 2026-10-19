# -*- coding: utf-8 -*-
"""表6 四结构 AIC/BIC 一致重算（与论文最终口径一致）。"""
import sys, io, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
import statsmodels.api as sm

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
OUT = os.path.dirname(os.path.abspath(__file__))

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
pool = {'T': T, 'T²': T**2, 'RH': RH, 'RH²': RH**2, 'FT': FT, 'Crack': Cr,
        '√FT': sFT, '√Crack': sCr, 'Water': W, 'Water²': W**2,
        'RH×Water': RH*W, '√FT×√Crack': sFT*sCr, 'T×√Crack': T*sCr,
        'RH×√Crack': RH*sCr, '√Crack×Water': sCr*W}

structs = {
    '朴素线性（5 项）': ['T', 'RH', 'FT', 'Crack', 'Water'],
    '非线性主效应（8 项）': ['T', 'RH', '√FT', '√Crack', 'Water', 'T²', 'RH²', 'Water²'],
    '精炼方程（11 项）': ['T', 'T²', 'RH²', '√FT', '√Crack', 'Water', 'Water²',
                       'RH×Water', '√FT×√Crack', 'T×√Crack', 'RH×√Crack'],
    '完整 SHAP 驱动（13 项）': ['T', 'RH', '√FT', '√Crack', 'Water', 'T²', 'RH²', 'Water²',
                              '√Crack×Water', 'RH×Water', 'T×√Crack', '√FT×√Crack', 'RH×√Crack'],
}
out = {}
for nm, terms in structs.items():
    X = np.column_stack([pool[t] for t in terms])
    m = sm.OLS(y, sm.add_constant(X)).fit()
    out[nm] = {'k': len(terms) + 1, 'train_r2': float(m.rsquared),
               'aic': float(m.aic), 'bic': float(m.bic)}
    print("%-22s k=%2d R2=%.4f AIC=%9.2f BIC=%9.2f" % (
        nm, len(terms)+1, m.rsquared, m.aic, m.bic))
with open(os.path.join(OUT, 'aic_bic_final.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
