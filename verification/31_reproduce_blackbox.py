# -*- coding: utf-8 -*-
"""黑箱基准 0.9821 可复现脚本（无权重 CatBoost 5 折 CV）。"""
import sys, io, json, os
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
OUT = os.path.dirname(os.path.abspath(__file__))
FEATURES = ['温度', '湿度', '冻融次数', '裂缝长度', '含水率']
TARGET = '导热系数'

df_deg = pd.read_excel(os.path.join(DATA_DIR, 'degradation_dataset.xlsx'), sheet_name='degradation_data')
df_deg.columns = ['样本', '温度', '湿度', '冻融次数', '裂缝长度', '含水率', '导热系数']
df_is = pd.read_excel(os.path.join(DATA_DIR, 'intact_conditions_summary.xlsx'),
                      sheet_name='summary')
df_int = pd.DataFrame({'温度': df_is['Temperature_C'], '湿度': df_is['RH_pct'], '冻融次数': 0,
                       '裂缝长度': 0.0, '含水率': 0, '导热系数': df_is['Thermal_Conductivity_Mean']})
df = pd.concat([df_deg, df_int], ignore_index=True)
X = df[FEATURES].values.astype(float)
y = df[TARGET].values.astype(float)

KF = KFold(n_splits=5, shuffle=True, random_state=42)
r2s, rmses, mapes = [], [], []
for tr, te in KF.split(X):
    m = CatBoostRegressor(iterations=300, depth=4, learning_rate=0.05,
                          random_seed=42, verbose=False)
    m.fit(X[tr], y[tr])
    p = m.predict(X[te])
    ss_res = np.sum((y[te]-p)**2); ss_tot = np.sum((y[te]-y[te].mean())**2)
    r2s.append(1-ss_res/ss_tot)
    rmses.append(np.sqrt(np.mean((y[te]-p)**2)))
    mapes.append(np.mean(np.abs((y[te]-p)/y[te]))*100)

res = {'n': len(df), 'cv_r2': float(np.mean(r2s)), 'cv_r2_std': float(np.std(r2s)),
       'cv_rmse': float(np.mean(rmses)), 'cv_mape': float(np.mean(mapes))}
print("黑箱(无权重 CatBoost) 5折CV: R²=%.4f±%.4f RMSE=%.6f MAPE=%.2f%%" % (
    res['cv_r2'], res['cv_r2_std'], res['cv_rmse'], res['cv_mape']))
with open(os.path.join(OUT, 'blackbox_repro.json'), 'w', encoding='utf-8') as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
print("saved:", os.path.join(OUT, 'blackbox_repro.json'))
