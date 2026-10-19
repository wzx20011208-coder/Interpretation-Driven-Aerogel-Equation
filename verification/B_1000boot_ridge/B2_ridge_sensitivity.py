# -*- coding: utf-8 -*-
"""
B2. Ridge 敏感性对照（审稿模拟 B 组）
====================================
目的：证明高 VIF（7 项 >10）没有扭曲方程系数的符号与量级趋势。
做法：对最终 11 项设计矩阵做 z-score 标准化，比较 OLS 与 Ridge
（alpha 网格 + 5 折 RidgeCV）的系数符号一致性。
"""
import os
import json
import sys
import io
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
OUT = os.path.join(BASE, 'B2_ridge_sensitivity.json')
RANDOM_STATE = 42
FEATURES = ['温度', '湿度', '冻融次数', '裂缝长度', '含水率']
TARGET = '导热系数'
EQ_NAMES = ['T', 'T²', 'RH²', '√FT', '√Crack', 'Water', 'Water²',
            'T×√Crack', 'RH×Water', '√FT×√Crack', 'RH×√Crack']


def identify_scenario(row):
    ft, crack, water = row['冻融次数'], row['裂缝长度'], row['含水率']
    if ft == 0 and crack == 0 and water == 0:
        return '完好'
    elif ft > 0 and crack == 0 and water == 0:
        return '纯冻融'
    elif ft == 0 and crack > 0 and water == 0:
        return '纯裂缝'
    elif ft == 0 and crack == 0 and water > 0:
        return '纯渗水'
    return '复合'


def load_data():
    df_deg = pd.read_excel(os.path.join(DATA_DIR, 'degradation_dataset.xlsx'), sheet_name='degradation_data')
    df_deg.columns = ['样本', '温度', '湿度', '冻融次数', '裂缝长度', '含水率', '导热系数']
    df_deg = df_deg[FEATURES + [TARGET]].copy()
    df_intact = pd.read_excel(
        os.path.join(DATA_DIR, 'intact_conditions_summary.xlsx'),
        sheet_name='summary'
    )
    df_intact = pd.DataFrame({
        '温度': df_intact['Temperature_C'],
        '湿度': df_intact['RH_pct'],
        '冻融次数': 0,
        '裂缝长度': 0.0,
        '含水率': 0,
        '导热系数': df_intact['Thermal_Conductivity_Mean'],
    })
    return pd.concat([df_deg, df_intact], ignore_index=True)


def build_X(df):
    T = df['温度'].values.astype(float)
    RH = df['湿度'].values.astype(float)
    FT = df['冻融次数'].values.astype(float)
    Crack = df['裂缝长度'].values.astype(float)
    Water = df['含水率'].values.astype(float)
    sFT, sCR = np.sqrt(FT), np.sqrt(Crack)
    return np.column_stack([T, T ** 2, RH ** 2, sFT, sCR, Water, Water ** 2,
                            T * sCR, RH * Water, sFT * sCR, RH * sCR])


def main():
    df_all = load_data()
    X = build_X(df_all)
    y = df_all[TARGET].values.astype(float)

    # 参考：原始尺度 OLS（与正文一致）+ VIF
    X_sm = sm.add_constant(X, has_constant='add')
    ols_raw = sm.OLS(y, X_sm).fit()
    vif = {EQ_NAMES[i]: float(variance_inflation_factor(X, i)) for i in range(X.shape[1])}

    # 标准化设计矩阵
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    ols_z = sm.OLS(y, sm.add_constant(Xz, has_constant='add')).fit()
    ols_z_coef = ols_z.params[1:]
    ols_signs = np.sign(ols_z_coef)

    # Ridge alpha 网格
    alphas = np.logspace(-5, 2, 15)
    grid = []
    for a in alphas:
        ridge = Ridge(alpha=a, fit_intercept=True, random_state=RANDOM_STATE)
        ridge.fit(Xz, y)
        coef = ridge.coef_
        agree = int(np.sum(np.sign(coef) == ols_signs))
        grid.append({
            'alpha': float(a),
            'sign_agreement': agree,
            'sign_agreement_frac': float(agree / len(EQ_NAMES)),
            'coefs': {EQ_NAMES[i]: float(coef[i]) for i in range(len(EQ_NAMES))},
        })

    # RidgeCV（5 折，与正文随机 5 折口径一致）
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    ridge_cv = RidgeCV(alphas=alphas, cv=kf, scoring='neg_mean_squared_error')
    ridge_cv.fit(Xz, y)
    cv_coef = ridge_cv.coef_
    cv_agree = int(np.sum(np.sign(cv_coef) == ols_signs))
    r2_cv = ridge_cv.score(Xz, y)

    print(f"参考 OLS（标准化）系数符号:")
    print("  " + " ".join(f"{n}:{'−' if s < 0 else '+'}" for n, s in zip(EQ_NAMES, ols_signs)))
    print(f"\nVIF（原始尺度）: " + ", ".join(f"{n}={vif[n]:.1f}" for n in EQ_NAMES))
    print(f"VIF>10 的项数: {sum(1 for v in vif.values() if v > 10)}")

    print(f"\nRidgeCV: alpha*={ridge_cv.alpha_:.5g}, 5折R²={r2_cv:.4f}, "
          f"符号一致 {cv_agree}/11")
    print("\nalpha 网格符号一致数:")
    for g in grid:
        print(f"  alpha={g['alpha']:.5g}: {g['sign_agreement']}/11")

    result = {
        'n': int(len(df_all)),
        'reference_ols_raw': {n: float(c) for n, c in zip(['const'] + EQ_NAMES, ols_raw.params)},
        'reference_ols_std': {EQ_NAMES[i]: float(ols_z_coef[i]) for i in range(len(EQ_NAMES))},
        'vif': vif,
        'ridge_grid': grid,
        'ridge_cv': {
            'alpha': float(ridge_cv.alpha_),
            'r2_full_sample': float(r2_cv),
            'sign_agreement': cv_agree,
            'sign_agreement_frac': float(cv_agree / len(EQ_NAMES)),
            'coefs_std': {EQ_NAMES[i]: float(cv_coef[i]) for i in range(len(EQ_NAMES))},
        },
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {OUT}")


if __name__ == '__main__':
    main()
