# -*- coding: utf-8 -*-
"""
B1. 1000 次分层 bootstrap 系数稳定性（审稿模拟 B 组）
=====================================================
两种分层口径都跑：
  (A) 按工况场景分层（与现稿 v4_supplement B1 口径一致，可复现 300 次的 100%）
  (B) 按温度-湿度格点分层（正文目前声称的口径，27 个格点内重采样）
输出：每项系数全量估计、bootstrap 均值、95% 百分位区间、符号稳定率。
"""
import os
import json
import sys
import io
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
OUT = os.path.join(BASE, 'B1_1000_bootstrap.json')
RANDOM_STATE = 42
FEATURES = ['温度', '湿度', '冻融次数', '裂缝长度', '含水率']
TARGET = '导热系数'
EQ_NAMES = ['T', 'T²', 'RH²', '√FT', '√Crack', 'Water', 'Water²',
            'T×√Crack', 'RH×Water', '√FT×√Crack', 'RH×√Crack']
NAMES = ['const'] + EQ_NAMES


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

    df_intact_summary = pd.read_excel(
        os.path.join(DATA_DIR, 'intact_conditions_summary.xlsx'),
        sheet_name='summary'
    )
    df_intact = pd.DataFrame({
        '温度': df_intact_summary['Temperature_C'],
        '湿度': df_intact_summary['RH_pct'],
        '冻融次数': 0,
        '裂缝长度': 0.0,
        '含水率': 0,
        '导热系数': df_intact_summary['Thermal_Conductivity_Mean'],
    })
    df_all = pd.concat([df_deg, df_intact], ignore_index=True)
    df_all['场景'] = df_all.apply(identify_scenario, axis=1)
    return df_all


def build_X(df):
    T = df['温度'].values.astype(float)
    RH = df['湿度'].values.astype(float)
    FT = df['冻融次数'].values.astype(float)
    Crack = df['裂缝长度'].values.astype(float)
    Water = df['含水率'].values.astype(float)
    sFT, sCR = np.sqrt(FT), np.sqrt(Crack)
    X = np.column_stack([T, T ** 2, RH ** 2, sFT, sCR, Water, Water ** 2,
                         T * sCR, RH * Water, sFT * sCR, RH * sCR])
    return X


def fit_ols(df):
    X = build_X(df)
    y = df[TARGET].values.astype(float)
    X_sm = sm.add_constant(X, has_constant='add')
    model = sm.OLS(y, X_sm).fit()
    return model, X, y


def bootstrap_coefs(df_all, stratify, n_boot=1000, seed=42):
    """分层有放回重采样，返回系数矩阵 (n_boot, 12)。"""
    rng = np.random.RandomState(seed)
    if stratify == 'scenario':
        keys = sorted(df_all['场景'].unique())
        strata = {k: df_all[df_all['场景'] == k] for k in keys}
    else:  # T-RH 格点
        df_all = df_all.copy()
        df_all['_cell'] = df_all['温度'].astype(str) + '_' + df_all['湿度'].astype(str)
        keys = sorted(df_all['_cell'].unique())
        strata = {k: df_all[df_all['_cell'] == k] for k in keys}
    matrix = []
    for it in range(n_boot):
        parts = []
        for k in keys:
            g = strata[k]
            idx = rng.choice(len(g), size=len(g), replace=True)
            parts.append(g.iloc[idx].reset_index(drop=True))
        df_boot = pd.concat(parts, ignore_index=True)
        m, _, _ = fit_ols(df_boot)
        matrix.append(m.params.astype(float))
    return np.array(matrix)


def summarize(full_params, matrix):
    out = {}
    for i, n in enumerate(NAMES):
        lo, hi = np.percentile(matrix[:, i], [2.5, 97.5])
        sign_stable = float(np.mean(np.sign(matrix[:, i]) == np.sign(full_params[i]))) * 100
        out[n] = {
            'full_fit': float(full_params[i]),
            'boot_mean': float(np.mean(matrix[:, i])),
            'ci95': [float(lo), float(hi)],
            'sign_stability_pct': sign_stable,
        }
    return out


def main():
    df_all = load_data()
    full_model, _, _ = fit_ols(df_all)
    full_params = full_model.params.astype(float)
    print(f"数据: {len(df_all)} 条；最终方程 11 项 + 常数")

    results = {}
    for label, strat in [('scenario', 'scenario'), ('t_rh_grid', 't_rh_grid')]:
        print(f"\n===== bootstrap 1000 次（{label} 分层） =====")
        matrix = bootstrap_coefs(df_all, strat, n_boot=1000)
        res = summarize(full_params, matrix)
        results[label] = {'n_bootstrap': 1000, 'stratify': strat, 'coef_ci': res}
        mins = min(v['sign_stability_pct'] for v in res.values())
        print(f"  最小符号稳定率: {mins:.1f}%")
        for n, v in res.items():
            print(f"  {n:14s} 全量={v['full_fit']:+.8f} "
                  f"CI=[{v['ci95'][0]:+.8f},{v['ci95'][1]:+.8f}] 符号稳定={v['sign_stability_pct']:.0f}%")

    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n已保存: {OUT}")


if __name__ == '__main__':
    main()
