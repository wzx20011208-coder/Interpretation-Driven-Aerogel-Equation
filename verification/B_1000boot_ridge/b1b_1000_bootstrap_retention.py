# -*- coding: utf-8 -*-
"""
B1b. 1000 次分层 bootstrap 结构选择稳定性（保留频率）
====================================================
复刻 v3 bootstrap_stability 方法论（同一随机流、同一消元逻辑），仅将次数 300 → 1000，
供论文图9(a) 与正文"1000 次分层 Bootstrap"口径对齐。

方法论（与 可解释性方程特征分析优化_v3.py::bootstrap_stability 完全一致）：
  1. 按工况场景分层有放回重采样
  2. 每次训练 CatBoost（无权重）并计算 SHAP 交互值，取 Top5 交互对
  3. 构造 13 项候选方程（8 主效应 + 5 交互），OLS 反向消元（p>0.05，平方项受保护）
  4. 记录每项保留频率
每 100 次迭代写一次检查点。
"""
import os
import json
import sys
import io
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
from catboost import CatBoostRegressor
import shap

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
OUT = os.path.join(OUT_DIR, 'B1b_1000_bootstrap_retention.json')
N_BOOT = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
RANDOM_STATE = 42
FEATURES = ['温度', '湿度', '冻融次数', '裂缝长度', '含水率']
TARGET = '导热系数'

CATBOOST_PARAMS = dict(
    iterations=300, depth=4, learning_rate=0.05,
    random_seed=RANDOM_STATE, verbose=False
)

V2_FINAL_TERMS = {'T', 'T²', 'RH²', '√FT', '√Crack', 'Water', 'Water²',
                  'T×√Crack', 'RH×Water', '√FT×√Crack', 'RH×√Crack'}

INTERACTION_MAP = {
    ('温度', '湿度'): 'T×RH',
    ('温度', '冻融次数'): 'T×√FT',
    ('温度', '裂缝长度'): 'T×√Crack',
    ('温度', '含水率'): 'T×Water',
    ('湿度', '冻融次数'): 'RH×√FT',
    ('湿度', '裂缝长度'): 'RH×√Crack',
    ('湿度', '含水率'): 'RH×Water',
    ('冻融次数', '裂缝长度'): '√FT×√Crack',
    ('冻融次数', '含水率'): '√FT×Water',
    ('裂缝长度', '含水率'): '√Crack×Water',
}

MAIN_NAMES = ['T', 'RH', '√FT', '√Crack', 'Water', 'T²', 'RH²', 'Water²']
ALL_CANDIDATES = MAIN_NAMES + list(INTERACTION_MAP.values())
PROTECT_PAIRS = [('T', 'T²'), ('RH', 'RH²'), ('Water', 'Water²')]


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
        os.path.join(DATA_DIR, 'intact_conditions_summary.xlsx'), sheet_name='summary')
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


def one_iteration(df_boot, boot_params):
    T = df_boot['温度'].values
    RH = df_boot['湿度'].values
    FT = df_boot['冻融次数'].values
    Crack = df_boot['裂缝长度'].values
    Water = df_boot['含水率'].values
    y = df_boot[TARGET].values

    sqrtFT, sqrtCrack = np.sqrt(FT), np.sqrt(Crack)

    X_cb = df_boot[FEATURES].values
    cb_model = CatBoostRegressor(**boot_params)
    cb_model.fit(X_cb, y)

    explainer = shap.TreeExplainer(cb_model)
    shap_inter = explainer.shap_interaction_values(X_cb)

    n_feat = len(FEATURES)
    pair_list = []
    for ii in range(n_feat):
        for jj in range(ii + 1, n_feat):
            strength = np.mean(np.abs(shap_inter[:, ii, jj]))
            pair_list.append((FEATURES[ii], FEATURES[jj], strength))
    pair_list.sort(key=lambda x: x[2], reverse=True)

    top5 = []
    for f1, f2, s in pair_list[:5]:
        eq_name = INTERACTION_MAP.get((f1, f2)) or INTERACTION_MAP.get((f2, f1))
        if eq_name:
            top5.append(eq_name)
    if len(top5) < 5:
        return None

    feature_builders = {
        'T': lambda: T, 'RH': lambda: RH, '√FT': lambda: sqrtFT,
        '√Crack': lambda: sqrtCrack, 'Water': lambda: Water,
        'T²': lambda: T**2, 'RH²': lambda: RH**2, 'Water²': lambda: Water**2,
        '√Crack×Water': lambda: sqrtCrack * Water, 'RH×Water': lambda: RH * Water,
        'T×√Crack': lambda: T * sqrtCrack, '√FT×√Crack': lambda: sqrtFT * sqrtCrack,
        'RH×√Crack': lambda: RH * sqrtCrack, 'T×RH': lambda: T * RH,
        'T×√FT': lambda: T * sqrtFT, 'T×Water': lambda: T * Water,
        'RH×√FT': lambda: RH * sqrtFT, '√FT×Water': lambda: sqrtFT * Water,
    }

    current_names = MAIN_NAMES + top5
    current_inter = list(top5)

    # 第一阶段：移除不显著交互项
    while True:
        X = np.column_stack([feature_builders[n]() for n in current_names])
        model_ols = sm.OLS(y, sm.add_constant(X)).fit()
        inter_pvals = [(n, float(model_ols.pvalues[current_names.index(n) + 1]))
                       for n in current_names if n in current_inter]
        if not inter_pvals:
            break
        inter_pvals.sort(key=lambda x: x[1], reverse=True)
        worst_name, worst_p = inter_pvals[0]
        if worst_p > 0.05:
            current_names.remove(worst_name)
            current_inter.remove(worst_name)
        else:
            break

    # 第二阶段：移除不显著主效应（保护平方项）
    while True:
        X = np.column_stack([feature_builders[n]() for n in current_names])
        model_ols = sm.OLS(y, sm.add_constant(X)).fit()
        main_in_model = [n for n in current_names if n not in current_inter]
        candidates = []
        for name in main_in_model:
            pval = float(model_ols.pvalues[current_names.index(name) + 1])
            protected = any(name == p2 and p1 in current_names
                            for p1, p2 in PROTECT_PAIRS)
            if not protected:
                candidates.append((name, pval))
        if not candidates:
            break
        candidates.sort(key=lambda x: x[1], reverse=True)
        worst_name, worst_p = candidates[0]
        if worst_p > 0.05:
            current_names.remove(worst_name)
        else:
            break

    return current_names


def main():
    df_all = load_data()
    print(f"数据: {len(df_all)} 条")
    scenarios = sorted(df_all['场景'].unique())
    strata = {s: df_all[df_all['场景'] == s] for s in scenarios}
    for s in scenarios:
        print(f"  {s}: {len(strata[s])} 条")

    boot_params = dict(CATBOOST_PARAMS)
    boot_params['allow_writing_files'] = False

    retention_count = {term: 0 for term in ALL_CANDIDATES}
    term_counts = []
    valid_iters = 0
    rng = np.random.RandomState(RANDOM_STATE)

    t0 = time.time()
    for it in range(N_BOOT):
        try:
            parts = []
            for s in scenarios:
                group = strata[s]
                idx = rng.choice(len(group), size=len(group), replace=True)
                parts.append(group.iloc[idx].reset_index(drop=True))
            df_boot = pd.concat(parts, ignore_index=True)

            kept = one_iteration(df_boot, boot_params)
            if kept is None:
                term_counts.append(0)
            else:
                for term in kept:
                    if term in retention_count:
                        retention_count[term] += 1
                term_counts.append(len(kept))
                valid_iters += 1
        except Exception:
            term_counts.append(0)

        if (it + 1) % 50 == 0:
            el = time.time() - t0
            eta = el / (it + 1) * (N_BOOT - it - 1)
            print(f"  进度 {it+1}/{N_BOOT} 有效{valid_iters} "
                  f"({el:.0f}s, ETA {eta/60:.1f}min)", flush=True)

        # 检查点
        if (it + 1) % 100 == 0:
            freq = {t: c / valid_iters * 100 if valid_iters else 0.0
                    for t, c in retention_count.items()}
            with open(OUT.replace('.json', f'_{N_BOOT}.json'), 'w', encoding='utf-8') as f:
                json.dump({
                    'n_bootstrap': it + 1,
                    'valid_iterations': valid_iters,
                    'retention_frequency': freq,
                    'term_count_distribution': term_counts,
                }, f, ensure_ascii=False, indent=2)

    freq = {t: c / valid_iters * 100 if valid_iters else 0.0
            for t, c in retention_count.items()}
    suffix = f"_{N_BOOT}" if N_BOOT != 1000 else ""
    result = {
        'n_bootstrap': N_BOOT,
        'valid_iterations': valid_iters,
        'retention_frequency': freq,
        'term_count_distribution': term_counts,
    }
    with open(OUT.replace('.json', f'{suffix}.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n完成: 有效 {valid_iters}/{N_BOOT}，耗时 {(time.time()-t0)/60:.1f} min")
    print(f"{'项':<16s}{'保留%':>8s}  最终方程")
    for term in sorted(freq, key=freq.get, reverse=True):
        print(f"{term:<16s}{freq[term]:>8.1f}  {'是' if term in V2_FINAL_TERMS else '否'}")
    print(f"已保存: {OUT}")


if __name__ == '__main__':
    main()
