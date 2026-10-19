# -*- coding: utf-8 -*-
r"""
C_PySR基线 第1步:5折CV 精度对比
=================================
对比对象(同一 KFold(5, shuffle, seed=42) 划分):
  1. 11项显式方程(本文,无权重 OLS)        —— 基准,已复现 R2=0.9594 / MAPE=1.76%
  2. CatBoost 黑箱(论文协议,加权,300/4/0.05) —— 上界参考
  3. PySR 配置A:通用算子 (+,-,*,/,square)    —— 盲搜基线
  4. PySR 配置B:物理增强 (+,-,*,/,square,sqrt,exp,log) —— 宽松基线
PySR 每折在训练集内部自行选模(model_selection='best'),不接触测试折。
输出: cv_results.json + 控制台汇总表
"""
import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from catboost import CatBoostRegressor

warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES = ['温度', '湿度', '冻融次数', '裂缝长度', '含水率']
FEATURES_EN = ['T', 'RH', 'FT', 'CR', 'W']
TARGET = '导热系数'
RANDOM_STATE = 42
KF = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

CATBOOST_PARAMS = dict(
    iterations=300, depth=4, learning_rate=0.05,
    random_seed=RANDOM_STATE, verbose=False
)

PYSR_COMMON = dict(
    niterations=60,
    populations=15,
    population_size=33,
    maxsize=25,
    model_selection='best',
    random_state=RANDOM_STATE,
    deterministic=True,
    parallelism='serial',
    verbosity=0,
)

CONFIGS = {
    'A_通用算子': dict(binary_operators=['+', '-', '*', '/'],
                       unary_operators=['square']),
    'B_物理增强': dict(binary_operators=['+', '-', '*', '/'],
                       unary_operators=['square', 'sqrt', 'exp', 'log']),
}


def load_data():
    df_deg = pd.read_excel(os.path.join(DATA_DIR, 'degradation_dataset.xlsx'), sheet_name='degradation_data')
    df_deg.columns = ['样本', '温度', '湿度', '冻融次数', '裂缝长度', '含水率', '导热系数']
    df_deg = df_deg[FEATURES + [TARGET]].copy()
    df_sum = pd.read_excel(os.path.join(DATA_DIR, 'intact_conditions_summary.xlsx'),
                           sheet_name='summary')
    df_int = pd.DataFrame({
        '温度': df_sum['Temperature_C'], '湿度': df_sum['RH_pct'], '冻融次数': 0,
        '裂缝长度': 0.0, '含水率': 0, '导热系数': df_sum['Thermal_Conductivity_Mean'],
        'sample_weight': df_sum['N_Readings'].astype(float),
    })
    df_deg['sample_weight'] = 1.0
    return pd.concat([df_deg, df_int], ignore_index=True)


def build_terms(df):
    T, RH, FT, CR, W = (df['温度'].values, df['湿度'].values,
                        df['冻融次数'].values, df['裂缝长度'].values, df['含水率'].values)
    sFT, sCR = np.sqrt(FT), np.sqrt(CR)
    return np.column_stack([
        T, T ** 2, RH ** 2, sFT, sCR, W, W ** 2,
        T * sCR, RH * W, sFT * sCR, RH * sCR])


def calc_metrics(y_true, y_pred):
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return float(r2), rmse, mape


def cv_ols_equation(df):
    X, y = build_terms(df), df[TARGET].values
    res = {'fold_r2': [], 'fold_mape': []}
    for tr, te in KF.split(X):
        m = LinearRegression().fit(X[tr], y[tr])
        r2, _, mape = calc_metrics(y[te], m.predict(X[te]))
        res['fold_r2'].append(r2)
        res['fold_mape'].append(mape)
    return res


def cv_catboost(df):
    X, y = df[FEATURES].values, df[TARGET].values
    w = df['sample_weight'].values
    res = {'fold_r2': [], 'fold_mape': []}
    for tr, te in KF.split(X):
        m = CatBoostRegressor(**CATBOOST_PARAMS)
        m.fit(X[tr], y[tr], sample_weight=w[tr])
        r2, _, mape = calc_metrics(y[te], m.predict(X[te]))
        res['fold_r2'].append(r2)
        res['fold_mape'].append(mape)
    return res


def cv_pysr(df, config_name, cfg):
    from pysr import PySRRegressor
    Xdf, y = df[FEATURES].rename(columns=dict(zip(FEATURES, FEATURES_EN))), df[TARGET].values
    res = {'fold_r2': [], 'fold_mape': [], 'fold_equations': [], 'fold_complexity': []}
    for k, (tr, te) in enumerate(KF.split(Xdf)):
        t0 = time.time()
        model = PySRRegressor(**PYSR_COMMON, **cfg)
        model.fit(Xdf.iloc[tr], y[tr])
        pred = model.predict(Xdf.iloc[te])
        r2, _, mape = calc_metrics(y[te], pred)
        best = model.get_best()
        res['fold_r2'].append(r2)
        res['fold_mape'].append(mape)
        res['fold_equations'].append(best.equation)
        res['fold_complexity'].append(int(best.complexity))
        print(f"  [{config_name}] fold{k + 1}: R2={r2:.4f} MAPE={mape:.2f}% "
              f"complexity={best.complexity} 用时{time.time() - t0:.0f}s", flush=True)
    return res


def main():
    df = load_data()
    out = {}

    print("== 1/4 11项显式方程 (本文基准) ==", flush=True)
    out['11项方程(本文)'] = cv_ols_equation(df)

    print("== 2/4 CatBoost 黑箱 (论文协议) ==", flush=True)
    out['CatBoost黑箱'] = cv_catboost(df)

    for name, cfg in CONFIGS.items():
        print(f"== PySR {name} 5折CV ==", flush=True)
        out[f'PySR {name}'] = cv_pysr(df, name, cfg)

    summary = []
    for name, res in out.items():
        summary.append({
            'method': name,
            'cv_R2_mean': float(np.mean(res['fold_r2'])),
            'cv_R2_std': float(np.std(res['fold_r2'])),
            'cv_MAPE_pct': float(np.mean(res['fold_mape'])),
        })
        if 'fold_complexity' in res:
            summary[-1]['mean_complexity'] = float(np.mean(res['fold_complexity']))

    print("\n===== 5折CV 汇总 =====")
    print(f"{'方法':<22}{'CV R²':<16}{'CV MAPE':<12}{'平均复杂度':<10}")
    for s in summary:
        comp = f"{s.get('mean_complexity', '-')}"
        print(f"{s['method']:<22}{s['cv_R2_mean']:.4f}±{s['cv_R2_std']:.4f}"
              f"{s['cv_MAPE_pct']:<12.2f}{comp:<10}")

    with open(os.path.join(HERE, 'cv_results.json'), 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'folds': out}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {os.path.join(HERE, 'cv_results.json')}")


if __name__ == '__main__':
    main()
