# -*- coding: utf-8 -*-
"""
可解释性方程特征分析优化 v4 补充验证（无新实验，仅气凝胶数据）
================================================================
在 v3 最终 11 项方程基础上，补齐"不做新实验也能做"的三类证据：

  A. 设计感知验证（替代独立测试集）
     A1 留一水平外推：冻融/裂缝/含水率 各水平分别留出，训练其余数据后预测
     A2 耦合项可辨识性：仅用完好+单因素场景时交互项是否可估计（论证复合实验的必要性）
     A3 留一 T-RH 格子交叉验证：27 个温度-湿度格子逐个留出（分组 CV，替代随机 5 折）
     A4 完好工况验证：1542 条原始测量 vs 27 条均值两种口径的基准层对比 + 格内留出

  B. 不确定度传播
     B1 300 次分层 bootstrap：系数 95% CI + 符号稳定性
     B2 解析 95% 置信/预测区间 + bootstrap 预测带（关键边际曲线）

  D. 显式方程方差分解
     D1 Sobol 一阶/总效应指数（设计域均匀抽样，Sobol 序列）
     D2 三个物理层贡献统计量 + 每项效应量（设计域内最大|贡献|）

结果写入 可解释性方程特征分析优化_结果/v4_supplement.json
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===== Elsevier 图形规范（印刷列宽：单栏 3.54in / 1.5栏 5.51in / 通栏 7.48in）=====
# 字号按最终印刷尺寸绝对设定：正文 7.5pt / 轴标 8pt / 标题 9pt / 刻度图例 7pt
# 拉丁字符 Arial，中文自动回退宋体；600 dpi；PDF 嵌入 TrueType（fonttype 42）
plt.rcParams.update({
    "font.family": ["Arial", "SimSun"],
    "font.size": 7.5,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "lines.linewidth": 1.0,
    "axes.linewidth": 0.8,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "axes.unicode_minus": False,
})
from scipy import stats
from scipy.stats import qmc
import statsmodels.api as sm

warnings.filterwarnings('ignore')
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
RANDOM_STATE = 42
FEATURES = ['温度', '湿度', '冻融次数', '裂缝长度', '含水率']
TARGET = '导热系数'

# v3 最终方程：11 项特征（不含截距）
EQ_NAMES = ['T', 'T²', 'RH²', '√FT', '√Crack', 'Water', 'Water²',
            'T×√Crack', 'RH×Water', '√FT×√Crack', 'RH×√Crack']


def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def calc_metrics(y_true, y_pred):
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return r2, rmse, mae, mape


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
    df_deg['数据来源'] = '劣化'

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
        '数据来源': '完好',
    })

    df_all = pd.concat([df_deg, df_intact], ignore_index=True)
    df_all['场景'] = df_all.apply(identify_scenario, axis=1)

    df_raw = pd.read_excel(
        os.path.join(DATA_DIR, 'intact_conditions_summary.xlsx'),
        sheet_name='raw_readings'
    )
    df_raw.columns = ['温度', '湿度', '导热系数']
    return df_all, df_raw


def build_X(df):
    T = df['温度'].values.astype(float)
    RH = df['湿度'].values.astype(float)
    FT = df['冻融次数'].values.astype(float)
    Crack = df['裂缝长度'].values.astype(float)
    Water = df['含水率'].values.astype(float)
    sFT = np.sqrt(FT)
    sCR = np.sqrt(Crack)
    X = np.column_stack([T, T ** 2, RH ** 2, sFT, sCR, Water, Water ** 2,
                         T * sCR, RH * Water, sFT * sCR, RH * sCR])
    return X, EQ_NAMES


def fit_equation(df, y=None):
    X, _ = build_X(df)
    if y is None:
        y = df[TARGET].values.astype(float)
    X_sm = sm.add_constant(X, has_constant='add')
    model = sm.OLS(y, X_sm).fit()
    return model, X, X_sm, y


def eq_predict_vectorized(T, RH, FT, Crack, Water, coefs):
    T = np.asarray(T, dtype=float)
    RH = np.asarray(RH, dtype=float)
    FT = np.asarray(FT, dtype=float)
    Crack = np.asarray(Crack, dtype=float)
    Water = np.asarray(Water, dtype=float)
    sFT = np.sqrt(np.maximum(FT, 0.0))
    sCR = np.sqrt(np.maximum(Crack, 0.0))
    return (coefs['const'] + coefs['T'] * T + coefs['T²'] * T ** 2
            + coefs['RH²'] * RH ** 2
            + coefs['√FT'] * sFT + coefs['√Crack'] * sCR
            + coefs['Water'] * Water + coefs['Water²'] * Water ** 2
            + coefs['T×√Crack'] * T * sCR + coefs['RH×Water'] * RH * Water
            + coefs['√FT×√Crack'] * sFT * sCR + coefs['RH×√Crack'] * RH * sCR)


def coefs_from_model(model):
    names = ['const'] + EQ_NAMES
    return {n: float(c) for n, c in zip(names, model.params)}


# ============================================================
# A. 设计感知验证
# ============================================================
def leave_one_level_out(df_all):
    """A1: 留一水平外推验证（每个劣化因子的每个非零水平）"""
    print("\n" + "=" * 60)
    print("A1. 留一水平外推验证")
    print("=" * 60)
    full_model, _, _, _ = fit_equation(df_all)
    factors = {
        '冻融次数': [18.0, 21.0, 24.0],
        '裂缝长度': [2.5, 5.0, 10.0],
        '含水率': [23.0, 38.0, 56.0],
    }
    results = []
    for factor, levels in factors.items():
        for lev in levels:
            mask = (df_all[factor] == lev)
            train = df_all[~mask]
            test = df_all[mask]
            Xte, _ = build_X(test)
            Xte_sm = sm.add_constant(Xte, has_constant='add')
            yte = test[TARGET].values.astype(float)

            m_loo, _, _, _ = fit_equation(train)
            pred_loo = m_loo.predict(Xte_sm)
            pred_full = full_model.predict(Xte_sm)
            r2_loo, rmse_loo, mae_loo, mape_loo = calc_metrics(yte, pred_loo)
            r2_full, rmse_full, mae_full, mape_full = calc_metrics(yte, pred_full)

            results.append({
                'factor': factor, 'level': float(lev), 'n_test': int(len(test)),
                'rmse_loo': float(rmse_loo), 'mape_loo': float(mape_loo), 'r2_loo': float(r2_loo),
                'rmse_full': float(rmse_full), 'mape_full': float(mape_full), 'r2_full': float(r2_full),
                'rmse_ratio': float(rmse_loo / rmse_full),
            })
            print(f"  [{factor}={lev:g}] n={len(test):3d}  "
                  f"留出拟合RMSE={rmse_loo:.6f} 全量拟合RMSE={rmse_full:.6f} 比值={rmse_loo/rmse_full:.2f}")
    return results


def coupling_identifiability(df_all):
    """A2: 无复合场景时耦合项是否可辨识（论证复合实验必要性）"""
    print("\n" + "=" * 60)
    print("A2. 耦合项可辨识性检查（无复合场景）")
    print("=" * 60)
    sub = df_all[df_all['场景'] != '复合']
    X, names = build_X(sub)
    X_sm = sm.add_constant(X, has_constant='add')
    rank = np.linalg.matrix_rank(X_sm)
    n_cols = X_sm.shape[1]
    print(f"  不含复合工况: {len(sub)}条, 设计矩阵秩={rank}/{n_cols}")
    # 检查每个交互列的方差
    inter_cols = {}
    for i, n in enumerate(names):
        if '×' in n:
            inter_cols[n] = float(np.var(X[:, i]))
    print("  交互列方差（0 表示完全不可辨识）:")
    for n, v in inter_cols.items():
        flag = '  <- 不可辨识' if v < 1e-20 else ''
        print(f"    {n:16s}: {v:.3e}{flag}")
    # 尝试拟合，捕获奇异
    try:
        model = sm.OLS(sub[TARGET].values.astype(float), X_sm).fit()
        print("  拟合成功（rank足够）")
    except Exception as e:
        print(f"  拟合失败: {type(e).__name__}: {str(e)[:120]}")
    return {
        'n_without_composite': int(len(sub)),
        'matrix_rank': int(rank),
        'n_columns': int(n_cols),
        'interaction_variances': inter_cols,
    }


def leave_one_cell_out(df_all):
    """A3: 留一 T-RH 格子交叉验证（27 格分组 CV）"""
    print("\n" + "=" * 60)
    print("A3. 留一 T-RH 格子交叉验证（27格分组CV）")
    print("=" * 60)
    cells = sorted(df_all.groupby(['温度', '湿度']).groups.items())
    y_all = np.zeros(len(df_all))
    pred_all = np.zeros(len(df_all))
    per_cell = []
    for (t, rh), idx in cells:
        test_idx = np.array(list(idx))
        train = df_all.drop(index=test_idx)
        test = df_all.loc[test_idx]
        m, _, _, _ = fit_equation(train)
        Xte, _ = build_X(test)
        Xte_sm = sm.add_constant(Xte, has_constant='add')
        yte = test[TARGET].values.astype(float)
        pred = m.predict(Xte_sm)
        y_all[test_idx] = yte
        pred_all[test_idx] = pred
        rmse_c = float(np.sqrt(np.mean((yte - pred) ** 2)))
        per_cell.append({'T': float(t), 'RH': float(rh), 'n': int(len(test_idx)), 'rmse': rmse_c})

    r2, rmse, mae, mape = calc_metrics(y_all, pred_all)
    print(f"  27格留出汇总: R²={r2:.4f}, RMSE={rmse:.6f}, MAE={mae:.6f}, MAPE={mape:.2f}%")
    cell_rmse = np.array([c['rmse'] for c in per_cell])
    print(f"  单格RMSE: 中位数={np.median(cell_rmse):.6f}, 最大={cell_rmse.max():.6f}")
    worst = per_cell[int(np.argmax(cell_rmse))]
    print(f"  最差格子: T={worst['T']:.0f}℃, RH={worst['RH']:.0f}%, RMSE={worst['rmse']:.6f}")
    return {
        'aggregate': {'r2': float(r2), 'rmse': float(rmse), 'mae': float(mae), 'mape': float(mape)},
        'per_cell': per_cell,
    }


def intact_validation(df_all, df_raw):
    """A4: 完好工况 1542 条原始 vs 27 条均值；格内留出验证"""
    print("\n" + "=" * 60)
    print("A4. 完好工况验证（原始测量 vs 聚合均值）")
    print("=" * 60)

    def fit_baseline(T, RH, y):
        X = np.column_stack([T, T ** 2, RH ** 2])
        return sm.OLS(y, sm.add_constant(X)).fit()

    intact = df_all[df_all['数据来源'] == '完好']
    T_m = intact['温度'].values.astype(float)
    RH_m = intact['湿度'].values.astype(float)
    y_m = intact[TARGET].values.astype(float)

    T_r = df_raw['温度'].values.astype(float)
    RH_r = df_raw['湿度'].values.astype(float)
    y_r = df_raw['导热系数'].values.astype(float)

    m_mean = fit_baseline(T_m, RH_m, y_m)
    m_raw = fit_baseline(T_r, RH_r, y_r)

    Xm = sm.add_constant(np.column_stack([T_m, T_m ** 2, RH_m ** 2]), has_constant='add')
    pred_mean_fit = m_mean.predict(Xm)
    pred_raw_fit = m_raw.predict(Xm)

    r2_mean, rmse_mean, _, mape_mean = calc_metrics(y_m, pred_mean_fit)
    r2_raw, rmse_raw, _, mape_raw = calc_metrics(y_m, pred_raw_fit)
    print(f"  均值口径基准层:  拟合RMSE(对27均值)={rmse_mean:.6f}, MAPE={mape_mean:.2f}%")
    print(f"  原始口径基准层:  预测RMSE(对27均值)={rmse_raw:.6f}, MAPE={mape_raw:.2f}%")
    print(f"  原始口径系数: const={m_raw.params[0]:.8f}, T={m_raw.params[1]:+.8f}, "
          f"T²={m_raw.params[2]:+.3e}, RH²={m_raw.params[3]:+.3e}")
    print(f"  均值口径系数: const={m_mean.params[0]:.8f}, T={m_mean.params[1]:+.8f}, "
          f"T²={m_mean.params[2]:+.3e}, RH²={m_mean.params[3]:+.3e}")

    # 格内留出：用其他格子的原始数据拟合，预测本格
    cells = sorted(df_raw.groupby(['温度', '湿度']).groups.items())
    y_hold = []
    pred_hold = []
    per_cell = []
    for (t, rh), idx in cells:
        test_idx = np.array(list(idx))
        train = df_raw.drop(index=test_idx)
        mb = fit_baseline(train['温度'].values.astype(float),
                          train['湿度'].values.astype(float),
                          train['导热系数'].values.astype(float))
        yte = y_r[test_idx]
        Xte = sm.add_constant(
            np.column_stack([T_r[test_idx], T_r[test_idx] ** 2, RH_r[test_idx] ** 2]),
            has_constant='add')
        p = mb.predict(Xte)
        y_hold.append(yte)
        pred_hold.append(p)
        per_cell.append({'T': float(t), 'RH': float(rh), 'n': int(len(test_idx)),
                         'rmse': float(np.sqrt(np.mean((yte - p) ** 2)))})
    y_hold = np.concatenate(y_hold)
    pred_hold = np.concatenate(pred_hold)
    r2_h, rmse_h, mae_h, mape_h = calc_metrics(y_hold, pred_hold)
    print(f"\n  原始数据格内留出(27格): R²={r2_h:.4f}, RMSE={rmse_h:.6f}, MAPE={mape_h:.2f}%")

    # 完整11项方程在完好均值上的表现（对照）
    full_model, _, _, _ = fit_equation(df_all)
    X_full_intact, _ = build_X(intact)
    pred_full_intact = full_model.predict(sm.add_constant(X_full_intact, has_constant='add'))
    rmse_full_intact = float(np.sqrt(np.mean((y_m - pred_full_intact) ** 2)))
    bias_full_intact = float(np.mean(y_m - pred_full_intact))
    print(f"  完整11项方程对完好均值: RMSE={rmse_full_intact:.6f}, 平均残差={bias_full_intact:+.6f}")

    return {
        'baseline_coefs': {
            'mean_fit': {k: float(v) for k, v in zip(['const', 'T', 'T²', 'RH²'], m_mean.params)},
            'raw_fit': {k: float(v) for k, v in zip(['const', 'T', 'T²', 'RH²'], m_raw.params)},
        },
        'mean_fit_on_means': {'rmse': float(rmse_mean), 'mape': float(mape_mean), 'r2': float(r2_mean)},
        'raw_fit_on_means': {'rmse': float(rmse_raw), 'mape': float(mape_raw), 'r2': float(r2_raw)},
        'raw_cell_holdout': {'r2': float(r2_h), 'rmse': float(rmse_h), 'mae': float(mae_h), 'mape': float(mape_h)},
        'full_equation_on_intact': {'rmse': rmse_full_intact, 'bias': bias_full_intact},
        'per_cell_raw': per_cell,
    }


# ============================================================
# B. 不确定度传播
# ============================================================
def bootstrap_coefficients(df_all, n_boot=300):
    """B1: 分层 bootstrap 系数分布 + 95% CI + 符号稳定性"""
    print("\n" + "=" * 60)
    print(f"B1. 分层bootstrap系数分布（{n_boot}次）")
    print("=" * 60)
    scenarios = sorted(df_all['场景'].unique())
    strata = {s: df_all[df_all['场景'] == s] for s in scenarios}
    rng = np.random.RandomState(RANDOM_STATE)

    full_model, _, _, _ = fit_equation(df_all)
    names = ['const'] + EQ_NAMES
    coef_matrix = []
    for it in range(n_boot):
        parts = []
        for s in scenarios:
            g = strata[s]
            idx = rng.choice(len(g), size=len(g), replace=True)
            parts.append(g.iloc[idx].reset_index(drop=True))
        df_boot = pd.concat(parts, ignore_index=True)
        m, _, _, _ = fit_equation(df_boot)
        coef_matrix.append(m.params.astype(float))
    coef_matrix = np.array(coef_matrix)

    full_params = full_model.params.astype(float)
    ci = {}
    for i, n in enumerate(names):
        lo, hi = np.percentile(coef_matrix[:, i], [2.5, 97.5])
        sign_stable = float(np.mean(np.sign(coef_matrix[:, i]) == np.sign(full_params[i]))) * 100
        ci[n] = {
            'full_fit': float(full_params[i]),
            'boot_mean': float(np.mean(coef_matrix[:, i])),
            'ci95': [float(lo), float(hi)],
            'sign_stability_pct': sign_stable,
        }
        print(f"  {n:14s}: 全量={full_params[i]:+.8f}  "
              f"bootstrap95%CI=[{lo:+.8f},{hi:+.8f}] 符号稳定={sign_stable:.0f}%")
    return {'n_bootstrap': n_boot, 'coef_ci': ci}


def prediction_bands(df_all, boot_results=None):
    """B2: 关键边际曲线的解析95%置信/预测区间 + bootstrap预测带"""
    print("\n" + "=" * 60)
    print("B2. 关键边际曲线预测区间")
    print("=" * 60)
    full_model, X_all, X_sm_all, y_all = fit_equation(df_all)

    grids = {
        '温度': np.linspace(5, 45, 50),
        '湿度': np.linspace(30, 90, 50),
        '冻融次数': np.linspace(0, 24, 50),
        '裂缝长度': np.linspace(0, 10, 50),
        '含水率': np.linspace(0, 56, 50),
    }
    fixed = {'温度': 25.0, '湿度': 60.0, '冻融次数': 0.0, '裂缝长度': 0.0, '含水率': 0.0}

    rng = np.random.RandomState(RANDOM_STATE)
    scenarios = sorted(df_all['场景'].unique())
    strata = {s: df_all[df_all['场景'] == s] for s in scenarios}
    boot_pred = {feat: [] for feat in grids}
    for it in range(300):
        parts = []
        for s in scenarios:
            g = strata[s]
            idx = rng.choice(len(g), size=len(g), replace=True)
            parts.append(g.iloc[idx].reset_index(drop=True))
        df_boot = pd.concat(parts, ignore_index=True)
        m, _, _, _ = fit_equation(df_boot)
        c = coefs_from_model(m)
        for feat, gv in grids.items():
            kw = dict(fixed)
            kw[feat] = gv
            boot_pred[feat].append(eq_predict_vectorized(
                kw['温度'], kw['湿度'], kw['冻融次数'], kw['裂缝长度'], kw['含水率'], c))

    band = {}
    fig, axes = plt.subplots(2, 3, figsize=(7.48, 4.57))
    axes = axes.flatten()
    for idx, (feat, gv) in enumerate(grids.items()):
        ax = axes[idx]
        bp = np.array(boot_pred[feat])  # 300 x 50
        lo_b, hi_b = np.percentile(bp, [5, 95], axis=0)

        kw = dict(fixed)
        kw[feat] = gv
        T = np.broadcast_to(kw['温度'], gv.shape).astype(float)
        RH = np.broadcast_to(kw['湿度'], gv.shape).astype(float)
        FT = np.broadcast_to(kw['冻融次数'], gv.shape).astype(float)
        CR = np.broadcast_to(kw['裂缝长度'], gv.shape).astype(float)
        WA = np.broadcast_to(kw['含水率'], gv.shape).astype(float)
        sFT, sCR = np.sqrt(np.maximum(FT, 0)), np.sqrt(np.maximum(CR, 0))
        Xg = np.column_stack([T, T ** 2, RH ** 2, sFT, sCR, WA, WA ** 2,
                              T * sCR, RH * WA, sFT * sCR, RH * sCR])
        Xg_sm = sm.add_constant(Xg, has_constant='add')
        pred_obj = full_model.get_prediction(Xg_sm)
        mean_pred = pred_obj.predicted_mean
        ci = pred_obj.conf_int(alpha=0.05)          # 均值置信区间
        pi = pred_obj.conf_int(alpha=0.05, obs=True)  # 新观测预测区间

        ax.plot(gv, mean_pred, 'b-', linewidth=2.5, label='方程均值')
        ax.fill_between(gv, ci[:, 0], ci[:, 1], color='blue', alpha=0.15, label='95% 置信区间')
        ax.fill_between(gv, pi[:, 0], pi[:, 1], color='gray', alpha=0.25, label='95% 预测区间')
        ax.fill_between(gv, lo_b, hi_b, color='orange', alpha=0.20, label='bootstrap 5-95%')
        ax.set_xlabel(feat, fontsize=9.0)
        ax.set_ylabel('导热系数 (W/(m·K))', fontsize=8.2)
        ax.set_title(f'{feat} 边际曲线与不确定度', fontsize=9.8, fontweight='bold')
        ax.legend(fontsize=6.8, loc='best')
        ax.grid(True, alpha=0.3)

        band[feat] = {
            'grid': gv.tolist(),
            'mean': mean_pred.tolist(),
            'ci_lo': ci[:, 0].tolist(), 'ci_hi': ci[:, 1].tolist(),
            'pi_lo': pi[:, 0].tolist(), 'pi_hi': pi[:, 1].tolist(),
            'boot_lo': lo_b.tolist(), 'boot_hi': hi_b.tolist(),
        }
        print(f"  [{feat}] 参考点均值={mean_pred[len(gv)//2]:.6f}, "
              f"95%CI宽度={np.mean(ci[:,1]-ci[:,0]):.2e}, "
              f"95%PI宽度={np.mean(pi[:,1]-pi[:,0]):.2e}")

    axes[5].set_visible(False)
    plt.suptitle('方程关键边际曲线与不确定度（解析区间 + bootstrap带）', fontsize=10.0, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(RESULTS_FOLDER, 'Fig_v4_Prediction_Bands.png')
    plt.savefig(fig_path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {os.path.basename(fig_path)}")
    return band


# ============================================================
# D. Sobol 方差分解（显式方程，设计域均匀抽样）
# ============================================================
def sobol_decomposition(coefs, n_samples=500000):
    """D1+D2: Sobol一阶/总效应 + 三层贡献统计 + 每项效应量"""
    print("\n" + "=" * 60)
    print(f"Sobol方差分解（设计域均匀独立抽样, 多种子平均, 总N={n_samples}）")
    print("=" * 60)
    ranges = np.array([[5, 45], [30, 90], [0, 24], [0, 10], [0, 56]], dtype=float)

    # 独立随机抽样（A/B 必须相互独立；低差异序列的行配对会导致一阶效应估计失效）
    n_seeds = 5
    sub_n = n_samples // n_seeds
    sum_s1 = np.zeros(5)
    sum_st = np.zeros(5)
    A_pool, B_pool, fA_pool, fB_pool = [], [], [], []
    for seed in range(RANDOM_STATE, RANDOM_STATE + n_seeds):
        rng = np.random.default_rng(seed)
        A = ranges[:, 0] + rng.random((sub_n, 5)) * (ranges[:, 1] - ranges[:, 0])
        B = ranges[:, 0] + rng.random((sub_n, 5)) * (ranges[:, 1] - ranges[:, 0])
        fA = eq_predict_vectorized(A[:, 0], A[:, 1], A[:, 2], A[:, 3], A[:, 4], coefs)
        fB = eq_predict_vectorized(B[:, 0], B[:, 1], B[:, 2], B[:, 3], B[:, 4], coefs)
        A_pool.append(A)
        fA_pool.append(fA)
        fB_pool.append(fB)
        for i in range(5):
            C = A.copy()
            C[:, i] = B[:, i]
            fC = eq_predict_vectorized(C[:, 0], C[:, 1], C[:, 2], C[:, 3], C[:, 4], coefs)
            sum_s1[i] += np.mean(fB * (fC - fA))
            sum_st[i] += np.mean((fA - fC) ** 2)

    fA_all = np.concatenate(fA_pool)
    fB_all = np.concatenate(fB_pool)
    A_all = np.concatenate(A_pool)
    f_all = np.concatenate([fA_all, fB_all])
    varY = float(np.var(f_all, ddof=1))
    mu = float(np.mean(f_all))

    names = ['温度T', '湿度RH', '冻融FT', '裂缝Crack', '含水率Water']
    s1, st = [], []
    for i in range(5):
        s1_i = float(sum_s1[i] / n_seeds / varY)
        st_i = float(sum_st[i] / n_seeds / (2 * varY))
        s1.append(s1_i)
        st.append(st_i)
        # 交叉验证：直接分箱估计 Var(E[Y|X_i])
        order = np.argsort(A_all[:, i])
        k = 50
        vals = np.array_split(fA_all[order], k)
        cond_mean = np.array([np.mean(v) for v in vals])
        s1_bin = float(np.var(cond_mean, ddof=1) / varY)
        print(f"  {names[i]:12s}: S1={s1_i:.4f}  ST={st_i:.4f}  "
              f"交互占比={st_i-s1_i:.4f}  [分箱校验S1={s1_bin:.4f}]")

    interaction_share_raw = 1.0 - sum(s1)
    interaction_share = max(0.0, interaction_share_raw)
    print(f"\n  总方差 Var(λ)={varY:.3e}")
    print(f"  交互(高阶)贡献 = 1-ΣS1 = {interaction_share:.4f}"
          f"{'（MC噪声范围内≈0）' if interaction_share_raw < 0 else ''}")

    # 三层贡献统计
    T, RH, FT, CR, WA = A[:, 0], A[:, 1], A[:, 2], A[:, 3], A[:, 4]
    sFT, sCR = np.sqrt(FT), np.sqrt(CR)
    c = coefs
    base = c['const'] + c['T'] * T + c['T²'] * T ** 2 + c['RH²'] * RH ** 2
    deg = c['√FT'] * sFT + c['√Crack'] * sCR + c['Water'] * WA + c['Water²'] * WA ** 2
    coup = (c['T×√Crack'] * T * sCR + c['RH×Water'] * RH * WA
            + c['√FT×√Crack'] * sFT * sCR + c['RH×√Crack'] * RH * sCR)
    total = base + deg + coup
    layer_stats = {
        'baseline': {'mean': float(np.mean(base)), 'std': float(np.std(base, ddof=1)),
                     'min': float(base.min()), 'max': float(base.max())},
        'degradation': {'mean': float(np.mean(deg)), 'std': float(np.std(deg, ddof=1)),
                        'min': float(deg.min()), 'max': float(deg.max())},
        'coupling': {'mean': float(np.mean(coup)), 'std': float(np.std(coup, ddof=1)),
                     'min': float(coup.min()), 'max': float(coup.max())},
        'total': {'mean': float(np.mean(total)), 'std': float(np.std(total, ddof=1)),
                  'min': float(total.min()), 'max': float(total.max())},
    }
    print("\n  设计域内三层贡献统计:")
    for k, v in layer_stats.items():
        print(f"    {k:12s}: 均值={v['mean']:.6f} 标准差={v['std']:.6f} "
              f"范围=[{v['min']:.6f},{v['max']:.6f}]")

    # 每项效应量（设计域内最大|贡献|与标准差）
    terms = {
        'T': c['T'] * T, 'T²': c['T²'] * T ** 2, 'RH²': c['RH²'] * RH ** 2,
        '√FT': c['√FT'] * sFT, '√Crack': c['√Crack'] * sCR,
        'Water': c['Water'] * WA, 'Water²': c['Water²'] * WA ** 2,
        'T×√Crack': c['T×√Crack'] * T * sCR, 'RH×Water': c['RH×Water'] * RH * WA,
        '√FT×√Crack': c['√FT×√Crack'] * sFT * sCR, 'RH×√Crack': c['RH×√Crack'] * RH * sCR,
    }
    effect = {}
    print("\n  每项效应量（设计域）:")
    for n, v in terms.items():
        effect[n] = {'max_abs': float(np.max(np.abs(v))), 'std': float(np.std(v, ddof=1))}
        print(f"    {n:12s}: 最大|贡献|={np.max(np.abs(v)):.3e}  std={np.std(v, ddof=1):.3e}")

    return {
        'var_total': varY, 'mean': mu,
        's1': {n: float(v) for n, v in zip(names, s1)},
        'st': {n: float(v) for n, v in zip(names, st)},
        'interaction_share': float(interaction_share),
        'layer_stats': layer_stats,
        'term_effect': effect,
    }


def plot_loo(results_a1, cell_out):
    """A1+A3 汇总图"""
    labels, rmse_full, rmse_loo = [], [], []
    for r in results_a1:
        labels.append(f"{r['factor']}={r['level']:g}")
        rmse_full.append(r['rmse_full'])
        rmse_loo.append(r['rmse_loo'])
    labels.append('T-RH格子留出')
    rmse_full.append(cell_out['aggregate']['rmse'])
    rmse_loo.append(cell_out['aggregate']['rmse'])

    fig, ax = plt.subplots(figsize=(7.48, 3.45))
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, rmse_full, w, label='全量拟合（同测试行）', color='#95a5a6', edgecolor='white')
    ax.bar(x + w / 2, rmse_loo, w, label='留出训练（外推）', color='#e74c3c', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.8, rotation=20, ha='right')
    ax.set_ylabel('RMSE (W/(m·K))', fontsize=9.0)
    ax.set_title('设计感知验证：留一水平/留一格子外推误差 vs 全量拟合误差', fontsize=9.8)
    ax.legend(fontsize=7.5)
    ax.grid(axis='y', alpha=0.3)
    for i in range(len(labels)):
        ax.text(i - w / 2, rmse_full[i] + 0.00001, f'{rmse_full[i]:.4f}', ha='center', fontsize=6.5)
        ax.text(i + w / 2, rmse_loo[i] + 0.00001, f'{rmse_loo[i]:.4f}', ha='center', fontsize=6.5)
    plt.tight_layout()
    path = os.path.join(RESULTS_FOLDER, 'Fig_v4_DesignAware_CV.png')
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {os.path.basename(path)}")


def plot_sobol(sobol):
    names = list(sobol['s1'].keys())
    s1 = [sobol['s1'][n] for n in names]
    st = [sobol['st'][n] for n in names]
    fig, ax = plt.subplots(figsize=(7.48, 4.49))
    y = np.arange(len(names))
    ax.barh(y + 0.18, s1, height=0.36, label='一阶效应 S1', color='#3498db', edgecolor='white')
    ax.barh(y - 0.18, st, height=0.36, label='总效应 ST', color='#e74c3c', edgecolor='white')
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8.2)
    ax.invert_yaxis()
    ax.set_xlabel('方差占比', fontsize=9.0)
    ax.set_title(f'Sobol方差分解（设计域均匀分布, 交互贡献={sobol["interaction_share"]:.2f}）', fontsize=9.8)
    ax.legend(fontsize=7.5)
    ax.grid(axis='x', alpha=0.3)
    for i, (a, b) in enumerate(zip(s1, st)):
        ax.text(a + 0.01, i + 0.18, f'{a:.3f}', va='center', fontsize=6.8)
        ax.text(b + 0.01, i - 0.18, f'{b:.3f}', va='center', fontsize=6.8)
    plt.tight_layout()
    path = os.path.join(RESULTS_FOLDER, 'Fig_v4_Sobol.png')
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {os.path.basename(path)}")


def plot_intact(intact_res, df_all):
    intact = df_all[df_all['数据来源'] == '完好']
    T_m = intact['温度'].values.astype(float)
    RH_m = intact['湿度'].values.astype(float)
    y_m = intact[TARGET].values.astype(float)
    full_model, _, _, _ = fit_equation(df_all)
    X_full_intact, _ = build_X(intact)
    pred_full = full_model.predict(sm.add_constant(X_full_intact, has_constant='add'))

    def fit_baseline(T, RH, y):
        X = np.column_stack([T, T ** 2, RH ** 2])
        return sm.OLS(y, sm.add_constant(X)).fit()

    m_raw = fit_baseline(intact['温度'].values.astype(float),
                         intact['湿度'].values.astype(float), y_m)
    pred_baseline = m_raw.predict(
        sm.add_constant(np.column_stack([T_m, T_m ** 2, RH_m ** 2]), has_constant='add'))

    fig, ax = plt.subplots(figsize=(5.51, 5.51))
    lim = [y_m.min() * 0.97, y_m.max() * 1.03]
    ax.scatter(y_m, pred_full, s=60, c='#e74c3c', alpha=0.75, edgecolors='white',
               label=f'完整11项方程 (RMSE={np.sqrt(np.mean((y_m-pred_full)**2)):.5f})')
    ax.scatter(y_m, pred_baseline, s=40, marker='s', c='#3498db', alpha=0.6, edgecolors='white',
               label=f'完好基准层(均值口径) (RMSE={np.sqrt(np.mean((y_m-pred_baseline)**2)):.5f})')
    ax.plot(lim, lim, 'k--', linewidth=1.2, label='y=x')
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel('完好工况实测均值 (W/(m·K))', fontsize=9.0)
    ax.set_ylabel('方程预测 (W/(m·K))', fontsize=9.0)
    ax.set_title('完好工况27点：完整方程 vs 基准层', fontsize=9.8)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_FOLDER, 'Fig_v4_Intact_Validation.png')
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {os.path.basename(path)}")


def main():
    ensure_folder(RESULTS_FOLDER)
    print("=" * 60)
    print("可解释性方程特征分析优化 v4 补充验证")
    print("=" * 60)

    df_all, df_raw = load_data()
    print(f"数据: 劣化{len(df_all)-27}条 + 完好均值27条 = {len(df_all)}条; "
          f"完好原始测量{len(df_raw)}条")

    # ---- A. 设计感知验证 ----
    a1 = leave_one_level_out(df_all)
    a2 = coupling_identifiability(df_all)
    a3 = leave_one_cell_out(df_all)
    a4 = intact_validation(df_all, df_raw)

    # ---- B. 不确定度传播 ----
    b1 = bootstrap_coefficients(df_all, n_boot=300)
    b2 = prediction_bands(df_all)

    # ---- D. Sobol ----
    full_model, _, _, _ = fit_equation(df_all)
    coefs = coefs_from_model(full_model)
    d1 = sobol_decomposition(coefs, n_samples=500000)

    # 绘图
    print("\n生成图表...")
    plot_loo(a1, a3)
    plot_sobol(d1)
    plot_intact(a4, df_all)

    # 保存 JSON
    out = {
        'A1_leave_one_level': a1,
        'A2_coupling_identifiability': a2,
        'A3_leave_one_cell': a3,
        'A4_intact_validation': a4,
        'B1_bootstrap_coef': b1,
        'B2_prediction_bands': {k: {kk: vv for kk, vv in v.items() if kk != 'grid'}
                                for k, v in b2.items()},
        'D_sobol': d1,
    }
    json_path = os.path.join(RESULTS_FOLDER, 'v4_supplement.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {json_path}")


if __name__ == '__main__':
    main()
