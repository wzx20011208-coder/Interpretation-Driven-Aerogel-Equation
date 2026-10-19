# -*- coding: utf-8 -*-
"""
v4 补充：7.4 升级 —— 跨场景外推验证（留一场景交叉验证）
============================================================
将 7.4 由"样本内分场景拟合对比"升级为真正的跨场景外推：
对五类工况（完好、纯冻融、纯裂缝、纯渗水、复合）逐一整体留出，
在其余四类工况上重新拟合去耦合方程（7 项）与完整方程（11 项），
预测被留出工况的全部样本，比较两模型的留出 RMSE/MAE/MAPE 与改善率。

关键处理：
- 训练集因场景留出可能出现零方差列（如留出复合工况时 √FT×√Crack 恒为 0），
  拟合时自动剔除零方差列并在预测时置零，保证可辨识性；
- 每个样本恰好被预测一次（其所属场景被留出时），可汇总整体留出误差；
- 完好（27 条均值）作为极端"零劣化"角点，其留出构成最难的外推场景。

输出：
- 可解释性方程特征分析优化_结果/v4_supplement.json 新增 A5_scenario_extrapolation
- 可解释性方程特征分析优化_结果/Fig_v4_Scenario_Extrapolation.png
"""

import os
import json

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
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')

FEATURES = ['温度', '湿度', '冻融次数', '裂缝长度', '含水率']
TARGET = '导热系数'

# v3 最终方程 11 项特征（不含截距）；前 7 项为基准层+劣化主效应
EQ_NAMES = ['T', 'T²', 'RH²', '√FT', '√Crack', 'Water', 'Water²',
            'T×√Crack', 'RH×Water', '√FT×√Crack', 'RH×√Crack']
MAIN_NAMES = EQ_NAMES[:7]

SCENARIOS = ['完好', '纯冻融', '纯裂缝', '纯渗水', '复合']


def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


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
    return df_all


def build_X(df, use_coupling=True):
    T = df['温度'].values.astype(float)
    RH = df['湿度'].values.astype(float)
    FT = df['冻融次数'].values.astype(float)
    Crack = df['裂缝长度'].values.astype(float)
    Water = df['含水率'].values.astype(float)
    sFT = np.sqrt(np.maximum(FT, 0.0))
    sCR = np.sqrt(np.maximum(Crack, 0.0))
    cols = [T, T ** 2, RH ** 2, sFT, sCR, Water, Water ** 2]
    names = list(MAIN_NAMES)
    if use_coupling:
        cols += [T * sCR, RH * Water, sFT * sCR, RH * sCR]
        names = list(EQ_NAMES)
    return np.column_stack(cols), names


def fit_predict(train, test, use_coupling):
    """在训练集上最小二乘拟合（自动剔除零方差列），预测测试集。"""
    Xtr, names = build_X(train, use_coupling)
    ytr = train[TARGET].values.astype(float)
    std = Xtr.std(axis=0)
    keep = std > 1e-12
    Xk = Xtr[:, keep]
    names_k = [n for n, k in zip(names, keep) if k]
    A = np.column_stack([np.ones(len(Xk)), Xk])
    beta, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    Xte, _ = build_X(test, use_coupling)
    Ate = np.column_stack([np.ones(len(Xte)), Xte[:, keep]])
    pred = Ate @ beta
    return pred, names_k, keep


def metrics(y, pred):
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
    mae = float(np.mean(np.abs(y - pred)))
    mape = float(np.mean(np.abs((y - pred) / y)) * 100)
    return r2, rmse, mae, mape


def leave_one_scenario_out(df_all):
    """A5: 留一场景外推验证（每个场景整体留出，其余场景训练）"""
    print("\n" + "=" * 60)
    print("A5. 跨场景外推验证（留一场景交叉验证）")
    print("=" * 60)

    pooled = {'y': [], 'pred_nocoup': [], 'pred_full': []}
    results = {}
    for s in SCENARIOS:
        test = df_all[df_all['场景'] == s]
        train = df_all[df_all['场景'] != s]
        y = test[TARGET].values.astype(float)

        pred_n, names_n, _ = fit_predict(train, test, use_coupling=False)
        pred_f, names_f, keep_f = fit_predict(train, test, use_coupling=True)
        dropped = [n for n, k in zip(EQ_NAMES, keep_f) if not k]

        r2_n, rmse_n, mae_n, mape_n = metrics(y, pred_n)
        r2_f, rmse_f, mae_f, mape_f = metrics(y, pred_f)
        improve = (rmse_n - rmse_f) / rmse_n * 100 if rmse_n > 0 else float('nan')

        results[s] = {
            'n': int(len(test)),
            'r2_nocoup': r2_n, 'rmse_nocoup': rmse_n,
            'mae_nocoup': mae_n, 'mape_nocoup': mape_n,
            'r2_full': r2_f, 'rmse_full': rmse_f,
            'mae_full': mae_f, 'mape_full': mape_f,
            'improve_pct': improve,
            'dropped_terms': dropped,
        }
        pooled['y'].append(y)
        pooled['pred_nocoup'].append(pred_n)
        pooled['pred_full'].append(pred_f)
        print(f"  留出 {s:<4s} n={len(test):>3d}  去耦合RMSE={rmse_n:.6f} "
              f"完整RMSE={rmse_f:.6f} 改善={improve:+.1f}%"
              + (f"  [剔除: {','.join(dropped)}]" if dropped else ""))

    # 整体留出误差（每个样本恰好被预测一次）
    y_all = np.concatenate(pooled['y'])
    pn = np.concatenate(pooled['pred_nocoup'])
    pf = np.concatenate(pooled['pred_full'])
    r2n, rmn, man, mpn = metrics(y_all, pn)
    r2f, rmf, maf, mpf = metrics(y_all, pf)
    pooled_metrics = {
        'r2_nocoup': r2n, 'rmse_nocoup': rmn, 'mae_nocoup': man, 'mape_nocoup': mpn,
        'r2_full': r2f, 'rmse_full': rmf, 'mae_full': maf, 'mape_full': mpf,
        'improve_pct': (rmn - rmf) / rmn * 100,
    }
    print(f"  整体留出(351条): 去耦合RMSE={rmn:.6f} 完整RMSE={rmf:.6f} "
          f"改善={pooled_metrics['improve_pct']:+.1f}%")

    # ---- 图：分场景 RMSE 对比 + 复合场景预测散点 ----
    fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.21))
    ax = axes[0]
    x = np.arange(len(SCENARIOS))
    w = 0.35
    ax.bar(x - w / 2, [results[s]['rmse_nocoup'] for s in SCENARIOS], w,
           label='去耦合方程（7项）', color='#95a5a6', alpha=0.85)
    ax.bar(x + w / 2, [results[s]['rmse_full'] for s in SCENARIOS], w,
           label='完整方程（11项）', color='#e74c3c', alpha=0.85)
    for i, s in enumerate(SCENARIOS):
        ax.text(i, max(results[s]['rmse_nocoup'], results[s]['rmse_full']) * 1.03,
                f"{results[s]['improve_pct']:+.0f}%", ha='center', fontsize=6.8,
                color='#2c3e50')
    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIOS, fontsize=8.2)
    ax.set_ylabel('留出 RMSE (W/(m·K))', fontsize=9.0)
    ax.set_title('留一场景外推：去耦合 vs 完整方程', fontsize=9.8)
    ax.legend(fontsize=6.8)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    mask = (df_all['场景'] == '复合').values
    y_comp = df_all.loc[mask, TARGET].values.astype(float)
    idx = SCENARIOS.index('复合')
    pred_n_c = pooled['pred_nocoup'][idx]
    pred_f_c = pooled['pred_full'][idx]
    ax.scatter(y_comp, pred_n_c, alpha=0.45, s=22, c='#95a5a6',
               label=f'去耦合 (RMSE={results["复合"]["rmse_nocoup"]:.6f})')
    ax.scatter(y_comp, pred_f_c, alpha=0.55, s=22, c='#e74c3c',
               label=f'完整方程 (RMSE={results["复合"]["rmse_full"]:.6f})')
    lo, hi = y_comp.min() * 0.97, y_comp.max() * 1.03
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1, label='y=x')
    ax.set_xlabel('实测导热系数 (W/(m·K))', fontsize=9.0)
    ax.set_ylabel('方程留出预测 (W/(m·K))', fontsize=9.0)
    ax.set_title('复合工况（训练时完全未见）：留出预测', fontsize=9.8)
    ax.legend(fontsize=6.8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_FOLDER, 'Fig_v4_Scenario_Extrapolation.png')
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {os.path.basename(path)}")

    return {'per_scenario': results, 'pooled': pooled_metrics}


def main():
    ensure_folder(RESULTS_FOLDER)
    df_all = load_data()
    print(f"数据: {len(df_all)} 条；场景构成: "
          f"{dict(df_all['场景'].value_counts())}")
    a5 = leave_one_scenario_out(df_all)

    json_path = os.path.join(RESULTS_FOLDER, 'v4_supplement.json')
    if os.path.exists(json_path):
        with open(json_path, encoding='utf-8') as f:
            out = json.load(f)
    else:
        out = {}
    out['A5_scenario_extrapolation'] = a5
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"结果已写入: {json_path} [A5_scenario_extrapolation]")


if __name__ == '__main__':
    main()
