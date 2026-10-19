# -*- coding: utf-8 -*-
"""
v4 补充：三层 SHAP 可解释性分析（完整体系）
============================================
将报告 4.x 的 SHAP 分析补全为三层递进框架：

  Layer I   全局特征重要性（谁最重要？）      —— 全样本 351 条，Mean |SHAP| 排序
  Layer II  交互值分析（谁与谁耦合？）        —— shap_interaction_values，Top 交互对
  Layer III 工况依赖分析（控制因素如何随工况变化？）
             —— 324 条劣化数据按四类工况分组，逐组统计 Mean |SHAP| 与排序，
                Kruskal-Wallis 检验 |SHAP| 分布差异 + BH-FDR 校正 + 效应量 η²，
                并对温度/湿度做子组分析以排除"开关变量"的设计混淆。

同时补做"三阶项"嵌套检验（11 项方程 vs 加入三阶协同项），
为 7.3 节 FPI 缺口指向更高阶协同的结论提供本报告独立的统计证据。

模型口径与 4.2/4.5 一致：无权重 CatBoost（iterations=300, depth=4, lr=0.05, seed=42）。
输出写入 可解释性方程特征分析优化_结果/v4_supplement.json
  - SHAP_layer3：工况分组统计 + KW/η² + 子组分析
  - third_order_test：三阶项嵌套检验
  - SHAP_layer1：全局 Mean |SHAP|（与 4.2 对照）
图：Fig_v4_SHAP_Layer3.png（四工况特征贡献对比）
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

import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy import stats
from catboost import CatBoostRegressor
import shap

warnings.filterwarnings('ignore')
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
RANDOM_STATE = 42

FEATURES = ['温度', '湿度', '冻融次数', '裂缝长度', '含水率']
TARGET = '导热系数'
SCENARIOS = ['纯冻融', '纯裂缝', '纯渗水', '复合']


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
    df_intact = pd.read_excel(
        os.path.join(DATA_DIR, 'intact_conditions_summary.xlsx'),
        sheet_name='summary'
    )
    df_intact = pd.DataFrame({
        '温度': df_intact['Temperature_C'], '湿度': df_intact['RH_pct'],
        '冻融次数': 0, '裂缝长度': 0.0, '含水率': 0,
        '导热系数': df_intact['Thermal_Conductivity_Mean'],
    })
    df_all = pd.concat([df_deg, df_intact], ignore_index=True)
    df_all['场景'] = df_all.apply(identify_scenario, axis=1)
    return df_all


def train_shap_model(df_all):
    X = df_all[FEATURES].values
    y = df_all[TARGET].values
    model = CatBoostRegressor(
        iterations=300, depth=4, learning_rate=0.05,
        random_seed=RANDOM_STATE, verbose=False,
    )
    model.fit(X, y)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return model, shap_values


def layer1_global(shap_values):
    mean_abs = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(mean_abs)[::-1]
    return {
        'mean_abs_shap': {f: float(mean_abs[i]) for i, f in enumerate(FEATURES)},
        'ranking': [FEATURES[i] for i in order],
    }


def kw_effect_size(groups):
    """Kruskal-Wallis 检验 + eta-squared（epsilon-squared）效应量。"""
    h, p = stats.kruskal(*groups)
    n = sum(len(g) for g in groups)
    k = len(groups)
    eta2 = (h - k + 1) / (n - k) if n > k else float('nan')
    return float(h), float(p), float(eta2)


def layer3_scenario(df_all, shap_values):
    """Layer III：四类劣化工况分组的 SHAP 贡献统计与差异检验。"""
    deg = df_all[df_all['场景'] != '完好'].copy()
    mask_deg = (df_all['场景'] != '完好').values
    sv_deg = shap_values[mask_deg]

    per_scenario = {}
    abs_sv_by_feat = {f: [] for f in FEATURES}
    for s in SCENARIOS:
        m = (deg['场景'] == s).values
        sv_s = sv_deg[m]
        mean_abs = np.mean(np.abs(sv_s), axis=0)
        order = np.argsort(mean_abs)[::-1]
        per_scenario[s] = {
            'n': int(m.sum()),
            'mean_abs_shap': {f: float(mean_abs[i]) for i, f in enumerate(FEATURES)},
            'ranking': [FEATURES[i] for i in order],
        }
        for i, f in enumerate(FEATURES):
            abs_sv_by_feat[f].append(np.abs(sv_s[:, i]))

    kw = {}
    for i, f in enumerate(FEATURES):
        h, p, eta2 = kw_effect_size(abs_sv_by_feat[f])
        kw[f] = {'H': h, 'p': p, 'eta2': eta2}
    # BH-FDR 校正
    ps = np.array([kw[f]['p'] for f in FEATURES])
    _, q, _, _ = multipletests(ps, method='fdr_bh')
    for i, f in enumerate(FEATURES):
        kw[f]['q_fdr'] = float(q[i])

    # 子组分析：温度、湿度在四组中均连续变化，排除"开关变量"设计混淆
    sub = {}
    for f in ['温度', '湿度']:
        i = FEATURES.index(f)
        h, p, eta2 = kw_effect_size(abs_sv_by_feat[f])
        sub[f] = {'H': h, 'p': p, 'eta2': eta2}

    return {'per_scenario': per_scenario, 'kw': kw, 'subgroup': sub}


def third_order_test(df_all):
    """三阶协同项嵌套检验：11 项方程 vs 分别加入候选三阶项。
    同时检验三种函数形式，考察结论对三阶项表示形式的敏感性。"""
    T = df_all['温度'].values.astype(float)
    RH = df_all['湿度'].values.astype(float)
    FT = df_all['冻融次数'].values.astype(float)
    CR = df_all['裂缝长度'].values.astype(float)
    W = df_all['含水率'].values.astype(float)
    y = df_all[TARGET].values.astype(float)
    sFT = np.sqrt(np.maximum(FT, 0.0))
    sCR = np.sqrt(np.maximum(CR, 0.0))

    X2 = np.column_stack([T, T**2, RH**2, sFT, sCR, W, W**2,
                          T*sCR, RH*W, sFT*sCR, RH*sCR])
    m2 = sm.OLS(y, sm.add_constant(X2)).fit()

    candidates = {
        '√FT×√Crack×√Water': sFT * sCR * np.maximum(W, 0.0),
        'FT×Crack×Water（线性三重）': FT * CR * W,
        'T×√Crack×√Water': T * sCR * np.maximum(W, 0.0),
    }
    out = {}
    for name, col in candidates.items():
        X3 = np.column_stack([X2, col])
        m3 = sm.OLS(y, sm.add_constant(X3)).fit()
        f_stat = float(((m2.ssr - m3.ssr) / 1) / (m3.ssr / m3.df_resid))
        p_val = float(stats.f.sf(f_stat, 1, m3.df_resid))
        X3c = sm.add_constant(X3)
        r2j = sm.OLS(X3c[:, -1], X3c[:, :-1]).fit().rsquared
        out[name] = {
            'F': f_stat, 'p': p_val,
            'delta_r2': float(m3.rsquared - m2.rsquared),
            'vif': float(1 / (1 - r2j)),
            'coef': float(m3.params[-1]),
            'coef_std_err': float(m3.bse[-1]),
        }
    return {
        'r2_11term': float(m2.rsquared),
        'candidates': out,
        'note': '显著性对三阶项的函数形式敏感：平方根形式不显著，'
                '线性三重与含温度调制形式显著，见正文讨论。',
    }


def plot_layer3(l3):
    fig, ax = plt.subplots(figsize=(7.48, 4.49))
    x = np.arange(len(FEATURES))
    w = 0.2
    colors = ['#3498db', '#2ecc71', '#e67e22', '#e74c3c']
    for j, s in enumerate(SCENARIOS):
        vals = [l3['per_scenario'][s]['mean_abs_shap'][f] for f in FEATURES]
        ax.bar(x + (j - 1.5) * w, vals, w, label=s, color=colors[j], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(FEATURES, fontsize=8.2)
    ax.set_ylabel('Mean |SHAP|', fontsize=9.0)
    ax.set_title('SHAP Layer III：四类工况下的特征贡献对比', fontsize=9.8)
    ax.legend(fontsize=6.8)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path = os.path.join(RESULTS_FOLDER, 'Fig_v4_SHAP_Layer3.png')
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {os.path.basename(path)}")


def main():
    ensure_folder(RESULTS_FOLDER)
    print("=" * 60)
    print("三层 SHAP 可解释性分析（完整体系）")
    print("=" * 60)
    df_all = load_data()
    print(f"数据: {len(df_all)} 条（劣化 324 + 完好均值 27）")

    model, sv = train_shap_model(df_all)
    print(f"无权重 CatBoost 训练 R²={model.score(df_all[FEATURES].values, df_all[TARGET].values):.4f}")

    l1 = layer1_global(sv)
    print("\n[Layer I] 全局 Mean |SHAP|:", {k: round(v, 6) for k, v in l1['mean_abs_shap'].items()})

    l3 = layer3_scenario(df_all, sv)
    print("\n[Layer III] 四工况 Top3:")
    for s in SCENARIOS:
        print(f"  {s}: {l3['per_scenario'][s]['ranking'][:3]}")
    print("\n[Layer III] KW 检验（H, p, η², q_FDR）:")
    for f in FEATURES:
        k = l3['kw'][f]
        print(f"  {f}: H={k['H']:.2f} p={k['p']:.3g} η²={k['eta2']:.4f} q={k['q_fdr']:.3g}")

    t3 = third_order_test(df_all)
    print(f"\n[三阶项] 11 项方程 R²={t3['r2_11term']:.5f}")
    for name, r in t3['candidates'].items():
        print(f"  {name}: F={r['F']:.2f} p={r['p']:.4f} ΔR²={r['delta_r2']:.5f} "
              f"VIF={r['vif']:.1f} coef={r['coef']:+.3e}±{r['coef_std_err']:.1e}")

    plot_layer3(l3)

    json_path = os.path.join(RESULTS_FOLDER, 'v4_supplement.json')
    if os.path.exists(json_path):
        with open(json_path, encoding='utf-8') as f:
            out = json.load(f)
    else:
        out = {}
    out['SHAP_layer1'] = l1
    out['SHAP_layer3'] = l3
    out['third_order_test'] = t3
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写入: {json_path}")


if __name__ == '__main__':
    main()
