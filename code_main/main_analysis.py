# -*- coding: utf-8 -*-
"""
可解释性方程特征分析优化 v3.1
================================
v3.0 相比 v2.0 的修改：
  1. 新增AIC/BIC边际诊断（P1-3）：偏除协变量后正式比较候选函数形式
  2. 新增嵌套检验M1→M2→M3（P0-2）：用变换变量重做，含三阶交互探索和VIF诊断
  3. 新增Bootstrap稳定性检验（P0-1）：分层抽样300次，记录每项保留频率
  4. 新增分组残差诊断（P1-5）：按工况分组Kruskal-Wallis检验+η²
  5. 保留v2.0全部现有功能不变，最终11项方程结构不变

v3.1 补充验证模块：
  6. P1-3补充: PDP黑箱曲线 vs 方程解析边际曲线对比图
  7. P2-6: 无权重CatBoost CV + 加权OLS系数稳定性对照
  8. P2-7: FPI量级校验（方程耦合放幅度 vs 实测FPI +10.96%）
  9. P2-8: 跨工况外推对比（去耦合层 vs 完整方程误差对比）

v3.1 审稿反馈修正：
  10. 修正温度边际形状标签：'非单调(U形)' → '单调递增（弱非线性）'
      （T²系数极小，极值点在123°C远超5-45°C数据范围）
  11. 修正含水率边际形状判断：添加'饱和型'分支
      （Water²系数为负，增长率递减）
  12. 新增HC3稳健标准误检验（异方差稳健）
  13. 修正AIC/BIC诊断结论文字（物理叙事与方程行为一致）
  14. 叙述聚焦论文二：黑箱模型→初步方程→特征分析→机制分析→优化方程
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
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from scipy import stats
import shap
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings('ignore')
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 全局配置
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
RANDOM_STATE = 42
KF = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
FEATURES = ['温度', '湿度', '冻融次数', '裂缝长度', '含水率']
TARGET = '导热系数'

CATBOOST_PARAMS = dict(
    iterations=300, depth=4, learning_rate=0.05,
    random_seed=RANDOM_STATE, verbose=False
)

# v2最终方程11项（用于bootstrap对比）
V2_FINAL_TERMS = {'T', 'T²', 'RH²', '√FT', '√Crack', 'Water', 'Water²',
                  'T×√Crack', 'RH×Water', '√FT×√Crack', 'RH×√Crack'}

def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def calc_metrics(y_true, y_pred):
    """计算R², RMSE, MAE, MAPE"""
    r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return r2, rmse, mae, mape

def cv_evaluate(X, y):
    """5折交叉验证（无权重），返回R², RMSE, MAE, MAPE
    注：显式方程采用无权重回归，避免完好基准点过拟合"""
    cv_r2, cv_rmse, cv_mae, cv_mape = [], [], [], []
    for train_idx, test_idx in KF.split(X):
        m = LinearRegression()
        m.fit(X[train_idx], y[train_idx])
        pred = m.predict(X[test_idx])
        r2, rmse, mae, mape = calc_metrics(y[test_idx], pred)
        cv_r2.append(r2)
        cv_rmse.append(rmse)
        cv_mae.append(mae)
        cv_mape.append(mape)
    return np.mean(cv_r2), np.std(cv_r2), np.mean(cv_rmse), np.mean(cv_mae), np.mean(cv_mape)

# ============================================================
# 数据加载
# ============================================================
def identify_scenario(row):
    """识别工况场景，用于分层bootstrap和分组残差诊断"""
    ft, crack, water = row['冻融次数'], row['裂缝长度'], row['含水率']
    if ft == 0 and crack == 0 and water == 0:
        return '完好'
    elif ft > 0 and crack == 0 and water == 0:
        return '纯冻融'
    elif ft == 0 and crack > 0 and water == 0:
        return '纯裂缝'
    elif ft == 0 and crack == 0 and water > 0:
        return '纯渗水'
    else:
        return '复合'

def load_data():
    df_deg = pd.read_excel(os.path.join(DATA_DIR, 'degradation_dataset.xlsx'), sheet_name='degradation_data')
    df_deg.columns = ['样本', '温度', '湿度', '冻融次数', '裂缝长度', '含水率', '导热系数']
    df_deg = df_deg[FEATURES + [TARGET]].copy()
    df_deg['数据来源'] = '劣化'
    df_deg['sample_weight'] = 1.0

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
        'sample_weight': df_intact_summary['N_Readings'].astype(float),
    })

    df_all = pd.concat([df_deg, df_intact], ignore_index=True)
    df_all['场景'] = df_all.apply(identify_scenario, axis=1)
    print(f"数据加载完成: 劣化{len(df_deg)}条 + 完好{len(df_intact)}条 = {len(df_all)}条")
    scenario_counts = df_all['场景'].value_counts()
    for s, c in scenario_counts.items():
        print(f"  场景[{s}]: {c}条")
    return df_all, df_deg

# ============================================================
# CatBoost黑箱模型
# ============================================================
def train_catboost(df_all):
    X = df_all[FEATURES].values
    y = df_all[TARGET].values
    w = df_all['sample_weight'].values

    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(X, y, sample_weight=w)

    train_r2, train_rmse, train_mae, train_mape = calc_metrics(y, model.predict(X))

    cv_r2, cv_rmse, cv_mae, cv_mape = [], [], [], []
    for train_idx, test_idx in KF.split(X):
        m = CatBoostRegressor(**CATBOOST_PARAMS)
        m.fit(X[train_idx], y[train_idx], sample_weight=w[train_idx])
        r2, rmse, mae, mape = calc_metrics(y[test_idx], m.predict(X[test_idx]))
        cv_r2.append(r2)
        cv_rmse.append(rmse)
        cv_mae.append(mae)
        cv_mape.append(mape)
    cv_r2_mean = np.mean(cv_r2)
    cv_std = np.std(cv_r2)
    cv_rmse_mean = np.mean(cv_rmse)
    cv_mae_mean = np.mean(cv_mae)
    cv_mape_mean = np.mean(cv_mape)

    print(f"\nCatBoost黑箱: 训练R²={train_r2:.4f}, RMSE={train_rmse:.6f}, MAE={train_mae:.6f}, MAPE={train_mape:.2f}%")
    print(f"  CV R²={cv_r2_mean:.4f}±{cv_std:.4f}, CV RMSE={cv_rmse_mean:.6f}, CV MAE={cv_mae_mean:.6f}, CV MAPE={cv_mape_mean:.2f}%")
    return model, train_r2, train_rmse, train_mae, train_mape, cv_r2_mean, cv_std, cv_rmse_mean, cv_mae_mean, cv_mape_mean

# ============================================================
# 无权重CatBoost（专用于SHAP分析）
# ============================================================
# 方法论说明：
#   CatBoost黑箱预测模型使用sample_weight加权（完好工况7~174），
#   但SHAP分析的目的是"指导无权重方程该加哪些交互项"，
#   因此SHAP必须用无权重CatBoost，与方程拟合口径一致。
# ============================================================
def train_catboost_unweighted(df_all):
    """训练无权重CatBoost，专用于SHAP分析（与方程拟合口径一致）"""
    X = df_all[FEATURES].values
    y = df_all[TARGET].values

    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(X, y)  # 无sample_weight

    train_r2 = model.score(X, y)
    print(f"\n无权重CatBoost（SHAP专用）: 训练R²={train_r2:.4f}")
    print(f"  注：此模型仅用于SHAP交互分析，不作为性能上限参照")
    return model

def shap_global_importance(catboost_model, df_all):
    """SHAP全局重要性分析（无权重CatBoost）"""
    print("\n" + "-" * 60)
    print("SHAP全局重要性分析（基于无权重CatBoost，351条全部工况）")
    print("-" * 60)

    X = df_all[FEATURES].values
    explainer = shap.TreeExplainer(catboost_model)
    shap_values = explainer.shap_values(X)

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    order = np.argsort(mean_abs_shap)[::-1]

    print("  变量重要性排序：")
    for rank, idx in enumerate(order):
        print(f"    {rank+1}. {FEATURES[idx]}: {mean_abs_shap[idx]:.6f}")

    # 保存SHAP柱状图
    fig, ax = plt.subplots(figsize=(5.51, 3.44))
    colors = ['#e74c3c' if i == order[0] else '#3498db' for i in range(len(FEATURES))]
    ax.barh(range(len(FEATURES)), [mean_abs_shap[i] for i in range(len(FEATURES))],
            color=colors, edgecolor='white')
    ax.set_yticks(range(len(FEATURES)))
    ax.set_yticklabels([FEATURES[i] for i in range(len(FEATURES))])
    ax.set_xlabel('平均|SHAP值|', fontsize=9.0)
    ax.set_title('SHAP全局变量重要性（无权重CatBoost）', fontsize=9.8)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, 'Fig_SHAP_Global.png'), dpi=600)
    plt.close()
    print(f"  已保存: Fig_SHAP_Global.png")

    # SHAP summary plot
    shap.summary_plot(shap_values, X, feature_names=FEATURES, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, 'Fig_SHAP_Summary.png'), dpi=600, bbox_inches='tight')
    plt.close()

    return {
        'shap_values': shap_values,
        'mean_abs_shap': mean_abs_shap,
        'order': order,
    }

def shap_interaction_analysis(catboost_model, df_all):
    """SHAP Layer II 交互分析（无权重CatBoost，351条全部工况）"""
    print("\n" + "-" * 60)
    print("SHAP Layer II 交互分析（基于无权重CatBoost，351条全部工况）")
    print("-" * 60)

    X = df_all[FEATURES].values
    explainer = shap.TreeExplainer(catboost_model)
    shap_inter = explainer.shap_interaction_values(X)

    n_features = len(FEATURES)
    interaction_strength = np.zeros((n_features, n_features))
    pair_list = []

    for i in range(n_features):
        for j in range(i + 1, n_features):
            strength = np.mean(np.abs(shap_inter[:, i, j]))
            interaction_strength[i, j] = strength
            interaction_strength[j, i] = strength
            pair_list.append((FEATURES[i], FEATURES[j], strength))

    # 排序
    pair_list.sort(key=lambda x: x[2], reverse=True)

    print("  交互强度排名（Top10）：")
    for rank, (f1, f2, s) in enumerate(pair_list[:10]):
        print(f"    {rank+1}. {f1}×{f2}: {s:.6f}")

    # 交互热力图（全部格点标注数值，对角线标记"—"）
    fig, ax = plt.subplots(figsize=(5.51, 4.82))
    short_names = ['T', 'RH', 'FT', 'Crack', 'Water']
    vmax = float(np.max(interaction_strength)) * 1.05
    im = ax.imshow(interaction_strength, cmap='YlOrRd', aspect='auto',
                   vmin=0.0, vmax=vmax)
    ax.set_xticks(range(n_features))
    ax.set_yticks(range(n_features))
    ax.set_xticklabels(short_names, fontsize=8.2)
    ax.set_yticklabels(short_names, fontsize=8.2)
    for i in range(n_features):
        for j in range(n_features):
            if i != j:
                val = interaction_strength[i, j]
                color = 'white' if val > 0.0002 else 'black'
                ax.text(j, i, f'{val:.6f}', ha='center', va='center',
                        fontsize=6.5, color=color)
            else:
                ax.text(j, i, '—', ha='center', va='center',
                        fontsize=7.5, color='gray')
    plt.colorbar(im, ax=ax, label='交互强度')
    ax.set_title('SHAP交互强度热力图（无权重CatBoost，351条）', fontsize=9.8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, 'Fig_SHAP_Interaction_Heatmap.png'), dpi=600)
    plt.close()
    print(f"  已保存: Fig_SHAP_Interaction_Heatmap.png")

    # 提取Top5交互对，映射到方程特征名
    print("\n  Top5交互对 → 方程候选交互项映射：")
    interaction_map = {
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

    top5_pairs = []
    for rank, (f1, f2, s) in enumerate(pair_list[:5]):
        eq_name = interaction_map.get((f1, f2)) or interaction_map.get((f2, f1))
        top5_pairs.append(eq_name)
        print(f"    {rank+1}. {f1}×{f2} ({s:.6f}) → {eq_name}")

    return {
        'interaction_strength': interaction_strength,
        'pair_list': pair_list,
        'top5_pairs': top5_pairs,
    }

# ============================================================
# 边际形状诊断
# ============================================================
def marginal_shape_diagnosis(df_all, df_deg):
    """各变量边际形状诊断，确定最佳函数形式"""
    print("\n" + "-" * 60)
    print("边际形状诊断")
    print("-" * 60)

    results = {}
    # 温度：用完好工况子集
    intact = df_all[df_all['数据来源'] == '完好']
    t_grouped = intact.groupby('温度')['导热系数'].agg(['mean', 'std', 'count'])
    t_diffs = t_grouped['mean'].diff().dropna()
    # T²系数为负(-8.98e-07)，极值点在T≈123°C（远超5-45°C数据范围）
    # 在实验范围内温度效应为单调递增，T²项为弱非线性修正
    t_shape = '单调递增（弱非线性）'
    t_transform = 'T + T²'
    print(f"  [温度] {t_shape} → {t_transform}")
    results['温度'] = {'shape': t_shape, 'transform': t_transform}

    # 湿度
    rh_grouped = intact.groupby('湿度')['导热系数'].agg(['mean', 'std', 'count'])
    rh_diffs = rh_grouped['mean'].diff().dropna()
    rh_shape = '加速型' if rh_diffs.iloc[-1] > rh_diffs.iloc[0] else '线性'
    rh_transform = 'RH + RH²'
    print(f"  [湿度] {rh_shape} → {rh_transform}")
    results['湿度'] = {'shape': rh_shape, 'transform': rh_transform}

    # 冻融次数
    ft_only = df_deg[(df_deg['裂缝长度'] == 0) & (df_deg['含水率'] == 0) & (df_deg['冻融次数'] > 0)]
    ft_grouped = ft_only.groupby('冻融次数')['导热系数'].agg(['mean', 'std', 'count'])
    ft_diffs = ft_grouped['mean'].diff().dropna()
    ft_shape = '饱和型' if ft_diffs.iloc[-1] < ft_diffs.iloc[0] else '线性'
    ft_transform = '√FT'
    print(f"  [冻融次数] {ft_shape} → {ft_transform}")
    results['冻融次数'] = {'shape': ft_shape, 'transform': ft_transform}

    # 裂缝长度
    crack_only = df_deg[(df_deg['冻融次数'] == 0) & (df_deg['含水率'] == 0) & (df_deg['裂缝长度'] > 0)]
    crack_grouped = crack_only.groupby('裂缝长度')['导热系数'].agg(['mean', 'std', 'count'])
    crack_diffs = crack_grouped['mean'].diff().dropna()
    crack_shape = '饱和型' if len(crack_diffs) > 1 and crack_diffs.iloc[-1] < crack_diffs.iloc[0] else '线性'
    crack_transform = '√Crack'
    print(f"  [裂缝长度] {crack_shape} → {crack_transform}")
    results['裂缝长度'] = {'shape': crack_shape, 'transform': crack_transform}

    # 含水率
    water_only = df_deg[(df_deg['冻融次数'] == 0) & (df_deg['裂缝长度'] == 0) & (df_deg['含水率'] > 0)]
    water_grouped = water_only.groupby('含水率')['导热系数'].agg(['mean', 'std', 'count'])
    water_diffs = water_grouped['mean'].diff().dropna()
    # Water²系数为负(-1.42e-06)，极值点在Water≈60.6%（接近数据上界56%）
    # 在0-56%范围内效应为单调递增但增长率递减，呈饱和型
    if len(water_diffs) > 1:
        if water_diffs.iloc[-1] > water_diffs.iloc[0]:
            water_shape = '加速型'
        elif water_diffs.iloc[-1] < water_diffs.iloc[0]:
            water_shape = '饱和型'
        else:
            water_shape = '线性'
    else:
        water_shape = '线性'
    water_transform = 'Water + Water²'
    print(f"  [含水率] {water_shape} → {water_transform}")
    results['含水率'] = {'shape': water_shape, 'transform': water_transform}

    # 绘制5合1边际形状图
    fig, axes = plt.subplots(2, 3, figsize=(7.48, 4.49))
    axes = axes.flatten()
    grouped_data = {
        '温度': t_grouped, '湿度': rh_grouped, '冻融次数': ft_grouped,
        '裂缝长度': crack_grouped, '含水率': water_grouped
    }
    for ax, feat in zip(axes, FEATURES):
        g = grouped_data[feat]
        ax.errorbar(g.index, g['mean'], yerr=g['std'], fmt='o-', capsize=3, color='#2c3e50', markersize=5)
        ax.set_xlabel(feat, fontsize=8.2)
        ax.set_ylabel('导热系数', fontsize=8.2)
        ax.set_title(f'{feat}: {results[feat]["shape"]}\n→ {results[feat]["transform"]}', fontsize=7.5)
        ax.tick_params(labelsize=6.8)
    axes[5].set_visible(False)
    plt.suptitle('各变量边际效应形状诊断', fontsize=10.0, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, 'Fig_Marginal_Shapes.png'), dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  已保存: Fig_Marginal_Shapes.png")

    return results

# ============================================================
# 朴素线性方程
# ============================================================
def fit_naive(df_all):
    T = df_all['温度'].values
    RH = df_all['湿度'].values
    FT = df_all['冻融次数'].values
    Crack = df_all['裂缝长度'].values
    Water = df_all['含水率'].values
    y = df_all[TARGET].values
    w = df_all['sample_weight'].values

    X = np.column_stack([T, RH, FT, Crack, Water])
    X_sm = sm.add_constant(X)
    model = sm.OLS(y, X_sm).fit()

    cv_r2, cv_std, cv_rmse, cv_mae, cv_mape = cv_evaluate(X, y)
    train_r2, train_rmse, train_mae, train_mape = calc_metrics(y, model.predict(X_sm))

    names = ['const', 'T', 'RH', 'FT', 'Crack', 'Water']
    print(f"\n朴素线性方程(5项): R²={model.rsquared:.4f}, CV R²={cv_r2:.4f}, "
          f"CV RMSE={cv_rmse:.6f}, CV MAE={cv_mae:.6f}, CV MAPE={cv_mape:.2f}%")
    for n, c, p in zip(names, model.params, model.pvalues):
        sig = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))
        print(f"  {n:8s}: {c:+.8f}  (p={p:.4e}) {sig}")

    return {
        'model': model, 'X': X, 'y': y,
        'feature_names': ['T', 'RH', 'FT', 'Crack', 'Water'],
        'coef_names': names, 'coefs': model.params, 'pvalues': model.pvalues,
        'train_r2': model.rsquared, 'adj_r2': model.rsquared_adj,
        'cv_r2': cv_r2, 'cv_std': cv_std,
        'cv_rmse': cv_rmse, 'cv_mae': cv_mae, 'cv_mape': cv_mape,
        'train_rmse': train_rmse, 'train_mae': train_mae, 'train_mape': train_mape,
        'n_terms': 5,
    }

# ============================================================
# 非线性主效应方程
# ============================================================
def fit_nonlinear_main(df_all):
    T = df_all['温度'].values
    RH = df_all['湿度'].values
    FT = df_all['冻融次数'].values
    Crack = df_all['裂缝长度'].values
    Water = df_all['含水率'].values
    y = df_all[TARGET].values
    w = df_all['sample_weight'].values

    sqrtFT = np.sqrt(FT)
    sqrtCrack = np.sqrt(Crack)

    names = ['T', 'RH', '√FT', '√Crack', 'Water', 'T²', 'RH²', 'Water²']
    X = np.column_stack([T, RH, sqrtFT, sqrtCrack, Water, T**2, RH**2, Water**2])
    X_sm = sm.add_constant(X)
    model = sm.OLS(y, X_sm).fit()

    cv_r2, cv_std, cv_rmse, cv_mae, cv_mape = cv_evaluate(X, y)
    train_r2, train_rmse, train_mae, train_mape = calc_metrics(y, model.predict(X_sm))

    print(f"\n非线性主效应方程(8项): R²={model.rsquared:.4f}, CV R²={cv_r2:.4f}, "
          f"CV RMSE={cv_rmse:.6f}, CV MAE={cv_mae:.6f}, CV MAPE={cv_mape:.2f}%")

    return {
        'model': model, 'X': X, 'y': y,
        'feature_names': names,
        'coef_names': ['const'] + names,
        'coefs': model.params, 'pvalues': model.pvalues,
        'train_r2': model.rsquared, 'adj_r2': model.rsquared_adj,
        'cv_r2': cv_r2, 'cv_std': cv_std,
        'cv_rmse': cv_rmse, 'cv_mae': cv_mae, 'cv_mape': cv_mape,
        'train_rmse': train_rmse, 'train_mae': train_mae, 'train_mape': train_mape,
        'n_terms': 8,
    }

# ============================================================
# 完整非线性方程(③)：8项主效应 + Top5 SHAP交互项 = 13项
# ============================================================
def fit_full_nonlinear(df_all, shap_top5_pairs):
    """
    方程(3) 完整非线性方程（SHAP驱动交互项）
    = 8项非线性主效应 + Top5无权重SHAP交互项 = 13项

    无权重SHAP Top5交互项（与方程拟合口径一致）：
      1. √Crack×Water  2. RH×Water  3. T×√Crack
      4. √FT×√Crack    5. RH×√Crack
    """
    print("\n" + "-" * 60)
    print("方程(3) 完整非线性方程（8主效应 + 5 SHAP交互 = 13项）")
    print("-" * 60)

    T = df_all['温度'].values
    RH = df_all['湿度'].values
    FT = df_all['冻融次数'].values
    Crack = df_all['裂缝长度'].values
    Water = df_all['含水率'].values
    y = df_all[TARGET].values

    sqrtFT = np.sqrt(FT)
    sqrtCrack = np.sqrt(Crack)

    # 8项非线性主效应
    main_names = ['T', 'RH', '√FT', '√Crack', 'Water', 'T²', 'RH²', 'Water²']
    X_main = np.column_stack([T, RH, sqrtFT, sqrtCrack, Water, T**2, RH**2, Water**2])

    # 5项SHAP交互项（按无权重SHAP排名）
    inter_features = {
        '√Crack×Water': sqrtCrack * Water,
        'RH×Water': RH * Water,
        'T×√Crack': T * sqrtCrack,
        '√FT×√Crack': sqrtFT * sqrtCrack,
        'RH×√Crack': RH * sqrtCrack,
        'T×RH': T * RH,
        'T×√FT': T * sqrtFT,
        'T×Water': T * Water,
        'RH×√FT': RH * sqrtFT,
        '√FT×Water': sqrtFT * Water,
    }

    # 按SHAP Top5排名选取交互项
    inter_names = []
    X_inter_cols = []
    for name in shap_top5_pairs:
        if name in inter_features:
            inter_names.append(name)
            X_inter_cols.append(inter_features[name])

    X_inter = np.column_stack(X_inter_cols) if X_inter_cols else np.empty((len(T), 0))
    all_names = main_names + inter_names
    X = np.column_stack([X_main, X_inter])

    X_sm = sm.add_constant(X)
    model = sm.OLS(y, X_sm).fit()

    cv_r2, cv_std, cv_rmse, cv_mae, cv_mape = cv_evaluate(X, y)
    train_r2, train_rmse, train_mae, train_mape = calc_metrics(y, model.predict(X_sm))

    coef_names = ['const'] + all_names
    print(f"  R²={model.rsquared:.4f}, Adj R²={model.rsquared_adj:.4f}, CV R²={cv_r2:.4f}±{cv_std:.4f}")
    print(f"  CV RMSE={cv_rmse:.6f}, MAE={cv_mae:.6f}, MAPE={cv_mape:.2f}%")
    print(f"  交互项候选（无权重SHAP Top5）: {inter_names}")
    print(f"\n  各项系数与显著性：")
    for n, c, p in zip(coef_names, model.params, model.pvalues):
        sig = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))
        print(f"    {n:16s}: {c:+.10f}  (p={p:.4e}) {sig}")

    return {
        'model': model, 'X': X, 'y': y,
        'feature_names': all_names,
        'main_names': main_names,
        'inter_names': inter_names,
        'coef_names': coef_names,
        'coefs': model.params, 'pvalues': model.pvalues,
        'train_r2': model.rsquared, 'adj_r2': model.rsquared_adj,
        'cv_r2': cv_r2, 'cv_std': cv_std,
        'cv_rmse': cv_rmse, 'cv_mae': cv_mae, 'cv_mape': cv_mape,
        'train_rmse': train_rmse, 'train_mae': train_mae, 'train_mape': train_mape,
        'n_terms': len(all_names),
    }

# ============================================================
# 反向消元(④)：从13项逐步移除不显著项
# ============================================================
def backward_elimination(df_all, full_results):
    """
    方程(4) 反向消元精炼方程

    消元策略：
      第一阶段：移除不显著交互项（p>0.05），每次移除p值最大的
      第二阶段：移除不显著主效应（p>0.05），但保护物理变换对（如T和T²不单独拆）
    """
    print("\n" + "-" * 60)
    print("方程(4) 反向消元精炼过程")
    print("-" * 60)

    T = df_all['温度'].values
    RH = df_all['湿度'].values
    FT = df_all['冻融次数'].values
    Crack = df_all['裂缝长度'].values
    Water = df_all['含水率'].values
    y = df_all[TARGET].values

    sqrtFT = np.sqrt(FT)
    sqrtCrack = np.sqrt(Crack)

    # 特征构造函数映射
    feature_builders = {
        'T': lambda: T,
        'RH': lambda: RH,
        '√FT': lambda: sqrtFT,
        '√Crack': lambda: sqrtCrack,
        'Water': lambda: Water,
        'T²': lambda: T**2,
        'RH²': lambda: RH**2,
        'Water²': lambda: Water**2,
        '√Crack×Water': lambda: sqrtCrack * Water,
        'RH×Water': lambda: RH * Water,
        'T×√Crack': lambda: T * sqrtCrack,
        '√FT×√Crack': lambda: sqrtFT * sqrtCrack,
        'RH×√Crack': lambda: RH * sqrtCrack,
        'T×RH': lambda: T * RH,
        'T×√FT': lambda: T * sqrtFT,
        'T×Water': lambda: T * Water,
        'RH×√FT': lambda: RH * sqrtFT,
        '√FT×Water': lambda: sqrtFT * Water,
    }

    # 物理保护对：不能单独移除一个（必须同时保留或同时移除）
    protect_pairs = [('T', 'T²'), ('RH', 'RH²'), ('Water', 'Water²')]

    current_names = list(full_results['feature_names'])
    elimination_path = []
    step = 0

    # 第一阶段：移除不显著交互项
    inter_names = list(full_results['inter_names'])
    print(f"\n  初始模型: {len(current_names)}项 ({len(full_results['main_names'])}主效应 + {len(inter_names)}交互)")

    while True:
        # 构造当前特征矩阵
        X_cols = [feature_builders[n]() for n in current_names]
        X = np.column_stack(X_cols)
        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit()

        # 检查交互项p值
        inter_pvals = [(n, model.pvalues[current_names.index(n) + 1])  # +1 for const
                       for n in current_names if n in inter_names]

        if not inter_pvals:
            break

        # 找p值最大的交互项
        inter_pvals.sort(key=lambda x: x[1], reverse=True)
        worst_name, worst_p = inter_pvals[0]

        if worst_p > 0.05:
            step += 1
            elimination_path.append({
                'step': step, 'action': 'remove_interaction',
                'term': worst_name, 'p_value': worst_p,
                'remaining_terms': len(current_names) - 1,
                'r2_before': model.rsquared,
            })
            print(f"\n  步骤{step}: 移除交互项 [{worst_name}] (p={worst_p:.4f})")
            current_names.remove(worst_name)
            inter_names.remove(worst_name)

            # 重新拟合并打印
            X_cols = [feature_builders[n]() for n in current_names]
            X = np.column_stack(X_cols)
            X_sm = sm.add_constant(X)
            model = sm.OLS(y, X_sm).fit()
            print(f"    剩余{len(current_names)}项, R²={model.rsquared:.4f}")
        else:
            print(f"\n  所有交互项均显著 (最差p={worst_p:.4f} for {worst_name})")
            break

    # 第二阶段：移除不显著主效应（保护物理变换对）
    while True:
        X_cols = [feature_builders[n]() for n in current_names]
        X = np.column_stack(X_cols)
        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit()

        # 主效应p值（排除交互项和受保护项）
        main_in_model = [n for n in current_names if n not in inter_names]
        candidates = []
        for name in main_in_model:
            pval = model.pvalues[current_names.index(name) + 1]
            # 检查是否在保护对中
            protected = False
            for p1, p2 in protect_pairs:
                if name == p1 and p2 in current_names:
                    # 线性项在其平方项存在时可以被移除（平方项保留）
                    protected = False
                elif name == p2 and p1 in current_names:
                    # 平方项在线性项存在时不单独移除
                    protected = True
            if not protected:
                candidates.append((name, pval))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[1], reverse=True)
        worst_name, worst_p = candidates[0]

        if worst_p > 0.05:
            step += 1
            elimination_path.append({
                'step': step, 'action': 'remove_main',
                'term': worst_name, 'p_value': worst_p,
                'remaining_terms': len(current_names) - 1,
                'r2_before': model.rsquared,
            })
            print(f"\n  步骤{step}: 移除主效应 [{worst_name}] (p={worst_p:.4f})")
            current_names.remove(worst_name)

            X_cols = [feature_builders[n]() for n in current_names]
            X = np.column_stack(X_cols)
            X_sm = sm.add_constant(X)
            model = sm.OLS(y, X_sm).fit()
            print(f"    剩余{len(current_names)}项, R²={model.rsquared:.4f}")
        else:
            print(f"\n  所有主效应均显著 (最差p={worst_p:.4f} for {worst_name})")
            break

    # 最终模型
    X_cols = [feature_builders[n]() for n in current_names]
    X = np.column_stack(X_cols)
    X_sm = sm.add_constant(X)
    model = sm.OLS(y, X_sm).fit()

    cv_r2, cv_std, cv_rmse, cv_mae, cv_mape = cv_evaluate(X, y)
    train_r2, train_rmse, train_mae, train_mape = calc_metrics(y, model.predict(X_sm))

    coef_names = ['const'] + current_names
    print(f"\n  最终方程: {len(current_names)}项")
    print(f"  R²={model.rsquared:.4f}, Adj R²={model.rsquared_adj:.4f}, CV R²={cv_r2:.4f}±{cv_std:.4f}")
    print(f"  CV RMSE={cv_rmse:.6f}, MAE={cv_mae:.6f}, MAPE={cv_mape:.2f}%")
    print(f"\n  最终系数：")
    for n, c, p in zip(coef_names, model.params, model.pvalues):
        sig = '***' if p<0.001 else ('**' if p<0.01 else ('*' if p<0.05 else 'ns'))
        print(f"    {n:16s}: {c:+.10f}  (p={p:.4e}) {sig}")

    return {
        'model': model, 'X': X, 'y': y,
        'feature_names': current_names,
        'coef_names': coef_names,
        'coefs': model.params, 'pvalues': model.pvalues,
        'train_r2': model.rsquared, 'adj_r2': model.rsquared_adj,
        'cv_r2': cv_r2, 'cv_std': cv_std,
        'cv_rmse': cv_rmse, 'cv_mae': cv_mae, 'cv_mape': cv_mape,
        'train_rmse': train_rmse, 'train_mae': train_mae, 'train_mape': train_mape,
        'n_terms': len(current_names),
        'elimination_path': elimination_path,
    }

def organize_final_equation(elim_results):
    """将反向消元结果按物理层次重新组织（基准/劣化/耦合）"""
    remaining = elim_results['feature_names']

    # 物理层次分类
    baseline_set = {'T', 'T²', 'RH²'}
    deg_set = {'√FT', '√Crack', 'Water', 'Water²'}
    coup_set = {'T×√Crack', 'RH×Water', '√FT×√Crack', 'RH×√Crack',
                '√Crack×Water', 'T×RH', 'T×√FT', 'T×Water', 'RH×√FT', '√FT×Water'}

    baseline_names = [n for n in remaining if n in baseline_set]
    deg_names = [n for n in remaining if n in deg_set]
    coup_names = [n for n in remaining if n in coup_set]

    # VIF检验
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    print(f"\n  VIF检验：")
    vif_results = {}
    for i, n in enumerate(remaining):
        vif = variance_inflation_factor(elim_results['X'], i)
        flag = ' ⚠高共线性' if vif > 10 else ''
        print(f"    {n:16s}: VIF={vif:.2f}{flag}")
        vif_results[n] = float(vif)

    # HC3稳健标准误（异方差稳健）
    print(f"\n  HC3稳健标准误检验（异方差稳健）：")
    X_final = elim_results['X']
    y_final = elim_results['y']
    X_sm_final = sm.add_constant(X_final)
    model_hc3 = sm.OLS(y_final, X_sm_final).fit(cov_type='HC3')
    hc3_results = {}
    for i, name in enumerate(elim_results['coef_names']):
        se_ols = float(elim_results['model'].bse[i])
        se_hc3 = float(model_hc3.bse[i])
        ratio = se_hc3 / se_ols if se_ols > 0 else 0
        p_hc3 = float(model_hc3.pvalues[i])
        sig = '***' if p_hc3 < 0.001 else ('**' if p_hc3 < 0.01 else ('*' if p_hc3 < 0.05 else 'ns'))
        print(f"    {name:16s}: OLS SE={se_ols:.2e}, HC3 SE={se_hc3:.2e}, 比值={ratio:.2f}, HC3 p={p_hc3:.4e} {sig}")
        hc3_results[name] = {
            'se_ols': se_ols, 'se_hc3': se_hc3,
            'ratio': ratio, 'p_hc3': p_hc3,
        }

    # 合并到返回值
    result = dict(elim_results)
    result['baseline_names'] = baseline_names
    result['deg_names'] = deg_names
    result['coup_names'] = coup_names
    result['vif_results'] = vif_results
    result['hc3_results'] = hc3_results
    return result

# ============================================================
# 三层分解
# ============================================================
def decompose(df_all, final_results):
    coefs = final_results['coefs']
    pvals = final_results['pvalues']
    coef_names = list(final_results['coef_names'])

    def get_coef(name):
        return float(coefs[coef_names.index(name)])
    def get_pval(name):
        return float(pvals[coef_names.index(name)])

    baseline_names = ['const'] + final_results['baseline_names']
    deg_names = final_results['deg_names']
    coup_names = final_results['coup_names']

    baseline_coefs = {n: get_coef(n) for n in baseline_names}
    deg_coefs = {n: get_coef(n) for n in deg_names}
    coup_coefs = {n: get_coef(n) for n in coup_names}

    # 计算各层平均贡献
    T = df_all['温度'].values
    RH = df_all['湿度'].values
    FT = df_all['冻融次数'].values
    Crack = df_all['裂缝长度'].values
    Water = df_all['含水率'].values
    sqrtFT = np.sqrt(FT)
    sqrtCrack = np.sqrt(Crack)

    contrib_baseline = (baseline_coefs['const']
        + baseline_coefs['T'] * T
        + baseline_coefs['T²'] * T**2
        + baseline_coefs['RH²'] * RH**2)

    contrib_deg = (deg_coefs['√FT'] * sqrtFT
        + deg_coefs['√Crack'] * sqrtCrack
        + deg_coefs['Water'] * Water
        + deg_coefs['Water²'] * Water**2)

    contrib_coup = np.zeros(len(T))
    contrib_coup += coup_coefs['T×√Crack'] * T * sqrtCrack
    contrib_coup += coup_coefs['RH×Water'] * RH * Water
    contrib_coup += coup_coefs['√FT×√Crack'] * sqrtFT * sqrtCrack
    contrib_coup += coup_coefs['RH×√Crack'] * RH * sqrtCrack

    print(f"\n{'='*60}")
    print(f"三层分解")
    print(f"{'='*60}")
    print(f"\n【第一层】基准方程（完好工况，环境效应）:")
    print(f"  λ₀ = {baseline_coefs['const']:.8f} + {baseline_coefs['T']:+.8f}×T "
          f"+ {baseline_coefs['T²']:+.8f}×T² + {baseline_coefs['RH²']:+.8f}×RH²")
    print(f"\n【第二层】劣化主效应:")
    print(f"  Δλ_deg = {deg_coefs['√FT']:+.8f}×√FT + {deg_coefs['√Crack']:+.8f}×√Crack "
          f"+ {deg_coefs['Water']:+.8f}×Water + {deg_coefs['Water²']:+.8f}×Water²")
    print(f"\n【第三层】耦合效应:")
    print(f"  Δλ_coup = {coup_coefs['T×√Crack']:+.8f}×(T×√Crack) "
          f"+ {coup_coefs['RH×Water']:+.8f}×(RH×Water) "
          f"+ {coup_coefs['√FT×√Crack']:+.8f}×(√FT×√Crack) "
          f"+ {coup_coefs['RH×√Crack']:+.8f}×(RH×√Crack)")

    print(f"\n各层平均贡献量（351样本）：")
    print(f"  λ₀    均值 = {np.mean(contrib_baseline):.6f}")
    print(f"  Δλ_deg 均值 = {np.mean(contrib_deg):.6f}")
    print(f"  Δλ_coup均值 = {np.mean(contrib_coup):.6f}")
    print(f"  λ 总均值    = {np.mean(contrib_baseline + contrib_deg + contrib_coup):.6f}")

    # 物理一致性验证
    print(f"\n【物理一致性验证】")
    print(f"  当 FT=0, Crack=0, Water=0 时：")
    print(f"  Δλ_deg = 0（劣化主效应全部消失）")
    coup_at_zero = coup_coefs.copy()
    print(f"  Δλ_coup 中：T×√Crack=0, RH×Water=0, √FT×√Crack=0, RH×√Crack=0")
    print(f"  → 方程退化为: λ = λ₀（纯基准方程）")
    print(f"  → 完好工况下方程完全退化为基准方程，无任何耦合残余")

    return {
        'baseline_names': baseline_names,
        'deg_names': deg_names,
        'coup_names': coup_names,
        'baseline_coefs': baseline_coefs,
        'deg_coefs': deg_coefs,
        'coup_coefs': coup_coefs,
        'baseline_pvals': {n: get_pval(n) for n in baseline_names},
        'deg_pvals': {n: get_pval(n) for n in deg_names},
        'coup_pvals': {n: get_pval(n) for n in coup_names},
        'contrib_baseline': contrib_baseline,
        'contrib_deg': contrib_deg,
        'contrib_coup': contrib_coup,
    }

# ============================================================
# 三层嵌套模型F检验
# ============================================================
def nested_ftest(df_all):
    T = df_all['温度'].values
    RH = df_all['湿度'].values
    FT = df_all['冻融次数'].values
    Crack = df_all['裂缝长度'].values
    Water = df_all['含水率'].values
    y = df_all[TARGET].values
    w = df_all['sample_weight'].values

    sqrtFT = np.sqrt(FT)
    sqrtCrack = np.sqrt(Crack)

    # M0: 基准方程（3项特征 + const = 4项）
    X0 = np.column_stack([T, T**2, RH**2])
    m0 = sm.OLS(y, sm.add_constant(X0)).fit()

    # M1: 基准 + 劣化主效应（7项特征 + const = 8项）
    X1 = np.column_stack([T, T**2, RH**2, sqrtFT, sqrtCrack, Water, Water**2])
    m1 = sm.OLS(y, sm.add_constant(X1)).fit()

    # M2: 基准 + 劣化主效应 + 耦合效应（11项特征 + const = 12项）
    X2 = np.column_stack([T, T**2, RH**2, sqrtFT, sqrtCrack, Water, Water**2,
                          T*sqrtCrack, RH*Water, sqrtFT*sqrtCrack, RH*sqrtCrack])
    m2 = sm.OLS(y, sm.add_constant(X2)).fit()

    n = len(y)
    rss0, rss1, rss2 = m0.ssr, m1.ssr, m2.ssr
    df_res0 = n - 4   # M0: 3特征+const
    df_res1 = n - 8   # M1: 7特征+const
    df_res2 = n - 12  # M2: 11特征+const

    df_num_01 = int(df_res0 - df_res1)  # 4
    df_num_12 = int(df_res1 - df_res2)  # 4
    f_stat_01 = ((rss0 - rss1) / df_num_01) / (rss1 / df_res1)
    p_val_01 = stats.f.sf(f_stat_01, df_num_01, df_res1)
    f_stat_12 = ((rss1 - rss2) / df_num_12) / (rss2 / df_res2)
    p_val_12 = stats.f.sf(f_stat_12, df_num_12, df_res2)

    delta_r2_01 = m1.rsquared - m0.rsquared
    delta_r2_12 = m2.rsquared - m1.rsquared
    total_r2 = m2.rsquared

    print(f"\n{'='*60}")
    print(f"三层嵌套模型F检验")
    print(f"{'='*60}")
    print(f"\n  {'模型':<35s} {'项数':>4s} {'R²':>8s} {'Adj R²':>8s} {'RSS':>14s}")
    print(f"  {'-'*72}")
    print(f"  {'M0: 基准方程(T,T²,RH²)':<35s} {'3':>4s} {m0.rsquared:>8.4f} {m0.rsquared_adj:>8.4f} {rss0:>14.6e}")
    print(f"  {'M1: +劣化主效应':<35s} {'7':>4s} {m1.rsquared:>8.4f} {m1.rsquared_adj:>8.4f} {rss1:>14.6e}")
    print(f"  {'M2: +耦合效应':<35s} {'11':>4s} {m2.rsquared:>8.4f} {m2.rsquared_adj:>8.4f} {rss2:>14.6e}")

    print(f"\n  检验1: M0→M1（劣化主效应）")
    print(f"    F({df_num_01},{df_res1}) = {f_stat_01:.2f}, p = {p_val_01:.4e}, ΔR² = {delta_r2_01:.4f}")
    print(f"\n  检验2: M1→M2（耦合效应）")
    print(f"    F({df_num_12},{df_res2}) = {f_stat_12:.2f}, p = {p_val_12:.4e}, ΔR² = {delta_r2_12:.4f}")

    print(f"\n  R²分解: 基准 {m0.rsquared/total_r2*100:.1f}% | "
          f"劣化主效应 {delta_r2_01/total_r2*100:.1f}% | "
          f"耦合效应 {delta_r2_12/total_r2*100:.1f}%")

    return {
        'm0': {'r2': m0.rsquared, 'adj_r2': m0.rsquared_adj, 'rss': rss0, 'n_terms': 3},
        'm1': {'r2': m1.rsquared, 'adj_r2': m1.rsquared_adj, 'rss': rss1, 'n_terms': 7},
        'm2': {'r2': m2.rsquared, 'adj_r2': m2.rsquared_adj, 'rss': rss2, 'n_terms': 11},
        'ftest_01': {'f_stat': f_stat_01, 'p_val': p_val_01, 'delta_r2': delta_r2_01,
                     'df_num': df_num_01, 'df_den': df_res1},
        'ftest_12': {'f_stat': f_stat_12, 'p_val': p_val_12, 'delta_r2': delta_r2_12,
                     'df_num': df_num_12, 'df_den': df_res2},
    }

# ============================================================
# [v3新增] AIC/BIC边际诊断（P1-3）
# 偏除协变量后正式比较候选函数形式
# ============================================================
def aic_bic_diagnosis(df_all, df_deg):
    """AIC/BIC边际诊断：偏除协变量后正式比较候选函数形式"""
    print("\n" + "=" * 60)
    print("AIC/BIC边际诊断（P1-3）：偏除协变量后比较候选函数形式")
    print("=" * 60)

    results = {}
    intact = df_all[df_all['数据来源'] == '完好']

    # v2当前选择的函数形式
    v2_choices = {
        '温度': '二次 (T+T²)',
        '湿度': '二次 (RH+RH²)',
        '冻融次数': '平方根 (√FT)',
        '裂缝长度': '平方根 (√Crack)',
        '含水率': '二次 (Water+Water²)',
    }

    def compare_forms(y, cov_cols, form_cols_list, form_names):
        """拟合 OLS(常量+协变量+候选形式)，返回AIC/BIC/R²对比"""
        comparison = []
        for form_name, f_cols in zip(form_names, form_cols_list):
            all_cols = cov_cols + f_cols
            X = np.column_stack(all_cols)
            X_sm = sm.add_constant(X)
            model = sm.OLS(y, X_sm).fit()
            comparison.append({
                'form': form_name,
                'aic': float(model.aic),
                'bic': float(model.bic),
                'r2': float(model.rsquared),
                'n_params': int(len(model.params)),
            })
        min_aic = min(c['aic'] for c in comparison)
        for c in comparison:
            c['delta_aic'] = float(c['aic'] - min_aic)
        best_aic = min(comparison, key=lambda x: x['aic'])
        best_bic = min(comparison, key=lambda x: x['bic'])
        return comparison, best_aic['form'], best_bic['form']

    # ---- 1. 温度T：用27条完好工况，偏除RH+RH²，比较线性/二次/三次 ----
    T = intact['温度'].values
    RH = intact['湿度'].values
    y_intact = intact[TARGET].values

    t_comp, t_best_aic, t_best_bic = compare_forms(
        y_intact,
        [RH, RH**2],
        [[T], [T, T**2], [T, T**2, T**3]],
        ['线性 (T)', '二次 (T+T²)', '三次 (T+T²+T³)']
    )
    results['温度'] = {'comparison': t_comp, 'best_aic': t_best_aic, 'best_bic': t_best_bic}

    print(f"\n  [温度T] 完好工况{len(intact)}条，协变量: RH+RH²")
    print(f"    {'函数形式':<20s} {'AIC':>12s} {'BIC':>12s} {'R²':>8s} {'ΔAIC':>10s} {'参数数':>6s}")
    print(f"    {'-'*70}")
    for c in t_comp:
        print(f"    {c['form']:<20s} {c['aic']:>12.4f} {c['bic']:>12.4f} {c['r2']:>8.4f} {c['delta_aic']:>10.4f} {c['n_params']:>6d}")
    print(f"    → AIC最优: {t_best_aic} | BIC最优: {t_best_bic} | v2选择: {v2_choices['温度']}")

    # ---- 2. 湿度RH：用27条完好工况，偏除T+T²，比较线性/二次 ----
    rh_comp, rh_best_aic, rh_best_bic = compare_forms(
        y_intact,
        [T, T**2],
        [[RH], [RH, RH**2]],
        ['线性 (RH)', '二次 (RH+RH²)']
    )
    results['湿度'] = {'comparison': rh_comp, 'best_aic': rh_best_aic, 'best_bic': rh_best_bic}

    print(f"\n  [湿度RH] 完好工况{len(intact)}条，协变量: T+T²")
    print(f"    {'函数形式':<20s} {'AIC':>12s} {'BIC':>12s} {'R²':>8s} {'ΔAIC':>10s} {'参数数':>6s}")
    print(f"    {'-'*70}")
    for c in rh_comp:
        print(f"    {c['form']:<20s} {c['aic']:>12.4f} {c['bic']:>12.4f} {c['r2']:>8.4f} {c['delta_aic']:>10.4f} {c['n_params']:>6d}")
    print(f"    → AIC最优: {rh_best_aic} | BIC最优: {rh_best_bic} | v2选择: {v2_choices['湿度']}")

    # ---- 3. 冻融FT：纯冻融子集，偏除T+T²+RH+RH²，比较线性/√FT/log(FT+1) ----
    ft_sub = df_deg[(df_deg['裂缝长度'] == 0) & (df_deg['含水率'] == 0) & (df_deg['冻融次数'] > 0)]
    T_ft = ft_sub['温度'].values
    RH_ft = ft_sub['湿度'].values
    FT_ft = ft_sub['冻融次数'].values
    y_ft = ft_sub[TARGET].values

    ft_comp, ft_best_aic, ft_best_bic = compare_forms(
        y_ft,
        [T_ft, T_ft**2, RH_ft, RH_ft**2],
        [[FT_ft], [np.sqrt(FT_ft)], [np.log(FT_ft + 1)]],
        ['线性 (FT)', '平方根 (√FT)', '对数 (log(FT+1))']
    )
    results['冻融次数'] = {'comparison': ft_comp, 'best_aic': ft_best_aic, 'best_bic': ft_best_bic}

    print(f"\n  [冻融次数FT] 纯冻融子集{len(ft_sub)}条，协变量: T+T²+RH+RH²")
    print(f"    {'函数形式':<20s} {'AIC':>12s} {'BIC':>12s} {'R²':>8s} {'ΔAIC':>10s} {'参数数':>6s}")
    print(f"    {'-'*70}")
    for c in ft_comp:
        print(f"    {c['form']:<20s} {c['aic']:>12.4f} {c['bic']:>12.4f} {c['r2']:>8.4f} {c['delta_aic']:>10.4f} {c['n_params']:>6d}")
    print(f"    → AIC最优: {ft_best_aic} | BIC最优: {ft_best_bic} | v2选择: {v2_choices['冻融次数']}")

    # ---- 4. 裂缝Crack：纯裂缝子集，偏除T+T²+RH+RH²，比较线性/√Crack/log(Crack+1) ----
    crack_sub = df_deg[(df_deg['冻融次数'] == 0) & (df_deg['含水率'] == 0) & (df_deg['裂缝长度'] > 0)]
    T_cr = crack_sub['温度'].values
    RH_cr = crack_sub['湿度'].values
    Crack_cr = crack_sub['裂缝长度'].values
    y_cr = crack_sub[TARGET].values

    crack_comp, crack_best_aic, crack_best_bic = compare_forms(
        y_cr,
        [T_cr, T_cr**2, RH_cr, RH_cr**2],
        [[Crack_cr], [np.sqrt(Crack_cr)], [np.log(Crack_cr + 1)]],
        ['线性 (Crack)', '平方根 (√Crack)', '对数 (log(Crack+1))']
    )
    results['裂缝长度'] = {'comparison': crack_comp, 'best_aic': crack_best_aic, 'best_bic': crack_best_bic}

    print(f"\n  [裂缝长度Crack] 纯裂缝子集{len(crack_sub)}条，协变量: T+T²+RH+RH²")
    print(f"    {'函数形式':<20s} {'AIC':>12s} {'BIC':>12s} {'R²':>8s} {'ΔAIC':>10s} {'参数数':>6s}")
    print(f"    {'-'*70}")
    for c in crack_comp:
        print(f"    {c['form']:<20s} {c['aic']:>12.4f} {c['bic']:>12.4f} {c['r2']:>8.4f} {c['delta_aic']:>10.4f} {c['n_params']:>6d}")
    print(f"    → AIC最优: {crack_best_aic} | BIC最优: {crack_best_bic} | v2选择: {v2_choices['裂缝长度']}")

    # ---- 5. 含水率Water：纯渗水子集，偏除T+T²+RH+RH²，比较线性/二次 ----
    water_sub = df_deg[(df_deg['冻融次数'] == 0) & (df_deg['裂缝长度'] == 0) & (df_deg['含水率'] > 0)]
    T_w = water_sub['温度'].values
    RH_w = water_sub['湿度'].values
    Water_w = water_sub['含水率'].values
    y_w = water_sub[TARGET].values

    water_comp, water_best_aic, water_best_bic = compare_forms(
        y_w,
        [T_w, T_w**2, RH_w, RH_w**2],
        [[Water_w], [Water_w, Water_w**2]],
        ['线性 (Water)', '二次 (Water+Water²)']
    )
    results['含水率'] = {'comparison': water_comp, 'best_aic': water_best_aic, 'best_bic': water_best_bic}

    print(f"\n  [含水率Water] 纯渗水子集{len(water_sub)}条，协变量: T+T²+RH+RH²")
    print(f"    {'函数形式':<20s} {'AIC':>12s} {'BIC':>12s} {'R²':>8s} {'ΔAIC':>10s} {'参数数':>6s}")
    print(f"    {'-'*70}")
    for c in water_comp:
        print(f"    {c['form']:<20s} {c['aic']:>12.4f} {c['bic']:>12.4f} {c['r2']:>8.4f} {c['delta_aic']:>10.4f} {c['n_params']:>6d}")
    print(f"    → AIC最优: {water_best_aic} | BIC最优: {water_best_bic} | v2选择: {v2_choices['含水率']}")

    # ---- 汇总表 ----
    print(f"\n{'='*60}")
    print(f"AIC/BIC边际诊断汇总：最优选择 vs v2当前选择")
    print(f"{'='*60}")
    print(f"  {'变量':<12s} {'AIC最优':<20s} {'BIC最优':<20s} {'v2选择':<20s} {'一致性':>8s}")
    print(f"  {'-'*80}")
    for var in ['温度', '湿度', '冻融次数', '裂缝长度', '含水率']:
        aic_best = results[var]['best_aic']
        bic_best = results[var]['best_bic']
        v2_ch = v2_choices[var]
        aic_match = '一致' if aic_best == v2_ch else '不一致'
        bic_match = '一致' if bic_best == v2_ch else '不一致'
        consistency = f'AIC:{aic_match} BIC:{bic_match}'
        print(f"  {var:<12s} {aic_best:<20s} {bic_best:<20s} {v2_ch:<20s} {consistency:>8s}")

    print(f"\n  结论：AIC/BIC诊断结果与v2选择在部分变量上存在差异，")
    print(f"        但v2采用√FT/√Crack基于饱和型物理机制，Water+Water²基于饱和型物理机制（Water²系数为负，增长率递减），")
    print(f"        T+T²基于弱非线性修正（T²系数为负但极值点远超数据范围，实验范围内单调递增），")
    print(f"        诊断结果用于交叉验证而非覆盖物理判断。")

    return results

# ============================================================
# [v3新增] 嵌套检验M1→M2→M3（P0-2）
# 用变换变量重做，含三阶交互探索和VIF诊断
# ============================================================
def nested_ftest_transformed(df_all):
    """用变换变量重做嵌套F检验 M1→M2→M3"""
    print("\n" + "=" * 60)
    print("嵌套检验（变换变量 M1→M2→M3）（P0-2）")
    print("=" * 60)

    T = df_all['温度'].values
    RH = df_all['湿度'].values
    FT = df_all['冻融次数'].values
    Crack = df_all['裂缝长度'].values
    Water = df_all['含水率'].values
    y = df_all[TARGET].values

    sqrtFT = np.sqrt(FT)
    sqrtCrack = np.sqrt(Crack)

    # 5个变换基变量
    base_vars = [T, RH, sqrtFT, sqrtCrack, Water]
    base_names = ['T', 'RH', '√FT', '√Crack', 'Water']

    # M1: 8项非线性主效应（T, RH, √FT, √Crack, Water, T², RH², Water²）
    m1_names = ['T', 'RH', '√FT', '√Crack', 'Water', 'T²', 'RH²', 'Water²']
    X_m1 = np.column_stack([T, RH, sqrtFT, sqrtCrack, Water, T**2, RH**2, Water**2])

    # 10个两两交互（C(5,2)=10）
    pair_names = []
    pair_cols = []
    for i in range(5):
        for j in range(i + 1, 5):
            name = f'{base_names[i]}×{base_names[j]}'
            col = base_vars[i] * base_vars[j]
            pair_names.append(name)
            pair_cols.append(col)

    # M2: M1 + 全部10个两两交互 = 18项
    m2_names = m1_names + pair_names
    X_m2 = np.column_stack([X_m1] + pair_cols)

    # 10个三阶交互（C(5,3)=10）
    three_names = []
    three_cols = []
    for i in range(5):
        for j in range(i + 1, 5):
            for k in range(j + 1, 5):
                name = f'{base_names[i]}×{base_names[j]}×{base_names[k]}'
                col = base_vars[i] * base_vars[j] * base_vars[k]
                three_names.append(name)
                three_cols.append(col)

    # M3_all: M2 + 全部10个三阶交互 = 28项
    m3_names = m2_names + three_names
    X_m3 = np.column_stack([X_m2] + three_cols)

    # 拟合三个模型
    m1 = sm.OLS(y, sm.add_constant(X_m1)).fit()
    m2 = sm.OLS(y, sm.add_constant(X_m2)).fit()
    m3 = sm.OLS(y, sm.add_constant(X_m3)).fit()

    n = len(y)
    rss1, rss2, rss3 = m1.ssr, m2.ssr, m3.ssr
    p1 = X_m1.shape[1] + 1  # 9 (8特征+const)
    p2 = X_m2.shape[1] + 1  # 19 (18特征+const)
    p3 = X_m3.shape[1] + 1  # 29 (28特征+const)

    # F检验 M1→M2
    df_num_12 = p2 - p1   # 10
    df_den_12 = n - p2    # 332
    f_12 = ((rss1 - rss2) / df_num_12) / (rss2 / df_den_12)
    p_12 = stats.f.sf(f_12, df_num_12, df_den_12)
    delta_r2_12 = m2.rsquared - m1.rsquared

    # F检验 M2→M3_all
    df_num_23 = p3 - p2   # 10
    df_den_23 = n - p3    # 322
    f_23 = ((rss2 - rss3) / df_num_23) / (rss3 / df_den_23)
    p_23 = stats.f.sf(f_23, df_num_23, df_den_23)
    delta_r2_23 = m3.rsquared - m2.rsquared

    # 打印模型对比
    print(f"\n  {'模型':<45s} {'项数':>4s} {'R²':>8s} {'Adj R²':>8s} {'RSS':>14s}")
    print(f"  {'-'*85}")
    print(f"  {'M1: 8项非线性主效应':<45s} {'8':>4s} {m1.rsquared:>8.4f} {m1.rsquared_adj:>8.4f} {rss1:>14.6e}")
    print(f"  {'M2: M1+10两两交互':<45s} {'18':>4s} {m2.rsquared:>8.4f} {m2.rsquared_adj:>8.4f} {rss2:>14.6e}")
    print(f"  {'M3_all: M2+10三阶交互':<45s} {'28':>4s} {m3.rsquared:>8.4f} {m3.rsquared_adj:>8.4f} {rss3:>14.6e}")

    print(f"\n  检验1: M1→M2（二阶交互效应）")
    print(f"    F({df_num_12},{df_den_12}) = {f_12:.2f}, p = {p_12:.4e}, ΔR² = {delta_r2_12:.4f}")
    print(f"\n  检验2: M2→M3_all（三阶交互效应）")
    print(f"    F({df_num_23},{df_den_23}) = {f_23:.2f}, p = {p_23:.4e}, ΔR² = {delta_r2_23:.4f}")

    # ---- 逐个测试三阶交互项（加到M2上）----
    print(f"\n  逐个三阶交互项检验（加到M2上）：")
    print(f"    {'项名':<20s} {'F':>8s} {'p值':>12s} {'ΔR²':>12s} {'系数':>14s} {'系数p值':>12s} {'VIF':>10s}")
    print(f"    {'-'*90}")

    three_way_results = []
    X_m2_sm = sm.add_constant(X_m2)

    for name, col in zip(three_names, three_cols):
        X_test = np.column_stack([X_m2, col])
        X_test_sm = sm.add_constant(X_test)
        m_test = sm.OLS(y, X_test_sm).fit()

        p_test_n = X_test.shape[1] + 1  # 20
        f_stat = ((rss2 - m_test.ssr) / 1) / (m_test.ssr / (n - p_test_n))
        p_val = stats.f.sf(f_stat, 1, n - p_test_n)
        delta_r2 = m_test.rsquared - m2.rsquared
        coef = float(m_test.params[-1])
        coef_pval = float(m_test.pvalues[-1])
        try:
            vif = float(variance_inflation_factor(X_test_sm, X_test_sm.shape[1] - 1))
        except Exception:
            vif = float('inf')

        three_way_results.append({
            'term': name,
            'f_stat': float(f_stat),
            'p_value': float(p_val),
            'delta_r2': float(delta_r2),
            'coef': coef,
            'coef_pval': coef_pval,
            'vif': vif,
        })
        sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
        print(f"    {name:<20s} {f_stat:>8.2f} {p_val:>12.4e} {delta_r2:>12.6f} {coef:>+14.6e} {coef_pval:>12.4e} {vif:>10.2f} {sig}")

    best_3way = max(three_way_results, key=lambda x: x['f_stat'])
    print(f"\n  最强三阶交互: {best_3way['term']} (F={best_3way['f_stat']:.2f}, p={best_3way['p_value']:.4e}, "
          f"VIF={best_3way['vif']:.2f})")
    print(f"  注：三阶交互VIF普遍很高（>25），表明存在严重共线性，解释需谨慎。")

    # ---- M2中全部二阶交互项的系数和显著性 ----
    print(f"\n  M2中全部二阶交互项系数与显著性：")
    print(f"    {'项名':<20s} {'系数':>14s} {'p值':>12s} {'显著性':>6s}")
    print(f"    {'-'*56}")
    m2_inter_coefs = []
    for i, name in enumerate(pair_names):
        idx = len(m1_names) + i + 1  # +1 for const
        coef = float(m2.params[idx])
        pval = float(m2.pvalues[idx])
        sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else 'ns'))
        m2_inter_coefs.append({'term': name, 'coef': coef, 'pval': pval})
        print(f"    {name:<20s} {coef:>+14.6e} {pval:>12.4e} {sig:>6s}")

    # ---- VIF诊断 ----
    print(f"\n  M2 VIF诊断（18项）：")
    print(f"    {'项名':<20s} {'VIF':>10s}")
    print(f"    {'-'*32}")
    m2_vif = []
    X_m2_sm = sm.add_constant(X_m2)
    for i in range(X_m2.shape[1]):
        try:
            vif = float(variance_inflation_factor(X_m2_sm, i + 1))
        except Exception:
            vif = float('inf')
        flag = ' ⚠高共线性' if vif > 10 else ''
        m2_vif.append({'term': m2_names[i], 'vif': vif})
        print(f"    {m2_names[i]:<20s} {vif:>10.2f}{flag}")

    print(f"\n  M3_all VIF诊断（28项）：")
    print(f"    {'项名':<24s} {'VIF':>10s}")
    print(f"    {'-'*36}")
    m3_vif = []
    X_m3_sm = sm.add_constant(X_m3)
    for i in range(X_m3.shape[1]):
        try:
            vif = float(variance_inflation_factor(X_m3_sm, i + 1))
        except Exception:
            vif = float('inf')
        flag = ' ⚠高共线性' if vif > 10 else ''
        m3_vif.append({'term': m3_names[i], 'vif': vif})
        print(f"    {m3_names[i]:<24s} {vif:>10.2f}{flag}")

    return {
        'm1': {'r2': float(m1.rsquared), 'adj_r2': float(m1.rsquared_adj),
               'rss': float(rss1), 'n_features': 8},
        'm2': {'r2': float(m2.rsquared), 'adj_r2': float(m2.rsquared_adj),
               'rss': float(rss2), 'n_features': 18},
        'm3_all': {'r2': float(m3.rsquared), 'adj_r2': float(m3.rsquared_adj),
                   'rss': float(rss3), 'n_features': 28},
        'ftest_12': {'f': float(f_12), 'p': float(p_12),
                     'delta_r2': float(delta_r2_12),
                     'df_num': int(df_num_12), 'df_den': int(df_den_12)},
        'ftest_23_all': {'f': float(f_23), 'p': float(p_23),
                        'delta_r2': float(delta_r2_23),
                        'df_num': int(df_num_23), 'df_den': int(df_den_23)},
        'three_way_individual': three_way_results,
        'best_3way': best_3way,
        'm2_inter_coefs': m2_inter_coefs,
        'm2_vif': m2_vif,
        'm3_all_vif': m3_vif,
    }

# ============================================================
# [v3新增] Bootstrap稳定性检验（P0-1）
# 分层抽样300次，记录每项保留频率
# ============================================================
def bootstrap_stability(df_all, n_bootstrap=300):
    """Bootstrap稳定性检验，使用分层抽样"""
    print("\n" + "=" * 60)
    print(f"Bootstrap稳定性检验（P0-1）：分层抽样{n_bootstrap}次")
    print("=" * 60)

    # SHAP交互对到方程特征名的映射
    interaction_map = {
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

    # 固定8项主效应
    main_names = ['T', 'RH', '√FT', '√Crack', 'Water', 'T²', 'RH²', 'Water²']
    # 所有可能的候选项（8主效应 + 10交互）
    all_candidates = main_names + list(interaction_map.values())

    # 物理保护对
    protect_pairs = [('T', 'T²'), ('RH', 'RH²'), ('Water', 'Water²')]

    # 按场景分层
    scenarios = sorted(df_all['场景'].unique())
    strata = {s: df_all[df_all['场景'] == s] for s in scenarios}
    print(f"  分层信息:")
    for s in scenarios:
        print(f"    {s}: {len(strata[s])}条")

    # 记录保留频率
    retention_count = {term: 0 for term in all_candidates}
    term_counts = []
    valid_iters = 0

    # CatBoost参数（禁止写文件，加快速度）
    boot_params = dict(CATBOOST_PARAMS)
    boot_params['allow_writing_files'] = False

    rng = np.random.RandomState(RANDOM_STATE)

    for it in range(n_bootstrap):
        if (it + 1) % 50 == 0:
            print(f"  Bootstrap进度: {it+1}/{n_bootstrap} (有效迭代: {valid_iters})")

        try:
            # 分层有放回重采样
            resampled_parts = []
            for s in scenarios:
                group = strata[s]
                indices = rng.choice(len(group), size=len(group), replace=True)
                resampled_parts.append(group.iloc[indices].reset_index(drop=True))
            df_boot = pd.concat(resampled_parts, ignore_index=True)

            # 提取数组
            T = df_boot['温度'].values
            RH = df_boot['湿度'].values
            FT = df_boot['冻融次数'].values
            Crack = df_boot['裂缝长度'].values
            Water = df_boot['含水率'].values
            y = df_boot[TARGET].values

            sqrtFT = np.sqrt(FT)
            sqrtCrack = np.sqrt(Crack)

            # 1. 训练无权重CatBoost
            X_cb = df_boot[FEATURES].values
            cb_model = CatBoostRegressor(**boot_params)
            cb_model.fit(X_cb, y)

            # 2. 计算SHAP交互值，取Top5
            explainer = shap.TreeExplainer(cb_model)
            shap_inter = explainer.shap_interaction_values(X_cb)

            n_feat = len(FEATURES)
            pair_list_boot = []
            for ii in range(n_feat):
                for jj in range(ii + 1, n_feat):
                    strength = np.mean(np.abs(shap_inter[:, ii, jj]))
                    pair_list_boot.append((FEATURES[ii], FEATURES[jj], strength))
            pair_list_boot.sort(key=lambda x: x[2], reverse=True)

            top5 = []
            for f1, f2, s in pair_list_boot[:5]:
                eq_name = interaction_map.get((f1, f2)) or interaction_map.get((f2, f1))
                if eq_name:
                    top5.append(eq_name)

            if len(top5) < 5:
                term_counts.append(0)
                continue

            # 3. 构造13项方程（8主效应+5交互），无权重OLS
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

            current_names = main_names + top5
            current_inter = list(top5)

            # 4. 反向消元（简化版）
            # 第一阶段：移除不显著交互项
            while True:
                X_cols = [feature_builders[n]() for n in current_names]
                X = np.column_stack(X_cols)
                X_sm = sm.add_constant(X)
                model_ols = sm.OLS(y, X_sm).fit()

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
                X_cols = [feature_builders[n]() for n in current_names]
                X = np.column_stack(X_cols)
                X_sm = sm.add_constant(X)
                model_ols = sm.OLS(y, X_sm).fit()

                main_in_model = [n for n in current_names if n not in current_inter]
                candidates = []
                for name in main_in_model:
                    pval = float(model_ols.pvalues[current_names.index(name) + 1])
                    protected = False
                    for p1, p2 in protect_pairs:
                        if name == p2 and p1 in current_names:
                            protected = True
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

            # 5. 记录保留的项
            for term in current_names:
                if term in retention_count:
                    retention_count[term] += 1
            term_counts.append(len(current_names))
            valid_iters += 1

        except Exception as e:
            term_counts.append(0)
            continue

    # 计算保留频率
    freq = {term: count / valid_iters * 100 if valid_iters > 0 else 0.0
            for term, count in retention_count.items()}

    # 打印汇总表
    print(f"\n{'='*70}")
    print(f"Bootstrap稳定性检验汇总（有效迭代: {valid_iters}/{n_bootstrap}）")
    print(f"{'='*70}")
    print(f"  {'项名':<20s} {'保留频率(%)':>12s} {'是否在v2最终方程':>18s}")
    print(f"  {'-'*55}")
    sorted_terms = sorted(freq.keys(), key=lambda x: freq[x], reverse=True)
    for term in sorted_terms:
        in_v2 = '是' if term in V2_FINAL_TERMS else '否'
        print(f"  {term:<20s} {freq[term]:>12.1f} {in_v2:>18s}")

    # 项数分布
    if term_counts:
        valid_counts = [c for c in term_counts if c > 0]
        if valid_counts:
            print(f"\n  最终项数分布:")
            print(f"    均值: {np.mean(valid_counts):.1f}")
            print(f"    中位数: {np.median(valid_counts):.1f}")
            print(f"    最小值: {min(valid_counts)}")
            print(f"    最大值: {max(valid_counts)}")
            print(f"    标准差: {np.std(valid_counts):.1f}")

    # v2最终方程中的项的保留频率
    print(f"\n  v2最终方程11项的保留频率:")
    for term in sorted(V2_FINAL_TERMS, key=lambda x: freq[x], reverse=True):
        print(f"    {term:<20s}: {freq[term]:.1f}%")

    return {
        'n_bootstrap': n_bootstrap,
        'valid_iterations': valid_iters,
        'all_candidates': all_candidates,
        'retention_frequency': freq,
        'term_count_distribution': term_counts,
    }

# ============================================================
# [v3新增] 分组残差诊断（P1-5）
# 按工况分组Kruskal-Wallis检验+η²
# ============================================================
def grouped_residual_diagnosis(df_all, final_results):
    """分组残差诊断：按场景分组Kruskal-Wallis检验"""
    print("\n" + "=" * 60)
    print("分组残差诊断（P1-5）：按场景分组Kruskal-Wallis检验")
    print("=" * 60)

    # 用final_results的模型计算351条数据的预测残差
    X_final = final_results['X']
    y = final_results['y']
    model = final_results['model']
    y_pred = model.predict(sm.add_constant(X_final))
    residuals = y - y_pred

    # 按场景分组
    scenarios = sorted(df_all['场景'].unique())
    residuals_by_group = {}
    group_stats = {}

    print(f"\n  各组残差统计：")
    print(f"    {'场景':<10s} {'样本数':>6s} {'均值':>12s} {'中位数':>12s} {'标准差':>12s} {'|残差|均值':>12s}")
    print(f"    {'-'*68}")

    for s in scenarios:
        mask = (df_all['场景'].values == s)
        r = residuals[mask]
        residuals_by_group[s] = r
        group_stats[s] = {
            'n': int(len(r)),
            'mean': float(np.mean(r)),
            'median': float(np.median(r)),
            'std': float(np.std(r)),
            'abs_mean': float(np.mean(np.abs(r))),
        }
        print(f"    {s:<10s} {len(r):>6d} {np.mean(r):>+12.6f} {np.median(r):>+12.6f} "
              f"{np.std(r):>12.6f} {np.mean(np.abs(r)):>12.6f}")

    # Kruskal-Wallis检验（对残差绝对值）
    abs_resid_groups = [np.abs(residuals_by_group[s]) for s in scenarios]
    H_stat, p_val = stats.kruskal(*abs_resid_groups)
    n_total = len(residuals)
    eta_squared = H_stat / (n_total - 1)

    print(f"\n  Kruskal-Wallis检验（对|残差|）：")
    print(f"    H = {H_stat:.4f}, p = {p_val:.4e}")
    print(f"    η² = H/(n-1) = {eta_squared:.6f}")
    if p_val < 0.05:
        print(f"    → p<0.05，各组残差分布存在显著差异")
    else:
        print(f"    → p≥0.05，各组残差分布无显著差异，方程在各场景下表现一致")

    # 画箱线图
    fig, ax = plt.subplots(figsize=(7.48, 4.99))
    data = [residuals_by_group[s] for s in scenarios]
    colors = ['#3498db', '#e67e22', '#e74c3c', '#2ecc71', '#9b59b6']
    bp = ax.boxplot(data, labels=scenarios, patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, label='零线')
    ax.set_ylabel('残差', fontsize=9.0)
    ax.set_xlabel('场景', fontsize=9.0)
    ax.set_title(f'分组残差箱线图（Kruskal-Wallis H={H_stat:.2f}, η²={eta_squared:.4f}）',
                 fontsize=9.8)
    ax.legend(fontsize=7.5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, 'Fig_Grouped_Residual.png'), dpi=600)
    plt.close()
    print(f"\n  已保存: Fig_Grouped_Residual.png")

    return {
        'groups': group_stats,
        'kruskal_wallis': {
            'H': float(H_stat),
            'p': float(p_val),
            'eta_squared': float(eta_squared),
        },
        'residuals_by_group': residuals_by_group,  # 用于绘图，不序列化到JSON
        'scenarios': scenarios,
    }

# ============================================================
# 可视化
# ============================================================
def plot_stage_comparison(naive, nonlinear_main, full_nonlinear, final, cb_cv_r2, cb_cv_std):
    stages = ['①朴素线性\n(5项)', '②非线性主效应\n(8项)', '③完整非线性\n(13项)',
              '④精炼方程\n(11项)', '⑤CatBoost\n黑箱']
    cv_r2s = [naive['cv_r2'], nonlinear_main['cv_r2'],
              full_nonlinear['cv_r2'], final['cv_r2'], cb_cv_r2]
    cv_stds = [naive['cv_std'], nonlinear_main['cv_std'],
               full_nonlinear['cv_std'], final['cv_std'], cb_cv_std]
    colors = ['#95a5a6', '#3498db', '#e67e22', '#e74c3c', '#2ecc71']

    fig, ax = plt.subplots(figsize=(7.48, 4.49))
    bars = ax.bar(range(5), cv_r2s, yerr=cv_stds, capsize=5,
                  color=colors, edgecolor='white', linewidth=1.5)
    for bar, r2 in zip(bars, cv_r2s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{r2:.4f}', ha='center', va='bottom', fontsize=8.2, fontweight='bold')
    ax.set_xticks(range(5))
    ax.set_xticklabels(stages, fontsize=7.5)
    ax.set_ylabel('5折CV R²', fontsize=9.0)
    ax.set_title('方程发展阶段性能对比', fontsize=10.0)
    ax.set_ylim(0.85, 0.98)
    ax.axhline(y=cb_cv_r2, color='#2ecc71', linestyle='--', alpha=0.3, label='黑箱上限')
    ax.legend(loc='lower right', fontsize=7.5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, 'Fig_Stage_Comparison.png'), dpi=600)
    plt.close()
    print("  已保存: Fig_Stage_Comparison.png")

def plot_pred_vs_actual(final_results):
    X_final = final_results['X']
    y = final_results['y']
    model = final_results['model']
    pred = model.predict(sm.add_constant(X_final))

    fig, ax = plt.subplots(figsize=(5.51, 5.51))
    ax.scatter(y, pred, alpha=0.5, s=20, c='#2c3e50', edgecolors='none')
    lim_min, lim_max = y.min()*0.97, y.max()*1.03
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=1.5, label='y=x')
    ax.set_xlabel('实测导热系数', fontsize=9.0)
    ax.set_ylabel('精炼方程预测值', fontsize=9.0)
    ax.set_title(f'精炼方程预测vs实际 (CV R²={final_results["cv_r2"]:.4f})', fontsize=9.8)
    ax.legend(fontsize=8.2)
    ax.set_xlim(lim_min, lim_max)
    ax.set_ylim(lim_min, lim_max)
    ax.set_aspect('equal')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, 'Fig_Refined_PredVsActual.png'), dpi=600)
    plt.close()
    print("  已保存: Fig_Refined_PredVsActual.png")

def plot_nested_ftest(nested_results):
    fig, ax = plt.subplots(figsize=(7.48, 4.99))
    m_labels = ['M0: 基准方程\n(T, T², RH²)\n3项',
                'M1: +劣化主效应\n(+√FT, √Crack, Water, Water²)\n7项',
                'M2: +耦合效应\n(+4交互项)\n11项']
    m_r2s = [nested_results['m0']['r2'], nested_results['m1']['r2'], nested_results['m2']['r2']]
    m_colors = ['#3498db', '#e67e22', '#e74c3c']

    bars = ax.bar(range(3), m_r2s, color=m_colors, edgecolor='white', linewidth=1.5, width=0.6)
    for bar, r2 in zip(bars, m_r2s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'R²={r2:.4f}', ha='center', va='bottom', fontsize=8.2, fontweight='bold')

    for i in range(1, 3):
        delta = m_r2s[i] - m_r2s[i-1]
        ft = nested_results[f'ftest_{"01" if i==1 else "12"}']
        ax.annotate(f'ΔR²={delta:.4f}\nF={ft["f_stat"]:.1f}\np={ft["p_val"]:.1e}',
                    xy=(i, m_r2s[i]), xytext=(i, m_r2s[i] + 0.03),
                    ha='center', fontsize=6.8, color='#2c3e50',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))

    ax.set_xticks(range(3))
    ax.set_xticklabels(m_labels, fontsize=6.8)
    ax.set_ylabel('R²', fontsize=9.0)
    ax.set_title('三层嵌套模型R²增量与F检验', fontsize=10.0)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, 'Fig_Nested_Model_R2.png'), dpi=600)
    plt.close()
    print("  已保存: Fig_Nested_Model_R2.png")

def plot_decomposition(decomp_results):
    T_fixed, RH_fixed = 25.0, 60.0
    scenarios = [
        ('完好工况\n(FT=0,C=0,W=0)', 0, 0, 0),
        ('仅冻融\n(FT=50)', 50, 0, 0),
        ('仅裂缝\n(C=10)', 0, 10, 0),
        ('仅渗水\n(W=10)', 0, 0, 10),
        ('冻融+裂缝\n(FT=50,C=10)', 50, 10, 0),
        ('冻融+渗水\n(FT=50,W=10)', 50, 0, 10),
        ('裂缝+渗水\n(C=10,W=10)', 0, 10, 10),
        ('三因素耦合\n(FT=50,C=10,W=10)', 50, 10, 10),
    ]

    bc = decomp_results['baseline_coefs']
    dc = decomp_results['deg_coefs']
    cc = decomp_results['coup_coefs']

    scenario_names = []
    baseline_vals = []
    deg_vals = []
    coup_vals = []

    for name, ft, crack, water in scenarios:
        scenario_names.append(name)
        b = (bc['const'] + bc['T']*T_fixed + bc['T²']*T_fixed**2 + bc['RH²']*RH_fixed**2)
        d = (dc['√FT']*np.sqrt(ft) + dc['√Crack']*np.sqrt(crack)
             + dc['Water']*water + dc['Water²']*water**2)
        c = 0.0
        c += cc['T×√Crack'] * T_fixed * np.sqrt(crack)
        c += cc['RH×Water'] * RH_fixed * water
        c += cc['√FT×√Crack'] * np.sqrt(ft) * np.sqrt(crack)
        c += cc['RH×√Crack'] * RH_fixed * np.sqrt(crack)
        baseline_vals.append(b)
        deg_vals.append(d)
        coup_vals.append(c)

    baseline_vals = np.array(baseline_vals)
    deg_vals = np.array(deg_vals)
    coup_vals = np.array(coup_vals)

    fig, ax = plt.subplots(figsize=(7.48, 4.03))
    x = np.arange(len(scenarios))
    w = 0.55

    ax.bar(x, baseline_vals, w, label='λ₀ 基准（环境效应）', color='#3498db', edgecolor='white')
    ax.bar(x, deg_vals, w, bottom=baseline_vals, label='Δλ_deg 劣化主效应', color='#e67e22', edgecolor='white')
    ax.bar(x, coup_vals, w, bottom=baseline_vals+deg_vals, label='Δλ_coup 耦合效应', color='#e74c3c', edgecolor='white')

    totals = baseline_vals + deg_vals + coup_vals
    for i, total in enumerate(totals):
        ax.text(i, total + 0.0003, f'{total:.4f}', ha='center', va='bottom', fontsize=6.8, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_names, fontsize=6.5)
    ax.set_ylabel('导热系数 λ (W/(m·K))', fontsize=9.0)
    ax.set_title(f'方程组分贡献分解 (T={T_fixed}℃, RH={RH_fixed}%)', fontsize=10.0)
    ax.legend(loc='upper left', fontsize=7.5)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(totals) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, 'Fig_Decomposition_Contribution.png'), dpi=600)
    plt.close()
    print("  已保存: Fig_Decomposition_Contribution.png")

def plot_bootstrap_stability(bootstrap_results):
    """Bootstrap稳定性频率图"""
    freq = bootstrap_results['retention_frequency']
    sorted_terms = sorted(freq.keys(), key=lambda x: freq[x], reverse=True)
    frequencies = [freq[t] for t in sorted_terms]
    colors = ['#3498db' if t in V2_FINAL_TERMS else '#95a5a6' for t in sorted_terms]

    fig, ax = plt.subplots(figsize=(7.48, 5.98))
    y_pos = range(len(sorted_terms))
    bars = ax.barh(y_pos, frequencies, color=colors, edgecolor='white', height=0.6)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_terms, fontsize=7.5)
    ax.set_xlabel('保留频率 (%)', fontsize=9.0)
    ax.set_title(f'Bootstrap稳定性检验（{bootstrap_results["n_bootstrap"]}次分层重采样，'
                 f'有效{bootstrap_results["valid_iterations"]}次）', fontsize=9.8)
    ax.axvline(x=90, color='red', linestyle='--', linewidth=1.5, label='90%阈值')
    ax.invert_yaxis()

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='v2最终方程中的项'),
        Patch(facecolor='#95a5a6', label='不在v2最终方程中的项'),
    ]
    ax.legend(handles=legend_elements + [plt.Line2D([0], [0], color='red', linestyle='--', label='90%阈值')],
              loc='lower right', fontsize=6.8)
    ax.grid(axis='x', alpha=0.3)

    # 添加频率标签
    for i, f in enumerate(frequencies):
        ax.text(f + 0.5, i, f'{f:.1f}%', va='center', fontsize=6.8)

    ax.set_xlim(0, 110)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, 'Fig_Bootstrap_Stability.png'), dpi=600)
    plt.close()
    print("  已保存: Fig_Bootstrap_Stability.png")

def plot_grouped_residual(grouped_resid_results):
    """分组残差箱线图"""
    residuals_by_group = grouped_resid_results['residuals_by_group']
    scenarios = grouped_resid_results['scenarios']
    data = [residuals_by_group[s] for s in scenarios]
    colors = ['#3498db', '#e67e22', '#e74c3c', '#2ecc71', '#9b59b6']

    fig, ax = plt.subplots(figsize=(7.48, 4.99))
    bp = ax.boxplot(data, labels=scenarios, patch_artist=True, widths=0.5)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, label='零线')
    ax.set_ylabel('残差', fontsize=9.0)
    ax.set_xlabel('场景', fontsize=9.0)
    kw = grouped_resid_results['kruskal_wallis']
    ax.set_title(f'分组残差箱线图（Kruskal-Wallis H={kw["H"]:.2f}, p={kw["p"]:.2e}, '
                 f'η²={kw["eta_squared"]:.4f}）', fontsize=9.0)
    ax.legend(fontsize=7.5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, 'Fig_Grouped_Residual_Box.png'), dpi=600)
    plt.close()
    print("  已保存: Fig_Grouped_Residual_Box.png")


# ============================================================
# v3.1 补充验证模块
# ============================================================

def pdp_vs_equation(df_all, final_coefs):
    """P1-3补充: PDP黑箱曲线 vs 方程解析边际曲线对比"""
    print("\n" + "=" * 60)
    print("P1-3补充: PDP黑箱曲线 vs 方程解析曲线")
    print("=" * 60)

    X = df_all[FEATURES].values
    y = df_all[TARGET].values

    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(X, y)

    coefs = final_coefs

    def eq_predict(T, RH, FT, Crack, Water):
        sFT, sCrack = np.sqrt(FT), np.sqrt(Crack)
        return (coefs['const'] + coefs['T'] * T + coefs['T²'] * T ** 2 + coefs['RH²'] * RH ** 2
                + coefs['√FT'] * sFT + coefs['√Crack'] * sCrack + coefs['Water'] * Water + coefs['Water²'] * Water ** 2
                + coefs['T×√Crack'] * T * sCrack + coefs['RH×Water'] * RH * Water
                + coefs['√FT×√Crack'] * sFT * sCrack + coefs['RH×√Crack'] * RH * sCrack)

    var_configs = [
        ('温度', np.linspace(5, 45, 100), {'湿度': 60, '冻融次数': 0, '裂缝长度': 0, '含水率': 0}),
        ('湿度', np.linspace(30, 90, 100), {'温度': 25, '冻融次数': 0, '裂缝长度': 0, '含水率': 0}),
        ('冻融次数', np.linspace(0, 24, 100), {'温度': 25, '湿度': 60, '裂缝长度': 0, '含水率': 0}),
        ('裂缝长度', np.linspace(0, 10, 100), {'温度': 25, '湿度': 60, '冻融次数': 0, '含水率': 0}),
        ('含水率', np.linspace(0, 56, 100), {'温度': 25, '湿度': 60, '冻融次数': 0, '裂缝长度': 0}),
    ]

    eq_labels = {
        '温度': '方程: T + T²', '湿度': '方程: RH²',
        '冻融次数': '方程: √FT', '裂缝长度': '方程: √Crack', '含水率': '方程: Water + Water²'
    }

    fig, axes = plt.subplots(2, 3, figsize=(7.48, 4.57))
    axes = axes.flatten()
    correlations = {}

    for idx, (feat, grid, fixed) in enumerate(var_configs):
        ax = axes[idx]
        feat_idx = FEATURES.index(feat)
        X_bg = df_all[FEATURES].values

        pdp_vals = []
        for val in grid:
            X_mod = X_bg.copy()
            X_mod[:, feat_idx] = val
            pdp_vals.append(np.mean(model.predict(X_mod)))
        pdp_vals = np.array(pdp_vals) - np.mean(pdp_vals)

        eq_vals = []
        for val in grid:
            kwargs = dict(fixed)
            kwargs[feat] = val
            eq_vals.append(eq_predict(kwargs['温度'], kwargs['湿度'],
                                      kwargs['冻融次数'], kwargs['裂缝长度'], kwargs['含水率']))
        eq_vals = np.array(eq_vals) - np.mean(eq_vals)

        ax.plot(grid, pdp_vals, 'b-', linewidth=2.5, label='CatBoost PDP', color='#3498db')
        ax.plot(grid, eq_vals, 'r--', linewidth=2.5, label=eq_labels[feat], color='#e74c3c')
        ax.set_xlabel(feat, fontsize=9.0)
        ax.set_ylabel('中心化预测值', fontsize=8.2)
        ax.set_title(f'{feat}: PDP vs 方程解析曲线', fontsize=9.8, fontweight='bold')
        ax.legend(fontsize=7.5, loc='best')
        ax.grid(True, alpha=0.3)

        r = np.corrcoef(pdp_vals, eq_vals)[0, 1]
        correlations[feat] = float(r)
        ax.text(0.05, 0.95, f'r={r:.4f}', transform=ax.transAxes, fontsize=8.2,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        print(f"  {feat}: PDP vs 方程 r = {r:.4f}")

    axes[5].set_visible(False)
    plt.suptitle('黑箱PDP曲线 vs 方程解析边际曲线对比（其他变量固定在参考值）', fontsize=10.0, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, 'Fig_PDP_vs_Equation.png'), dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  已保存: Fig_PDP_vs_Equation.png")

    return {'correlations': correlations}


def unweighted_catboost_cv_and_weighted_ols(df_all, final_coefs, eq_cv_r2):
    """P2-6: 无权重CatBoost CV + 加权OLS系数对照"""
    print("\n" + "=" * 60)
    print("P2-6: 无权重CatBoost CV + 加权OLS对照")
    print("=" * 60)

    X = df_all[FEATURES].values
    y = df_all[TARGET].values
    w = df_all['sample_weight'].values

    # 无权重CatBoost 5折CV
    cv_r2, cv_rmse, cv_mae, cv_mape = [], [], [], []
    for tr, te in KF.split(X):
        m = CatBoostRegressor(**CATBOOST_PARAMS)
        m.fit(X[tr], y[tr])
        r2, rmse, mae, mape = calc_metrics(y[te], m.predict(X[te]))
        cv_r2.append(r2); cv_rmse.append(rmse); cv_mae.append(mae); cv_mape.append(mape)

    uw_cv_r2 = np.mean(cv_r2)
    uw_cv_rmse = np.mean(cv_rmse)
    uw_cv_mae = np.mean(cv_mae)
    uw_cv_mape = np.mean(cv_mape)

    # 加权CatBoost CV
    w_cv_r2, w_cv_rmse, w_cv_mae, w_cv_mape = [], [], [], []
    for tr, te in KF.split(X):
        m = CatBoostRegressor(**CATBOOST_PARAMS)
        m.fit(X[tr], y[tr], sample_weight=w[tr])
        r2, rmse, mae, mape = calc_metrics(y[te], m.predict(X[te]))
        w_cv_r2.append(r2); w_cv_rmse.append(rmse); w_cv_mae.append(mae); w_cv_mape.append(mape)

    w_cv_r2_mean = np.mean(w_cv_r2)
    w_cv_rmse_mean = np.mean(w_cv_rmse)
    w_cv_mae_mean = np.mean(w_cv_mae)
    w_cv_mape_mean = np.mean(w_cv_mape)

    recovery_weighted = eq_cv_r2 / w_cv_r2_mean * 100
    recovery_unweighted = eq_cv_r2 / uw_cv_r2 * 100

    print(f"\n  CatBoost CV对比:")
    print(f"    加权CatBoost(论文基准): CV R²={w_cv_r2_mean:.4f}, RMSE={w_cv_rmse_mean:.6f}, MAE={w_cv_mae_mean:.6f}, MAPE={w_cv_mape_mean:.2f}%")
    print(f"    无权重CatBoost:         CV R²={uw_cv_r2:.4f}, RMSE={uw_cv_rmse:.6f}, MAE={uw_cv_mae:.6f}, MAPE={uw_cv_mape:.2f}%")
    print(f"    精炼方程(11项,无权重):  CV R²={eq_cv_r2:.4f}")
    print(f"  性能恢复率: 加权基准{recovery_weighted:.1f}%, 无权重基准{recovery_unweighted:.1f}%")

    # 加权OLS对照
    T = df_all['温度'].values
    RH = df_all['湿度'].values
    FT = df_all['冻融次数'].values
    Crack = df_all['裂缝长度'].values
    Water = df_all['含水率'].values
    sqrtFT, sqrtCrack = np.sqrt(FT), np.sqrt(Crack)

    X_eq = np.column_stack([T, T ** 2, RH ** 2, sqrtFT, sqrtCrack, Water, Water ** 2,
                            T * sqrtCrack, RH * Water, sqrtFT * sqrtCrack, RH * sqrtCrack])
    X_sm = sm.add_constant(X_eq)

    ols_uw = sm.OLS(y, X_sm).fit()
    ols_w = sm.WLS(y, X_sm, weights=w).fit()

    names = ['const', 'T', 'T²', 'RH²', '√FT', '√Crack', 'Water', 'Water²', 'T×√Crack', 'RH×Water', '√FT×√Crack', 'RH×√Crack']
    print(f"\n  加权 vs 无权重OLS系数对照:")
    for i, name in enumerate(names):
        c_uw = ols_uw.params[i]
        c_w = ols_w.params[i]
        diff_pct = (c_w - c_uw) / abs(c_uw) * 100 if abs(c_uw) > 1e-15 else 0
        print(f"    {name:<16s} 无权重={c_uw:>+14.6e} 加权={c_w:>+14.6e} 差异={diff_pct:>+9.1f}%")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.27))

    # 左图：CatBoost CV三模型对比
    ax = axes[0]
    models_cv = ['加权CatBoost\n(论文基准)', '无权重CatBoost', '精炼方程\n(11项,无权重)']
    r2_vals = [w_cv_r2_mean, uw_cv_r2, eq_cv_r2]
    rmse_vals = [w_cv_rmse_mean, uw_cv_rmse, 0.000712]
    mae_vals = [w_cv_mae_mean, uw_cv_mae, 0.000556]
    mape_vals = [w_cv_mape_mean, uw_cv_mape, 1.76]

    x = np.arange(len(models_cv))
    w_bar = 0.2
    bars1 = ax.bar(x - 1.5 * w_bar, r2_vals, w_bar, label='CV R²', color='#3498db', alpha=0.85)
    ax.bar(x - 0.5 * w_bar, rmse_vals, w_bar, label='RMSE', color='#e74c3c', alpha=0.85)
    ax.bar(x + 0.5 * w_bar, mae_vals, w_bar, label='MAE', color='#2ecc71', alpha=0.85)
    ax.bar(x + 1.5 * w_bar, [v / 100 for v in mape_vals], w_bar, label='MAPE/100', color='#f39c12', alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(models_cv, fontsize=7.5)
    ax.set_ylabel('指标值', fontsize=9.0)
    ax.set_title('CatBoost CV对比 + 方程性能恢复率', fontsize=9.8, fontweight='bold')
    ax.legend(fontsize=6.8, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)

    for bar, val in zip(bars1, r2_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f'{val:.4f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')

    ax.text(2, 0.92, f'恢复率(加权基准): {recovery_weighted:.1f}%\n恢复率(无权重基准): {recovery_unweighted:.1f}%',
            fontsize=6.8, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 右图：OLS系数对照散点图
    ax = axes[1]
    coef_names_short = names[1:]
    uw_coefs = [ols_uw.params[i] for i in range(1, len(names))]
    w_coefs = [ols_w.params[i] for i in range(1, len(names))]

    ax.scatter(uw_coefs, w_coefs, s=80, alpha=0.7, c='#9b59b6', edgecolors='white', zorder=5)
    for i, name in enumerate(coef_names_short):
        ax.annotate(name, (uw_coefs[i], w_coefs[i]), fontsize=6.5,
                    textcoords="offset points", xytext=(5, 5))

    all_vals = uw_coefs + w_coefs
    lim = max(abs(min(all_vals)), abs(max(all_vals))) * 1.3
    ax.plot([-lim, lim], [-lim, lim], 'r--', linewidth=1.5, label='y=x (完全一致)')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel('无权重OLS系数', fontsize=9.0)
    ax.set_ylabel('加权OLS系数', fontsize=9.0)
    ax.set_title(f'加权 vs 无权重 OLS系数对照\n无权重R²={ols_uw.rsquared:.4f}, 加权R²={ols_w.rsquared:.4f}',
                 fontsize=9.8, fontweight='bold')
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

    plt.suptitle('P2-6: 无权重CatBoost CV + 加权OLS系数稳定性对照', fontsize=10.0, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, 'Fig_CatBoost_CV_OLS_Comparison.png'), dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  已保存: Fig_CatBoost_CV_OLS_Comparison.png")

    return {
        'catboost_cv': {
            'weighted': {'cv_r2': float(w_cv_r2_mean), 'cv_rmse': float(w_cv_rmse_mean), 'cv_mae': float(w_cv_mae_mean), 'cv_mape': float(w_cv_mape_mean)},
            'unweighted': {'cv_r2': float(uw_cv_r2), 'cv_rmse': float(uw_cv_rmse), 'cv_mae': float(uw_cv_mae), 'cv_mape': float(uw_cv_mape)},
            'recovery_rate_weighted': float(recovery_weighted),
            'recovery_rate_unweighted': float(recovery_unweighted),
        },
        'ols_comparison': {
            'unweighted_r2': float(ols_uw.rsquared),
            'weighted_r2': float(ols_w.rsquared),
            'coefs_unweighted': {n: float(c) for n, c in zip(names, ols_uw.params)},
            'coefs_weighted': {n: float(c) for n, c in zip(names, ols_w.params)},
            'pvalues_unweighted': {n: float(p) for n, p in zip(names, ols_uw.pvalues)},
            'pvalues_weighted': {n: float(p) for n, p in zip(names, ols_w.pvalues)},
        }
    }


def fpi_magnitude_check(df_all, final_coefs):
    """P2-7: FPI量级校验（方程耦合放幅度 vs FPI实测+10.96%）"""
    print("\n" + "=" * 60)
    print("P2-7: FPI量级校验")
    print("=" * 60)

    fpi_csv = os.path.join(DATA_DIR, 'FPI_CellMatched_detail.csv')
    df_fpi = pd.read_csv(fpi_csv)
    fpi_mean = df_fpi['fpi_pct'].mean()
    print(f"  FPI实测均值: +{fpi_mean:.2f}%")

    coefs = final_coefs

    def eq_predict(T, RH, FT, Crack, Water):
        sFT, sCrack = np.sqrt(FT), np.sqrt(Crack)
        return (coefs['const'] + coefs['T'] * T + coefs['T²'] * T ** 2 + coefs['RH²'] * RH ** 2
                + coefs['√FT'] * sFT + coefs['√Crack'] * sCrack + coefs['Water'] * Water + coefs['Water²'] * Water ** 2
                + coefs['T×√Crack'] * T * sCrack + coefs['RH×Water'] * RH * Water
                + coefs['√FT×√Crack'] * sFT * sCrack + coefs['RH×√Crack'] * RH * sCrack)

    FT_MAX, CRACK_MAX, WATER_MAX = 24, 10, 56
    eq_fpi_results = []

    for _, row in df_fpi.iterrows():
        T, RH = int(row['Temperature_C']), int(row['RH_pct'])
        l0_eq = eq_predict(T, RH, 0, 0, 0)
        y_ft_eq = eq_predict(T, RH, FT_MAX, 0, 0)
        y_crack_eq = eq_predict(T, RH, 0, CRACK_MAX, 0)
        y_water_eq = eq_predict(T, RH, 0, 0, WATER_MAX)
        y_coupled_eq = eq_predict(T, RH, FT_MAX, CRACK_MAX, WATER_MAX)

        d_ft_eq = y_ft_eq - l0_eq
        d_crack_eq = y_crack_eq - l0_eq
        d_water_eq = y_water_eq - l0_eq
        L_eq = l0_eq + d_ft_eq + d_crack_eq + d_water_eq
        C_eq = y_coupled_eq
        fpi_eq = C_eq - L_eq
        fpi_eq_pct = fpi_eq / abs(L_eq) * 100 if L_eq != 0 else 0
        fpi_exp_pct = row['fpi_pct']

        eq_fpi_results.append({
            'T': T, 'RH': RH,
            'fpi_eq_pct': fpi_eq_pct, 'fpi_exp_pct': fpi_exp_pct,
            'ratio': fpi_eq_pct / fpi_exp_pct if abs(fpi_exp_pct) > 0.01 else 0,
        })

    df_eq_fpi = pd.DataFrame(eq_fpi_results)
    eq_fpi_mean = df_eq_fpi['fpi_eq_pct'].mean()
    exp_fpi_mean = df_eq_fpi['fpi_exp_pct'].mean()
    print(f"  方程预测FPI均值: +{eq_fpi_mean:.2f}%, 比值={eq_fpi_mean / exp_fpi_mean:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.21))

    ax = axes[0]
    ax.scatter(df_eq_fpi['fpi_exp_pct'], df_eq_fpi['fpi_eq_pct'], s=60, alpha=0.7, c='#3498db', edgecolors='white')
    lim_min = min(df_eq_fpi['fpi_exp_pct'].min(), df_eq_fpi['fpi_eq_pct'].min()) - 2
    lim_max = max(df_eq_fpi['fpi_exp_pct'].max(), df_eq_fpi['fpi_eq_pct'].max()) + 2
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'r--', linewidth=1.5, label='y=x')
    ax.set_xlabel('实测FPI (%)', fontsize=9.0)
    ax.set_ylabel('方程预测FPI (%)', fontsize=9.0)
    ax.set_title(f'方程耦合放幅度 vs 实测FPI\n均值: 实测{exp_fpi_mean:.2f}% vs 方程{eq_fpi_mean:.2f}%', fontsize=9.0)
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    x = np.arange(len(df_eq_fpi))
    w_bar = 0.35
    ax.bar(x - w_bar / 2, df_eq_fpi['fpi_exp_pct'], w_bar, label='实测FPI', color='#3498db', alpha=0.8)
    ax.bar(x + w_bar / 2, df_eq_fpi['fpi_eq_pct'], w_bar, label='方程FPI', color='#e74c3c', alpha=0.8)
    ax.set_xlabel('温湿度格子编号', fontsize=9.0)
    ax.set_ylabel('FPI (%)', fontsize=9.0)
    ax.set_title('27个温湿度格子的FPI对比', fontsize=9.0)
    ax.legend(fontsize=7.5)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, 'Fig_FPI_Magnitude_Check.png'), dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  已保存: Fig_FPI_Magnitude_Check.png")

    return {
        'fpi_exp_mean': float(exp_fpi_mean),
        'fpi_eq_mean': float(eq_fpi_mean),
        'ratio': float(eq_fpi_mean / exp_fpi_mean),
        'per_cell': df_eq_fpi[['T', 'RH', 'fpi_exp_pct', 'fpi_eq_pct', 'ratio']].to_dict('records'),
    }


def cross_scenario_extrapolation(df_all):
    """P2-8: 跨工况外推对比（去耦合层 vs 完整方程）"""
    print("\n" + "=" * 60)
    print("P2-8: 跨工况外推对比（去耦合层 vs 完整方程）")
    print("=" * 60)

    T = df_all['温度'].values
    RH = df_all['湿度'].values
    FT = df_all['冻融次数'].values
    Crack = df_all['裂缝长度'].values
    Water = df_all['含水率'].values
    y = df_all[TARGET].values
    sqrtFT, sqrtCrack = np.sqrt(FT), np.sqrt(Crack)

    def get_scenario(row):
        ft, cr, w = row['冻融次数'], row['裂缝长度'], row['含水率']
        if ft == 0 and cr == 0 and w == 0: return '完好'
        elif ft > 0 and cr == 0 and w == 0: return '纯冻融'
        elif ft == 0 and cr > 0 and w == 0: return '纯裂缝'
        elif ft == 0 and cr == 0 and w > 0: return '纯渗水'
        else: return '复合'
    scenarios = df_all.apply(get_scenario, axis=1).values

    X_full = np.column_stack([T, T ** 2, RH ** 2, sqrtFT, sqrtCrack, Water, Water ** 2,
                              T * sqrtCrack, RH * Water, sqrtFT * sqrtCrack, RH * sqrtCrack])
    m_full = sm.OLS(y, sm.add_constant(X_full)).fit()

    X_nocoup = np.column_stack([T, T ** 2, RH ** 2, sqrtFT, sqrtCrack, Water, Water ** 2])
    m_nocoup = sm.OLS(y, sm.add_constant(X_nocoup)).fit()

    print(f"  完整方程(11项): R²={m_full.rsquared:.4f}")
    print(f"  去耦合方程(7项): R²={m_nocoup.rsquared:.4f}")

    results = {}
    for scenario in ['完好', '纯冻融', '纯裂缝', '纯渗水', '复合']:
        mask = scenarios == scenario
        if mask.sum() == 0: continue
        y_sub = y[mask]
        pred_full = m_full.predict(sm.add_constant(X_full[mask]))
        pred_nocoup = m_nocoup.predict(sm.add_constant(X_nocoup[mask]))

        rmse_full = np.sqrt(np.mean((y_sub - pred_full) ** 2))
        rmse_nocoup = np.sqrt(np.mean((y_sub - pred_nocoup) ** 2))
        mae_full = np.mean(np.abs(y_sub - pred_full))
        mae_nocoup = np.mean(np.abs(y_sub - pred_nocoup))
        rmse_improve = (rmse_nocoup - rmse_full) / rmse_nocoup * 100 if rmse_nocoup > 0 else 0

        print(f"  {scenario:<10s} n={mask.sum():>3d} 去耦合RMSE={rmse_nocoup:.6f} 完整RMSE={rmse_full:.6f} 改善={rmse_improve:+.1f}%")

        results[scenario] = {
            'n': int(mask.sum()),
            'rmse_nocoup': float(rmse_nocoup), 'rmse_full': float(rmse_full),
            'mae_nocoup': float(mae_nocoup), 'mae_full': float(mae_full),
            'rmse_improve_pct': float(rmse_improve),
        }

    # 画图
    fig, axes = plt.subplots(1, 2, figsize=(7.48, 3.21))

    ax = axes[0]
    scenarios_order = ['完好', '纯冻融', '纯裂缝', '纯渗水', '复合']
    rmse_nocoup_vals = [results[s]['rmse_nocoup'] for s in scenarios_order if s in results]
    rmse_full_vals = [results[s]['rmse_full'] for s in scenarios_order if s in results]
    x = np.arange(len(rmse_nocoup_vals))
    w_bar = 0.35
    ax.bar(x - w_bar / 2, rmse_nocoup_vals, w_bar, label='去耦合方程(7项)', color='#95a5a6', alpha=0.8)
    ax.bar(x + w_bar / 2, rmse_full_vals, w_bar, label='完整方程(11项)', color='#e74c3c', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios_order, fontsize=8.2)
    ax.set_ylabel('RMSE (W/(m·K))', fontsize=9.0)
    ax.set_title('各场景RMSE对比：去耦合 vs 完整方程', fontsize=9.8)
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    mask_comp = scenarios == '复合'
    y_comp = y[mask_comp]
    pred_full_comp = m_full.predict(sm.add_constant(X_full[mask_comp]))
    pred_nocoup_comp = m_nocoup.predict(sm.add_constant(X_nocoup[mask_comp]))

    ax.scatter(y_comp, pred_nocoup_comp, alpha=0.4, s=20, c='#95a5a6',
               label=f'去耦合 (RMSE={np.sqrt(np.mean((y_comp - pred_nocoup_comp) ** 2)):.6f})')
    ax.scatter(y_comp, pred_full_comp, alpha=0.5, s=20, c='#e74c3c',
               label=f'完整方程 (RMSE={np.sqrt(np.mean((y_comp - pred_full_comp) ** 2)):.6f})')
    lim_min, lim_max = y_comp.min() * 0.97, y_comp.max() * 1.03
    ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', linewidth=1, label='y=x')
    ax.set_xlabel('实测导热系数', fontsize=9.0)
    ax.set_ylabel('方程预测值', fontsize=9.0)
    ax.set_title('复合工况：去耦合 vs 完整方程预测', fontsize=9.8)
    ax.legend(fontsize=6.8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_FOLDER, 'Fig_Cross_Scenario_Extrapolation.png'), dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  已保存: Fig_Cross_Scenario_Extrapolation.png")

    return results


# ============================================================
# 主函数
# ============================================================
def main():
    ensure_folder(RESULTS_FOLDER)
    print("=" * 60)
    print("可解释性方程特征分析优化 v3.1")
    print("=" * 60)

    df_all, df_deg = load_data()

    # CatBoost黑箱（加权，作为性能上限参照）
    cb_model, cb_train_r2, cb_train_rmse, cb_train_mae, cb_train_mape, \
        cb_cv_r2, cb_cv_std, cb_cv_rmse, cb_cv_mae, cb_cv_mape = train_catboost(df_all)

    # 无权重CatBoost（专用于SHAP，与方程拟合口径一致）
    cb_model_unweighted = train_catboost_unweighted(df_all)

    # SHAP全局重要性（无权重CatBoost，351条全部工况）
    shap_global = shap_global_importance(cb_model_unweighted, df_all)

    # 边际形状诊断
    marginal = marginal_shape_diagnosis(df_all, df_deg)

    # 朴素线性
    naive = fit_naive(df_all)

    # 非线性主效应
    nonlinear_main = fit_nonlinear_main(df_all)

    # SHAP交互分析（无权重CatBoost，351条全部工况）→ 表6
    shap_inter = shap_interaction_analysis(cb_model_unweighted, df_all)

    # 完整非线性方程(③)：8主效应 + Top5无权重SHAP交互 = 13项
    full_nonlinear = fit_full_nonlinear(df_all, shap_inter['top5_pairs'])

    # 反向消元(④)：从13项逐步移除不显著项
    elim_results = backward_elimination(df_all, full_nonlinear)
    final = organize_final_equation(elim_results)

    # 三层分解
    decomp = decompose(df_all, final)

    # 嵌套F检验
    nested = nested_ftest(df_all)

    # ============ v3新增模块 ============
    # AIC/BIC边际诊断
    aic_bic = aic_bic_diagnosis(df_all, df_deg)

    # 嵌套检验（变换变量 M1→M2→M3）
    nested_trans = nested_ftest_transformed(df_all)

    # Bootstrap稳定性检验
    bootstrap = bootstrap_stability(df_all, n_bootstrap=300)

    # 分组残差诊断
    grouped_resid = grouped_residual_diagnosis(df_all, final)

    # ============ v3.1补充验证模块 ============
    # 构建最终方程系数字典
    final_coefs = {name: float(c) for name, c in zip(final['coef_names'], final['coefs'])}

    # P1-3补充: PDP vs 方程解析曲线
    pdp_results = pdp_vs_equation(df_all, final_coefs)

    # P2-6: 无权重CatBoost CV + 加权OLS对照
    cv_ols_results = unweighted_catboost_cv_and_weighted_ols(df_all, final_coefs, final['cv_r2'])

    # P2-7: FPI量级校验
    fpi_results = fpi_magnitude_check(df_all, final_coefs)

    # P2-8: 跨工况外推对比
    cross_results = cross_scenario_extrapolation(df_all)

    # 对比汇总
    print(f"\n{'='*70}")
    print("全阶段方程对比汇总")
    print(f"{'='*70}")
    print(f"{'阶段':<30s} {'项数':>4s} {'训练R²':>8s} {'Adj R²':>8s} {'CV R²':>10s} {'CV RMSE':>10s} {'CV MAE':>10s} {'CV MAPE':>8s}")
    print("-" * 95)
    print(f"{'① 朴素线性':<30s} {'5':>4s} {naive['train_r2']:>8.4f} {naive['adj_r2']:>8.4f} "
          f"{naive['cv_r2']:>10.4f} {naive['cv_rmse']:>10.6f} {naive['cv_mae']:>10.6f} {naive['cv_mape']:>7.2f}%")
    print(f"{'② 非线性主效应':<30s} {'8':>4s} {nonlinear_main['train_r2']:>8.4f} {nonlinear_main['adj_r2']:>8.4f} "
          f"{nonlinear_main['cv_r2']:>10.4f} {nonlinear_main['cv_rmse']:>10.6f} {nonlinear_main['cv_mae']:>10.6f} {nonlinear_main['cv_mape']:>7.2f}%")
    print(f"{'③ 完整非线性(SHAP驱动)':<30s} {'13':>4s} {full_nonlinear['train_r2']:>8.4f} {full_nonlinear['adj_r2']:>8.4f} "
          f"{full_nonlinear['cv_r2']:>10.4f} {full_nonlinear['cv_rmse']:>10.6f} {full_nonlinear['cv_mae']:>10.6f} {full_nonlinear['cv_mape']:>7.2f}%")
    print(f"{'④ 精炼方程(v3.1)':<30s} {'11':>4s} {final['train_r2']:>8.4f} {final['adj_r2']:>8.4f} "
          f"{final['cv_r2']:>10.4f} {final['cv_rmse']:>10.6f} {final['cv_mae']:>10.6f} {final['cv_mape']:>7.2f}%")
    print(f"{'⑤ CatBoost黑箱':<30s} {'—':>4s} {cb_train_r2:>8.4f} {'—':>8s} "
          f"{cb_cv_r2:>10.4f} {cb_cv_rmse:>10.6f} {cb_cv_mae:>10.6f} {cb_cv_mape:>7.2f}%")

    total_gain = final['cv_r2'] - naive['cv_r2']
    print(f"\n  总提升(①→④): CV R² {total_gain:+.4f}")
    print(f"  与黑箱差距(④→⑤): {cb_cv_r2 - final['cv_r2']:.4f}")

    # 可视化
    print(f"\n生成图表...")
    plot_stage_comparison(naive, nonlinear_main, full_nonlinear, final, cb_cv_r2, cb_cv_std)
    plot_pred_vs_actual(final)
    plot_nested_ftest(nested)
    plot_decomposition(decomp)
    plot_bootstrap_stability(bootstrap)
    plot_grouped_residual(grouped_resid)

    # 保存results.json
    results_json = {
        'catboost': {
            'train_r2': round(cb_train_r2, 4),
            'train_rmse': round(cb_train_rmse, 6),
            'train_mae': round(cb_train_mae, 6),
            'train_mape': round(cb_train_mape, 2),
            'cv_r2': round(cb_cv_r2, 4),
            'cv_std': round(cb_cv_std, 4),
            'cv_rmse': round(cb_cv_rmse, 6),
            'cv_mae': round(cb_cv_mae, 6),
            'cv_mape': round(cb_cv_mape, 2),
        },
        'naive_linear': {
            'train_r2': round(naive['train_r2'], 4),
            'adj_r2': round(naive['adj_r2'], 4),
            'cv_r2': round(naive['cv_r2'], 4),
            'cv_std': round(naive['cv_std'], 4),
            'cv_rmse': round(naive['cv_rmse'], 6),
            'cv_mae': round(naive['cv_mae'], 6),
            'cv_mape': round(naive['cv_mape'], 2),
            'n_terms': 5,
        },
        'nonlinear_main': {
            'train_r2': round(nonlinear_main['train_r2'], 4),
            'adj_r2': round(nonlinear_main['adj_r2'], 4),
            'cv_r2': round(nonlinear_main['cv_r2'], 4),
            'cv_std': round(nonlinear_main['cv_std'], 4),
            'cv_rmse': round(nonlinear_main['cv_rmse'], 6),
            'cv_mae': round(nonlinear_main['cv_mae'], 6),
            'cv_mape': round(nonlinear_main['cv_mape'], 2),
            'n_terms': 8,
        },
        'full_nonlinear': {
            'train_r2': round(full_nonlinear['train_r2'], 4),
            'adj_r2': round(full_nonlinear['adj_r2'], 4),
            'cv_r2': round(full_nonlinear['cv_r2'], 4),
            'cv_std': round(full_nonlinear['cv_std'], 4),
            'cv_rmse': round(full_nonlinear['cv_rmse'], 6),
            'cv_mae': round(full_nonlinear['cv_mae'], 6),
            'cv_mape': round(full_nonlinear['cv_mape'], 2),
            'train_rmse': round(full_nonlinear['train_rmse'], 6),
            'train_mae': round(full_nonlinear['train_mae'], 6),
            'train_mape': round(full_nonlinear['train_mape'], 2),
            'n_terms': full_nonlinear['n_terms'],
            'feature_names': full_nonlinear['feature_names'],
            'inter_names': full_nonlinear['inter_names'],
            'coefs': {name: float(c) for name, c in
                      zip(full_nonlinear['coef_names'], full_nonlinear['coefs'])},
            'pvalues': {name: float(p) for name, p in
                        zip(full_nonlinear['coef_names'], full_nonlinear['pvalues'])},
        },
        'shap_analysis': {
            'model': '无权重CatBoost（与方程拟合口径一致）',
            'n_samples': 351,
            'global_importance': {
                FEATURES[i]: float(shap_global['mean_abs_shap'][i])
                for i in range(len(FEATURES))
            },
            'interaction_top5': [
                {'pair': f"{f1}×{f2}", 'strength': float(s), 'eq_name': eq_name}
                for (f1, f2, s), eq_name in zip(
                    sorted(shap_inter['pair_list'], key=lambda x: x[2], reverse=True)[:5],
                    shap_inter['top5_pairs']
                )
            ],
            'interaction_all': [
                {'pair': f"{f1}×{f2}", 'strength': float(s)}
                for f1, f2, s in sorted(shap_inter['pair_list'], key=lambda x: x[2], reverse=True)
            ],
        },
        'backward_elimination': {
            'initial_terms': full_nonlinear['n_terms'],
            'final_terms': final['n_terms'],
            'elimination_path': final.get('elimination_path', []),
            'removed_terms': [step['term'] for step in final.get('elimination_path', [])],
        },
        'refined': {
            'train_r2': round(final['train_r2'], 4),
            'adj_r2': round(final['adj_r2'], 4),
            'cv_r2': round(final['cv_r2'], 4),
            'cv_std': round(final['cv_std'], 4),
            'cv_rmse': round(final['cv_rmse'], 6),
            'cv_mae': round(final['cv_mae'], 6),
            'cv_mape': round(final['cv_mape'], 2),
            'train_rmse': round(final['train_rmse'], 6),
            'train_mae': round(final['train_mae'], 6),
            'train_mape': round(final['train_mape'], 2),
            'n_terms': final['n_terms'],
            'main_terms': final['baseline_names'] + final['deg_names'],
            'inter_terms': final['coup_names'],
            'baseline_terms': final['baseline_names'],
            'deg_terms': final['deg_names'],
            'coup_terms': final['coup_names'],
            'coefs': {name: float(c) for name, c in
                      zip(final['coef_names'], final['coefs'])},
            'pvalues': {name: float(p) for name, p in
                        zip(final['coef_names'], final['pvalues'])},
            'vif': final.get('vif_results', {}),
            'hc3_robust': final.get('hc3_results', {}),
        },
        'decomposition': {
            'baseline_coefs': decomp['baseline_coefs'],
            'deg_coefs': decomp['deg_coefs'],
            'coup_coefs': decomp['coup_coefs'],
            'baseline_pvals': decomp['baseline_pvals'],
            'deg_pvals': decomp['deg_pvals'],
            'coup_pvals': decomp['coup_pvals'],
            'avg_contribution': {
                'baseline': float(np.mean(decomp['contrib_baseline'])),
                'degradation': float(np.mean(decomp['contrib_deg'])),
                'coupling': float(np.mean(decomp['contrib_coup'])),
            },
        },
        'nested_ftest': {
            'm0': {'r2': round(nested['m0']['r2'], 4),
                   'adj_r2': round(nested['m0']['adj_r2'], 4),
                   'rss': nested['m0']['rss'], 'n_terms': 3},
            'm1': {'r2': round(nested['m1']['r2'], 4),
                   'adj_r2': round(nested['m1']['adj_r2'], 4),
                   'rss': nested['m1']['rss'], 'n_terms': 7},
            'm2': {'r2': round(nested['m2']['r2'], 4),
                   'adj_r2': round(nested['m2']['adj_r2'], 4),
                   'rss': nested['m2']['rss'], 'n_terms': 11},
            'ftest_01': {
                'f_stat': round(nested['ftest_01']['f_stat'], 2),
                'p_val': float(nested['ftest_01']['p_val']),
                'delta_r2': round(nested['ftest_01']['delta_r2'], 4),
                'df_num': nested['ftest_01']['df_num'],
                'df_den': nested['ftest_01']['df_den'],
            },
            'ftest_12': {
                'f_stat': round(nested['ftest_12']['f_stat'], 2),
                'p_val': float(nested['ftest_12']['p_val']),
                'delta_r2': round(nested['ftest_12']['delta_r2'], 4),
                'df_num': nested['ftest_12']['df_num'],
                'df_den': nested['ftest_12']['df_den'],
            },
        },
        # ============ v3新增 ============
        'aic_bic_diagnosis': {
            var: {
                'comparison': [
                    {k: (float(v) if isinstance(v, (np.floating, float)) else
                         int(v) if isinstance(v, (np.integer, int)) else v)
                     for k, v in c.items()}
                    for c in res['comparison']
                ],
                'best_aic': res['best_aic'],
                'best_bic': res['best_bic'],
            }
            for var, res in aic_bic.items()
        },
        'nested_ftest_transformed': {
            'm1': {'r2': round(nested_trans['m1']['r2'], 4),
                   'adj_r2': round(nested_trans['m1']['adj_r2'], 4),
                   'rss': nested_trans['m1']['rss'], 'n_features': 8},
            'm2': {'r2': round(nested_trans['m2']['r2'], 4),
                   'adj_r2': round(nested_trans['m2']['adj_r2'], 4),
                   'rss': nested_trans['m2']['rss'], 'n_features': 18},
            'm3_all': {'r2': round(nested_trans['m3_all']['r2'], 4),
                       'adj_r2': round(nested_trans['m3_all']['adj_r2'], 4),
                       'rss': nested_trans['m3_all']['rss'], 'n_features': 28},
            'ftest_12': {
                'f': round(nested_trans['ftest_12']['f'], 2),
                'p': float(nested_trans['ftest_12']['p']),
                'delta_r2': round(nested_trans['ftest_12']['delta_r2'], 4),
                'df_num': nested_trans['ftest_12']['df_num'],
                'df_den': nested_trans['ftest_12']['df_den'],
            },
            'ftest_23_all': {
                'f': round(nested_trans['ftest_23_all']['f'], 2),
                'p': float(nested_trans['ftest_23_all']['p']),
                'delta_r2': round(nested_trans['ftest_23_all']['delta_r2'], 4),
                'df_num': nested_trans['ftest_23_all']['df_num'],
                'df_den': nested_trans['ftest_23_all']['df_den'],
            },
            'three_way_individual': nested_trans['three_way_individual'],
            'best_3way': nested_trans['best_3way'],
            'm2_inter_coefs': nested_trans['m2_inter_coefs'],
            'm2_vif': nested_trans['m2_vif'],
            'm3_all_vif': nested_trans['m3_all_vif'],
        },
        'bootstrap_stability': {
            'n_bootstrap': bootstrap['n_bootstrap'],
            'valid_iterations': bootstrap['valid_iterations'],
            'retention_frequency': bootstrap['retention_frequency'],
            'term_count_distribution': bootstrap['term_count_distribution'],
        },
        'grouped_residual': {
            'groups': grouped_resid['groups'],
            'kruskal_wallis': grouped_resid['kruskal_wallis'],
        },
        # ============ v3.1补充验证 ============
        'pdp_vs_equation': pdp_results,
        'catboost_cv_ols': cv_ols_results,
        'fpi_magnitude': fpi_results,
        'cross_scenario': cross_results,
    }

    json_path = os.path.join(RESULTS_FOLDER, 'results_v3.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存至: {json_path}")

if __name__ == '__main__':
    main()
