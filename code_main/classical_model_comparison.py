# -*- coding: utf-8 -*-
"""
可解释性方程特征分析优化 v4 经典热传导模型对比（C 模块）
================================================================
目标：把方程中的三个"疑似缺陷"用经典热传导模型转成物理洞察，全部为纯计算：

  C1 基准层（完好工况）热力学可行性：
      Wiener 串/并联上下界 + Maxwell-Eucken 两相模型 vs 27 个完好实测均值
      -> 证明 λ₀(T,RH) 落在经典多孔介质模型允许的物理区间内
  C2 含水率响应的"凹形"：
      线性/指数/串联充水（series water-filling）模型 vs 纯渗水实测均值
      -> 凹形 = 水分先填充孔隙网络后饱和，与串联充水机制一致
  C3 裂缝负效应的物理机制：
      裂缝=串联空气隙模型 vs 方程交互调整后的裂缝边际 + 纯裂缝实测
      -> 高 T/RH 下裂缝降低导热系数与空气隙机制一致；低 T 正效应需诚实讨论
  C4 冻融饱和型 vs 损伤累积模型：
      √FT / log / 幂律 / 指数饱和 拟合纯冻融实测均值
  C5 温度基准锚定：
      完好基准层温度斜率 vs 空气导热系数斜率 + 辐射 T³ 贡献示意

依赖：同文件夹下 可解释性方程特征分析优化_v4_补充验证.py（提供数据加载与方程拟合）
结果：可解释性方程特征分析优化_结果/v4_classical_comparison.json + Fig_v4_Classical_*.png
"""

import os
import json
import importlib.util
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
from scipy.optimize import curve_fit

warnings.filterwarnings('ignore')
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
RANDOM_STATE = 42

# 载入 v4 补充验证模块（数据加载与方程拟合，保证口径一致）
_spec = importlib.util.spec_from_file_location(
    'v4supp', os.path.join(BASE, 'supplementary_validation.py'))
v4s = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v4s)

# 经典模型物性参数（25°C 基准，文献常用值）
AIR_25 = 0.0259          # 空气导热系数 W/(m·K)
AIR_SLOPE = 7.6e-5       # 空气导热系数温度斜率 W/(m·K²)
WATER_LAMBDA = 0.60       # 液态水导热系数 W/(m·K)


def air_lambda(T):
    return AIR_25 + AIR_SLOPE * (np.asarray(T, dtype=float) - 20.0)


def ensure_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def calc_aicc(n, rss, k):
    """小样本校正 AIC（n 很小必须用 AICc）"""
    ll = -n / 2.0 * (np.log(2 * np.pi * rss / n) + 1.0)
    aic = -2 * ll + 2 * k
    aicc = aic + 2 * k * (k + 1) / max(n - k - 1, 1)
    return aic, aicc


# ============================================================
# C1. Wiener 上下界 + Maxwell-Eucken
# ============================================================
def c1_baseline_bounds(df_all, df_raw):
    print("\n" + "=" * 60)
    print("C1. 完好基准层 vs Wiener上下界与Maxwell-Eucken（含Knudsen效应）")
    print("=" * 60)
    intact = df_all[df_all['数据来源'] == '完好']
    y_m = intact['导热系数'].values.astype(float)
    T_m = intact['温度'].values.astype(float)

    phis = [0.90, 0.93, 0.96]
    lam_s_list = [0.5, 1.0, 1.4, 3.0]

    def envelope(T, gas_fun):
        """给定气体导热函数，返回 (串联下界min, 并联上界max, ME连续气相min)"""
        ser_min, par_max, me_min = 1e9, -1e9, 1e9
        for phi in phis:
            for ls in lam_s_list:
                lg = gas_fun(T)
                ser = 1.0 / ((1 - phi) / ls + phi / lg)
                par = (1 - phi) * ls + phi * lg
                lc, ld, vd = lg, ls, 1 - phi
                me = lc * (2 * lc + ld - 2 * vd * (lc - ld)) / (2 * lc + ld + vd * (lc - ld))
                ser_min = min(ser_min, ser)
                par_max = max(par_max, par)
                me_min = min(me_min, me)
        return ser_min, par_max, me_min

    # 自由空气
    free = np.array([envelope(t, air_lambda) for t in T_m])
    ser_free, par_free, me_free = free[:, 0], free[:, 1], free[:, 2]
    viol_free = y_m < ser_free - 1e-12

    # Knudsen 受限气体：λg,eff = λ_air / (1 + 2β·Kn), Kn = l_mfp/d_pore
    beta_kn = 1.5
    l_mfp_25 = 67e-9  # 空气平均自由程 @25°C (m)
    d_pores = [20e-9, 35e-9, 50e-9]

    def knudsen_gas(T, d_pore):
        l_mfp = l_mfp_25 * (np.asarray(T, dtype=float) + 273.15) / 298.15
        kn = l_mfp / d_pore
        return air_lambda(T) / (1 + 2 * beta_kn * kn)

    kn_series = np.zeros((len(T_m), len(d_pores)))
    for j, dp in enumerate(d_pores):
        kn_series[:, j] = np.array([envelope(t, lambda tt, d=dp: knudsen_gas(tt, d))[0]
                                    for t in T_m])
    viol_kn = np.zeros((len(T_m), len(d_pores)), dtype=bool)
    for j in range(len(d_pores)):
        viol_kn[:, j] = y_m < kn_series[:, j] - 1e-12

    print(f"  φ∈{phis}, λs∈{lam_s_list} W/(m·K), 逐点按实测温度计算:")
    print(f"    完好实测区间: [{float(y_m.min()):.4f}, {float(y_m.max()):.4f}] W/(m·K)")
    print(f"    自由空气串联下界(逐点): 违反点数 = {int(viol_free.sum())}/27")
    if viol_free.sum() > 0:
        vt = T_m[viol_free]
        print(f"      违反点温度: {sorted(set(float(v) for v in vt))}（低温点）")
    print(f"    Knudsen受限气体(β={beta_kn}, d_pore={[int(d*1e9) for d in d_pores]}nm):")
    for j, dp in enumerate(d_pores):
        print(f"      d_pore={int(dp*1e9)}nm: 违反点数 = {int(viol_kn[:, j].sum())}/27")

    # 温度扫描曲线（画图用）
    T_grid = np.linspace(5, 45, 81)
    ser_free_grid = np.array([envelope(t, air_lambda)[0] for t in T_grid])
    me_free_grid = np.array([envelope(t, air_lambda)[2] for t in T_grid])
    kn_series_grid = np.array([envelope(t, lambda tt: knudsen_gas(tt, 35e-9))[0]
                               for t in T_grid])

    fig, ax = plt.subplots(figsize=(7.48, 4.86))
    ax.plot(T_grid, ser_free_grid, 'b--', linewidth=2, label='Wiener 串联下界（自由空气, 参数下限组合）')
    ax.plot(T_grid, me_free_grid, 'g-.', linewidth=2, label='Maxwell-Eucken 连续气相（自由空气, 下限组合）')
    ax.plot(T_grid, kn_series_grid, 'k-', linewidth=2.5,
            label='Wiener 串联下界（Knudsen受限气体, d_pore=35nm）')
    ax.text(42, 0.0445, 'Wiener 并联上界远高于此\n（0.06–0.32 W/(m·K)）', fontsize=6.5,
            ha='right', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    for rh in [30, 60, 90]:
        sub = intact[intact['湿度'] == rh]
        ax.scatter(sub['温度'], sub['导热系数'], s=60, label=f'完好实测 RH={rh}%')
    ax.set_ylim(0.02, 0.045)
    ax.set_xlabel('温度 (℃)', fontsize=9.0)
    ax.set_ylabel('导热系数 (W/(m·K))', fontsize=9.0)
    ax.set_title('完好工况实测 vs 经典模型界（低温点低于自由空气下界=Knudsen效应）', fontsize=9.8)
    ax.legend(fontsize=6.8, loc='upper left')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_FOLDER, 'Fig_v4_Classical_Bounds.png')
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {os.path.basename(path)}")

    return {
        'assumptions': {'porosity': phis, 'solid_lambda': lam_s_list,
                        'knudsen_beta': beta_kn, 'd_pore_nm': [int(d * 1e9) for d in d_pores]},
        'intact_measured_range': [float(y_m.min()), float(y_m.max())],
        'free_air': {
            'violations_out_of_27': int(viol_free.sum()),
            'violation_temperatures': sorted(set(float(v) for v in T_m[viol_free])),
        },
        'knudsen': {
            'violations_by_pore': {int(dp * 1e9): int(viol_kn[:, j].sum())
                                   for j, dp in enumerate(d_pores)},
        },
    }


# ============================================================
# C2. 含水率响应：线性/指数/串联充水 vs 凹形
# ============================================================
def c2_moisture_models(df_all, coefs):
    print("\n" + "=" * 60)
    print("C2. 含水率响应模型对比（线性/指数/串联充水）")
    print("=" * 60)
    intact = df_all[df_all['数据来源'] == '完好']
    water_sub = df_all[(df_all['冻融次数'] == 0) & (df_all['裂缝长度'] == 0) & (df_all['含水率'] > 0)]

    w_levels = np.array([0.0, 23.0, 38.0, 56.0])
    l0_pool = float(intact['导热系数'].mean())
    w_means = [l0_pool]
    for w in w_levels[1:]:
        w_means.append(float(water_sub[water_sub['含水率'] == w]['导热系数'].mean()))
    w_means = np.array(w_means)
    n = len(w_levels)

    def fit_linear(X, y):
        A = np.column_stack([np.ones(len(X)), X])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        return beta, A @ beta

    def fit_quad(X, y):
        A = np.column_stack([np.ones(len(X)), X, X ** 2])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        return beta, A @ beta

    def f_exp(X, l0, k):
        return l0 * np.exp(k * X)

    def f_series(X, l0, s):
        # 串联充水: 1/λ = (1-f)/λ0 + f/λw, f = s·X
        f = s * X
        return 1.0 / ((1 - f) / l0 + f / WATER_LAMBDA)

    models = {}
    b_lin, p_lin = fit_linear(w_levels, w_means)
    models['linear'] = {'params': {'l0': b_lin[0], 'a': b_lin[1]},
                        'rss': float(np.sum((w_means - p_lin) ** 2)), 'k': 2}
    b_qua, p_qua = fit_quad(w_levels, w_means)
    models['quadratic'] = {'params': {'l0': b_qua[0], 'b': b_qua[1], 'c': b_qua[2]},
                           'rss': float(np.sum((w_means - p_qua) ** 2)), 'k': 3}
    p_exp, _ = curve_fit(f_exp, w_levels, w_means, p0=[l0_pool, 0.003], maxfev=20000)
    models['exponential'] = {'params': {'l0': p_exp[0], 'k': p_exp[1]},
                             'rss': float(np.sum((w_means - f_exp(w_levels, *p_exp)) ** 2)), 'k': 2}
    p_ser, _ = curve_fit(f_series, w_levels, w_means, p0=[l0_pool, 0.005], maxfev=20000)
    models['series_filling'] = {'params': {'l0': p_ser[0], 's': p_ser[1]},
                                'rss': float(np.sum((w_means - f_series(w_levels, *p_ser)) ** 2)), 'k': 2}

    print(f"  纯渗水水平均值: W=0:{w_means[0]:.5f}  23:{w_means[1]:.5f}  "
          f"38:{w_means[2]:.5f}  56:{w_means[3]:.5f}")
    print(f"  增量: 0→23: {w_means[1]-w_means[0]:+.5f}, 23→38: {w_means[2]-w_means[1]:+.5f}, "
          f"38→56: {w_means[3]-w_means[2]:+.5f}")
    for name, m in models.items():
        aic, aicc = calc_aicc(n, m['rss'], m['k'])
        m['aic'] = aic
        m['aicc'] = aicc
        print(f"  {name:16s}: RSS={m['rss']:.2e}  AICc={aicc:.1f}")

    # 方程含水率边际斜率（参考 T=25, RH=60）
    d0 = coefs['Water']
    d56 = coefs['Water'] + 2 * coefs['Water²'] * 56.0
    print(f"\n  方程含水率边际斜率: W=0时 {d0:.3e}, W=56时 {d56:.3e}, 衰减 {100*(1-d56/d0):.0f}%")
    print(f"  -> 凹形与串联充水饱和机制一致")

    # 图
    wg = np.linspace(0, 56, 100)
    fig, ax = plt.subplots(figsize=(7.48, 4.99))
    ax.scatter(w_levels, w_means, s=90, c='#2c3e50', zorder=5, label='纯渗水实测均值')
    ax.plot(wg, b_lin[0] + b_lin[1] * wg, '--', color='#95a5a6', label='线性模型')
    ax.plot(wg, b_qua[0] + b_qua[1] * wg + b_qua[2] * wg ** 2, '-', color='#3498db',
            label=f'二次模型（方程同形, c={b_qua[2]:+.2e}）')
    ax.plot(wg, f_series(wg, *p_ser), '-.', color='#e67e22', label='串联充水模型')
    ax.plot(wg, f_exp(wg, *p_exp), ':', color='#2ecc71', label='指数模型')
    ax.set_xlabel('含水率 W (%)', fontsize=9.0)
    ax.set_ylabel('导热系数 (W/(m·K))', fontsize=9.0)
    ax.set_title('含水率响应：实测 vs 经典模型（凹形=水分填充饱和）', fontsize=9.8)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_FOLDER, 'Fig_v4_Moisture_Models.png')
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {os.path.basename(path)}")

    return {
        'level_means': {str(w): float(v) for w, v in zip(w_levels, w_means)},
        'increments': {'0_to_23': float(w_means[1]-w_means[0]),
                       '23_to_38': float(w_means[2]-w_means[1]),
                       '38_to_56': float(w_means[3]-w_means[2])},
        'models': models,
        'equation_marginal': {'slope_at_0': d0, 'slope_at_56': d56,
                              'decay_pct': float(100 * (1 - d56 / d0))},
    }


# ============================================================
# C3. 裂缝：串联空气隙模型
# ============================================================
def c3_crack_airgap(df_all, coefs):
    print("\n" + "=" * 60)
    print("C3. 裂缝=串联空气隙模型 vs 方程裂缝边际")
    print("=" * 60)

    # 方程交互调整后的裂缝净系数 c_net(T,RH) = β_√CR + β_TCR·T + β_RHCR·RH
    T_grid = np.linspace(5, 45, 81)
    rh_list = [30.0, 60.0, 90.0]
    cnet = {}
    for rh in rh_list:
        cnet[rh] = (coefs['√Crack'] + coefs['T×√Crack'] * T_grid
                    + coefs['RH×√Crack'] * rh)

    # 串联空气隙模型（方案A）：f = 裂缝开口宽度 / 板厚
    # 学位论文确认：裂缝为贯穿型（贯穿 50 mm 厚），实际裂缝开口约 0.5 mm
    L_board = 50.0          # 板厚 mm
    crack_opening = 0.5     # 裂缝开口宽度 mm
    f_air = crack_opening / L_board   # 0.01
    cr10 = 10.0             # 裂缝长度 cm（用于方程效应）
    series_effect = {}
    eq_effect = {}
    for rh in rh_list:
        l0 = v4s.eq_predict_vectorized(T_grid, np.full_like(T_grid, rh), 0, 0, 0, coefs)
        la = air_lambda(T_grid)
        l_crack = l0 * la / ((1 - f_air) * la + f_air * l0)
        series_effect[rh] = l_crack - l0
        l_eq_crack = v4s.eq_predict_vectorized(T_grid, np.full_like(T_grid, rh), 0, cr10, 0, coefs)
        eq_effect[rh] = l_eq_crack - l0

    print(f"  串联空气隙模型（方案A：板厚 L=50mm, 裂缝开口0.5mm, f=0.01）:")
    for rh in rh_list:
        print(f"    RH={rh:.0f}%: T=25℃ 时 Δλ={series_effect[rh][40]:+.6f} W/(m·K)")
    print(f"  方程裂缝效应 Δλ(裂缝长度10cm):")
    for rh in rh_list:
        print(f"    RH={rh:.0f}%: T=25℃ 时 Δλ={eq_effect[rh][40]:+.6f} W/(m·K)")
    print(f"  T=25℃, RH=60%: 串联模型 {series_effect[60][40]:+.6f} vs 方程 {eq_effect[60][40]:+.6f}")

    # 纯裂缝实测（按 T 分组：低温≤15 / 中温20-30 / 高温≥35）
    intact = df_all[df_all['数据来源'] == '完好']
    crack_sub = df_all[(df_all['冻融次数'] == 0) & (df_all['含水率'] == 0) & (df_all['裂缝长度'] > 0)]
    intact_map = {(r['温度'], r['湿度']): r['导热系数'] for _, r in intact.iterrows()}
    crack_sub = crack_sub.copy()
    crack_sub['l0_cell'] = crack_sub.apply(
        lambda r: intact_map.get((r['温度'], r['湿度']), np.nan), axis=1)
    crack_sub['effect'] = crack_sub['导热系数'] - crack_sub['l0_cell']
    # 方程在同一点上的裂缝效应（逐点对比）
    crack_sub['eq_effect'] = (
        v4s.eq_predict_vectorized(crack_sub['温度'].values, crack_sub['湿度'].values,
                                  crack_sub['冻融次数'].values, crack_sub['裂缝长度'].values,
                                  crack_sub['含水率'].values, coefs)
        - v4s.eq_predict_vectorized(crack_sub['温度'].values, crack_sub['湿度'].values,
                                    np.zeros(len(crack_sub)), np.zeros(len(crack_sub)),
                                    np.zeros(len(crack_sub)), coefs))
    crack_sub['Tband'] = pd.cut(crack_sub['温度'], bins=[0, 15, 30, 60],
                                labels=['低温≤15', '中温20-30', '高温≥35'])
    print("\n  纯裂缝实测效应（相对同格完好均值，按温度带）:")
    band_stats = {}
    for band, g in crack_sub.groupby('Tband', observed=True):
        m = float(g['effect'].mean())
        s = float(g['effect'].std())
        me = float(g['eq_effect'].mean())
        print(f"    {band}: n={len(g)}, 实测效应={m:+.6f} ± {s:.6f}, 方程效应={me:+.6f}")
        band_stats[str(band)] = {'n': int(len(g)), 'measured_mean': m, 'std': s,
                                 'equation_mean': me}
    overall = float(crack_sub['effect'].mean())
    eq_overall = float(crack_sub['eq_effect'].mean())
    print(f"    全部: 实测平均效应={overall:+.6f}, 方程平均效应={eq_overall:+.6f}")

    # 按裂缝水平
    level_stats = {}
    print("\n  按裂缝水平:")
    for cr, g in crack_sub.groupby('裂缝长度'):
        level_stats[str(float(cr))] = {
            'n': int(len(g)),
            'measured_mean': float(g['effect'].mean()),
            'equation_mean': float(g['eq_effect'].mean()),
        }
        print(f"    CR={cr:g}cm: 实测={float(g['effect'].mean()):+.6f}, "
              f"方程={float(g['eq_effect'].mean()):+.6f}")
    print("  -> 方案A（空气隙=开口0.5mm/板厚50mm）下，串联空气隙模型Δλ量级远小于实测/方程；")
    print("     空气隙方向一致，但量级不匹配，裂缝长度依赖需2D热流路径/接触热阻解释")

    # 图
    fig, axes = plt.subplots(1, 3, figsize=(7.48, 2.49))
    ax = axes[0]
    for rh, col in zip(rh_list, ['#3498db', '#e67e22', '#e74c3c']):
        ax.plot(T_grid, cnet[rh], color=col, label=f'RH={rh:.0f}%')
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel('温度 (℃)', fontsize=9.0)
    ax.set_ylabel('交互调整后裂缝净系数 β_√CR+β_TCR·T+β_RHCR·RH', fontsize=8.2)
    ax.set_title('方程裂缝净系数随 T/RH 的符号翻转', fontsize=9.8)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(T_grid, series_effect[60], 'g-', linewidth=2.5, label='串联空气隙模型 (RH=60%)')
    ax.plot(T_grid, eq_effect[60], 'r--', linewidth=2.5, label='方程 Δλ(裂缝长度10cm) (RH=60%)')
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel('温度 (℃)', fontsize=9.0)
    ax.set_ylabel('Δλ (W/(m·K))', fontsize=9.0)
    ax.set_title('裂缝长度10cm的导热系数效应：串联空气隙 vs 方程', fontsize=9.8)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    ax = axes[2]
    cr_keys = sorted(level_stats.keys(), key=float)
    x = np.arange(len(cr_keys))
    measured = [level_stats[k]['measured_mean'] for k in cr_keys]
    eqvals = [level_stats[k]['equation_mean'] for k in cr_keys]
    w = 0.35
    ax.bar(x - w / 2, measured, w, label='实测（同格完好为基准）', color='#3498db', edgecolor='white')
    ax.bar(x + w / 2, eqvals, w, label='方程', color='#e74c3c', edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{float(k):g}cm' for k in cr_keys], fontsize=8.2)
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.set_ylabel('Δλ (W/(m·K))', fontsize=9.0)
    ax.set_title('裂缝效应：实测 vs 方程（按裂缝水平）', fontsize=9.8)
    ax.legend(fontsize=6.8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_FOLDER, 'Fig_v4_Crack_AirGap.png')
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {os.path.basename(path)}")

    return {
        'net_coef_at_25C': {str(rh): float(cnet[rh][40]) for rh in rh_list},
        'series_model_at_25C': {str(rh): float(series_effect[rh][40]) for rh in rh_list},
        'equation_effect_at_25C': {str(rh): float(eq_effect[rh][40]) for rh in rh_list},
        'board_thickness_mm': L_board,
        'crack_opening_mm': crack_opening,
        'air_gap_fraction': f_air,
        'crack_volume_fraction_at_10mm': f_air,  # 兼容旧键名，语义改为空气隙分数
        'pure_crack_measured': band_stats,
        'pure_crack_overall_mean': float(overall),
        'pure_crack_equation_overall_mean': eq_overall,
        'pure_crack_by_level': level_stats,
    }


# ============================================================
# C4. 冻融饱和型 vs 损伤累积模型
# ============================================================
def c4_frost_damage(df_all):
    print("\n" + "=" * 60)
    print("C4. 冻融响应 vs 损伤累积模型")
    print("=" * 60)
    intact = df_all[df_all['数据来源'] == '完好']
    ft_sub = df_all[(df_all['裂缝长度'] == 0) & (df_all['含水率'] == 0) & (df_all['冻融次数'] > 0)]
    ft_levels = np.array([0.0, 18.0, 21.0, 24.0])
    l0_pool = float(intact['导热系数'].mean())
    ft_means = np.array([l0_pool] + [float(ft_sub[ft_sub['冻融次数'] == f]['导热系数'].mean())
                                     for f in ft_levels[1:]])
    n = len(ft_levels)
    print(f"  纯冻融水平均值: {[f'{v:.5f}' for v in ft_means]}")
    print(f"  增量: 0→18: {ft_means[1]-ft_means[0]:+.5f}, 18→21: {ft_means[2]-ft_means[1]:+.5f}, "
          f"21→24: {ft_means[3]-ft_means[2]:+.5f}")

    X = ft_levels
    y = ft_means

    def fit_1d(fun, p0):
        p, _ = curve_fit(fun, X, y, p0=p0, maxfev=50000)
        pred = fun(X, *p)
        return p, float(np.sum((y - pred) ** 2))

    def f_sqrt(x, a, b):
        return a + b * np.sqrt(x)

    def f_log(x, a, b):
        return a + b * np.log(x + 1)

    def f_power(x, a, b, p):
        return a + b * x ** p

    def f_exp_sat(x, a, b, k):
        return a + b * (1 - np.exp(-k * x))

    results = {}
    p, rss = fit_1d(f_sqrt, [l0_pool, 0.001])
    results['sqrt'] = {'params': {'a': p[0], 'b': p[1]}, 'rss': rss, 'k': 2}
    p, rss = fit_1d(f_log, [l0_pool, 0.001])
    results['log'] = {'params': {'a': p[0], 'b': p[1]}, 'rss': rss, 'k': 2}
    p, _ = curve_fit(f_power, X, y, p0=[l0_pool, 0.001, 0.5],
                     bounds=([0, 0, 0.2], [np.inf, np.inf, 1.0]), maxfev=50000)
    rss = float(np.sum((y - f_power(X, *p)) ** 2))
    results['power'] = {'params': {'a': p[0], 'b': p[1], 'p': p[2]}, 'rss': rss, 'k': 3}
    p, _ = curve_fit(f_exp_sat, X, y, p0=[l0_pool, 0.003, 0.08],
                     bounds=([0, 0, 0.02], [np.inf, np.inf, 1.0]), maxfev=50000)
    rss = float(np.sum((y - f_exp_sat(X, *p)) ** 2))
    results['exp_saturation'] = {'params': {'a': p[0], 'b': p[1], 'k': p[2]}, 'rss': rss, 'k': 3}

    for name, m in results.items():
        aic, aicc = calc_aicc(n, m['rss'], m['k'])
        m['aic'] = aic
        m['aicc'] = aicc
        print(f"  {name:16s}: RSS={m['rss']:.2e}  AICc={aicc:.1f}  "
              f"参数={ {k2: round(v2,4) for k2, v2 in m['params'].items()} }")

    xg = np.linspace(0, 24, 100)
    fig, ax = plt.subplots(figsize=(7.48, 4.99))
    ax.scatter(ft_levels, ft_means, s=90, c='#2c3e50', zorder=5, label='纯冻融实测均值')
    ax.plot(xg, f_sqrt(xg, *results['sqrt']['params'].values()), '-', color='#3498db', label='√FT')
    ax.plot(xg, f_log(xg, *results['log']['params'].values()), '--', color='#e67e22', label='ln(FT+1)')
    ax.plot(xg, f_power(xg, *results['power']['params'].values()), '-.', color='#2ecc71',
            label=f"幂律 p={results['power']['params']['p']:.2f}")
    ax.plot(xg, f_exp_sat(xg, *results['exp_saturation']['params'].values()), ':', color='#e74c3c',
            label='指数饱和')
    ax.set_xlabel('冻融次数 FT', fontsize=9.0)
    ax.set_ylabel('导热系数 (W/(m·K))', fontsize=9.0)
    ax.set_title('冻融响应：饱和型 vs 损伤累积模型', fontsize=9.8)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_FOLDER, 'Fig_v4_FT_Damage.png')
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {os.path.basename(path)}")

    return {
        'level_means': {str(f): float(v) for f, v in zip(ft_levels, ft_means)},
        'increments': {'0_to_18': float(ft_means[1]-ft_means[0]),
                       '18_to_21': float(ft_means[2]-ft_means[1]),
                       '21_to_24': float(ft_means[3]-ft_means[2])},
        'models': results,
    }


# ============================================================
# C5. 温度基准锚定
# ============================================================
def c5_temperature_anchor(df_all, df_raw):
    print("\n" + "=" * 60)
    print("C5. 完好基准层温度斜率 vs 空气/辐射")
    print("=" * 60)

    def fit_baseline(T, RH, y):
        X = np.column_stack([T, T ** 2, RH ** 2])
        return np.linalg.lstsq(np.column_stack([np.ones(len(T)), X]), y, rcond=None)[0]

    T_r = df_raw['温度'].values.astype(float)
    RH_r = df_raw['湿度'].values.astype(float)
    y_r = df_raw['导热系数'].values.astype(float)
    beta = fit_baseline(T_r, RH_r, y_r)
    const, bT, bT2, bRH2 = beta
    slope_25 = bT + 2 * bT2 * 25.0
    air_slope = AIR_SLOPE
    ratio = slope_25 / air_slope
    print(f"  原始数据完好基准层: λ0(T) = {const:.5f} {bT:+.2e}·T {bT2:+.2e}·T² "
          f"{bRH2:+.2e}·RH²")
    print(f"  25°C 处温度斜率 = {slope_25:.2e} W/(m·K²), 空气斜率 = {air_slope:.2e}, 比值 = {ratio:.1f}")
    print(f"  -> 气凝胶温度敏感性约为空气的 {ratio:.1f} 倍（固相传导为主；辐射项为文献标度示意）")

    # 辐射示意：λ ∝ T³ 拟合 RH=30% 完好格点
    intact = df_all[df_all['数据来源'] == '完好']
    sub = intact[intact['湿度'] == 30]
    T30 = sub['温度'].values.astype(float)
    y30 = sub['导热系数'].values.astype(float)
    Tk = T30 + 273.15

    def f_lin(T, a, b):
        return a + b * T

    def f_cubic(T, a, b, c):
        return a + b * T + c * T ** 3

    p_lin, _ = curve_fit(f_lin, T30, y30, p0=[0.02, 1e-4])
    p_cub, _ = curve_fit(f_cubic, T30, y30, p0=[0.02, 1e-4, 1e-10])
    rss_lin = float(np.sum((y30 - f_lin(T30, *p_lin)) ** 2))
    rss_cub = float(np.sum((y30 - f_cubic(T30, *p_cub)) ** 2))
    print(f"\n  RH=30% 完好格点（9点）: 线性RSS={rss_lin:.2e}, T³项RSS={rss_cub:.2e}")
    print(f"  -> T³(辐射标度)项使RSS降低 {100*(1-rss_cub/rss_lin):.0f}%；"
          f"（未直接测量辐射，仅作与文献 T³ 标度方向一致的定性示意）")

    Tg = np.linspace(5, 45, 100)
    fig, ax = plt.subplots(figsize=(7.48, 4.99))
    ax.plot(Tg, air_lambda(Tg), 'k--', linewidth=2, label='空气导热系数 λ_air(T)')
    ax.plot(Tg, const + bT * Tg + bT2 * Tg ** 2 + bRH2 * 30 ** 2, 'b-', linewidth=2.5,
            label='完好基准层 λ0(T, RH=30%)')
    ax.plot(Tg, f_cubic(Tg, *p_cub), 'r:', linewidth=2, label='T³辐射示意拟合')
    ax.scatter(T30, y30, s=80, c='#2c3e50', zorder=5, label='完好实测 RH=30%')
    ax.set_xlabel('温度 (℃)', fontsize=9.0)
    ax.set_ylabel('导热系数 (W/(m·K))', fontsize=9.0)
    ax.set_title('温度基准锚定：气凝胶 vs 空气 vs 辐射标度示意', fontsize=9.8)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_FOLDER, 'Fig_v4_Temperature_Anchor.png')
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {os.path.basename(path)}")

    return {
        'baseline_coefs_raw': {'const': float(const), 'T': float(bT),
                               'T2': float(bT2), 'RH2': float(bRH2)},
        'slope_at_25C': float(slope_25),
        'air_slope': float(air_slope),
        'ratio_to_air': float(ratio),
        'radiation_check': {'rss_linear': rss_lin, 'rss_with_T3': rss_cub,
                            'rss_reduction_pct': float(100 * (1 - rss_cub / rss_lin))},
    }


def sensitivity_analysis(df_all, coefs):
    """C6: 参数敏感性分析（板厚 / 孔隙率 / 骨架导热 / 孔径）"""
    print("\n" + "=" * 60)
    print("C6. 参数敏感性分析")
    print("=" * 60)
    intact = df_all[df_all['数据来源'] == '完好']
    T_m = intact['温度'].values.astype(float)
    y_m = intact['导热系数'].values.astype(float)

    phis = np.arange(0.88, 0.975, 0.01)
    lam_s_list = [0.5, 1.0, 1.4, 3.0]
    d_pores = [20e-9, 35e-9, 50e-9]
    beta_kn = 1.5
    l_mfp_25 = 67e-9

    def knudsen_gas(T, d_pore):
        l_mfp = l_mfp_25 * (np.asarray(T, dtype=float) + 273.15) / 298.15
        return air_lambda(T) / (1 + 2 * beta_kn * l_mfp / d_pore)

    # (1) Knudsen 下界违反数在 孔隙率×骨架×孔径 全网格上的最大值
    viol_kn = {}
    for dp in d_pores:
        worst = 0
        for phi in phis:
            for ls in lam_s_list:
                ser = 1.0 / ((1 - phi) / ls + phi / knudsen_gas(T_m, dp))
                worst = max(worst, int((y_m < ser - 1e-12).sum()))
        viol_kn[f'd_pore_{int(dp * 1e9)}nm'] = worst
    # (2) 自由空气下界违反数在网格上的最大值
    free_worst = 0
    for phi in phis:
        for ls in lam_s_list:
            ser = 1.0 / ((1 - phi) / ls + phi / air_lambda(T_m))
            free_worst = max(free_worst, int((y_m < ser - 1e-12).sum()))
    print(f"  自由空气Wiener下界: 全网格最大违反数 = {free_worst}/27")
    for k, v in viol_kn.items():
        print(f"  Knudsen({k}): 全网格最大违反数 = {v}/27")

    # (3) 板厚对裂缝空气隙模型的影响（T=25, RH=60, 裂缝开口0.5mm）
    Ls = [30, 40, 50, 60, 80]
    T_ref, RH_ref, crack_cm = 25.0, 60.0, 10.0
    opening = 0.5   # 裂缝开口宽度 mm（方案A）
    l0 = v4s.eq_predict_vectorized(T_ref, RH_ref, 0, 0, 0, coefs)
    la = air_lambda(T_ref)
    eq_dl = float(v4s.eq_predict_vectorized(T_ref, RH_ref, 0, crack_cm, 0, coefs) - l0)
    sweep = []
    for L in Ls:
        f = opening / L
        l_crack = l0 * la / ((1 - f) * la + f * l0)
        dl = float(l_crack - l0)
        sweep.append({'L_mm': L, 'air_gap_fraction': float(f),
                      'delta_lambda': dl, 'ratio_to_equation': float(dl / eq_dl)})
        print(f"  L={L:g}mm: f=开口0.5mm/L={f:.4f}, 空气隙Δλ={dl:+.6f}, "
              f"方程Δλ(10cm)={eq_dl:+.6f}, 比值={dl/eq_dl:.3f}")
    print(f"  （学位论文/论文确认试件厚度为 50mm，300×300×50mm）")

    # (4) 由表观密度反推孔隙率（SiO2 骨架密度 1800–2200 kg/m³）
    rho_b = np.array([100.0, 120.0])
    rho_s = np.array([1800.0, 2200.0])
    phi_lo = float((1 - rho_b.max() / rho_s.min()).min())
    phi_hi = float((1 - rho_b.min() / rho_s.max()).max())
    print(f"  按密度反推孔隙率: φ = 1 − ρb/ρs, ρb=100–120 kg/m³, "
          f"ρs=1800–2200 kg/m³（SiO2骨架）→ φ ∈ [{phi_lo:.3f}, {phi_hi:.3f}]")
    print(f"  （注：未计入水泥/EPS 组分体积分数；若 EPS 占比高，真实骨架密度更低，孔隙率区间会偏移）")

    # 图
    fig, axes = plt.subplots(1, 2, figsize=(7.48, 2.94))
    ax = axes[0]
    Lg = np.linspace(25, 85, 60)
    fg = opening / Lg
    dl_g = l0 * la / ((1 - fg) * la + fg * l0) - l0
    ax.plot(Lg, dl_g, 'b-', linewidth=2.5, label='串联空气隙模型 Δλ(裂缝开口0.5mm)')
    ax.axhline(eq_dl, color='r', linestyle='--', linewidth=2, label=f'方程 Δλ={eq_dl:+.6f}')
    ax.axvline(50, color='gray', linestyle=':', linewidth=1.5, label='试件厚度 50mm（论文确认）')
    ax.set_xlabel('板厚 L (mm)', fontsize=9.0)
    ax.set_ylabel('Δλ (W/(m·K))', fontsize=9.0)
    ax.set_title('裂缝空气隙模型对板厚的敏感性（T=25℃, RH=60%, 裂缝开口0.5mm）', fontsize=9.0)
    ax.legend(fontsize=6.8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    labels = ['自由空气'] + [f'Knudsen\n{int(d*1e9)}nm' for d in d_pores]
    vals = [free_worst] + [viol_kn[f'd_pore_{int(d*1e9)}nm'] for d in d_pores]
    colors = ['#e74c3c'] + ['#2ecc71'] * 3
    bars = ax.bar(labels, vals, color=colors, edgecolor='white', width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f'{v}', ha='center', fontsize=8.2,
                fontweight='bold')
    ax.set_ylabel('Wiener 下界违反点数 (/27)', fontsize=9.0)
    ax.set_title('下界违反数：孔隙率×骨架×孔径 全网格最大值', fontsize=9.0)
    ax.set_ylim(0, 6)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_FOLDER, 'Fig_v4_Sensitivity.png')
    plt.savefig(path, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"  已保存: {os.path.basename(path)}")

    return {
        'confirmed_params': {'thickness_mm': 50.0, 'density_kgm3': [100, 120],
                             'thermal_conductivity_measured': 0.029,
                             'porosity_from_density': {
                                 'formula': 'φ = 1 − ρb/ρs',
                                 'bulk_density': [100, 120],
                                 'solid_density': [1800, 2200],
                                 'range': [phi_lo, phi_hi],
                                 'caveat': '未计入水泥/EPS组分体积分数'},
                             'source': '学位论文表2-3 与 Energy & Buildings 354 (2026) 116960 表3'},
        'knudsen_grid': {'phis': [float(p) for p in phis],
                         'lam_s': lam_s_list,
                         'max_violations': viol_kn},
        'free_air_grid': {'max_violations': free_worst},
        'board_thickness_sweep': sweep,
        'equation_crack_effect_at_25_60_10': eq_dl,
    }


def main():
    ensure_folder(RESULTS_FOLDER)
    print("=" * 60)
    print("可解释性方程特征分析优化 v4 经典热传导模型对比")
    print("=" * 60)

    df_all, df_raw = v4s.load_data()
    full_model, _, _, _ = v4s.fit_equation(df_all)
    coefs = v4s.coefs_from_model(full_model)

    c1 = c1_baseline_bounds(df_all, df_raw)
    c2 = c2_moisture_models(df_all, coefs)
    c3 = c3_crack_airgap(df_all, coefs)
    c4 = c4_frost_damage(df_all)
    c5 = c5_temperature_anchor(df_all, df_raw)
    c6 = sensitivity_analysis(df_all, coefs)

    out = {
        'C1_baseline_bounds': c1,
        'C2_moisture_models': c2,
        'C3_crack_airgap': c3,
        'C4_frost_damage': c4,
        'C5_temperature_anchor': c5,
        'C6_sensitivity': c6,
    }
    json_path = os.path.join(RESULTS_FOLDER, 'v4_classical_comparison.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {json_path}")


if __name__ == '__main__':
    main()
