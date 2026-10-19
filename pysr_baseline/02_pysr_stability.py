r"""
C_PySR基线 第2步:结构稳定性 + 物理一致性检验
================================================
1. 稳定性: PySR(配置B) 全量数据 × 5个种子独立搜索,比较
   - 方程结构是否复现(字符串)
   - 预测函数一致性(定义域内 3000 点网格上的两两 Pearson 相关)
   - 变量/交互项清单
2. 物理一致性: 对每个种子方程与本文11项方程(对照),在351个数据点上做前向差分:
   - dλ/dW 负值占比 (物理预期:含水率升高使λ增大)
   - dλ/dT 负值占比 (预期:总体单调增)
   - dλ/dFT 负值占比 (预期:冻融损伤累积使λ增大)
   - 网格上非物理预测 (λ<0 或 λ>0.1) 占比, 是否含除法/指数导致奇异
输出: stability_results.json + 控制台汇总
"""
import os
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES = ['温度', '湿度', '冻融次数', '裂缝长度', '含水率']
FEATURES_EN = ['T', 'RH', 'FT', 'CR', 'W']
TARGET = '导热系数'
SEEDS = [0, 1, 2, 3, 4]
N_GRID = 3000
DOMAIN = {'温度': (5, 45), '湿度': (30, 90), '冻融次数': (0, 24),
          '裂缝长度': (0.0, 10.0), '含水率': (0, 56)}
STEPS = {'温度': 1.0, '湿度': 2.0, '冻融次数': 0.5, '裂缝长度': 0.2, '含水率': 1.0}
RNG = np.random.RandomState(42)


def load_data():
    df_deg = pd.read_excel(os.path.join(DATA_DIR, 'degradation_dataset.xlsx'), sheet_name='degradation_data')
    df_deg.columns = ['样本', '温度', '湿度', '冻融次数', '裂缝长度', '含水率', '导热系数']
    df_deg = df_deg[FEATURES + [TARGET]].copy()
    df_sum = pd.read_excel(os.path.join(DATA_DIR, 'intact_conditions_summary.xlsx'),
                           sheet_name='summary')
    df_int = pd.DataFrame({
        '温度': df_sum['Temperature_C'], '湿度': df_sum['RH_pct'], '冻融次数': 0,
        '裂缝长度': 0.0, '含水率': 0, '导热系数': df_sum['Thermal_Conductivity_Mean'],
    })
    return pd.concat([df_deg, df_int], ignore_index=True)


def build_terms(df):
    T, RH, FT, CR, W = (df['温度'].values, df['湿度'].values,
                        df['冻融次数'].values, df['裂缝长度'].values, df['含水率'].values)
    sFT, sCR = np.sqrt(FT), np.sqrt(CR)
    return np.column_stack([
        T, T ** 2, RH ** 2, sFT, sCR, W, W ** 2,
        T * sCR, RH * W, sFT * sCR, RH * sCR])


class OlsRef:
    """本文11项方程(全量拟合),接口与 PySRRegressor.predict 一致"""

    def __init__(self, df):
        X, y = build_terms(df), df[TARGET].values
        self.m = LinearRegression().fit(X, y)
        self.eq = ('λ = b0 + b1·T + b2·T² + b3·RH² + b4·√FT + b5·√CR + b6·W + b7·W² '
                   '+ b8·T√CR + b9·RH·W + b10·√FT√CR + b11·RH√CR')

    def predict(self, X):
        Xv = np.asarray(X, dtype=float)
        T, RH = Xv[:, 0], Xv[:, 1]
        FT, CR, W = Xv[:, 2], Xv[:, 3], Xv[:, 4]
        sFT, sCR = np.sqrt(FT), np.sqrt(CR)
        M = np.column_stack([T, T ** 2, RH ** 2, sFT, sCR, W, W ** 2,
                             T * sCR, RH * W, sFT * sCR, RH * sCR])
        return self.m.predict(M)


def make_grid():
    arr = np.column_stack([RNG.uniform(*DOMAIN[f], N_GRID) for f in FEATURES])
    return pd.DataFrame(arr, columns=FEATURES_EN)


def safe_stats(pred):
    """网格预测的数值健康度"""
    p = np.asarray(pred, dtype=float)
    n_nan = int(np.isnan(p).sum())
    p0 = np.nan_to_num(p, nan=0.0, posinf=1e6, neginf=-1e6)
    n_neg = int((p0 < 0).sum())
    n_extreme = int((p0 > 0.1).sum())
    return {'NaN数': n_nan, '负λ数': n_neg, 'λ>0.1数': n_extreme,
            '网格λ范围': [float(np.nanmin(p)), float(np.nanmax(p))]}


def derivative_stats(predict_fn, df):
    """在351个数据点上,对各变量做前向差分(批量预测),统计导数为负的样本占比"""
    X = df[FEATURES].values.copy()
    out = {}
    for j, f in enumerate(FEATURES):
        lo, hi = DOMAIN[f]
        h = STEPS[f]
        base_rows, pert_rows, denoms = [], [], []
        for i in range(len(X)):
            x = X[i]
            if x[j] + h <= hi:
                xp = x.copy(); xp[j] = x[j] + h
                base_rows.append(x); pert_rows.append(xp); denoms.append(h)
            elif x[j] - h >= lo:
                xm = x.copy(); xm[j] = x[j] - h
                base_rows.append(xm); pert_rows.append(x); denoms.append(h)
        B = np.asarray(predict_fn(np.array(base_rows)), dtype=float)
        P = np.asarray(predict_fn(np.array(pert_rows)), dtype=float)
        d = (P - B) / np.array(denoms)
        neg = int((d < 0).sum()); pos = int((d > 0).sum())
        out[f'dλ/d{FEATURES_EN[j]}负占比'] = round(neg / max(neg + pos, 1), 3)
    return out


def main():
    df = load_data()
    grid = make_grid()
    results = {'seeds': [], 'pairwise_grid_corr': None, 'ols_ref': {}}

    # ---- 本文11项方程(对照) ----
    ref = OlsRef(df)
    results['ols_ref'] = {
        'equation': ref.eq,
        'grid_health': safe_stats(ref.predict(grid)),
        'derivatives': derivative_stats(ref.predict, df),
    }
    print(f"[对照] 11项方程 网格健康度: {results['ols_ref']['grid_health']}")
    print(f"[对照] 11项方程 导数负占比: {results['ols_ref']['derivatives']}\n")

    # ---- PySR 多种子 ----
    from pysr import PySRRegressor
    Xdf = df[FEATURES].rename(columns=dict(zip(FEATURES, FEATURES_EN)))
    y = df[TARGET].values
    preds = []
    for seed in SEEDS:
        t0 = time.time()
        model = PySRRegressor(
            binary_operators=['+', '-', '*', '/'],
            unary_operators=['square', 'sqrt', 'exp', 'log'],
            niterations=100, populations=15, population_size=33,
            maxsize=25, model_selection='best', random_state=seed,
            deterministic=True, parallelism='serial',
            verbosity=0,
        )
        model.fit(Xdf, y)
        best = model.get_best()
        gpred = model.predict(grid)

        def pred_arr(xarr):
            return model.predict(pd.DataFrame(xarr, columns=FEATURES_EN))
        entry = {
            'seed': seed,
            'equation': best.equation,
            'complexity': int(best.complexity),
            'loss': float(best.loss),
            'contains_div': '/' in best.equation,
            'contains_exp': 'exp(' in best.equation,
            'contains_log': 'log(' in best.equation,
            'grid_health': safe_stats(gpred),
            'derivatives': derivative_stats(pred_arr, df),
        }
        results['seeds'].append(entry)
        preds.append(np.nan_to_num(np.asarray(gpred, dtype=float),
                                   nan=np.nan, posinf=np.nan, neginf=np.nan))
        print(f"[seed {seed}] complexity={entry['complexity']} loss={entry['loss']:.2e} "
              f"用时{time.time() - t0:.0f}s")
        print(f"   eq: {best.equation[:120]}")
        print(f"   grid: {entry['grid_health']}")
        print(f"   deriv neg: {entry['derivatives']}\n", flush=True)

    # ---- 种子间预测一致性 ----
    n = len(SEEDS)
    corr = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            m = ~(np.isnan(preds[i]) | np.isnan(preds[j]))
            if m.sum() > 10:
                corr[i, j] = corr[j, i] = np.corrcoef(preds[i][m], preds[j][m])[0, 1]
    results['pairwise_grid_corr'] = corr.round(3).tolist()
    off = corr[np.triu_indices(n, 1)]
    print(f"种子间网格预测 Pearson 相关: 均值={off.mean():.3f} 最小={off.min():.3f}")
    from collections import Counter
    cnt = Counter(s['equation'] for s in results['seeds'])
    repro = sum(1 for s in results['seeds'] if cnt[s['equation']] > 1)
    print(f"结构可复现种子数: {repro}/{n}(与其余至少一个种子给出相同方程)")

    with open(os.path.join(HERE, 'stability_results.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {os.path.join(HERE, 'stability_results.json')}")


if __name__ == '__main__':
    main()
