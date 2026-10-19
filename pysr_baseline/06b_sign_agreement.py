r"""
C_PySR基线 第6b步:符号模式一致性分析
=====================================
问题:B60 fold3/fold5 出现了 dλ/dCR 双号,但其负号点的分布
     是否与本文方程的环境依赖结构(干燥→负,湿润→正)一致?
做法:在 CR>0 有限点上,比较各 PySR 方程导数符号与本文方程导数符号的一致率;
     并统计各方程负号点的平均湿度/含水率(检验"干燥→负"物理模式)。
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import sympy as sp
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES = ['温度', '湿度', '冻融次数', '裂缝长度', '含水率']
TARGET = '导热系数'

T, RH, FT, CR, W = sp.symbols('T RH FT CR W', real=True)
LOCALS = {'T': T, 'RH': RH, 'FT': FT, 'CR': CR, 'W': W,
          'square': lambda x: x ** 2, 'sqrt': sp.sqrt, 'exp': sp.exp, 'log': sp.log}
OUR_TERMS = [T, T ** 2, RH ** 2, sp.sqrt(FT), sp.sqrt(CR), W, W ** 2,
             T * sp.sqrt(CR), RH * W, sp.sqrt(FT) * sp.sqrt(CR), RH * sp.sqrt(CR)]


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


def our_deriv(df):
    T_, RH_, FT_, CR_, W_ = (df['温度'].values, df['湿度'].values,
                             df['冻融次数'].values, df['裂缝长度'].values, df['含水率'].values)
    sFT, sCR = np.sqrt(FT_), np.sqrt(CR_)
    X = np.column_stack([T_, T_ ** 2, RH_ ** 2, sFT, sCR, W_, W_ ** 2,
                         T_ * sCR, RH_ * W_, sFT * sCR, RH_ * sCR])
    m = LinearRegression().fit(X, df[TARGET].values)
    expr = m.intercept_ + sum(c * t for c, t in zip(m.coef_, OUR_TERMS))
    return sp.diff(expr, CR)


def eval_deriv(d, df):
    f = sp.lambdify((T, RH, FT, CR, W), d, 'numpy')
    return np.array([float(f(r.温度, r.湿度, r.冻融次数, r.裂缝长度, r.含水率))
                     for _, r in df.iterrows()], dtype=float)


def main():
    df = load_data()
    d_ours = our_deriv(df)
    v_ours = eval_deriv(d_ours, df)
    mask = (df['裂缝长度'].values > 0)  # CR>0:162 点,本文方程在该子集导数均有限
    s_ours = np.sign(v_ours[mask])

    neg_pts = df.values[mask][s_ours < 0]
    pos_pts = df.values[mask][s_ours > 0]
    cols = FEATURES
    print('本文方程负号点环境 vs 正号点环境(均值):')
    for j, c in enumerate(cols):
        print(f'  {c:<6}: 负号点 {neg_pts[:, j].mean():8.2f} | 正号点 {pos_pts[:, j].mean():8.2f}')

    out = []
    for fn, tag in [('maxsize_results_A40.json', 'A40'), ('maxsize_results_B40.json', 'B40'),
                    ('maxsize_results_B60.json', 'B60')]:
        d = json.load(open(os.path.join(HERE, fn), encoding='utf-8'))
        for i, (eq, has) in enumerate(zip(d['fold_equations'], d['fold_has_CR'])):
            if has:
                out.append((f'{tag} fold{i + 1}', eq))
    d = json.load(open(os.path.join(HERE, 'stability_results.json'), encoding='utf-8'))
    for s in d['seeds']:
        if 'CR' in s['equation']:
            out.append((f"种子{s.get('seed', '?')}", s['equation']))

    print('\nCR>0 子集上与本文方程的符号一致率:')
    for label, eq in out:
        v = eval_deriv(sp.diff(sp.sympify(eq, locals=LOCALS), CR), df)
        vm, fin = v[mask], np.isfinite(v[mask])
        agree = np.mean(np.sign(vm[fin]) == s_ours[fin])
        n_neg = int(np.sum(vm[fin] < 0))
        print(f'  {label:<12} 有限 {fin.sum():>3}/162 | 负号点 {n_neg:>3} | 一致率 {agree * 100:5.1f}%')

    json.dump({'note': '符号一致率=CR>0且双方导数有限点上 sign(pysr)==sign(ours) 的比例;机会水平50%'},
              open(os.path.join(HERE, 'sign_agreement_summary.txt'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
