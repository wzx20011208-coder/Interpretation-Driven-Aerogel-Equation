r"""
C_PySR基线 第6步:CR 符号反转复现核验
=====================================
主张待核验:"盲搜(即使保留 CR)未复现裂缝贡献的环境依赖符号反转"
做法:
  1. 本文 11 项方程(全集 OLS,与论文终版同法)求 ∂λ/∂CR,统计正负号分布(基准)
  2. 全部含 CR 的 PySR 方程(A40/B40/B60 CV 折 + maxsize=25 稳定性种子)做同样计算
  3. 比较负号占比与"是否出现双号"(=符号反转)
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


T, RH, FT, CR, W = sp.symbols('T RH FT CR W', real=True)
LOCALS = {'T': T, 'RH': RH, 'FT': FT, 'CR': CR, 'W': W,
          'square': lambda x: x ** 2, 'sqrt': sp.sqrt, 'exp': sp.exp, 'log': sp.log}

OUR_TERMS = [T, T ** 2, RH ** 2, sp.sqrt(FT), sp.sqrt(CR), W, W ** 2,
             T * sp.sqrt(CR), RH * W, sp.sqrt(FT) * sp.sqrt(CR), RH * sp.sqrt(CR)]


def our_equation(df):
    T_, RH_, FT_, CR_, W_ = (df['温度'].values, df['湿度'].values,
                             df['冻融次数'].values, df['裂缝长度'].values, df['含水率'].values)
    sFT, sCR = np.sqrt(FT_), np.sqrt(CR_)
    X = np.column_stack([T_, T_ ** 2, RH_ ** 2, sFT, sCR, W_, W_ ** 2,
                         T_ * sCR, RH_ * W_, sFT * sCR, RH_ * sCR])
    m = LinearRegression().fit(X, df[TARGET].values)
    return m.intercept_ + sum(c * t for c, t in zip(m.coef_, OUR_TERMS))


def sign_stats(expr, df, label):
    d = sp.diff(expr, CR)
    f = sp.lambdify((T, RH, FT, CR, W), d, 'numpy')
    vals = np.array([float(f(r.温度, r.湿度, r.冻融次数, r.裂缝长度, r.含水率))
                     for _, r in df.iterrows()], dtype=float)
    finite = np.isfinite(vals)
    neg = np.sum(vals[finite] < 0)
    pos = np.sum(vals[finite] > 0)
    both = neg > 0 and pos > 0
    cr_pos = (df['裂缝长度'].values > 0) & finite
    neg_crp = np.sum((vals < 0)[cr_pos])
    pos_crp = np.sum((vals > 0)[cr_pos])
    print(f"{label:<28} 有限点 {finite.sum():>3}/351 | 负 {neg:>3}({neg / max(finite.sum(), 1) * 100:4.1f}%) "
          f"正 {pos:>3} | CR>0子集 负 {neg_crp}/{cr_pos.sum()} | 双号(符号反转): {'是' if both else '否'}")
    return {'label': label, 'n_finite': int(finite.sum()), 'n_neg': int(neg), 'n_pos': int(pos),
            'sign_reversal': bool(both), 'crpos_neg': int(neg_crp), 'crpos_total': int(cr_pos.sum())}


def pysr_eqs():
    items = []
    for fn, tag in [('maxsize_results_A40.json', 'A40'), ('maxsize_results_B40.json', 'B40'),
                    ('maxsize_results_B60.json', 'B60')]:
        d = json.load(open(os.path.join(HERE, fn), encoding='utf-8'))
        for i, (eq, has) in enumerate(zip(d['fold_equations'], d['fold_has_CR'])):
            if has:
                items.append((f'PySR {tag} fold{i + 1}', eq))
    d = json.load(open(os.path.join(HERE, 'stability_results.json'), encoding='utf-8'))
    for s in d['seeds']:
        if 'CR' in s['equation']:
            items.append((f"PySR 种子{s.get('seed', '?')} (maxsize=25)", s['equation']))
    return items


def main():
    df = load_data()
    results = []

    print('===== 本文 11 项方程(全集 OLS)=====')
    expr = our_equation(df)
    print('方程:', sp.simplify(sp.N(expr, 4)), '\n')
    results.append(sign_stats(expr, df, '本文 11 项方程'))

    print('\n===== 含 CR 的 PySR 方程 =====')
    for label, eq in pysr_eqs():
        print(f'{label}: {eq[:90]}...' if len(eq) > 90 else f'{label}: {eq}')
        e = sp.sympify(eq, locals=LOCALS)
        results.append(sign_stats(e, df, label))
        print()

    fn = os.path.join(HERE, 'sign_check_results.json')
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    n_rev = sum(r['sign_reversal'] for r in results[1:])
    print(f'== 结论:含 CR 的 PySR 方程共 {len(results) - 1} 个,复现符号反转的 {n_rev} 个 ==')
    print('saved', fn)


if __name__ == '__main__':
    main()
