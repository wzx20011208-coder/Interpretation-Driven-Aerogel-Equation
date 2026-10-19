r"""
C_PySR基线 第4步:复杂度预算公平性检验(maxsize=40/60)
=====================================================
背景:第1/3步的 maxsize=25 按 PySR 节点计数(变量/常数/算子各计1)无法表达
本文11项方程(估算约55-61节点),因此"匹配复杂度预算"的精度结论存疑。
本步仅放宽 maxsize,其余协议与 03 完全一致(populations=15, niterations=300,
deterministic, serial, 同一 KFold(5, shuffle, seed=42)),隔离单变量。

用法:
    python 04_pysr_maxsize.py A 40
    python 04_pysr_maxsize.py B 40
    python 04_pysr_maxsize.py B 60
输出: maxsize_results_{配置}{size}.json
"""
import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore')
from pysr import PySRRegressor

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
HERE = os.path.dirname(os.path.abspath(__file__))
FEATURES = ['温度', '湿度', '冻融次数', '裂缝长度', '含水率']
FEATURES_EN = ['T', 'RH', 'FT', 'CR', 'W']
TARGET = '导热系数'
KF = KFold(n_splits=5, shuffle=True, random_state=42)

CONFIGS = {
    'A': dict(binary_operators=['+', '-', '*', '/'], unary_operators=['square']),
    'B': dict(binary_operators=['+', '-', '*', '/'],
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
    })
    return pd.concat([df_deg, df_int], ignore_index=True)


def calc_metrics(y_true, y_pred):
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return float(r2), mape


def main():
    which = sys.argv[1].upper() if len(sys.argv) > 1 else 'B'
    maxsize = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    cfg = CONFIGS[which]

    df = load_data()
    Xdf = df[FEATURES].rename(columns=dict(zip(FEATURES, FEATURES_EN)))
    y = df[TARGET].values

    r2s, mapes, eqs, comps, crs = [], [], [], [], []
    for k, (tr, te) in enumerate(KF.split(Xdf)):
        t0 = time.time()
        m = PySRRegressor(niterations=300, populations=15, population_size=33,
                          maxsize=maxsize, model_selection='best', random_state=42,
                          deterministic=True, parallelism='serial', verbosity=0, **cfg)
        m.fit(Xdf.iloc[tr], y[tr])
        r2, mape = calc_metrics(y[te], m.predict(Xdf.iloc[te]))
        best = m.get_best()
        has_cr = 'CR' in best.equation
        r2s.append(r2); mapes.append(mape); eqs.append(best.equation)
        comps.append(int(best.complexity)); crs.append(has_cr)
        print(f'fold{k + 1}: R2={r2:.4f} MAPE={mape:.2f}% c={best.complexity} '
              f'CR={has_cr} 用时{time.time() - t0:.0f}s', flush=True)

    out = {'config': which, 'maxsize': maxsize, 'fold_r2': r2s, 'fold_mape': mapes,
           'fold_equations': eqs, 'fold_complexity': comps, 'fold_has_CR': crs,
           'cv_R2': float(np.mean(r2s)), 'cv_R2_std': float(np.std(r2s)),
           'cv_MAPE': float(np.mean(mapes))}
    fn = os.path.join(HERE, f'maxsize_results_{which}{maxsize}.json')
    with open(fn, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f'=> CV R²={out["cv_R2"]:.4f}±{out["cv_R2_std"]:.4f} MAPE={out["cv_MAPE"]:.2f}% '
          f'CR出现折数={sum(crs)}/5')
    print('saved', fn)


if __name__ == '__main__':
    main()
