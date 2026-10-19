r"""
C_PySR基线 第3步:迭代预算敏感性(防"训练不足"质疑)
- 配置A/B,niterations=300(5倍预算),其余同 01
- 同一 KFold(5, shuffle, seed=42)
"""
import os
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
    'A_通用算子_300it': dict(binary_operators=['+', '-', '*', '/'],
                             unary_operators=['square']),
    'B_物理增强_300it': dict(binary_operators=['+', '-', '*', '/'],
                             unary_operators=['square', 'sqrt', 'exp', 'log']),
}
COMMON = dict(niterations=300, populations=15, population_size=33,
              maxsize=25, model_selection='best', random_state=42,
              deterministic=True, parallelism='serial', verbosity=0)


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
    df = load_data()
    Xdf = df[FEATURES].rename(columns=dict(zip(FEATURES, FEATURES_EN)))
    y = df[TARGET].values
    out = {}
    for name, cfg in CONFIGS.items():
        print(f'== {name} ==', flush=True)
        r2s, mapes = [], []
        for k, (tr, te) in enumerate(KF.split(Xdf)):
            t0 = time.time()
            m = PySRRegressor(**COMMON, **cfg)
            m.fit(Xdf.iloc[tr], y[tr])
            r2, mape = calc_metrics(y[te], m.predict(Xdf.iloc[te]))
            r2s.append(r2); mapes.append(mape)
            print(f'  fold{k + 1}: R2={r2:.4f} MAPE={mape:.2f}% '
                  f'c={m.get_best().complexity} 用时{time.time() - t0:.0f}s', flush=True)
        out[name] = {'fold_r2': r2s, 'fold_mape': mapes,
                     'cv_R2': float(np.mean(r2s)), 'cv_R2_std': float(np.std(r2s)),
                     'cv_MAPE': float(np.mean(mapes))}
        print(f'  => CV R²={out[name]["cv_R2"]:.4f}±{out[name]["cv_R2_std"]:.4f} '
              f'MAPE={out[name]["cv_MAPE"]:.2f}%', flush=True)

    with open(os.path.join(HERE, 'budget_results.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('saved budget_results.json')


if __name__ == '__main__':
    main()
