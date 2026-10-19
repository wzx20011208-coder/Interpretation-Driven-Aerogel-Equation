r"""
C_PySR基线 第5步:CatBoost 无权重主协议复现
正文 3.1: 无权重 CatBoost 5折CV R²=0.9821, MAPE=1.10%(主协议)
本脚本在同一 KFold(5, shuffle, seed=42) 下复现,作为 Table 7 黑箱上界行
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from catboost import CatBoostRegressor

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
FEATURES = ['温度', '湿度', '冻融次数', '裂缝长度', '含水率']
TARGET = '导热系数'
KF = KFold(n_splits=5, shuffle=True, random_state=42)
PARAMS = dict(iterations=300, depth=4, learning_rate=0.05, random_seed=42, verbose=False)


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


def main():
    df = load_data()
    X, y = df[FEATURES].values, df[TARGET].values
    r2s, mapes, rmses = [], [], []
    for tr, te in KF.split(X):
        m = CatBoostRegressor(**PARAMS)
        m.fit(X[tr], y[tr])
        p = m.predict(X[te])
        r2s.append(1 - np.sum((y[te] - p) ** 2) / np.sum((y[te] - np.mean(y[te])) ** 2))
        mapes.append(np.mean(np.abs((y[te] - p) / y[te])) * 100)
        rmses.append(np.sqrt(np.mean((y[te] - p) ** 2)))
    print(f'无权重 CatBoost 5折CV: R²={np.mean(r2s):.4f}±{np.std(r2s):.4f} '
          f'MAPE={np.mean(mapes):.2f}% RMSE={np.mean(rmses):.3e}')
    print('各折R²:', [f'{v:.4f}' for v in r2s])
    print('正文参考: R²=0.9821, RMSE=4.69e-4, MAPE=1.10%')
    print(f'方程恢复率: 0.9594/{np.mean(r2s):.4f} = {0.9594 / np.mean(r2s) * 100:.1f}%')


if __name__ == '__main__':
    main()
