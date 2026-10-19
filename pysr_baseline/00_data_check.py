# -*- coding: utf-8 -*-
r"""
C_PySR基线 第0步:数据加载与 11 项显式方程指标复现
- 数据: ../data (324 劣化 + 27 完好均值 = 351)
- 方程: v2 最终 11 项,无权重 OLS,5折CV (KFold shuffle seed=42)
- 目标: 复现论文 R2=0.9594 / MAPE=1.76%,作为 PySR 对比基准行
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
FEATURES = ['温度', '湿度', '冻融次数', '裂缝长度', '含水率']
TARGET = '导热系数'
RANDOM_STATE = 42
KF = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def load_data():
    df_deg = pd.read_excel(os.path.join(DATA_DIR, 'degradation_dataset.xlsx'), sheet_name='degradation_data')
    df_deg.columns = ['样本', '温度', '湿度', '冻融次数', '裂缝长度', '含水率', '导热系数']
    df_deg = df_deg[FEATURES + [TARGET]].copy()

    df_intact_summary = pd.read_excel(
        os.path.join(DATA_DIR, 'intact_conditions_summary.xlsx'), sheet_name='summary')
    df_intact = pd.DataFrame({
        '温度': df_intact_summary['Temperature_C'],
        '湿度': df_intact_summary['RH_pct'],
        '冻融次数': 0,
        '裂缝长度': 0.0,
        '含水率': 0,
        '导热系数': df_intact_summary['Thermal_Conductivity_Mean'],
    })
    df_all = pd.concat([df_deg, df_intact], ignore_index=True)
    return df_all


def calc_metrics(y_true, y_pred):
    r2 = 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return r2, mape


def build_terms(df):
    """v2 最终 11 项设计矩阵(与论文一致)"""
    T, RH, FT, CR, W = (df['温度'].values, df['湿度'].values,
                        df['冻融次数'].values, df['裂缝长度'].values, df['含水率'].values)
    sFT, sCR = np.sqrt(FT), np.sqrt(CR)
    X = np.column_stack([
        T, T ** 2, RH ** 2, sFT, sCR, W, W ** 2,
        T * sCR, RH * W, sFT * sCR, RH * sCR,
    ])
    names = ['T', 'T²', 'RH²', '√FT', '√Crack', 'Water', 'Water²',
             'T×√Crack', 'RH×Water', '√FT×√Crack', 'RH×√Crack']
    return X, names


def main():
    df = load_data()
    print(f"总数据: {len(df)} 条 (劣化 {len(df) - 27} + 完好 27)")
    print(f"特征范围: T[{df['温度'].min()},{df['温度'].max()}] RH[{df['湿度'].min()},{df['湿度'].max()}] "
          f"FT[{df['冻融次数'].min()},{df['冻融次数'].max()}] CR[{df['裂缝长度'].min()},{df['裂缝长度'].max()}] "
          f"W[{df['含水率'].min()},{df['含水率'].max()}]")
    print(f"λ范围: [{df['导热系数'].min():.4f}, {df['导热系数'].max():.4f}] W/(m·K)")

    X, names = build_terms(df)
    y = df[TARGET].values

    # 全量拟合(仅报告)
    m = LinearRegression().fit(X, y)
    r2_tr, mape_tr = calc_metrics(y, m.predict(X))
    print(f"\n[11项方程 全量拟合] R²={r2_tr:.4f}  MAPE={mape_tr:.2f}%")

    # 5折CV(与论文一致)
    r2s, mapes = [], []
    for tr, te in KF.split(X):
        m = LinearRegression().fit(X[tr], y[tr])
        r2, mape = calc_metrics(y[te], m.predict(X[te]))
        r2s.append(r2)
        mapes.append(mape)
    print(f"[11项方程 5折CV]   R²={np.mean(r2s):.4f}±{np.std(r2s):.4f}  MAPE={np.mean(mapes):.2f}%")
    print(f"  各折R²: {[f'{v:.4f}' for v in r2s]}")
    print(f"\n论文参考值: R²=0.9594  MAPE=1.76%")


if __name__ == '__main__':
    main()
