import io
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import KFold

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')


def load_data():
    df_deg = pd.read_excel(
        os.path.join(DATA_DIR, "degradation_dataset.xlsx"), sheet_name="degradation_data"
    )
    df_deg.columns = ["样本", "温度", "湿度", "冻融次数", "裂缝长度", "含水率", "导热系数"]
    df_deg = df_deg[["温度", "湿度", "冻融次数", "裂缝长度", "含水率", "导热系数"]].copy()
    df_deg["数据来源"] = "劣化"

    df_intact_summary = pd.read_excel(
        os.path.join(DATA_DIR, "intact_conditions_summary.xlsx"),
        sheet_name="summary",
    )
    df_intact = pd.DataFrame(
        {
            "温度": df_intact_summary["Temperature_C"],
            "湿度": df_intact_summary["RH_pct"],
            "冻融次数": 0,
            "裂缝长度": 0.0,
            "含水率": 0,
            "导热系数": df_intact_summary["Thermal_Conductivity_Mean"],
            "数据来源": "完好",
        }
    )
    return pd.concat([df_deg, df_intact], ignore_index=True), df_deg


def build_X(df):
    T = df["温度"].values.astype(float)
    RH = df["湿度"].values.astype(float)
    FT = df["冻融次数"].values.astype(float)
    Crack = df["裂缝长度"].values.astype(float)
    Water = df["含水率"].values.astype(float)
    sFT, sCrack = np.sqrt(FT), np.sqrt(Crack)
    cols = {
        "T": T,
        "RH": RH,
        "√FT": sFT,
        "√Crack": sCrack,
        "Water": Water,
        "T²": T**2,
        "RH²": RH**2,
        "Water²": Water**2,
        "T×√Crack": T * sCrack,
        "RH×Water": RH * Water,
        "√FT×√Crack": sFT * sCrack,
        "RH×√Crack": RH * sCrack,
    }
    order = [
        "T",
        "RH",
        "√FT",
        "√Crack",
        "Water",
        "T²",
        "RH²",
        "Water²",
        "T×√Crack",
        "RH×Water",
        "√FT×√Crack",
        "RH×√Crack",
    ]
    return np.column_stack([cols[n] for n in order]), order


def metrics(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return r2, rmse, mape


df_all, df_deg = load_data()
y = df_all["导热系数"].values.astype(float)
X, order = build_X(df_all)
Xc = sm.add_constant(X)
model = sm.OLS(y, Xc).fit()

pred_all = model.predict(Xc)
r2_all, rmse_all, mape_all = metrics(y, pred_all)

deg_mask = (df_all["数据来源"] == "劣化").values
pred_deg = pred_all[deg_mask]
y_deg = y[deg_mask]
r2_deg, rmse_deg, mape_deg = metrics(y_deg, pred_deg)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
pred_cv = np.empty_like(y)
for tr, te in kf.split(X):
    m = sm.OLS(y[tr], sm.add_constant(X[tr])).fit()
    pred_cv[te] = m.predict(sm.add_constant(X[te]))
r2_cv, rmse_cv, mape_cv = metrics(y, pred_cv)

print("n_total", len(y), "n_degraded", int(deg_mask.sum()))
print(
    "FULL 351:    R²=%.4f  RMSE=%.6f  MAPE=%.4f%%"
    % (r2_all, rmse_all, mape_all)
)
print(
    "DEGRADED 324: R²=%.4f  RMSE=%.6f  MAPE=%.4f%%"
    % (r2_deg, rmse_deg, mape_deg)
)
print(
    "5-FOLD CV (351 refit): R²=%.4f  RMSE=%.6f  MAPE=%.4f%%"
    % (r2_cv, rmse_cv, mape_cv)
)
