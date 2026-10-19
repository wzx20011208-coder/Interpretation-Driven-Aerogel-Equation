# -*- coding: utf-8 -*-
"""表6补齐：按工况（完好/单因素/耦合）计算方程、黑箱、EN ISO 的 R²/RMSE/MAPE。"""
import io
import json
import os
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import curve_fit
from sklearn.model_selection import KFold
import statsmodels.api as sm

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
OUT = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["温度", "湿度", "冻融次数", "裂缝长度", "含水率"]
TARGET = "导热系数"
RANDOM_STATE = 42
CB_PARAMS = dict(iterations=300, depth=4, learning_rate=0.05, verbose=0, random_seed=RANDOM_STATE)

# 最终显式方程系数（11 项 + 常数，与论文一致）
COEFS = {
    "const": 2.338e-2, "T": 2.217e-4, "T²": -8.980e-7, "RH²": 2.983e-7,
    "√FT": 5.084e-4, "√Crack": 5.687e-4, "Water": 1.724e-4, "Water²": -1.423e-6,
    "T×√Crack": -2.123e-5, "RH×Water": -2.909e-7, "√FT×√Crack": 1.604e-4,
    "RH×√Crack": -4.117e-6,
}
TERMS = ["const", "T", "T²", "RH²", "√FT", "√Crack", "Water", "Water²",
         "T×√Crack", "RH×Water", "√FT×√Crack", "RH×√Crack"]


def load_data():
    df_deg = pd.read_excel(os.path.join(DATA_DIR, "degradation_dataset.xlsx"), sheet_name="degradation_data")
    df_deg.columns = ["样本", "温度", "湿度", "冻融次数", "裂缝长度", "含水率", "导热系数"]
    df_deg = df_deg[FEATURES + [TARGET]].copy()
    df_intact = pd.read_excel(
        os.path.join(DATA_DIR, "intact_conditions_summary.xlsx"),
        sheet_name="summary",
    )
    df_intact = pd.DataFrame({
        "温度": df_intact["Temperature_C"],
        "湿度": df_intact["RH_pct"],
        "冻融次数": 0,
        "裂缝长度": 0.0,
        "含水率": 0,
        "导热系数": df_intact["Thermal_Conductivity_Mean"],
    })
    return pd.concat([df_deg, df_intact], ignore_index=True)


def scenario(row):
    ft, cr, wt = row["冻融次数"], row["裂缝长度"], row["含水率"]
    if ft == 0 and cr == 0 and wt == 0:
        return "完好"
    if ft > 0 and cr == 0 and wt == 0:
        return "纯冻融"
    if ft == 0 and cr > 0 and wt == 0:
        return "纯裂缝"
    if ft == 0 and cr == 0 and wt > 0:
        return "纯渗水"
    return "复合"


def build_design(df):
    T = df["温度"].values.astype(float)
    RH = df["湿度"].values.astype(float)
    FT = df["冻融次数"].values.astype(float)
    Cr = df["裂缝长度"].values.astype(float)
    W = df["含水率"].values.astype(float)
    sFT, sCr = np.sqrt(FT), np.sqrt(Cr)
    return np.column_stack([
        np.ones_like(T), T, T**2, RH**2, sFT, sCr, W, W**2,
        T*sCr, RH*W, sFT*sCr, RH*sCr,
    ])


def eq_predict(df, coefs=None):
    c = coefs or COEFS
    return build_design(df) @ np.array([c[k] for k in TERMS])


def metrics(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    mape = float(np.mean(np.abs((y - p) / y)) * 100)
    return r2, rmse, mape


def en_iso(x, lref, fT, fm):
    T, RH = x
    return lref * np.exp(fT * (T - 20)) * np.exp(fm * (RH - 50))


def en_iso_damage(x, lref, fT, fm, kFT, kCR, kWT):
    T, RH, FT, Cr, W = x
    return lref * np.exp(fT * (T - 20)) * np.exp(fm * (RH - 50)) * (1 + kFT * FT + kCR * Cr + kWT * W)


def main():
    df = load_data()
    df["工况"] = df.apply(scenario, axis=1)
    df["分组"] = np.where(df["工况"] == "完好", "完好",
                  np.where(df["工况"] == "复合", "复合", "单因素"))

    y = df[TARGET].values.astype(float)
    Xf = df[FEATURES].values.astype(float)
    y_arr = df[TARGET].values.astype(float)

    # 1) 显式方程（固定系数，全量拟合口径）
    p_eq = eq_predict(df)

    # 2) 黑箱 CatBoost（全量训练，样本内）
    cb = CatBoostRegressor(**CB_PARAMS)
    cb.fit(Xf, y_arr)
    p_cb = cb.predict(Xf)

    # 3) EN ISO 10456 基础版（完好拟合）
    intact_mask = df["分组"] == "完好"
    T27 = df.loc[intact_mask, "温度"].values.astype(float)
    RH27 = df.loc[intact_mask, "湿度"].values.astype(float)
    y27 = df.loc[intact_mask, TARGET].values.astype(float)
    popt, _ = curve_fit(en_iso, (T27, RH27), y27, p0=[0.028, 0.005, 0.002], maxfev=100000)
    p_en = en_iso((df["温度"].values.astype(float), df["湿度"].values.astype(float)), *popt)

    # 4) EN ISO 10456 损伤扩展版（全量拟合）
    XX = (df["温度"].values.astype(float), df["湿度"].values.astype(float),
          df["冻融次数"].values.astype(float), df["裂缝长度"].values.astype(float),
          df["含水率"].values.astype(float))
    poptd, _ = curve_fit(en_iso_damage, XX, y_arr,
                         p0=[0.028, 0.005, 0.002, 0.001, 0.001, 0.001], maxfev=200000)
    p_end = en_iso_damage(XX, *poptd)

    # 5) 5 折 CV OOF：方程（逐折 OLS 重拟合）与黑箱
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    eq_oof = np.zeros_like(y_arr)
    cb_oof = np.zeros_like(y_arr)
    for tr, te in kf.split(Xf):
        ols = sm.OLS(y_arr[tr], build_design(df.iloc[tr])).fit()
        eq_oof[te] = ols.predict(build_design(df.iloc[te]))
        m = CatBoostRegressor(**CB_PARAMS)
        m.fit(Xf[tr], y_arr[tr])
        cb_oof[te] = m.predict(Xf[te])

    models = {
        "显式方程(样本内)": p_eq,
        "黑箱CatBoost(样本内)": p_cb,
        "EN ISO 10456": p_en,
        "EN ISO 10456扩展": p_end,
        "显式方程(5折CV OOF)": eq_oof,
        "黑箱CatBoost(5折CV OOF)": cb_oof,
    }

    groups = ["完好", "纯冻融", "纯裂缝", "纯渗水", "复合", "单因素", "全部"]
    out = {}
    header = "%-22s %8s %8s %8s %8s %8s %8s" % (
        "模型", "完好27", "纯冻融81", "纯裂缝81", "纯渗水81", "复合81", "全部351")
    print(header)
    for mname, pred in models.items():
        out[mname] = {}
        row = []
        for g in groups:
            mask = df["分组"] == g if g in ("完好", "单因素", "复合") else (
                df["工况"] == g if g != "全部" else np.ones(len(df), dtype=bool))
            if g == "全部":
                mask = np.ones(len(df), dtype=bool)
            yy = y_arr[mask]
            pp = pred[mask]
            r2, rmse, mape = metrics(yy, pp)
            out[mname][g] = {"n": int(mask.sum()), "r2": r2, "rmse": rmse, "mape": mape}
            row.append("%.3f/%.2f%%" % (r2, mape))
        print("%-22s %s" % (mname, " | ".join(row)))

    with open(os.path.join(OUT, "table6_matrix.json"), "w", encoding="utf-8") as f:
        json.dump({"en_iso_params": {"base": popt.tolist(), "damage": poptd.tolist()},
                   "models": out}, f, ensure_ascii=False, indent=1)
    print("saved:", os.path.join(OUT, "table6_matrix.json"))


if __name__ == "__main__":
    main()
