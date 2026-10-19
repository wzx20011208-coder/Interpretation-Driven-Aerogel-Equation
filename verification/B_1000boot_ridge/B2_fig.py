# -*- coding: utf-8 -*-
"""B2 系数路径图：标准化系数随 alpha 变化，与 OLS 参考对比。"""
import os
import sys
import io
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
plt.rcParams['font.family'] = ['DejaVu Sans', 'SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

BASE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(BASE, 'B2_ridge_sensitivity.json'), encoding='utf-8'))

names = list(data['vif'].keys())
alphas = [g['alpha'] for g in data['ridge_grid']]
coefs = {n: [g['coefs'][n] for g in data['ridge_grid']] for n in names}
ols = data['reference_ols_std']
alpha_star = data['ridge_cv']['alpha']

fig, ax = plt.subplots(figsize=(9, 6))
for n in names:
    ax.plot(alphas, coefs[n], marker='o', ms=3, lw=1.3, label=n)
for n in names:
    ax.axhline(ols[n], color='gray', lw=0.6, ls='--', alpha=0.5)
ax.axvline(alpha_star, color='black', lw=1.2, ls=':', label=f'RidgeCV α={alpha_star:g}')
ax.set_xscale('log')
ax.set_xlabel('Ridge 惩罚系数 α（标准化尺度）')
ax.set_ylabel('标准化回归系数')
ax.set_title('Ridge 敏感性：系数符号与 OLS 对照')
ax.legend(ncol=2, fontsize=8, frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()
out_png = os.path.join(BASE, 'Fig_B2_Ridge_Path.png')
fig.savefig(out_png, dpi=300)
print("saved:", out_png)
