"""
Production quality analysis: statistical decision making.
Run: python production_analysis.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)
os.makedirs("figures", exist_ok=True)
SEED = 42

# =====================================================================
# 1. LOAD AND CHECK THE DATA
# =====================================================================
df = pd.read_csv("data/production_data.csv", parse_dates=["ProductionDate"])
print("Shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nDate range:", df["ProductionDate"].min(), "to", df["ProductionDate"].max())

CONT = ["ProductionTime_min", "Temperature_C", "Humidity_pct",
        "Pressure_kPa", "Speed_units_per_min"]
Y = "QualityScore"

q = df[Y]
print("\nQualityScore summary:\n", q.describe().round(2))
print("Skewness:", round(stats.skew(q), 2), " Excess kurtosis:", round(stats.kurtosis(q), 2))
print("Scores equal to 100 (ceiling):", int((q == 100).sum()), " Scores below 60:", int((q < 60).sum()))

# QualityLabel and DefectCount are outcomes of the process, not inputs: exclude from models
print("\nQualityScore range by QualityLabel:\n",
      df.groupby("QualityLabel")[Y].agg(["min", "max", "count"]))

# Are rows with missing values different?
miss_any = df[CONT].isnull().any(axis=1)
print("\nRows with at least one missing value:", int(miss_any.sum()))
print("Mean score: missing rows =", round(df.loc[miss_any, Y].mean(), 2),
      " complete rows =", round(df.loc[~miss_any, Y].mean(), 2))
print("Welch t-test:", stats.ttest_ind(df.loc[miss_any, Y], df.loc[~miss_any, Y], equal_var=False))

# Median imputation (keeps all 1,000 rows); listwise deletion is used as a sensitivity check
data = df.copy()
for c in CONT:
    data[c] = data[c].fillna(data[c].median())

# =====================================================================
# 2. EXPLORATORY DATA ANALYSIS
# =====================================================================
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(q, bins=30, kde=True, ax=ax[0]); ax[0].set_title("Distribution of QualityScore")
stats.probplot(q, plot=ax[1]); ax[1].set_title("Normal Q-Q plot of QualityScore")
plt.tight_layout(); plt.savefig("figures/quality_distribution.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 3, figsize=(14, 4))
for a, c in zip(ax, ["MachineID", "Shift", "Product"]):
    sns.boxplot(data=data, x=c, y=Y, ax=a); a.set_title(f"QualityScore by {c}")
plt.tight_layout(); plt.savefig("figures/quality_by_group.png", dpi=150); plt.close()

num = CONT + [Y]
corr = data[num].corr()
print("\nCorrelation matrix:\n", corr.round(2))
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Matrix")
plt.tight_layout(); plt.savefig("figures/correlation.png", dpi=150); plt.close()

print("\nCorrelation with QualityScore (Pearson r, p-value; Spearman rho):")
for c in CONT:
    r, p = stats.pearsonr(data[c], data[Y])
    rho, _ = stats.spearmanr(data[c], data[Y])
    print(f"  {c:22s} r={r:6.3f}  p={p:.2e}  rho={rho:6.3f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
sns.regplot(data=data, x="Humidity_pct", y=Y, scatter_kws={"s": 12, "alpha": .5}, ax=ax[0])
sns.regplot(data=data, x="Temperature_C", y=Y, scatter_kws={"s": 12, "alpha": .5}, ax=ax[1])
plt.tight_layout(); plt.savefig("figures/humidity_temperature_scatter.png", dpi=150); plt.close()

# =====================================================================
# 3. HYPOTHESIS TESTS (ANOVA / t-test)
# =====================================================================
print("\n" + "=" * 60 + "\nANOVA FOR EACH CATEGORICAL FACTOR\n" + "=" * 60)
anova_rows = []
for factor in ["MachineID", "Shift", "Product", "OperatorID"]:
    groups = [g[Y].values for _, g in data.groupby(factor)]
    F, p = stats.f_oneway(*groups)
    H, p_kw = stats.kruskal(*groups)
    lev_stat, lev_p = stats.levene(*groups)
    grand = data[Y].mean()
    ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
    ss_total = ((data[Y] - grand) ** 2).sum()
    eta2 = ss_between / ss_total
    anova_rows.append([factor, len(groups), F, p, eta2, p_kw, lev_p])
anova = pd.DataFrame(anova_rows, columns=["Factor", "Groups", "F", "p_ANOVA",
                                          "eta_sq", "p_KruskalWallis", "p_Levene"])
# Holm correction for testing four factors
order = anova["p_ANOVA"].argsort().values
holm = np.empty(len(anova))
running = 0
for rank, i in enumerate(order):
    running = max(running, min(1, (len(anova) - rank) * anova.loc[i, "p_ANOVA"]))
    holm[i] = running
anova["p_Holm"] = holm
print(anova.round(4).to_string(index=False))

print("\nTukey HSD post-hoc test for MachineID:")
mach = [g[Y].values for _, g in data.groupby("MachineID")]
names = sorted(data["MachineID"].unique())
tk = stats.tukey_hsd(*mach)
rows = []
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        rows.append([f"{names[i]} vs {names[j]}", mach[i].mean() - mach[j].mean(), tk.pvalue[i, j]])
print(pd.DataFrame(rows, columns=["Pair", "MeanDiff", "p_adj"]).round(4).to_string(index=False))

print("\nShift comparisons (Welch t-tests):")
for a, b in [("Morning", "Night"), ("Morning", "Afternoon"), ("Afternoon", "Night")]:
    t = stats.ttest_ind(data.loc[data.Shift == a, Y], data.loc[data.Shift == b, Y], equal_var=False)
    print(f"  {a} vs {b}: t={t.statistic:.3f}, p={t.pvalue:.4f}")

# =====================================================================
# 4. REGRESSION (OLS BUILT WITH LINEAR ALGEBRA)
# =====================================================================
def ols(X, y, names):
    """OLS via least squares. Returns a results dict (classical and HC3 robust SEs)."""
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)          # solves min ||y - X b||^2
    fitted = X @ beta
    resid = y - fitted
    XtX_inv = np.linalg.inv(X.T @ X)
    sigma2 = resid @ resid / (n - k)
    se = np.sqrt(np.diag(sigma2 * XtX_inv))
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)            # leverage (hat-matrix diagonal)
    meat = X.T @ (X * (resid / (1 - h))[:, None] ** 2)
    se_hc3 = np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), n - k)
    t3 = beta / se_hc3
    p3 = 2 * stats.t.sf(np.abs(t3), n - k)
    tss = ((y - y.mean()) ** 2).sum()
    rss = resid @ resid
    r2 = 1 - rss / tss
    adj = 1 - (1 - r2) * (n - 1) / (n - k)
    F = ((tss - rss) / (k - 1)) / (rss / (n - k))
    table = pd.DataFrame({"coef": beta, "se": se, "t": t, "p": p,
                          "se_HC3": se_hc3, "p_HC3": p3}, index=names)
    return dict(beta=beta, fitted=fitted, resid=resid, table=table, r2=r2, adj=adj,
                F=F, pF=stats.f.sf(F, k - 1, n - k), rss=rss, n=n, k=k, X=X)

def design(frame, cont, cats):
    parts = [frame[cont].astype(float)]
    for c in cats:
        parts.append(pd.get_dummies(frame[c], prefix=c, drop_first=True).astype(float))
    D = pd.concat(parts, axis=1)
    D.insert(0, "Intercept", 1.0)
    return D

cats = ["MachineID", "Shift"]
D = design(data, CONT, cats)
y = data[Y].to_numpy(float)
m1 = ols(D.to_numpy(), y, list(D.columns))
print("\n" + "=" * 60 + "\nMULTIPLE REGRESSION\n" + "=" * 60)
print(m1["table"].round(4))
print(f"R2={m1['r2']:.4f}  Adj R2={m1['adj']:.4f}  F={m1['F']:.2f}  p(F)={m1['pF']:.2e}  n={m1['n']}")

# ----- Linear algebra of the design matrix -----
Xm = D.to_numpy()
print("\nCondition number of raw design matrix:", f"{np.linalg.cond(Xm):.3e}")
cols_scaled = [c for c in D.columns if c != "Intercept"]
Z = (D[cols_scaled] - D[cols_scaled].mean()) / D[cols_scaled].std()
Zm = np.column_stack([np.ones(len(Z)), Z.to_numpy()])
print("Condition number after standardising:", round(np.linalg.cond(Zm), 2))
print("Singular values of standardised predictors:",
      np.round(np.linalg.svd(Z.to_numpy(), compute_uv=False), 2))
R = np.corrcoef(D[CONT].to_numpy(), rowvar=False)
vif = pd.Series(np.diag(np.linalg.inv(R)), index=CONT)
print("\nVIF (continuous predictors):\n", vif.round(3))
print("\nPressure_kPa: mean =", round(data["Pressure_kPa"].mean(), 2),
      " sd =", round(data["Pressure_kPa"].std(), 2))

# ----- Residual diagnostics -----
res, fit = m1["resid"], m1["fitted"]
sh = stats.shapiro(res)
jb = stats.jarque_bera(res)
print("\nDIAGNOSTICS")
print("Residual skew:", round(stats.skew(res), 3), " excess kurtosis:", round(stats.kurtosis(res), 3))
print("Shapiro-Wilk (all residuals): W=%.4f p=%.2e" % (sh.statistic, sh.pvalue))
print("Jarque-Bera: JB=%.1f p=%.2e" % (jb.statistic, jb.pvalue))
# Breusch-Pagan (LM version): regress squared residuals on X
aux = ols(Xm, res ** 2, list(D.columns))
lm = len(res) * aux["r2"]
print("Breusch-Pagan LM=%.3f p=%.4f" % (lm, stats.chi2.sf(lm, Xm.shape[1] - 1)))
dw = np.sum(np.diff(res) ** 2) / np.sum(res ** 2)
print("Durbin-Watson:", round(dw, 3))
lev = np.einsum("ij,jk,ik->i", Xm, np.linalg.inv(Xm.T @ Xm), Xm)
cooks = (res ** 2 / (m1["rss"] / (m1["n"] - m1["k"]) * m1["k"])) * lev / (1 - lev) ** 2
print("Max Cook's distance:", round(cooks.max(), 3), " obs with D > 4/n:", int((cooks > 4 / m1["n"]).sum()))

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].scatter(fit, res, s=10, alpha=.6); ax[0].axhline(0, color="k")
ax[0].set(xlabel="Fitted", ylabel="Residual", title="Residuals vs Fitted")
stats.probplot(res, plot=ax[1]); ax[1].set_title("Residual Q-Q plot")
ax[2].hist(res, bins=40); ax[2].set_title("Residual histogram")
plt.tight_layout(); plt.savefig("figures/residual_diagnostics.png", dpi=150); plt.close()

# ----- Sensitivity checks -----
print("\nSENSITIVITY CHECKS (coefficients, HC3 p-values)")
def quick(frame, label):
    Dx = design(frame, CONT, cats)
    m = ols(Dx.to_numpy(), frame[Y].to_numpy(float), list(Dx.columns))
    t = m["table"].loc[["Temperature_C", "Humidity_pct", "Speed_units_per_min", "ProductionTime_min"]]
    print(f"{label}: R2={m['r2']:.3f}")
    print(t[["coef", "p_HC3"]].round(4).T.to_string())
quick(df.dropna(subset=CONT), "Listwise deletion (n=%d)" % df.dropna(subset=CONT).shape[0])
quick(data[data[Y] < 100], "Excluding scores capped at 100 (n=%d)" % (data[Y] < 100).sum())
quick(data[data[Y] >= 60], "Excluding scores below 60 (n=%d)" % (data[Y] >= 60).sum())

# =====================================================================
# 5. NON-LINEAR EFFECTS AND OPTIMUM (CALCULUS)
# =====================================================================
d2 = data.copy()
for c in ["Temperature_C", "Humidity_pct"]:
    d2[c + "_c"] = d2[c] - d2[c].mean()
    d2[c + "_c2"] = d2[c + "_c"] ** 2
cont2 = ["ProductionTime_min", "Temperature_C_c", "Temperature_C_c2",
         "Humidity_pct_c", "Humidity_pct_c2", "Pressure_kPa", "Speed_units_per_min"]
D2 = design(d2, cont2, cats)
m2 = ols(D2.to_numpy(), y, list(D2.columns))
print("\n" + "=" * 60 + "\nQUADRATIC MODEL\n" + "=" * 60)
print(m2["table"].loc[["Temperature_C_c", "Temperature_C_c2", "Humidity_pct_c", "Humidity_pct_c2"]].round(4))
print(f"R2={m2['r2']:.4f}  Adj R2={m2['adj']:.4f}")
# Nested F-test: do the two squared terms improve the linear model?
df_num = m2["k"] - m1["k"]
Fnest = ((m1["rss"] - m2["rss"]) / df_num) / (m2["rss"] / (m2["n"] - m2["k"]))
print("Nested F-test (quadratic vs linear): F=%.3f, p=%.4f" % (Fnest, stats.f.sf(Fnest, df_num, m2["n"] - m2["k"])))
for v in ["Temperature_C", "Humidity_pct"]:
    b1 = m2["table"].loc[v + "_c", "coef"]; b2 = m2["table"].loc[v + "_c2", "coef"]
    mu = d2[v].mean()
    print(f"\n{v}: dQ/dx = b1 + 2*b2*x_c = {b1:.4f} + 2*({b2:.4f})*x_c")
    x_star = mu - b1 / (2 * b2)
    kind = "maximum" if b2 < 0 else "minimum"
    print(f"  stationary point at {x_star:.2f} ({kind}); observed range {d2[v].min():.1f} to {d2[v].max():.1f}")

# =====================================================================
# 6. RANDOM FOREST
# =====================================================================
X_rf = pd.get_dummies(data[CONT + cats], drop_first=True).astype(float)
Xtr, Xte, ytr, yte = train_test_split(X_rf, data[Y], test_size=0.2, random_state=SEED)
rf = RandomForestRegressor(n_estimators=300, random_state=SEED).fit(Xtr, ytr)
pred = rf.predict(Xte)
print("\n" + "=" * 60 + "\nRANDOM FOREST\n" + "=" * 60)
print("Test R2: %.4f  RMSE: %.2f" % (r2_score(yte, pred), np.sqrt(mean_squared_error(yte, pred))))
lin = ols(np.column_stack([np.ones(len(Xtr)), Xtr.to_numpy()]), ytr.to_numpy(), ["i"] + list(Xtr.columns))
lin_pred = np.column_stack([np.ones(len(Xte)), Xte.to_numpy()]) @ lin["beta"]
print("Linear regression test R2 (same split): %.4f  RMSE: %.2f" % (
    r2_score(yte, lin_pred), np.sqrt(mean_squared_error(yte, lin_pred))))
pi = permutation_importance(rf, Xte, yte, n_repeats=30, random_state=SEED)
imp = pd.DataFrame({"perm_importance": pi.importances_mean, "sd": pi.importances_std,
                    "impurity_importance": rf.feature_importances_}, index=X_rf.columns)
imp = imp.sort_values("perm_importance", ascending=False)
print(imp.round(4).head(10))
imp["perm_importance"].head(8).iloc[::-1].plot.barh(xerr=imp["sd"].head(8).iloc[::-1], figsize=(7, 4))
plt.title("Permutation importance (drop in test R2)"); plt.tight_layout()
plt.savefig("figures/permutation_importance.png", dpi=150); plt.close()