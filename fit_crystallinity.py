import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit

parser = argparse.ArgumentParser()
parser.add_argument("tbl", nargs="?", default="melts-liquid.tbl",
                    help="Path to melts-liquid.tbl")
args = parser.parse_args()
tbl_path = Path(args.tbl)
out_dir  = tbl_path.parent

df = pd.read_csv(tbl_path, header=0)
df.columns = df.columns.str.strip()

V1 = df.loc[df["Index"] == 1, "liq V (cc)"].values[0]
P_MPa = df["P (kbars)"].values * 100.0
crystallinity = 1.0 - df["liq V (cc)"].values / V1

def r2_score(C_obs, C_pred):
    ss_res = np.sum((C_obs - C_pred)**2)
    ss_tot = np.sum((C_obs - C_obs.mean())**2)
    return 1.0 - ss_res / ss_tot

def aic_score(C_obs, C_pred, k):
    n = len(C_obs)
    ss_res = np.sum((C_obs - C_pred)**2)
    return n * np.log(ss_res / n) + 2 * k

# ---------- Model A: exponential only  C = A * exp(-k * P) ----------
def exp_only(P, A, k):
    return A * np.exp(-k * P)

# Robust initial guess via log-linear fit on positive-crystallinity points
pos = crystallinity > 0
if pos.sum() >= 2:
    log_c = np.log(crystallinity[pos])
    slope, intercept = np.polyfit(P_MPa[pos], log_c, 1)
    k0 = max(-slope, 1e-6)   # k must be positive
    A0 = np.exp(intercept)
else:
    k0, A0 = 0.04, 1.0

popt_exp, _ = curve_fit(exp_only, P_MPa, crystallinity, p0=[A0, k0], maxfev=50000)
A_exp, k_exp = popt_exp
C_pred_exp = exp_only(P_MPa, *popt_exp)
r2_exp  = r2_score(crystallinity, C_pred_exp)
aic_exp = aic_score(crystallinity, C_pred_exp, k=2)

print(f"Model A — Exponential only:")
print(f"  C = {A_exp:.6f} * exp(-{k_exp:.6f} * P)")
print(f"  R² = {r2_exp:.6f},  AIC = {aic_exp:.2f}")

# ---------- Model B: piecewise  exp (P>=BP) + quadratic (P<BP), BP fitted ----------
def piecewise(params, P):
    A, k, a3, b3, BP = params
    v2 = A * np.exp(-k * BP)
    m_hi = P >= BP
    m_lo = P < BP
    C = np.empty(len(P))
    C[m_hi] = A * np.exp(-k * P[m_hi])
    C[m_lo] = a3 * (P[m_lo] - BP)**2 + b3 * (P[m_lo] - BP) + v2
    return C

def cost_pw(params):
    BP = params[4]
    if BP <= P_MPa.min() or BP >= P_MPa.max():
        return 1e10
    return np.sum((crystallinity - piecewise(params, P_MPa))**2)

# Multi-start: grid search over BP candidates, keep best result
BP_candidates = np.linspace(P_MPa.min() + 5, P_MPa.max() - 5, 20)
best_res = None
for BP0 in BP_candidates:
    mask_lo0 = P_MPa < BP0
    if mask_lo0.sum() < 3 or (~mask_lo0).sum() < 3:
        continue
    c3u = np.polyfit(P_MPa[mask_lo0], crystallinity[mask_lo0], 2)
    x0 = [A_exp, k_exp, c3u[0], c3u[1], BP0]
    res = minimize(cost_pw, x0, method="Nelder-Mead",
                   options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 200000})
    if best_res is None or res.fun < best_res.fun:
        best_res = res
res_pw = best_res
A_pw, k_pw, a3_pw, b3_pw, BP_pw = res_pw.x
v2_pw = A_pw * np.exp(-k_pw * BP_pw)
C_pred_pw = piecewise(res_pw.x, P_MPa)
r2_pw  = r2_score(crystallinity, C_pred_pw)
aic_pw = aic_score(crystallinity, C_pred_pw, k=5)

print(f"\nModel B — Piecewise (exp P≥BP + quadratic P<BP, BP fitted):")
print(f"  BP = {BP_pw:.2f} MPa")
print(f"  C = {A_pw:.6f} * exp(-{k_pw:.6f} * P)  [P≥{BP_pw:.2f}]")
print(f"  C = {a3_pw:.6f}*(P-{BP_pw:.2f})² + {b3_pw:.6f}*(P-{BP_pw:.2f}) + {v2_pw:.6f}  [P<{BP_pw:.2f}]")
print(f"  R² = {r2_pw:.6f},  AIC = {aic_pw:.2f}")

# ---------- Model selection by AIC ----------
if aic_exp <= aic_pw:
    model_name = "Exponential only"
    C_pred = C_pred_exp
    r2_all = r2_exp
    print(f"\n>>> Selected: Model A (Exponential only)  — better AIC ({aic_exp:.2f} ≤ {aic_pw:.2f})")
else:
    model_name = f"Piecewise (exp + quadratic, BP={BP_pw:.1f} MPa)"
    C_pred = C_pred_pw
    r2_all = r2_pw
    print(f"\n>>> Selected: Model B (Piecewise, BP={BP_pw:.1f} MPa)  — better AIC ({aic_pw:.2f} < {aic_exp:.2f})")

# ---------- Plot ----------
P_plt = np.linspace(P_MPa.min(), P_MPa.max(), 500)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.scatter(P_MPa, crystallinity, s=4, alpha=0.3, color="steelblue", label="MELTS data")

if aic_exp <= aic_pw:
    ax.plot(P_plt, exp_only(P_plt, *popt_exp), "r-", linewidth=2,
            label=f"Exp only: C={A_exp:.4f}·exp(-{k_exp:.4f}·P)  R²={r2_exp:.4f}")
    # also show piecewise for reference
    P_hi_plt = np.linspace(BP, P_MPa.max(), 300)
    P_lo_plt = np.linspace(P_MPa.min(), BP, 300)
    ax.plot(P_hi_plt, piecewise(res_pw.x, P_hi_plt), "g--", linewidth=1.5, alpha=0.6,
            label=f"Piecewise (ref)  R²={r2_pw:.4f}")
    ax.plot(P_lo_plt, piecewise(res_pw.x, P_lo_plt), "g--", linewidth=1.5, alpha=0.6)
else:
    mask_hi_pw = P_MPa >= BP_pw
    mask_lo_pw = P_MPa <  BP_pw
    P_hi_plt = np.linspace(BP_pw, P_MPa.max(), 300)
    P_lo_plt = np.linspace(P_MPa.min(), BP_pw, 300)
    ax.plot(P_hi_plt, piecewise(res_pw.x, P_hi_plt), "r-", linewidth=2,
            label=f"Exp (P≥{BP_pw:.1f})  R²={r2_score(crystallinity[mask_hi_pw], C_pred_pw[mask_hi_pw]):.4f}")
    ax.plot(P_lo_plt, piecewise(res_pw.x, P_lo_plt), "m-", linewidth=2,
            label=f"Quad (P<{BP_pw:.1f})  R²={r2_score(crystallinity[mask_lo_pw], C_pred_pw[mask_lo_pw]):.4f}")
    ax.axvline(BP_pw, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    # also show exp-only for reference
    ax.plot(P_plt, exp_only(P_plt, *popt_exp), "g--", linewidth=1.5, alpha=0.6,
            label=f"Exp only (ref)  R²={r2_exp:.4f}")

ax.set_xlabel("P (MPa)", fontsize=12)
ax.set_ylabel("Crystallinity", fontsize=12)
ax.set_title(f"Crystallinity vs Pressure — {model_name}  (R²={r2_all:.4f})", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
res_all = crystallinity - C_pred
ax2.scatter(P_MPa, res_all, s=4, alpha=0.4, color="steelblue",
            label=f"{model_name}  R²={r2_all:.4f}")
# residuals of the non-selected model (gray reference)
res_other = crystallinity - (C_pred_pw if aic_exp <= aic_pw else C_pred_exp)
ax2.scatter(P_MPa, res_other, s=4, alpha=0.2, color="gray",
            label="Other model (ref)")
ax2.axhline(0, color="black", linewidth=1)
ax2.set_xlabel("P (MPa)", fontsize=12)
ax2.set_ylabel("Residuals (crystallinity)", fontsize=12)
ax2.set_title("Residuals", fontsize=13)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_dir / "crystallinity_vs_P.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_dir / 'crystallinity_vs_P.png'}")
