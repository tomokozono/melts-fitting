import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

# ---------- Piecewise fitting with continuity at BP2=40 MPa ----------
from scipy.optimize import minimize

BP2 = 40.0   # single breakpoint (MPa)

mask_hi = P_MPa >= BP2   # P >= 40 : exponential decay
mask_lo = P_MPa <  BP2   # P <  40 : quadratic

def piecewise(params, P):
    """
    params = [A, k, a3, b3]
    Hi (P>=BP2): C = A * exp(-k * P)               (exponential decay)
    Lo (P<BP2):  C = a3*(P-BP2)^2 + b3*(P-BP2) + v2  (quadratic through (BP2, v2))
    Continuity guaranteed: v2 = A * exp(-k * BP2)
    """
    A, k, a3, b3 = params
    v2 = A * np.exp(-k * BP2)          # junction value (continuity enforced)
    m_hi = P >= BP2
    m_lo = P < BP2
    C = np.empty(len(P))
    C[m_hi] = A * np.exp(-k * P[m_hi])
    C[m_lo] = a3 * (P[m_lo] - BP2)**2 + b3 * (P[m_lo] - BP2) + v2
    return C

def cost(params):
    return np.sum((crystallinity - piecewise(params, P_MPa))**2)

# Initial guess: estimate A and k from data at P=40 and P=100
C40  = crystallinity[np.argmin(np.abs(P_MPa - 40))]
C100 = crystallinity[np.argmin(np.abs(P_MPa - 100))]
k0 = np.log(C40 / C100) / (100 - 40)   # from C=A*exp(-k*P)
A0 = C40 / np.exp(-k0 * 40)
c3u = np.polyfit(P_MPa[mask_lo], crystallinity[mask_lo], 2)
x0 = [A0, k0, c3u[0], c3u[1]]

result = minimize(cost, x0, method="Nelder-Mead",
                  options={"xatol": 1e-12, "fatol": 1e-14, "maxiter": 200000})
A, k, a3, b3 = result.x
v2 = A * np.exp(-k * BP2)

def r2(C_obs, C_pred):
    ss_res = np.sum((C_obs - C_pred)**2)
    ss_tot = np.sum((C_obs - C_obs.mean())**2)
    return 1.0 - ss_res / ss_tot

C_pred = piecewise(result.x, P_MPa)
r2_hi  = r2(crystallinity[mask_hi], C_pred[mask_hi])
r2_lo  = r2(crystallinity[mask_lo], C_pred[mask_lo])
r2_all = r2(crystallinity, C_pred)

print(f"Continuity check at P={BP2} MPa:  exp={A*np.exp(-k*BP2):.6f}  quad={v2:.6f}")
print(f"\nHi (P>={BP2} MPa) exponential: C = {A:.6f} * exp(-{k:.6f} * P)   R²={r2_hi:.6f}")
print(f"Lo (P< {BP2} MPa) quadratic:   C = {a3:.6f}*(P-{BP2})² + {b3:.6f}*(P-{BP2}) + {v2:.6f}   R²={r2_lo:.6f}")
print(f"Overall R² = {r2_all:.6f}")

P_hi_plt = np.linspace(BP2, P_MPa.max(), 300)
P_lo_plt = np.linspace(P_MPa.min(), BP2, 300)

# ---------- Plot ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
ax.scatter(P_MPa, crystallinity, s=4, alpha=0.3, color="steelblue", label="MELTS data")
ax.plot(P_hi_plt, piecewise(result.x, P_hi_plt), "r-", linewidth=2,
        label=f"Exponential (P≥{BP2:.0f} MPa)  R²={r2_hi:.4f}")
ax.plot(P_lo_plt, piecewise(result.x, P_lo_plt), "m-", linewidth=2,
        label=f"Quadratic   (P< {BP2:.0f} MPa)  R²={r2_lo:.4f}")
ax.axvline(BP2, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax.set_xlabel("P (MPa)", fontsize=12)
ax.set_ylabel("Crystallinity", fontsize=12)
ax.set_title(f"Crystallinity vs Pressure — continuous piecewise fit  (overall R²={r2_all:.4f})", fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Residuals
ax2 = axes[1]
res_all = crystallinity - C_pred
ax2.scatter(P_MPa[mask_hi], res_all[mask_hi], s=4, alpha=0.4, color="red",     label=f"Exponential (P≥{BP2:.0f})")
ax2.scatter(P_MPa[mask_lo], res_all[mask_lo], s=4, alpha=0.4, color="magenta", label=f"Quadratic   (P< {BP2:.0f})")
ax2.axhline(0, color="black", linewidth=1)
ax2.axvline(BP2, color="gray", linestyle="--", linewidth=1, alpha=0.7)
ax2.set_xlabel("P (MPa)", fontsize=12)
ax2.set_ylabel("Residuals (crystallinity)", fontsize=12)
ax2.set_title("Residuals", fontsize=13)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_dir / "crystallinity_vs_P.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_dir / 'crystallinity_vs_P.png'}")
