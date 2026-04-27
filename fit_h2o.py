import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("tbl", nargs="?", default="melts-liquid.tbl",
                    help="Path to melts-liquid.tbl")
args = parser.parse_args()
tbl_path = Path(args.tbl)
out_dir  = tbl_path.parent

df = pd.read_csv(tbl_path, header=0)
df.columns = df.columns.str.strip()

P_kbar = df["P (kbars)"].values
H2O_wt = df["wt% H2O"].values

mask = (P_kbar > 0) & (H2O_wt > 0)
P = P_kbar[mask] * 1e8       # kbar -> Pa
H2O = H2O_wt[mask] / 100.0  # wt% -> fraction

def power_law(p, a, b):
    return a * p ** b

popt, pcov = curve_fit(power_law, P, H2O, p0=[1.0, 0.5])
a, b = popt
perr = np.sqrt(np.diag(pcov))

P_fit = np.linspace(P.min(), P.max(), 300)
H2O_fit = power_law(P_fit, a, b)

residuals = H2O - power_law(P, a, b)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((H2O - np.mean(H2O))**2)
r2 = 1 - ss_res / ss_tot

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

H2O_bd74 = 4.11e-6 * P_fit ** 0.5

ax = axes[0]
ax.scatter(P, H2O, s=5, alpha=0.4, color="steelblue", label="MELTS data")
ax.plot(P_fit, H2O_fit, "r-", linewidth=2,
        label=f"This fit: H₂O = {a:.4e} × P^{b:.4f}\nR² = {r2:.6f}")
ax.plot(P_fit, H2O_bd74, "g--", linewidth=2,
        label="BD1974: H₂O = 4.11×10⁻⁶ × P^0.5")
ax.set_xlabel("P (Pa)", fontsize=12)
ax.set_ylabel("H₂O (fraction)", fontsize=12)
ax.set_title("H₂O solubility vs Pressure", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.scatter(P, residuals, s=5, alpha=0.4, color="steelblue")
ax2.axhline(0, color="red", linewidth=1)
ax2.set_xlabel("P (Pa)", fontsize=12)
ax2.set_ylabel("Residuals (H₂O fraction)", fontsize=12)
ax2.set_title("Residuals", fontsize=13)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_dir / "h2o_fitting.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"Fitting result: H2O = {a:.6e} * P^{b:.6f}")
print(f"  a = {a:.6e} ± {perr[0]:.6e}")
print(f"  b = {b:.6f} ± {perr[1]:.6f}")
print(f"  R² = {r2:.8f}")
print(f"  N = {len(P)} points")
print(f"  P range: {P.min():.3e} – {P.max():.3e} Pa")
print(f"  H2O range: {H2O.min():.6f} – {H2O.max():.6f} (fraction)")
print(f"Saved: {out_dir / 'h2o_fitting.png'}")

txt_path = out_dir / "h2o_fitting_coeffs.txt"
with open(txt_path, "w") as f:
    f.write("=== H2O solubility fitting results ===\n")
    f.write(f"Model: H2O (fraction) = a * P (Pa) ^ b\n\n")
    f.write(f"a = {a:.8e}  ±  {perr[0]:.8e}\n")
    f.write(f"b = {b:.8f}  ±  {perr[1]:.8f}\n\n")
    f.write(f"R2      = {r2:.8f}\n")
    f.write(f"N       = {len(P)}\n")
    f.write(f"P range = {P.min():.4e} – {P.max():.4e} Pa\n")
    f.write(f"H2O range = {H2O.min():.6f} – {H2O.max():.6f} (fraction)\n\n")
    f.write("Reference: BD1974  H2O = 4.11e-6 * P^0.5\n")
print(f"Saved: {txt_path}")
