import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Giordano et al. (2008) VTF viscosity model ----------
MFW = {
    "SiO2": 60.0843, "TiO2": 79.8658, "Al2O3": 101.961276,
    "FeOt": 71.8444, "MnO": 70.937449, "MgO": 40.3044,
    "CaO": 56.0774, "Na2O": 61.97894, "K2O": 94.196,
    "P2O5": 141.9446, "H2O": 18.01528,
}

# Model coefficients (Table 1, Giordano et al. 2008)
B_COEFF = {"b1": 159.56, "b2": -173.34, "b3": 72.13, "b4": 75.69,
           "b5": -38.98, "b6": -84.08, "b7": 141.54,
           "b11": -2.43, "b12": -0.91, "b13": 17.62}
C_COEFF = {"c1": 2.75, "c2": 15.72, "c3": 8.32, "c4": 10.20,
           "c5": -12.29, "c6": -99.54, "c11": 0.30}


def giordano2008_logeta(T_C, SiO2, TiO2, Al2O3, Fe2O3, FeO, MnO, MgO,
                         CaO, Na2O, K2O, P2O5, H2O_wt):
    """
    Returns log10(viscosity [Pa·s]).
    Inputs: T in °C, oxides in wt%.
    """
    # Convert Fe2O3 + FeO to total FeOt
    FeOt = 0.9 * Fe2O3 + FeO

    # Normalize anhydrous oxides to (100 - H2O_wt), then add H2O as-is
    anhydrous = np.array([SiO2, TiO2, Al2O3, FeOt, MnO, MgO, CaO, Na2O, K2O, P2O5])
    sum1 = anhydrous.sum(axis=0)
    factor = (100.0 - H2O_wt) / sum1
    Si, Ti, Al, Fe, Mn, Mg, Ca, Na, K, P = [ox * factor for ox in anhydrous]
    Hw = H2O_wt  # H2O stays as-is

    # Compute mol% (each wt%_norm / MFW * GFW)
    wt_arr = np.stack([Si, Ti, Al, Fe, Mn, Mg, Ca, Na, K, P, Hw], axis=0)  # (11, N)
    mfw_arr = np.array([MFW[k] for k in
                        ["SiO2","TiO2","Al2O3","FeOt","MnO","MgO",
                         "CaO","Na2O","K2O","P2O5","H2O"]])[:, None]      # (11, 1)
    n = wt_arr / mfw_arr                # proportional to moles, shape (11, N)
    GFW = 100.0 / n.sum(axis=0)        # average formula weight, shape (N,)
    m = n * GFW                         # mol%, shape (11, N)

    Si, Ti, Al, Fe, Mn, Mg, Ca, Na, K, P, Hw = m  # each shape (N,)

    b = B_COEFF
    c = C_COEFF

    B = (b["b1"]  * (Si + Ti)
       + b["b2"]  * Al
       + b["b3"]  * (Fe + Mn + P)
       + b["b4"]  * Mg
       + b["b5"]  * Ca
       + b["b6"]  * (Na + Hw)
       + b["b7"]  * (Hw + np.log(1.0 + Hw))
       + b["b11"] * (Si + Ti) * (Fe + Mn + Mg)
       + b["b12"] * (Si + Ti + Al + P) * (Na + K + Hw)
       + b["b13"] * Al * (Na + K))

    C = (c["c1"]  * Si
       + c["c2"]  * (Ti + Al)
       + c["c3"]  * (Fe + Mn + Mg)
       + c["c4"]  * Ca
       + c["c5"]  * (Na + K)
       + c["c6"]  * np.log(1.0 + Hw)
       + c["c11"] * (Al + Fe + Mn + Mg + Ca - P) * (Na + K + Hw))

    T_K = T_C + 273.15
    return -4.55 + B / (T_K - C)   # log10(Pa·s)


# ---------- Load melts-liquid.tbl ----------
parser = argparse.ArgumentParser()
parser.add_argument("tbl", nargs="?", default="melts-liquid.tbl",
                    help="Path to melts-liquid.tbl")
args = parser.parse_args()
tbl_path = Path(args.tbl)
out_dir  = tbl_path.parent

df = pd.read_csv(tbl_path, header=0)
df.columns = df.columns.str.strip()

mask = df["wt% H2O"] > 0
d = df[mask].copy()

# Full Giordano: all oxides vary with P
log_eta = giordano2008_logeta(
    T_C    = d["T (C)"].values,
    SiO2   = d["wt% SiO2"].values,
    TiO2   = d["wt% TiO2"].values,
    Al2O3  = d["wt% Al2O3"].values,
    Fe2O3  = d["wt% Fe2O3"].values,
    FeO    = d["wt% FeO"].values,
    MnO    = d["wt% MnO"].values,
    MgO    = d["wt% MgO"].values,
    CaO    = d["wt% CaO"].values,
    Na2O   = d["wt% Na2O"].values,
    K2O    = d["wt% K2O"].values,
    P2O5   = d["wt% P2O5"].values,
    H2O_wt = d["wt% H2O"].values,
)

# Fixed-composition Giordano: oxides fixed at index=1, only H2O varies
row1 = df[df["Index"] == 1].iloc[0]
n = len(d)
log_eta_fixcomp = giordano2008_logeta(
    T_C    = d["T (C)"].values,
    SiO2   = np.full(n, row1["wt% SiO2"]),
    TiO2   = np.full(n, row1["wt% TiO2"]),
    Al2O3  = np.full(n, row1["wt% Al2O3"]),
    Fe2O3  = np.full(n, row1["wt% Fe2O3"]),
    FeO    = np.full(n, row1["wt% FeO"]),
    MnO    = np.full(n, row1["wt% MnO"]),
    MgO    = np.full(n, row1["wt% MgO"]),
    CaO    = np.full(n, row1["wt% CaO"]),
    Na2O   = np.full(n, row1["wt% Na2O"]),
    K2O    = np.full(n, row1["wt% K2O"]),
    P2O5   = np.full(n, row1["wt% P2O5"]),
    H2O_wt = d["wt% H2O"].values,
)

H2O_frac = d["wt% H2O"].values / 100.0
T_C      = d["T (C)"].values

# MELTS liq vis: log10(poise) -> Pa·s
log_vis_melts = d["liq vis (log 10 poise)"].values - 1.0  # log10(Pa·s) = log10(poise) + log10(0.1)

# Hess & Dingwell (1996)
def hess_dingwell(T_C, H2O_frac):
    x = np.log(100.0 * H2O_frac)  # ln(wt% H2O)
    return -3.545 + 0.833*x + (9601.0 - 2368.0*x) / (T_C + 273.0 - (195.7 + 32.25*x))

# ---------- 3rd degree polynomial fitting ----------
deg = 3
coeffs = np.polyfit(H2O_frac, log_eta, deg)
log_eta_pred = np.polyval(coeffs, H2O_frac)
residuals_poly = log_eta - log_eta_pred
ss_res = np.sum(residuals_poly**2)
ss_tot = np.sum((log_eta - np.mean(log_eta))**2)
r2_poly = 1 - ss_res / ss_tot
n_pts, k = len(H2O_frac), deg + 1
aic_poly = n_pts * np.log(ss_res / n_pts) + 2 * k
bic_poly = n_pts * np.log(ss_res / n_pts) + k * np.log(n_pts)

print(f"\n3rd-degree polynomial fit:")
print(f"  Coefficients (high→low): {coeffs}")
print(f"  R² = {r2_poly:.8f},  AIC = {aic_poly:.2f},  BIC = {bic_poly:.2f}")

H2O_fit = np.linspace(H2O_frac.min(), H2O_frac.max(), 400)
log_eta_fit = np.polyval(coeffs, H2O_fit)

# ---------- H&D-form fitting (6 parameters) ----------
from scipy.optimize import curve_fit

def hd_form(X, a, b, c, d, e, f):
    """log10(η) = a + b*x + (c + d*x) / (T_K - (e + f*x)),  x = ln(wt% H2O)"""
    T_C_arr, H2O_frac_arr = X
    x = np.log(100.0 * H2O_frac_arr)
    T_K = T_C_arr + 273.0
    return a + b*x + (c + d*x) / (T_K - (e + f*x))

p0 = [-3.545, 0.833, 9601.0, -2368.0, 195.7, 32.25]   # H&D (1996) as initial guess
popt_hd, pcov_hd = curve_fit(hd_form, (T_C, H2O_frac), log_eta,
                              p0=p0, maxfev=20000)
a_hd, b_hd, c_hd, d_hd, e_hd, f_hd = popt_hd
perr_hd = np.sqrt(np.diag(pcov_hd))

log_eta_hd_fit_data = hd_form((T_C, H2O_frac), *popt_hd)
residuals_hd = log_eta - log_eta_hd_fit_data
ss_res_hd = np.sum(residuals_hd**2)
r2_hd  = 1 - ss_res_hd / ss_tot
aic_hd = n_pts * np.log(ss_res_hd / n_pts) + 2 * 6
bic_hd = n_pts * np.log(ss_res_hd / n_pts) + 6 * np.log(n_pts)

print(f"\nH&D-form fit (6 parameters):")
print(f"  a = {a_hd:.4f} ± {perr_hd[0]:.4f}  (H&D: -3.545)")
print(f"  b = {b_hd:.4f} ± {perr_hd[1]:.4f}  (H&D:  0.833)")
print(f"  c = {c_hd:.2f} ± {perr_hd[2]:.2f}  (H&D: 9601)")
print(f"  d = {d_hd:.2f} ± {perr_hd[3]:.2f}  (H&D: -2368)")
print(f"  e = {e_hd:.4f} ± {perr_hd[4]:.4f}  (H&D: 195.7)")
print(f"  f = {f_hd:.4f} ± {perr_hd[5]:.4f}  (H&D: 32.25)")
print(f"  R² = {r2_hd:.8f},  AIC = {aic_hd:.2f},  BIC = {bic_hd:.2f}")

log_eta_hd_fit_curve = hd_form((np.full(len(H2O_fit), T_C.mean()), H2O_fit), *popt_hd)

# H&D-form fit for fixed-composition Giordano
ss_tot_fc = np.sum((log_eta_fixcomp - np.mean(log_eta_fixcomp))**2)
popt_hd_fc, pcov_hd_fc = curve_fit(hd_form, (T_C, H2O_frac), log_eta_fixcomp,
                                    p0=p0, maxfev=20000)
perr_hd_fc = np.sqrt(np.diag(pcov_hd_fc))
log_eta_hd_fc_data  = hd_form((T_C, H2O_frac), *popt_hd_fc)
log_eta_hd_fc_curve = hd_form((np.full(len(H2O_fit), T_C.mean()), H2O_fit), *popt_hd_fc)
residuals_hd_fc = log_eta_fixcomp - log_eta_hd_fc_data
ss_res_hd_fc = np.sum(residuals_hd_fc**2)
r2_hd_fc  = 1 - ss_res_hd_fc / ss_tot_fc
aic_hd_fc = n_pts * np.log(ss_res_hd_fc / n_pts) + 2 * 6
bic_hd_fc = n_pts * np.log(ss_res_hd_fc / n_pts) + 6 * np.log(n_pts)

a_hd_fc, b_hd_fc, c_hd_fc, d_hd_fc, e_hd_fc, f_hd_fc = popt_hd_fc
print(f"\nH&D-form fit — fixed composition (6 parameters):")
print(f"  a = {a_hd_fc:.4f} ± {perr_hd_fc[0]:.4f}")
print(f"  b = {b_hd_fc:.4f} ± {perr_hd_fc[1]:.4f}")
print(f"  c = {c_hd_fc:.2f} ± {perr_hd_fc[2]:.2f}")
print(f"  d = {d_hd_fc:.2f} ± {perr_hd_fc[3]:.2f}")
print(f"  e = {e_hd_fc:.4f} ± {perr_hd_fc[4]:.4f}")
print(f"  f = {f_hd_fc:.4f} ± {perr_hd_fc[5]:.4f}")
print(f"  R² = {r2_hd_fc:.8f},  AIC = {aic_hd_fc:.2f},  BIC = {bic_hd_fc:.2f}")

# ---------- Plot ----------
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

H2O_hd = np.linspace(H2O_frac.min(), H2O_frac.max(), 400)
log_eta_hd = hess_dingwell(T_C.mean(), H2O_hd)

ax = axes[0]
ax.scatter(H2O_frac, log_eta, s=4, alpha=0.35, color="steelblue",
           label="Giordano (2008) — full composition")
ax.scatter(H2O_frac, log_eta_fixcomp, s=4, alpha=0.35, color="mediumpurple",
           label="Giordano (2008) — fixed comp. (index=1), H₂O only")
ax.scatter(H2O_frac, log_vis_melts, s=4, alpha=0.35, color="orange",
           label="MELTS liq vis")
ax.plot(H2O_fit, log_eta_fit, "r-", linewidth=2,
        label=f"Degree-3 poly fit  R²={r2_poly:.6f}")
ax.plot(H2O_fit, log_eta_hd_fit_curve, "r--", linewidth=2,
        label=f"H&D-form fit (full comp.)  R²={r2_hd:.6f}")
ax.plot(H2O_fit, log_eta_hd_fc_curve, color="mediumpurple", linestyle="--", linewidth=2,
        label=f"H&D-form fit (fixed comp.)  R²={r2_hd_fc:.6f}")
ax.plot(H2O_hd, log_eta_hd, "g--", linewidth=2,
        label="Hess & Dingwell (1996)")
ax.set_xlabel("H₂O (fraction)", fontsize=12)
ax.set_ylabel("log₁₀ Viscosity (Pa·s)", fontsize=12)
ax.set_title("Viscosity vs H₂O", fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.scatter(H2O_frac, residuals_poly,   s=4, alpha=0.35, color="red",
            label=f"Poly-3 (full)  R²={r2_poly:.6f}")
ax2.scatter(H2O_frac, residuals_hd,    s=4, alpha=0.35, color="darkred",
            label=f"H&D-form (full)  R²={r2_hd:.6f}")
ax2.scatter(H2O_frac, residuals_hd_fc, s=4, alpha=0.35, color="mediumpurple",
            label=f"H&D-form (fixed)  R²={r2_hd_fc:.6f}")
ax2.axhline(0, color="black", linewidth=1)
ax2.legend(fontsize=9)
ax2.set_xlabel("H₂O (fraction)", fontsize=12)
ax2.set_ylabel("Residuals (log₁₀ Pa·s)", fontsize=12)
ax2.set_title("Residuals", fontsize=13)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(out_dir / "viscosity_giordano_poly3.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out_dir / 'viscosity_giordano_poly3.png'}")

# ---------- Save plot data ----------
df_scatter = pd.DataFrame({
    "H2O_fraction":                    H2O_frac,
    "log10_vis_Giordano2008_full":     log_eta,
    "log10_vis_Giordano2008_fixcomp":  log_eta_fixcomp,
    "log10_vis_MELTS":                 log_vis_melts,
    "log10_vis_poly4_fit":             np.polyval(coeffs, H2O_frac),
    "log10_vis_HDform_fit_full":        log_eta_hd_fit_data,
    "log10_vis_HDform_fit_fixcomp":    log_eta_hd_fc_data,
    "residual_poly4":                  residuals_poly,
    "residual_HDform_full":            residuals_hd,
    "residual_HDform_fixcomp":         residuals_hd_fc,
})
df_scatter.to_csv(out_dir / "viscosity_scatter_data.csv", index=False, float_format="%.8f")

df_curves = pd.DataFrame({
    "H2O_fraction":              H2O_fit,
    "log10_vis_poly4_fit":        log_eta_fit,
    "log10_vis_HDform_fit_full":   log_eta_hd_fit_curve,
    "log10_vis_HDform_fit_fixcomp": log_eta_hd_fc_curve,
    "log10_vis_HessDingwell1996": log_eta_hd,
})
df_curves.to_csv(out_dir / "viscosity_curve_data.csv", index=False, float_format="%.8f")

print(f"Saved: {out_dir / 'viscosity_scatter_data.csv'}, {out_dir / 'viscosity_curve_data.csv'}")
