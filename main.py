"""
Assignment 01 - Falkner-Skan stagnation-point (Hiemenz) flow.

Runs all three solvers, performs convergence / sensitivity studies, writes
figures to report/figures/ and LaTeX tables to report/tables/.

Usage:  python main.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.problem import FPP0_REFERENCE
from src import method_bvp, method_shooting_scipy, method_shooting_rk4

ROOT = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ROOT, "report", "figures")
TAB_DIR = os.path.join(ROOT, "report", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})
COLORS = ["#1f4e79", "#c0504d", "#4f9a3a"]


def thicknesses(res):
    """Displacement and momentum thickness from a solution profile."""
    eta = np.linspace(0, res["eta"][-1], 4001)
    fp = res["dense"](eta)[1]
    delta_star = np.trapezoid(1.0 - fp, eta)
    theta = np.trapezoid(fp * (1.0 - fp), eta)
    eta99 = eta[np.argmax(fp >= 0.99)]
    return delta_star, theta, eta99


def save(fig, name):
    fig.savefig(os.path.join(FIG_DIR, name))
    plt.close(fig)
    print(f"  saved figures/{name}")


def main():
    print("Solving with three methods (eta_inf = 10) ...")
    results = [method_bvp.solve(), method_shooting_scipy.solve(),
               method_shooting_rk4.solve()]
    for r in results:
        r["delta_star"], r["theta"], r["eta99"] = thicknesses(r)
        assert abs(r["fpp0"] - FPP0_REFERENCE) < 1e-5, f"{r['name']} gave a wrong root"
        print(f"  {r['name']:<32s} f''(0) = {r['fpp0']:.10f}  "
              f"err = {abs(r['fpp0'] - FPP0_REFERENCE):.2e}  t = {r['time']*1e3:7.1f} ms")

    # ---------------------------------------------------------------- Fig 1
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    labels = [r"$f(\eta)$", r"$f'(\eta)$", r"$f''(\eta)$"]
    eta_plot = np.linspace(0, 6, 300)
    for j, ax in enumerate(axes):
        for r, c, ls in zip(results, COLORS, ["-", "--", ":"]):
            ax.plot(eta_plot, r["dense"](eta_plot)[j], ls, color=c, lw=2,
                    label=r["name"])
        ax.set_xlabel(r"$\eta$"); ax.set_title(labels[j])
    axes[1].axhline(1.0, color="k", lw=0.8, ls="-.")
    axes[0].legend(fontsize=8)
    fig.suptitle("Hiemenz stagnation-point flow: similarity profiles")
    save(fig, "profiles.png")

    # ---------------------------------------------------------------- Fig 2
    # Newton convergence on a single domain (eta_inf = 4) from alpha_0 = 1
    _, hist_ivp = method_shooting_scipy.newton(1.0, 4.0)
    _, hist_rk4, _, _ = method_shooting_rk4.newton(1.0, 4.0, 0.01)
    fig, ax = plt.subplots(figsize=(6, 4))
    for hist, c, m, lab in [(hist_ivp, COLORS[1], "o", "solve_ivp + Newton"),
                            (hist_rk4, COLORS[2], "s", "RK4 + Newton")]:
        F = [h[1] for h in hist]
        ax.semilogy(range(len(F)), np.maximum(F, 1e-17), m + "-", color=c, label=lab)
    ax.set_xlabel("Newton iteration $k$")
    ax.set_ylabel(r"$|F(\alpha_k)| = |f'(\eta_\infty;\alpha_k)-1|$")
    ax.set_title(r"Newton convergence ($\eta_\infty = 4$, $\alpha_0 = 1$)")
    ax.legend()
    save(fig, "newton_convergence.png")
    newton_rows = [(k, a, F) for k, (a, F) in enumerate(hist_ivp)]

    # ---------------------------------------------------------------- Fig 2b
    # Shooting landscape F(alpha) -> why continuation is needed
    print("Shooting landscape ...")
    alphas = np.linspace(0.9, 1.5, 601)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for L, c in zip([3, 5, 8, 10], ["#1f4e79", "#4f9a3a", "#e69f00", "#c0504d"]):
        F = []
        for a in alphas:
            with np.errstate(all="ignore"):
                F.append(method_shooting_rk4._shoot(a, L, 0.02)[2])
        F = np.array(F)
        F[~np.isfinite(F) | (np.abs(F) > 50)] = np.nan   # blown-up trajectories
        ax.plot(alphas, F, color=c, lw=1.8, label=rf"$\eta_\infty = {L}$")
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(FPP0_REFERENCE, color="k", ls="--", lw=0.8)
    ax.text(FPP0_REFERENCE + 0.005, 1.6, "true $f''(0)$", fontsize=9)
    ax.set_ylim(-2, 2)
    ax.set_xlabel(r"$\alpha = f''(0)$")
    ax.set_ylabel(r"$F(\alpha) = f'(\eta_\infty;\alpha) - 1$")
    ax.set_title("Shooting residual vs. guessed wall shear")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "shooting_landscape.png")
    nocont_rows = []
    for L in [3, 4, 5, 6, 8, 10]:
        try:
            v = method_shooting_scipy.solve(eta_inf=L, continuation=False)["fpp0"]
        except RuntimeError:
            v = float("nan")
        w = method_shooting_scipy.solve(eta_inf=L, continuation=True)["fpp0"]
        nocont_rows.append((L, v, w))

    # ---------------------------------------------------------------- Fig 3
    print("RK4 step-size study ...")
    hs = np.array([0.4, 0.2, 0.1, 0.05, 0.025, 0.0125])
    ref = method_shooting_rk4.solve(h=0.0025)["fpp0"]
    errs = np.array([abs(method_shooting_rk4.solve(h=h)["fpp0"] - ref) for h in hs])
    order = np.polyfit(np.log(hs[:-1]), np.log(errs[:-1]), 1)[0]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(hs, errs, "o-", color=COLORS[2], label="RK4 error in $f''(0)$")
    ax.loglog(hs, errs[0] * (hs / hs[0]) ** 4, "k--", lw=1, label=r"$\mathcal{O}(h^4)$")
    ax.set_xlabel("step size $h$"); ax.set_ylabel("absolute error")
    ax.set_xticks(hs); ax.set_xticklabels([f"{h:g}" for h in hs])
    ax.minorticks_off()
    ax.set_title(f"RK4 convergence (observed order = {order:.2f})")
    ax.legend()
    save(fig, "rk4_convergence.png")
    rk4_rows = list(zip(hs, errs))

    # ---------------------------------------------------------------- Fig 4
    print("Far-field truncation study ...")
    eta_infs = np.array([3, 4, 5, 6, 7, 8, 10, 12, 15])
    vals = {"bvp": [], "ivp": [], "rk4": []}
    for L in eta_infs:
        vals["bvp"].append(method_bvp.solve(eta_inf=L)["fpp0"])
        vals["ivp"].append(method_shooting_scipy.solve(eta_inf=L)["fpp0"])
        vals["rk4"].append(method_shooting_rk4.solve(eta_inf=L)["fpp0"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for (k, v), c, m in zip(vals.items(), COLORS, ["o", "s", "^"]):
        a1.plot(eta_infs, v, m + "-", color=c, label=k)
        a2.semilogy(eta_infs, np.abs(np.array(v) - FPP0_REFERENCE) + 1e-16,
                    m + "-", color=c, label=k)
    a1.axhline(FPP0_REFERENCE, color="k", lw=0.8, ls="--", label="reference")
    a1.set_xlabel(r"$\eta_\infty$"); a1.set_ylabel("$f''(0)$"); a1.legend()
    a2.set_xlabel(r"$\eta_\infty$"); a2.set_ylabel("|error|"); a2.legend()
    fig.suptitle(r"Sensitivity to the truncated domain length $\eta_\infty$")
    save(fig, "eta_inf_study.png")

    # ---------------------------------------------------------------- Fig 5
    fig, ax = plt.subplots(figsize=(6, 4))
    e = np.linspace(0, 10, 500)
    base = results[0]["dense"](e)[1]
    for r, c in zip(results[1:], COLORS[1:]):
        ax.semilogy(e, np.abs(r["dense"](e)[1] - base) + 1e-18, color=c,
                    label=f"{r['name']} vs solve_bvp")
    ax.set_xlabel(r"$\eta$"); ax.set_ylabel(r"$|\Delta f'(\eta)|$")
    ax.set_title("Pointwise difference between methods")
    ax.legend(fontsize=8)
    save(fig, "method_difference.png")

    # ---------------------------------------------------------------- Tables
    with open(os.path.join(TAB_DIR, "comparison.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lcccccc}\n\\toprule\n")
        fh.write("Method & $f''(0)$ & $|$Error$|$ & Iter. & $\\delta^*$ & "
                 "$\\theta$ & Time (ms) \\\\\n\\midrule\n")
        for r in results:
            name = r["name"].replace("_", r"\_")
            fh.write(f"{name} & {r['fpp0']:.8f} & "
                     f"{abs(r['fpp0']-FPP0_REFERENCE):.1e} & {r['iterations']} & "
                     f"{r['delta_star']:.5f} & {r['theta']:.5f} & {r['time']*1e3:.1f} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(TAB_DIR, "eta_inf.tex"), "w") as fh:
        fh.write("\\begin{tabular}{cccc}\n\\toprule\n$\\eta_\\infty$ & solve\\_bvp & "
                 "solve\\_ivp + Newton & RK4 + Newton \\\\\n\\midrule\n")
        for i, L in enumerate(eta_infs):
            fh.write(f"{L} & {vals['bvp'][i]:.8f} & {vals['ivp'][i]:.8f} & "
                     f"{vals['rk4'][i]:.8f} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(TAB_DIR, "rk4.tex"), "w") as fh:
        fh.write("\\begin{tabular}{cc}\n\\toprule\n$h$ & $|f''_h(0)-f''_{h\\to0}(0)|$ \\\\\n\\midrule\n")
        for h, err in rk4_rows:
            fh.write(f"{h} & {err:.3e} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(TAB_DIR, "summary.tex"), "w") as fh:
        r = results[0]
        fh.write(f"\\newcommand{{\\fppzero}}{{{r['fpp0']:.8f}}}\n")
        fh.write(f"\\newcommand{{\\deltastar}}{{{r['delta_star']:.4f}}}\n")
        fh.write(f"\\newcommand{{\\thetamom}}{{{r['theta']:.4f}}}\n")
        fh.write(f"\\newcommand{{\\etaninetynine}}{{{r['eta99']:.2f}}}\n")
        fh.write(f"\\newcommand{{\\rkorder}}{{{order:.2f}}}\n")
    with open(os.path.join(TAB_DIR, "newton.tex"), "w") as fh:
        fh.write("\\begin{tabular}{ccc}\n\\toprule\n$k$ & $\\alpha_k$ & $|F(\\alpha_k)|$ \\\\\n\\midrule\n")
        for k, a, F in newton_rows:
            fh.write(f"{k} & {a:.12f} & {F:.3e} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(TAB_DIR, "continuation.tex"), "w") as fh:
        fh.write("\\begin{tabular}{ccc}\n\\toprule\n$\\eta_\\infty$ & Newton from $\\alpha_0=1$ & "
                 "with continuation \\\\\n\\midrule\n")
        for L, v, w in nocont_rows:
            fh.write(f"{L} & {v:.8f} & {w:.8f} \\\\\n")
        fh.write("\\bottomrule\n\\end{tabular}\n")
    print("  saved tables/*.tex\nDone.")


if __name__ == "__main__":
    main()
