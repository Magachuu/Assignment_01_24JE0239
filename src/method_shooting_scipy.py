"""
Method 2 - Shooting with scipy.integrate.solve_ivp (adaptive RK45/DOP853)
and Newton-Raphson on the unknown wall shear alpha = f''(0).

The Newton derivative dF/dalpha is obtained exactly by integrating the
variational equations alongside the state, so no finite differences are needed.

Why continuation in eta_inf?  For large eta_inf the map alpha -> f'(eta_inf)
is extremely steep (|dF/dalpha| ~ 1e5 for alpha = 1, eta_inf = 10) and
trajectories with alpha below the true value diverge; Newton started from
alpha = 1 then lands on a *spurious* root (f''(0) ~ 1.151).  Solving first on
a short domain (eta_inf = 3), where the problem is benign, and using that
alpha as the initial guess for progressively longer domains avoids this.
"""
import time
import numpy as np
from scipy.integrate import solve_ivp

from .problem import augmented_rhs, initial_state, ETA_INF_DEFAULT


def _residual(alpha, eta_inf, rtol, atol, method):
    with np.errstate(over="ignore", invalid="ignore"):
        sol = solve_ivp(augmented_rhs, (0.0, eta_inf),
                    initial_state(alpha, with_sensitivity=True),
                        method=method, rtol=rtol, atol=atol)
    z_end = sol.y[:, -1]
    F = z_end[1] - 1.0          # mismatch in f'(eta_inf)
    dF = z_end[4]               # d f'(eta_inf) / d alpha
    return F, dF


def newton(alpha0, eta_inf, tol=1e-10, max_iter=50,
           rtol=1e-10, atol=1e-12, method="DOP853"):
    """Damped Newton iteration on F(alpha) = f'(eta_inf; alpha) - 1."""
    history = []
    alpha = alpha0
    for _ in range(max_iter):
        F, dF = _residual(alpha, eta_inf, rtol, atol, method)
        history.append((alpha, abs(F)))
        if abs(F) < tol:
            break
        step = F / dF
        # damping keeps the trajectory from blowing up for poor initial guesses
        lam = 1.0
        while lam > 1e-4:
            trial = alpha - lam * step
            Ft, _ = _residual(trial, eta_inf, rtol, atol, method)
            if np.isfinite(Ft) and abs(Ft) < abs(F):
                break
            lam *= 0.5
        alpha = trial
    else:
        raise RuntimeError("Newton shooting (solve_ivp) did not converge")
    return alpha, history


def solve(eta_inf=ETA_INF_DEFAULT, alpha0=1.0, continuation=True, d_eta=1.0,
          tol=1e-10, rtol=1e-10, atol=1e-12, method="DOP853"):
    t0 = time.perf_counter()
    stages = (list(np.arange(3.0, eta_inf, d_eta)) + [eta_inf]
              if continuation and eta_inf > 3.0 else [eta_inf])
    alpha, total_iter = alpha0, 0
    for L in stages:
        alpha, history = newton(alpha, L, tol, rtol=rtol, atol=atol, method=method)
        total_iter += len(history) - 1
    elapsed = time.perf_counter() - t0

    eta = np.linspace(0.0, eta_inf, 401)
    final = solve_ivp(augmented_rhs, (0.0, eta_inf),
                      initial_state(alpha, True), method=method,
                      rtol=rtol, atol=atol, t_eval=eta, dense_output=True)
    return {
        "name": "solve_ivp + Newton shooting",
        "fpp0": float(alpha),
        "eta": final.t,
        "y": final.y[:3],
        "dense": lambda x: final.sol(x)[:3],
        "iterations": total_iter,
        "stages": len(stages),
        "history": history,
        "time": elapsed,
        "residual": history[-1][1],
    }


if __name__ == "__main__":
    r = solve()
    print(f"[{r['name']}] f''(0) = {r['fpp0']:.10f}  "
          f"Newton iters = {r['iterations']}  t = {r['time']*1e3:.1f} ms")
    for k, (a, F) in enumerate(r["history"]):
        print(f"  k={k:2d}  alpha={a:.12f}  |F|={F:.3e}")
