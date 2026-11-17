"""
Method 3 - Shooting with a hand-written fixed-step classical RK4 integrator
and Newton-Raphson. No SciPy is used in this file.

Newton derivative: exact, via the variational equations integrated with the
same RK4 scheme (so it is the exact derivative of the *discrete* map).
"""
import time
import numpy as np

from .problem import augmented_rhs, initial_state, ETA_INF_DEFAULT


def rk4(fun, y0, eta_inf, h):
    """Classical 4th-order Runge-Kutta on [0, eta_inf] with step ~h."""
    n = int(round(eta_inf / h))
    h = eta_inf / n
    eta = np.linspace(0.0, eta_inf, n + 1)
    Y = np.empty((len(y0), n + 1))
    Y[:, 0] = y = np.asarray(y0, dtype=float)
    for i in range(n):
        x = eta[i]
        k1 = fun(x, y)
        k2 = fun(x + 0.5 * h, y + 0.5 * h * k1)
        k3 = fun(x + 0.5 * h, y + 0.5 * h * k2)
        k4 = fun(x + h, y + h * k3)
        y = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        Y[:, i + 1] = y
    return eta, Y


def hermite(x, eta, Y):
    """
    Cubic Hermite interpolation of the RK4 nodal values, using the ODE itself
    for the slopes (y' = rhs(y)), so dense output is O(h^4) like the nodes.
    """
    from .problem import rhs
    x = np.atleast_1d(np.asarray(x, dtype=float))
    dY = rhs(None, Y)
    i = np.clip(np.searchsorted(eta, x) - 1, 0, len(eta) - 2)
    h = eta[i + 1] - eta[i]
    t = (x - eta[i]) / h
    h00, h10 = 2*t**3 - 3*t**2 + 1, t**3 - 2*t**2 + t
    h01, h11 = -2*t**3 + 3*t**2, t**3 - t**2
    return (h00 * Y[:, i] + h10 * h * dY[:, i]
            + h01 * Y[:, i + 1] + h11 * h * dY[:, i + 1])


def _shoot(alpha, eta_inf, h):
    with np.errstate(over="ignore", invalid="ignore"):
        eta, Z = rk4(augmented_rhs, initial_state(alpha, True), eta_inf, h)
    return eta, Z, Z[1, -1] - 1.0, Z[4, -1]


def newton(alpha0, eta_inf, h, tol=1e-10, max_iter=50):
    """Damped Newton iteration on F(alpha) = f'(eta_inf; alpha) - 1."""
    alpha = alpha0
    history = []
    for _ in range(max_iter):
        eta, Z, F, dF = _shoot(alpha, eta_inf, h)
        history.append((alpha, abs(F)))
        if abs(F) < tol:
            break
        # damped Newton: halve the step until |F| decreases (and stays finite)
        step, lam = F / dF, 1.0
        while lam > 1e-4:
            trial = alpha - lam * step
            Ft = _shoot(trial, eta_inf, h)[2]
            if np.isfinite(Ft) and abs(Ft) < abs(F):
                break
            lam *= 0.5
        alpha = trial
    else:
        raise RuntimeError("Newton shooting (RK4) did not converge")
    return alpha, history, eta, Z


def solve(eta_inf=ETA_INF_DEFAULT, h=0.01, alpha0=1.0, continuation=True,
          d_eta=1.0, tol=1e-10):
    """Newton shooting with continuation in eta_inf (see method 2 for why)."""
    t0 = time.perf_counter()
    stages = (list(np.arange(3.0, eta_inf, d_eta)) + [eta_inf]
              if continuation and eta_inf > 3.0 else [eta_inf])
    alpha, total_iter = alpha0, 0
    for L in stages:
        alpha, history, eta, Z = newton(alpha, L, h, tol)
        total_iter += len(history) - 1
    elapsed = time.perf_counter() - t0
    return {
        "name": "Scratch RK4 + Newton shooting",
        "fpp0": float(alpha),
        "eta": eta,
        "y": Z[:3],
        "dense": lambda x: hermite(x, eta, Z[:3]),
        "iterations": total_iter,
        "stages": len(stages),
        "history": history,
        "time": elapsed,
        "residual": history[-1][1],
        "h": eta[1] - eta[0],
    }


if __name__ == "__main__":
    r = solve()
    print(f"[{r['name']}] f''(0) = {r['fpp0']:.10f}  h = {r['h']}  "
          f"Newton iters = {r['iterations']}  t = {r['time']*1e3:.1f} ms")
