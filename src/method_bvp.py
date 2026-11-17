"""
Method 1 - Global collocation with scipy.integrate.solve_bvp.

The infinite domain is truncated at eta_inf and the far-field condition
f'(eta_inf) = 1 is imposed there.
"""
import time
import numpy as np
from scipy.integrate import solve_bvp

from .problem import rhs, ETA_INF_DEFAULT


def _bc(ya, yb):
    return np.array([ya[0], ya[1], yb[1] - 1.0])


def solve(eta_inf=ETA_INF_DEFAULT, n_mesh=101, tol=1e-8):
    eta = np.linspace(0.0, eta_inf, n_mesh)
    # Initial guess that already satisfies the BCs: f' = 1 - exp(-eta)
    guess = np.vstack([eta - 1.0 + np.exp(-eta),
                       1.0 - np.exp(-eta),
                       np.exp(-eta)])
    t0 = time.perf_counter()
    sol = solve_bvp(rhs, _bc, eta, guess, tol=tol, max_nodes=100000)
    elapsed = time.perf_counter() - t0
    if not sol.success:
        raise RuntimeError(f"solve_bvp failed: {sol.message}")
    return {
        "name": "solve_bvp (collocation)",
        "fpp0": float(sol.sol(0.0)[2]),
        "eta": sol.x,
        "y": sol.y,
        "dense": sol.sol,
        "iterations": sol.niter,
        "nodes": sol.x.size,
        "time": elapsed,
        "residual": float(np.max(sol.rms_residuals)),
    }


if __name__ == "__main__":
    r = solve()
    print(f"[{r['name']}] f''(0) = {r['fpp0']:.10f}  "
          f"nodes = {r['nodes']}  iters = {r['iterations']}  t = {r['time']*1e3:.1f} ms")
