"""
Problem definition shared by all solvers.

Falkner-Skan equation for 2-D stagnation-point (Hiemenz) flow, beta = 1:

    f''' + f f'' - (f')^2 + 1 = 0
    f(0) = 0,  f'(0) = 0,  f'(eta -> inf) = 1

Written as a first-order system with y = [f, f', f'']:

    y0' = y1
    y1' = y2
    y2' = y1^2 - y0*y2 - 1
"""
import numpy as np

# Reference value of the wall shear f''(0) for beta = 1 (literature value, 9 d.p.)
FPP0_REFERENCE = 1.232587657

ETA_INF_DEFAULT = 10.0


def rhs(eta, y):
    """Right-hand side of the first-order system (works for 1-D or 2-D y)."""
    f, fp, fpp = y[0], y[1], y[2]
    return np.array([fp, fpp, fp * fp - f * fpp - 1.0])


def jacobian(y):
    """d(rhs)/dy, used to integrate the variational (sensitivity) equations."""
    f, fp, fpp = y
    return np.array([[0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0],
                     [-fpp, 2.0 * fp, -f]])


def augmented_rhs(eta, z):
    """
    State + sensitivity system for exact Newton derivatives.

    z = [f, f', f'', s0, s1, s2] where s = dy/d(alpha), alpha = f''(0).
    The sensitivity obeys s' = J(y) s with s(0) = [0, 0, 1].
    """
    y, s = z[:3], z[3:]
    return np.concatenate([rhs(eta, y), jacobian(y) @ s])


def initial_state(alpha, with_sensitivity=False):
    y0 = np.array([0.0, 0.0, alpha])
    if with_sensitivity:
        return np.concatenate([y0, [0.0, 0.0, 1.0]])
    return y0
