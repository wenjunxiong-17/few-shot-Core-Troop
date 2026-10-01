import os
"""Self-test: verifies the FD-HVP MAML meta-gradient against finite differences
of the true meta-objective."""
import sys
import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WS)
from ml.core import (MLP, ce_grad, p_flat, inner_adapt, maml_meta_grad, p_unflat)

rng = np.random.default_rng(0)
net = MLP([4, 16, 8, 3], seed=1)
P = net.P
X = rng.normal(size=(40, 4))
y = rng.integers(0, 3, size=40)
Xs, ys = X[:12], y[:12]
Xq, yq = X[12:], y[12:]

alpha = 0.1
steps = 2
inner = lambda Q: ce_grad(net, Q, Xs, ys)
outer = lambda Q: ce_grad(net, Q, Xq, yq)


def meta_obj(v):
    Q2 = p_unflat(v, P)
    thetas = inner_adapt(Q2, inner, alpha, steps)
    return outer(thetas[-1])[0]


for so in (True, False):
    thetas = inner_adapt(P, inner, alpha, steps)
    g = maml_meta_grad(thetas, outer, inner, alpha, second_order=so)
    gv = p_flat(g)
    fd = np.zeros_like(gv)
    eps = 1e-6
    for i in range(len(fd)):
        dv = np.zeros_like(gv)
        dv[i] = eps
        fd[i] = (meta_obj(dv) - meta_obj(-dv)) / (2 * eps)
    cos = float(gv @ fd / (np.linalg.norm(gv) * np.linalg.norm(fd) + 1e-300))
    rel = float(np.linalg.norm(gv - fd) / (np.linalg.norm(fd) + 1e-300))
    print(f"second_order={so}: |g|={np.linalg.norm(gv):.6e} |fd|={np.linalg.norm(fd):.6e} "
          f"cos={cos:.8f} rel_err={rel:.4e}")
    print(f"   grad[:5] = {np.round(gv[:5], 6)}")
    print(f"   fd  [:5] = {np.round(fd[:5], 6)}")
