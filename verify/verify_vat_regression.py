"""VAT rank theorem -- scalar regression case.  [data-free]

For a scalar output f and the perturbation r,

    D(r) = ( f(x + r) - f(x) )^2

so

    grad_r D |_{r=0} = 0        (r = 0 is a stationary point)
    Hess_r D |_{r=0} = 2 * grad_x f (grad_x f)^T    ->  rank 1

with

    lambda_1 = 2 * ||grad_x f||^2 ,      u_1 = grad_x f / ||grad_x f|| .

The rank is 1 regardless of the input dimension I, because a scalar output
has only one "class direction".  Hence for regression the VAT power
iteration is redundant and the adversarial perturbation is simply the
normalised input gradient -- unsupervised FGSM on the prediction.

This is the reason the water-dataset experiments selected the VAT /
FixMatch semi-supervised weight to exactly 0: the method has no extra
direction to exploit, while Pseudo-Label with a lagged EMA teacher does.

Needs NO dataset: trains a small MLP on a synthetic scalar-regression task.
"""
import os
import sys
import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WS)
from ml.core import MLP, mse_grad, Adam   # noqa: E402

I = 11                            # input dim (same as the water covariates)
N = 4000
rng = np.random.default_rng(0)

# ---- synthetic scalar task: fixed random teacher ----------------------
teacher = MLP([I, 16, 1], seed=1, last_scale=1.0)
X = rng.normal(0, 1, size=(N, I))
T = teacher.forward(teacher.P, X)
T = (T - T.mean()) / (T.std() + 1e-12)          # standardise the target
X = (X - X.mean(0)) / (X.std(0) + 1e-12)

# ---- fit a student ---------------------------------------------------
net = MLP([I, 64, 64, 1], seed=0, last_scale=0.1)
P = net.P
opt = Adam(P, lr=3e-3)
for step in range(3000):
    i = rng.choice(N, 128, replace=False)
    _, g = mse_grad(net, P, X[i], T[i].reshape(-1, 1))
    P = opt.step(P, g)

resid = float(np.mean((net.forward(P, X) - T) ** 2))
print(f"synthetic scalar-regression task: student MSE = {resid:.4f} "
      f"(target variance = 1.0)\n")

delta = 1e-5
Xq = X[:8]


def dD_dr(P, x_ref, x):
    """grad_r D at r = x - x_ref, with D = (f(x) - f(x_ref))^2 and the
    reference argument detached."""
    f0 = net.forward(P, x_ref)
    fp = net.forward(P, x)
    dZ = 2.0 * (fp - f0)                     # dD/df(x)
    return net.backward_input(P, x, dZ)[0]


def grad_f(P, x):
    """grad_x f by central differences.  x: (1, I) -> (I,)."""
    g = np.zeros(x.shape[1])
    for j in range(x.shape[1]):
        for sgn in (1, -1):
            xp = x.copy()
            xp[0, j] += sgn * delta
            g[j] += sgn * net.forward(P, xp)[0, 0] / (2 * delta)
    return g


print(f"{'pt':>3} {'rank(H)':>8} {'lam1':>10} {'2|grad f|^2':>12} "
      f"{'|cos(u,g)|':>11} {'rel.err':>10}")
worst_cos, worst_rel, ranks = 1.0, 0.0, []
for i in range(len(Xq)):
    x0 = Xq[i:i + 1]
    H = np.zeros((I, I))
    for j in range(I):
        for sgn in (1, -1):
            xp = x0.copy()
            xp[0, j] += sgn * delta
            H[:, j] += sgn * dD_dr(P, x0, xp) / (2 * delta)
    H = 0.5 * (H + H.T)
    ev, V = np.linalg.eigh(H)
    lam1, u = ev[-1], V[:, -1]
    g = grad_f(P, x0)
    pred = 2.0 * float(g @ g)
    cos = abs(float(u @ g) / (np.linalg.norm(g) + 1e-12))
    rel = abs(lam1 - pred) / max(lam1, 1e-12)
    rank = int((np.abs(ev) > 1e-6 * abs(lam1)).sum())
    ranks.append(rank)
    worst_cos, worst_rel = min(worst_cos, cos), max(worst_rel, rel)
    print(f"{i:>3} {rank:>8} {lam1:>10.4f} {pred:>12.4f} {cos:>11.6f} "
          f"{rel:>10.2e}")

print(f"\n[check] rank(H) = {set(ranks)}  (expected {{1}} for a scalar output)")
print(f"[check] min |cos(u_1, grad f)| = {worst_cos:.6f}   (expected 1.0)")
print(f"[check] max relative error of lambda_1 = {worst_rel:.2e}")

# ---- the practical consequence --------------------------------------
eps, xi = 0.15, 1e-6
d = rng.normal(size=Xq.shape)
d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
f0 = net.forward(P, Xq)
fp = net.forward(P, Xq + xi * d)
r_vat = eps * net.backward_input(P, Xq + xi * d, 2.0 * (fp - f0))
r_vat /= (np.linalg.norm(r_vat, axis=1, keepdims=True) + 1e-12)
gf = np.stack([grad_f(P, Xq[i:i + 1]) for i in range(len(Xq))])
gf /= (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-12)
print(f"[check] per-sample |cos(VAT direction, normalised grad f)| "
      f"= {np.abs((r_vat * gf).sum(1)).mean():.6f}   (expected 1.0)")

print("\n=> scalar-regression VAT is exactly rank-1: the perturbation "
      "direction is\n   simply grad_x f, i.e. unsupervised FGSM on the "
      "prediction.\n   All the power iteration buys you is that same "
      "direction, at extra cost.")
