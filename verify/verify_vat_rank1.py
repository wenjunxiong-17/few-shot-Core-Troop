"""VAT rank theorem: for a softmax classifier with C classes,

    D(r) = D_KL( p(y|x) || p(y|x + r) )     (first argument detached)

    grad_r D |_{r=0} = 0
    H = Hess_r D |_{r=0} = sum_c p_c (J_c - Jbar)(J_c - Jbar)^T

where J_c = grad_x z_c and Jbar = sum_c p_c J_c.  Because
sum_c p_c (J_c - Jbar) = 0, the C vectors span at most a
(C-1)-dimensional subspace, hence

    rank(H) = min(I, C - 1).

Consequences
------------
* BINARY classification (C = 2):  rank 1, and with g = grad_x(z1 - z0),

      H = p0 p1 g g^T ,   lambda_1 = p0 p1 ||g||^2 ,   u_1 = g / ||g|| .

  One step of VAT's power iteration from ANY random direction d returns
  normalise(H d) = +-g/||g||: the power iteration is redundant and VAT
  degenerates into *unsupervised FGSM* along the decision-boundary normal.

* SCALAR REGRESSION: rank 1 for the same reason -- a scalar output has a
  single "class direction" (see verify_vat_regression.py).

* MULTI-CLASS, C >= 3:  rank(H) = min(I, C-1) > 1, the adversarial
  direction is genuinely ambiguous, and the power iteration / a proper
  eigen-solver does buy something.  This is the only regime in which VAT
  is a real method rather than a relabelled FGSM.

The identity is *pointwise* in (x, theta) -- it holds at every parameter
value -- so this script verifies it at randomly initialised parameters.
No dataset and no training are needed.

Numerical note: the finite-difference Hessian is only meaningful where the
model is not saturated.  If some p_c is ~1e-16 the curvature itself is
~1e-12 and double precision cannot resolve it, so such query points are
filtered out (they are reported, not hidden).
"""
import os
import sys
import numpy as np

WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WS)
from ml.core import MLP, softmax   # noqa: E402

I = 9                       # input dimension (same as the water covariates)
NQ = 160                    # candidate query points
MAXKEEP = 24                # non-degenerate points actually used
PMIN = 1e-3                 # "not saturated" threshold
delta = 1e-4                # central-difference step
xi = 1e-5                   # VAT finite-difference step
eps = 0.15                  # VAT perturbation size

rng = np.random.default_rng(0)
X = rng.normal(0, 1, size=(NQ, I))


def kl_grad_at(P, net, x_ref, x_pert):
    """grad_{x_pert} D_KL( p(.|x_ref) || p(.|x_pert) ), first arg detached.
    d D_KL / d logits = (p_pert - p_ref)."""
    p = softmax(net.forward(P, x_ref))
    pp = softmax(net.forward(P, x_pert))
    return net.backward_input(P, x_pert, pp - p)


def numeric_hess(P, net, x):
    """Hessian of D_KL w.r.t. r at r = 0, central differences."""
    H = np.zeros((I, I))
    for j in range(I):
        for sgn in (1, -1):
            xp = x.copy()
            xp[0, j] += sgn * delta
            H[:, j] += sgn * kl_grad_at(P, net, x, xp)[0] / (2 * delta)
    return 0.5 * (H + H.T)


def analytic_hess(P, net, x, C):
    """H = sum_c p_c (J_c - Jbar)(J_c - Jbar)^T, exactly."""
    J = np.zeros((C, I))
    for c in range(C):
        dZ = np.zeros((1, C))
        dZ[0, c] = 1.0
        J[c] = net.backward_input(P, x, dZ)[0]
    p = softmax(net.forward(P, x))[0]
    A = J - (p[:, None] * J).sum(0)
    return (p[:, None] * A).T @ A, J, p


def stable_rank(H, tol=1e-6):
    ev = np.linalg.eigvalsh(0.5 * (H + H.T))
    return int((np.abs(ev) > tol * max(abs(ev[-1]), 1e-300)).sum())


def pick_points(P, net):
    """Query points where the model is not saturated."""
    keep, dropped = [], 0
    for t in range(NQ):
        p = softmax(net.forward(P, X[t:t + 1]))[0]
        if p.min() > PMIN:
            keep.append(t)
            if len(keep) == MAXKEEP:
                break
        else:
            dropped += 1
    return keep, dropped


print(f"input dim I = {I}, {NQ} candidate query points, random parameters, "
      f"p_min > {PMIN:g} filter\n")
print(f"{'C':>2} {'rank(H) FD':>11} {'min(I,C-1)':>11} "
      f"{'max |dH|/|H|':>14} {'kept':>5} {'dropped':>8}")

results = {}
for C in (2, 3, 5, 9):
    net = MLP([I, 32, 32, C], seed=C, last_scale=1.0)
    P = net.P
    keep, dropped = pick_points(P, net)
    ranks, rels = [], []
    for t in keep:
        x = X[t:t + 1]
        Hn = numeric_hess(P, net, x)
        Ha, _, _ = analytic_hess(P, net, x, C)
        ranks.append(stable_rank(Hn))
        rels.append(np.linalg.norm(Hn - Ha) / (np.linalg.norm(Ha) + 1e-300))
    results[C] = dict(ranks=ranks, rels=rels, net=net, P=P, keep=keep)
    print(f"{C:>2} {sorted(set(ranks))!s:>11} {min(I, C - 1):>11} "
          f"{max(rels):>14.2e} {len(keep):>5} {dropped:>8}")

print("\n[check 1] rank(H) == min(I, C-1) at every kept point, for every C: "
      f"{all(sorted(set(results[C]['ranks'])) == [min(I, C - 1)] for C in results)}")
print("[check 2] the finite-difference Hessian matches the closed form "
      f"H = sum_c p_c (J_c-Jbar)(J_c-Jbar)^T: max rel.err = "
      f"{max(max(r['rels']) for r in results.values()):.2e}")

# ------------------------------------------------------------------
# Binary case in detail: the closed form.
# ------------------------------------------------------------------
C = 2
net, P, keep = results[C]["net"], results[C]["P"], results[C]["keep"]
print(f"\n--- binary case (C = {C}): closed form ---")
print(f"{'pt':>4} {'rank':>5} {'lam1':>10} {'p0p1|g|^2':>10} "
      f"{'|cos(u,g)|':>11} {'rel.err':>9}")
worst_cos, worst_rel = 1.0, 0.0
for k, t in enumerate(keep[:8]):
    x = X[t:t + 1]
    Hn = numeric_hess(P, net, x)
    _, J, p = analytic_hess(P, net, x, C)
    g = J[0] - J[1]                                   # grad_x (z0 - z1)
    ev, V = np.linalg.eigh(Hn)
    lam1, u = ev[-1], V[:, -1]
    pred = p[0] * p[1] * float(g @ g)
    cos = abs(float(u @ g) / (np.linalg.norm(g) + 1e-12))
    rel = abs(lam1 - pred) / max(lam1, 1e-12)
    worst_cos, worst_rel = min(worst_cos, cos), max(worst_rel, rel)
    print(f"{k:>4} {stable_rank(Hn):>5} {lam1:>10.4f} {pred:>10.4f} "
          f"{cos:>11.6f} {rel:>9.2e}")
print(f"  => min |cos(u_1, g)| = {worst_cos:.6f}, "
      f"max rel.err(lambda_1) = {worst_rel:.2e}")

# ------------------------------------------------------------------
# The practical consequence: is VAT's power iteration redundant?
#   one step from a random d returns normalise(H d), which equals the top
#   eigenvector u_1 exactly when rank(H) = 1.
# ------------------------------------------------------------------
print("\n--- is VAT's power iteration redundant? ---")
for C in (2, 3, 5, 9):
    r = results[C]
    net, P, keep = r["net"], r["P"], r["keep"]
    xb = X[keep]
    d = rng.normal(size=xb.shape)
    d /= (np.linalg.norm(d, axis=1, keepdims=True) + 1e-12)
    rv = eps * kl_grad_at(P, net, xb, xb + xi * d)     # ~ xi * H d
    rv /= (np.linalg.norm(rv, axis=1, keepdims=True) + 1e-12)
    cos = np.zeros(len(keep))
    for i, t in enumerate(keep):
        Ha, _, _ = analytic_hess(P, net, X[t:t + 1], C)
        _, V = np.linalg.eigh(Ha)
        cos[i] = abs(float(rv[i] @ V[:, -1]))          # vs top eigenvector
    print(f"  C={C}: rank(H) = {min(I, C - 1)}, mean |cos(one-step VAT "
          f"direction, top eigenvector)| = {cos.mean():.6f} "
          f"(min {cos.min():.6f})")

print("\n=> C = 2: rank 1, so the one-step VAT direction is already exact --"
      "\n   VAT == unsupervised FGSM along the decision-boundary normal and"
      "\n   the power iteration buys nothing."
      "\n   C >= 3: rank(H) = min(I, C-1) > 1, the direction is genuinely"
      "\n   ambiguous, and the power iteration does matter.  VAT is only a"
      "\n   real method in the multi-class regime.")
