"""Plot the type I / type II error trade-off curve of Gaussian DP (GDP).

A mechanism is mu-GDP iff distinguishing any two neighbouring datasets is at
least as hard as distinguishing N(0, 1) from N(mu, 1).  The optimal test at
type I error alpha therefore has type II error

    G_mu(alpha) = Phi(Phi^{-1}(1 - alpha) - mu),

which is exactly the curve drawn here.

Usage:
    python plot_gaussian_dp_tradeoff.py                       # mu = 0, 0.5, 1, 2, 4
    python plot_gaussian_dp_tradeoff.py --mu 1.0 --mu 3.0
    python plot_gaussian_dp_tradeoff.py --mu 2.0 --alpha 0.1 --out gdp.png
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm


def gdp_tradeoff(alpha, mu):
    """Type II error beta at type I error alpha for a mu-GDP mechanism."""
    alpha = np.asarray(alpha, dtype=float)
    return norm.cdf(norm.ppf(1.0 - alpha) - mu)


def gdp_delta(eps, mu):
    """The (eps, delta) curve implied by mu-GDP (Bu et al., dual of G_mu)."""
    eps = np.asarray(eps, dtype=float)
    if mu == 0:
        return np.zeros_like(eps)
    return norm.cdf(-eps / mu + mu / 2) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2)


def compute_empirical_tradeoff(losses_in, losses_out):
    """Compute empirical (alpha, beta) trade-off points and lower convex hull."""
    scores_in = np.asarray(losses_in, dtype=float).ravel()
    scores_out = np.asarray(losses_out, dtype=float).ravel()

    thresholds = np.unique(np.concatenate([scores_in, scores_out]))
    # Predict IN if score > threshold (canary score is -MSE, higher means IN)
    alphas = np.mean(scores_out[:, None] > thresholds, axis=0)
    betas = np.mean(scores_in[:, None] <= thresholds, axis=0)

    alphas = np.concatenate([[0.0], alphas, [1.0]])
    betas = np.concatenate([[1.0], betas, [0.0]])

    # Monotone chain algorithm for lower convex hull
    pts = sorted(set(zip(alphas, betas)))
    hull = []
    for p in pts:
        while len(hull) >= 2:
            o, a = hull[-2], hull[-1]
            if (a[0] - o[0]) * (p[1] - o[1]) - (a[1] - o[1]) * (p[0] - o[0]) <= 0:
                hull.pop()
            else:
                break
        hull.append(p)

    hull_a = np.array([p[0] for p in hull])
    hull_b = np.array([p[1] for p in hull])
    return alphas, betas, hull_a, hull_b


def plot_tradeoff(mus, alpha_mark=0.05, out=None, show_pdf_panel=True, empirical=None):
    alphas = np.linspace(0.0, 1.0, 1001)

    ncols = 2 if show_pdf_panel else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6.0 * ncols, 5.0))
    axes = np.atleast_1d(axes)
    ax = axes[0]

    for mu in mus:
        ax.plot(alphas, gdp_tradeoff(alphas, mu), lw=2, label=rf"$\mu={mu:g}$")

    ax.plot([0, 1], [1, 0], "k--", lw=1, alpha=0.6,
            label=r"$\mu=0$ (perfect privacy, $\beta=1-\alpha$)")

    if empirical is not None:
        alphas_emp, betas_emp, hull_a, hull_b = empirical
        ax.scatter(alphas_emp, betas_emp, s=15, color="dodgerblue", alpha=0.5,
                   label="Empirical points", zorder=4)
        ax.plot(hull_a, hull_b, color="crimson", lw=2,
                label="Empirical lower bound", zorder=5)
    ax.set_xlabel(r"type I error  $\alpha$   (false positive rate)")
    ax.set_ylabel(r"type II error  $\beta$   (false negative rate)")
    ax.set_title("Gaussian DP trade-off function\n"
                 r"$G_\mu(\alpha)=\Phi(\Phi^{-1}(1-\alpha)-\mu)$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # Annotate one operating point on the largest-mu curve.
    mu_mark = max(mus)
    if mu_mark > 0:
        beta_mark = float(gdp_tradeoff(alpha_mark, mu_mark))
        ax.plot([alpha_mark], [beta_mark], "o", color="crimson", zorder=5)
        ax.annotate(rf"$\alpha={alpha_mark:g}\Rightarrow\beta={beta_mark:.3f}$",
                    xy=(alpha_mark, beta_mark), xytext=(alpha_mark + 0.12, beta_mark + 0.12),
                    arrowprops=dict(arrowstyle="->", color="crimson"),
                    color="crimson", fontsize=9)

    if show_pdf_panel:
        # Where those two errors actually come from: one threshold, two Gaussians.
        ax2 = axes[1]
        mu_p = mu_mark if mu_mark > 0 else 1.0
        x = np.linspace(-4, mu_p + 4, 1000)
        thr = norm.ppf(1.0 - alpha_mark)                 # reject H0 when X > thr
        beta = float(norm.cdf(thr - mu_p))

        ax2.plot(x, norm.pdf(x, 0, 1), lw=2, color="tab:blue",
                 label=r"$H_0$: neighbour absent  $N(0,1)$")
        ax2.plot(x, norm.pdf(x, mu_p, 1), lw=2, color="tab:orange",
                 label=rf"$H_1$: neighbour present  $N({mu_p:g},1)$")
        ax2.fill_between(x, 0, norm.pdf(x, 0, 1), where=(x > thr),
                         color="tab:blue", alpha=0.35,
                         label=rf"false positive $=\alpha={alpha_mark:g}$")
        ax2.fill_between(x, 0, norm.pdf(x, mu_p, 1), where=(x <= thr),
                         color="tab:orange", alpha=0.35,
                         label=rf"false negative $=\beta={beta:.3f}$")
        ax2.axvline(thr, color="k", ls=":", lw=1.5)
        ax2.text(thr, ax2.get_ylim()[1] * 0.96, "  threshold", fontsize=9, va="top")
        ax2.set_xlabel("test statistic")
        ax2.set_ylabel("density")
        ax2.set_title(rf"one operating point ($\mu={mu_p:g}$): moving the"
                      "\n"
                      r"threshold trades $\alpha$ against $\beta$")
        ax2.legend(fontsize=8, loc="upper left")
        ax2.grid(alpha=0.3)

    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"saved -> {out}")
    return fig


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mu", type=float, action="append", default=None,
                   help="GDP mu (repeatable). Default: 0.5 1 2 4")
    p.add_argument("--alpha", type=float, default=0.05,
                   help="type I error to annotate (default 0.05)")
    p.add_argument("--results_dir", type=str, default=None,
                   help="Directory containing losses_in.npy and losses_out.npy")
    p.add_argument("--out", default="gdp_tradeoff.png", help="output image path")
    p.add_argument("--no-show", action="store_true", help="do not open a window")
    args = p.parse_args()

    mus = args.mu if args.mu else [0.5, 1.0, 2.0, 4.0]

    for mu in mus:
        b = float(gdp_tradeoff(args.alpha, mu))
        print(f"mu={mu:>5g} | alpha={args.alpha:g} -> beta={b:.4f} "
              f"(power {1 - b:.4f}) | delta(eps=1)={float(gdp_delta(1.0, mu)):.4g}")

    empirical = None
    if args.results_dir:
        import os
        fin = os.path.join(args.results_dir, "losses_in.npy")
        fout = os.path.join(args.results_dir, "losses_out.npy")
        lin = np.load(fin, allow_pickle=True)
        lout = np.load(fout, allow_pickle=True)
        empirical = compute_empirical_tradeoff(lin, lout)
        print(f"Loaded {len(lin)} IN and {len(lout)} OUT models from {args.results_dir}")

    plot_tradeoff(mus, alpha_mark=args.alpha, out=args.out, empirical=empirical)
    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
