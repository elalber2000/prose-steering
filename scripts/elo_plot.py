import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

prose_scores = [
    [-2, 4],
    [9, -2],
    [4, -9],
    [0, -2],
    [-2, -9],
    [0, -2],
    [9, 2],
    [-9, -4],
    [9, 0],
    [-2, 4],
    [4, 2],
    [4, 9],
    [-9, -4],
    [4, 9],
    [-9, 0],
    [-9, 0],
]

coherence_scores = [
    [-2, -4],
    [-2, 9],
    [4, -9],
    [-2, 0],
    [-2, -9],
    [-2, 0],
    [9, 2],
    [-4, -9],
    [0, 9],
    [-2, 4],
    [4, 2],
    [4, 0],
    [4, 9],
    [-9, -4],
    [4, 9],
    [-9, 0],
    [0, -9],
]



def spline_curve(x, y, n_points=300, k=3):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    xs = np.linspace(x.min(), x.max(), n_points)
    spline = make_interp_spline(x, y, k=min(k, len(x) - 1))
    ys = spline(xs)
    return xs, ys


K = 32
BASE_RATING = 1000
BOOTSTRAPS = 5000

BG = "#171717"
PROSE = "#82B366"
COHERENCE = "#9673A6"


def expected_score(ra, rb):
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))


def run_elo(matches, k=K, base_rating=BASE_RATING):
    models = sorted({x for a, b in matches for x in (a, b)}, key=float)
    ratings = {m: float(base_rating) for m in models}

    for winner, loser in matches:
        rw = ratings[winner]
        rl = ratings[loser]
        ew = expected_score(rw, rl)
        el = expected_score(rl, rw)
        ratings[winner] = rw + k * (1.0 - ew)
        ratings[loser] = rl + k * (0.0 - el)

    return ratings


def bootstrap_final_ratings(matches, n_boot=BOOTSTRAPS, seed=0):
    rng = np.random.default_rng(seed)
    models = sorted({x for a, b in matches for x in (a, b)}, key=float)
    finals = {m: [] for m in models}
    matches = np.array(matches, dtype=int)

    for _ in range(n_boot):
        shuffled = matches[rng.permutation(len(matches))].tolist()
        ratings = run_elo(shuffled)
        for m in models:
            finals[m].append(ratings[m])

    for m in models:
        finals[m] = np.array(finals[m])

    return models, finals


def summarize(finals, models):
    mean = np.array([finals[m].mean() for m in models])
    lo = np.array([np.percentile(finals[m], 2.5) for m in models])
    hi = np.array([np.percentile(finals[m], 97.5) for m in models])
    return mean, lo, hi


def plot_overlay_bootstrap(prose_scores, coherence_scores):
    prose_models, prose_finals = bootstrap_final_ratings(prose_scores, seed=1)
    coh_models, coh_finals = bootstrap_final_ratings(coherence_scores, seed=2)

    models = sorted(set(prose_models) | set(coh_models), key=float)

    prose_mean, prose_lo, prose_hi = summarize(prose_finals, models)
    coh_mean, coh_lo, coh_hi = summarize(coh_finals, models)

    x = np.arange(len(models))
    offset = 0.12

    plt.figure(figsize=(10, 6), facecolor=BG)
    ax = plt.gca()
    ax.set_facecolor(BG)

    for i in range(len(models)):
        ax.vlines(x[i] - offset, prose_lo[i], prose_hi[i], color=PROSE, linewidth=3, alpha=0.95)
        ax.scatter(x[i] - offset, prose_mean[i], color=PROSE, s=70, zorder=3)

        ax.vlines(x[i] + offset, coh_lo[i], coh_hi[i], color=COHERENCE, linewidth=3, alpha=0.95)
        ax.scatter(x[i] + offset, coh_mean[i], color=COHERENCE, s=70, zorder=3)

    prose_xs, prose_ys = spline_curve(x - offset, prose_mean)
    coh_xs, coh_ys = spline_curve(x + offset, coh_mean)

    ax.plot(prose_xs, prose_ys, color=PROSE, linewidth=1.5, alpha=0.9, linestyle="dashed")
    ax.plot(coh_xs, coh_ys, color=COHERENCE, linewidth=1.5, alpha=0.9, linestyle="dashed")

    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in models], color="white")
    ax.set_xlabel("Steering alpha", color="white")
    ax.set_ylabel("Elo Score", color="white")
    ax.set_title("Bootstrap Elo Comparison: Prose vs Coherence", color="white", pad=12)

    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
        spine.set_alpha(0.15)

    ax.grid(True, axis="y", alpha=0.12, color="white")

    legend = ax.legend(
        handles=[
            plt.Line2D([0], [0], color=PROSE, marker="o", linewidth=3, label="Prose"),
            plt.Line2D([0], [0], color=COHERENCE, marker="o", linewidth=3, label="Coherence"),
        ],
        facecolor=BG,
        edgecolor="#333333",
        labelcolor="white",
        loc="best",
    )
    for text in legend.get_texts():
        text.set_color("white")

    plt.tight_layout()
    plt.show()

    print("Model  Prose(mean [2.5,97.5])      Coherence(mean [2.5,97.5])")
    for m in models:
        pm = prose_finals[m].mean()
        pl = np.percentile(prose_finals[m], 2.5)
        ph = np.percentile(prose_finals[m], 97.5)

        cm = coh_finals[m].mean()
        cl = np.percentile(coh_finals[m], 2.5)
        ch = np.percentile(coh_finals[m], 97.5)

        print(f"{m:>4}  {pm:7.2f} [{pl:7.2f}, {ph:7.2f}]   {cm:7.2f} [{cl:7.2f}, {ch:7.2f}]")


plot_overlay_bootstrap(prose_scores, coherence_scores)