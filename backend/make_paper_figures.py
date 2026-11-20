"""
make_paper_figures.py -- builds figs/qualitative.png and figs/histogram.pdf for the paper
from the folder written by run_full_mpc.py.

Usage (from backend/, venv active):
    python make_paper_figures.py mpc_full_run_4.2.03_512
Output goes to ./figs/ (created if missing).
"""
import sys, os, json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED = "#1f2937", "#6b7280"

def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python make_paper_figures.py <run folder written by run_full_mpc.py>")
    d = sys.argv[1]
    need = ["original_image.png", "marked_image.png", "recovered_image.png", "hist.npz", "full_run_results.json"]
    missing = [f for f in need if not os.path.exists(os.path.join(d, f))]
    if missing:
        sys.exit("missing in %s: %s" % (d, ", ".join(missing)))
    os.makedirs("figs", exist_ok=True)

    orig = np.array(Image.open(os.path.join(d, "original_image.png")).convert("L"), dtype=np.int64)
    marked = np.array(Image.open(os.path.join(d, "marked_image.png")).convert("L"), dtype=np.int64)
    rec = np.array(Image.open(os.path.join(d, "recovered_image.png")).convert("L"), dtype=np.int64)
    res = json.load(open(os.path.join(d, "full_run_results.json")))
    hist = np.load(os.path.join(d, "hist.npz"))["hist"].astype(np.int64)
    ME, LE = int(res["ME"]), int(res["LE"])

    # ---- sanity checks the figure caption relies on ----
    identical = bool(np.array_equal(orig, rec))
    diff = np.abs(marked - orig)
    changed = int((diff > 0).sum())
    print("recovered == original:", identical)
    print("pixels changed by embedding: %d of %d (%.1f%%), max |diff| = %d" % (
        changed, diff.size, 100.0 * changed / diff.size, diff.max()))
    if not identical:
        print("WARNING: recovered image differs from the original; do not use this figure as-is")

    # ---- qualitative.png ----
    k = 50
    amp = np.clip(diff * k, 0, 255).astype(np.uint8)
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.55))
    panels = [(orig, "Original", None),
              (marked, "Marked (secrets embedded)", "PSNR %.2f dB" % res["psnr2_mpc_marked"]),
              (rec, "Recovered", "identical to original" if identical else "NOT identical"),
              (amp, "|Marked \u2212 Original| \u00d7 %d" % k, "%.1f%% of pixels changed" % (100.0 * changed / diff.size))]
    for ax, (im, title, sub) in zip(axes, panels):
        ax.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(title, fontsize=10.5, color=INK, fontweight="bold")
        if sub:
            ax.set_xlabel(sub, fontsize=9.5, color=MUTED)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#d1d5db")
    fig.tight_layout()
    fig.savefig("figs/qualitative.png", dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- histogram.pdf ----
    xmax = min(len(hist) - 1, LE + 12)
    bins = np.arange(xmax + 1)
    counts = hist[: xmax + 1]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    cols = []
    for b in bins:
        cols.append("#c0504d" if b == ME else ("#e0a458" if ME < b < LE else "#9db7d5"))
    nz = counts > 0                      # empty bins (e.g. LE itself) are left blank on the log axis
    ax.bar(bins[nz], counts[nz], width=0.85, color=[c for c, k in zip(cols, nz) if k], edgecolor="none")
    ax.set_yscale("log")
    ax.axvline(LE, color=INK, ls="--", lw=1.1)
    ax.annotate("ME = %d\n%d pixels carry one bit each" % (ME, hist[ME]), xy=(ME, hist[ME]),
                xytext=(ME + max(3, xmax * 0.12), hist[ME] * 0.9), fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    ax.text(LE + 0.6, ax.get_ylim()[1] * 0.55, "LE = %d" % LE, fontsize=9, color=INK, rotation=90, va="top")
    ax.set_xlabel("absolute prediction error |e|", fontsize=10.5)
    ax.set_ylabel("number of pixels (log scale)", fontsize=10.5)
    ax.set_xlim(-1, xmax + 1)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc="#c0504d", label="peak bin: data embedded"),
                       Patch(fc="#e0a458", label="between ME and LE: shifted by 1"),
                       Patch(fc="#9db7d5", label="other bins: unchanged")],
              fontsize=8.6, frameon=False, loc="upper right")
    ax.grid(axis="y", color="#e5e7eb", lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig("figs/histogram.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig("figs/histogram.png", dpi=250, bbox_inches="tight", facecolor="white")
    print("saved figs/qualitative.png, figs/histogram.pdf (and histogram.png)")

if __name__ == "__main__":
    main()