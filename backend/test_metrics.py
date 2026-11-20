"""
metrics_mpc.py -- metrics for the FULL MPC-protocol PEHS pipeline
(SGAP prediction -> |e| -> histogram -> histogram-shift embed -> sequential extract),
computed entirely on additive shares via Pi_Mul / Pi_RS / Pi_PriCmp / Pi_PriEq / Pi_B2A.

What it measures
  A. Correctness, in the format of the paper's Table II (PSNR1..PSNR4), plus bit-exactness
  B. Wall-clock time per pipeline stage, versus the plaintext PEHS baseline
  C. Communication rounds and bits per stage (counted, not estimated)
  D. Per-protocol micro-benchmark: measured rounds / bits vs the paper's Table VI
  E. Extrapolation of measured per-pixel / per-bin cost to a full 512x512 image

Run from backend/ (venv active):
    python metrics_mpc.py                    # sizes 16 24 32 48, up to 2 images from test_images/
    python metrics_mpc.py --sizes 16         # quick check
    python metrics_mpc.py --sizes 16 24 32 48 64 --images 3

Outputs: mpc_metrics_report.txt (readable) and mpc_metrics_results.json (raw).
"""
import sys
import os
import time
import json
import glob
import math
import argparse
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.ass import ASS
from app.core.pehs import pehs_embed_plaintext, pehs_extract_plaintext, psnr, sgap_predict
from app.protocols.offline import Offline
from app.protocols import pehs_mpc
from app.protocols.mul import pi_mul as _orig_pi_mul
from app.protocols.pricmp import gen_beta, pi_pricmp_2pc
from app.protocols.prieq import pi_prieq_2pc
from app.protocols.b2a import gen_random_bit_both_fields, pi_b2a
from app.protocols.rshift import pi_rs_2pc_correct


# ----------------------------------------------------------------------------
# Communication accounting
#   rounds : one per "reveal" (ASS.reconstruct call inside a protocol). The two reveals inside
#            a single Pi_Mul happen in parallel in a real deployment, so a whole Pi_Mul = 1 round.
#   bits   : total bits sent by ALL parties for that reveal = n*(n-1) * elements * field_bits.
# ----------------------------------------------------------------------------
class Counter:
    def __init__(self):
        self.rounds = 0
        self.bits = 0
        self.in_mul = False

    def reset(self):
        self.rounds = 0
        self.bits = 0
        self.in_mul = False


COUNTER = Counter()


class CountingASS(ASS):
    """ASS whose reconstruct() (== a reveal inside a protocol) is counted."""

    def reconstruct(self, shares):
        elems = int(np.size(shares[0]))
        COUNTER.bits += self.n * (self.n - 1) * elems * self.l
        if not COUNTER.in_mul:
            COUNTER.rounds += 1
        return super().reconstruct(shares)


def counting_pi_mul(x_shares_list, nbte, ass):
    was = COUNTER.in_mul
    COUNTER.in_mul = True
    try:
        out = _orig_pi_mul(x_shares_list, nbte, ass)
    finally:
        COUNTER.in_mul = was
    if not was:
        COUNTER.rounds += 1
    return out


# pehs_mpc looks up pi_mul as a module global at call time, so this patch reaches every use.
pehs_mpc.pi_mul = counting_pi_mul


def run_stage(stats, name, fn):
    r0, b0 = COUNTER.rounds, COUNTER.bits
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    stats[name] = {"time_sec": dt, "rounds": COUNTER.rounds - r0, "comm_bits": COUNTER.bits - b0}
    return out


# ----------------------------------------------------------------------------
# Test images
# ----------------------------------------------------------------------------
def synthetic_image(size, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    base = 128 + 60 * np.sin(xx / 5.0) + 50 * np.cos(yy / 7.0)
    img = base + rng.normal(0, 6, size=(size, size))
    return np.clip(img, 0, 255).astype(np.int64)


def find_images(folder, max_images):
    paths = []
    for ext in ("png", "jpg", "jpeg", "tiff", "tif", "bmp"):
        paths += glob.glob(os.path.join(folder, "*." + ext))
    return sorted(paths)[:max_images]


def load_gray(path, size):
    im = Image.open(path).convert("L").resize((size, size))
    return np.array(im, dtype=np.int64)


# ----------------------------------------------------------------------------
# Full-pipeline run on one image
# ----------------------------------------------------------------------------
def run_pipeline(img, n, l, verbose=True, seed=0):
    COUNTER.reset()
    h, w = img.shape
    ass = CountingASS(l=l, n=n)
    ass_bool = CountingASS(l=1, n=n)
    plain = ASS(l=l, n=n)  # non-counting; used only to check results (receiver-side combine)
    off_bool = Offline(ass_bool)
    off_arith = Offline(ass)
    rng = np.random.default_rng(seed)
    stats = {}

    # ---- plaintext ground truth + baseline timing ----
    pred_p, mask_p = sgap_predict(img)
    max_e = int(np.abs(img - pred_p)[mask_p].max())
    data_bits = rng.integers(0, 2, size=h * w).tolist()  # more than any image can hold -> max capacity

    t0 = time.perf_counter()
    marked_p, ME_p, LE_p, n_emb_p = pehs_embed_plaintext(img, data_bits)
    t_plain_embed = time.perf_counter() - t0
    t0 = time.perf_counter()
    pehs_extract_plaintext(marked_p, ME_p, LE_p)
    t_plain_extract = time.perf_counter() - t0

    # ---- MPC pipeline (Hiders compute on shares only) ----
    img_shares = run_stage(stats, "encrypt", lambda: ass.share(img))
    pred_shares, mask = run_stage(stats, "prediction", lambda: pehs_mpc.sgap_predict_on_shares(
        img_shares, l, ass, ass_bool, off_bool))
    e_shares = run_stage(stats, "error", lambda: pehs_mpc.compute_prediction_error_on_shares(
        img_shares, pred_shares, ass))
    abs_e, sign = run_stage(stats, "abs_value", lambda: pehs_mpc.abs_value_on_shares(
        e_shares, l, ass, ass_bool, off_bool, off_arith))
    hist, ME, LE = run_stage(stats, "histogram", lambda: pehs_mpc.histogram_on_shares(
        abs_e, mask, l, ass, ass_bool, off_bool, off_arith, max_e_estimate=max_e))
    marked_shares, n_emb = run_stage(stats, "embed", lambda: pehs_mpc.histogram_shift_embed_on_shares(
        img_shares, abs_e, sign, ME, LE, data_bits, mask, l, ass, ass_bool, off_bool, off_arith))
    marked_mpc = run_stage(stats, "receiver_combine", lambda: plain.reconstruct(marked_shares))
    if verbose:
        print("      sequential extraction (one pixel at a time, by design)...")
    rec_shares, bits_mpc = run_stage(stats, "extract_sequential", lambda: pehs_mpc.histogram_extract_sequential_on_shares(
        marked_shares, ME, LE, mask, l, ass, ass_bool, off_bool, off_arith, verbose=verbose))
    rec_mpc = plain.reconstruct(rec_shares)

    # ---- correctness ----
    rec3, bits3 = pehs_extract_plaintext(marked_mpc, ME, LE)  # plaintext-side extraction on reconstructed MI

    return {
        "h": h, "w": w, "n": n, "l": l,
        "valid_pixels": int(mask.sum()),
        "ME": ME, "LE": LE, "hist_bins": max_e + 1,
        "ME_LE_match_plaintext": bool(ME == ME_p and LE == LE_p),
        "embed_bit_exact_vs_plaintext": bool(np.array_equal(marked_p, marked_mpc)),
        "psnr1_plain_marked": psnr(img, marked_p),
        "psnr2_mpc_marked": psnr(img, marked_mpc),
        "psnr3_plain_extract_on_MI": psnr(img, rec3),
        "psnr4_mpc_extract_on_shares": psnr(img, rec_mpc),
        "capacity_bits": int(n_emb),
        "embedding_rate_bpp": n_emb / float(h * w),
        "bits_correct": bool(bits_mpc[:n_emb] == data_bits[:n_emb]),
        "extraction_paths_agree": bool(bits_mpc == bits3),
        "overflow_pixels_in_marked": int(np.sum((marked_mpc < 0) | (marked_mpc > 255))),
        "data_expansion_total": n * l / 8.0,
        "data_expansion_per_hider": l / 8.0,
        "plain_embed_sec": t_plain_embed,
        "plain_extract_sec": t_plain_extract,
        "stages": stats,
    }


# ----------------------------------------------------------------------------
# Per-protocol micro-benchmark, measured vs. paper Table VI ("Ours" row)
# ----------------------------------------------------------------------------
def paper_table6(n, l, p=1):
    rs_bits = 2 * p * sum((i - 1) * math.comb(n, i) for i in range(2, n + 1)) * l
    return {
        "Mul": (1, "O(N)"),
        "RS": (p, rs_bits),
        "B2A": (1, 2 * (n - 1)),
        "Cmp": (2, (n - 1) * (l * l + l - 2)),
        "Eq": (2, 4 * (n - 1) * l),
    }


def bench_protocols(n, l, E=2000):
    ass = CountingASS(l=l, n=n)
    ass_bool = CountingASS(l=1, n=n)
    plain = ASS(l=l, n=n)
    off_a = Offline(ass)
    off_b = Offline(ass_bool)
    rng = np.random.default_rng(1)
    x = rng.integers(-100, 100, size=E).astype(np.int64)
    y = rng.integers(-100, 100, size=E).astype(np.int64)
    y[: E // 2] = x[: E // 2]  # half equal so Eq is exercised both ways
    xs, ys = ass.share(x), ass.share(y)
    out = {}

    def measure(name, fn, check):
        COUNTER.reset()
        t0 = time.perf_counter()
        res = fn()
        dt = time.perf_counter() - t0
        out[name] = {
            "rounds": COUNTER.rounds,
            "comm_bits_per_elem_all_parties": COUNTER.bits / E,
            "time_us_per_elem": dt / E * 1e6,
            "correct": bool(check(res)),
        }

    nbte = off_a.gen_nbte(N=2, shape=(E,))
    measure("Mul", lambda: counting_pi_mul([xs, ys], nbte, ass),
            lambda r: np.array_equal(plain.reconstruct(r), plain._wrap_signed(x * y)))

    bits = rng.integers(0, 2, size=E).astype(np.int64)
    bsh = ass_bool.share(bits)
    rbB, rbA = gen_random_bit_both_fields((E,), ass_bool, ass)
    measure("B2A", lambda: pi_b2a(bsh, rbB, rbA, ass_bool, ass),
            lambda r: np.array_equal(plain.reconstruct(r), bits))

    bA, bbits = gen_beta(l, (E,), ass, ass_bool)
    measure("RS", lambda: pi_rs_2pc_correct(xs, bA, bbits, l, ass, ass_bool, off_b, counting_pi_mul,
                                            gen_random_bit_both_fields, pi_b2a),
            lambda r: np.array_equal(plain.reconstruct(r), x >> 1))

    bA, bbits = gen_beta(l, (E,), ass, ass_bool)
    measure("Cmp", lambda: pi_pricmp_2pc(xs, ys, bA, bbits, l, ass, ass_bool, off_b, counting_pi_mul),
            lambda r: np.array_equal(np.mod(sum(r), 2), (x >= y).astype(np.int64)))

    bA, bbits = gen_beta(l, (E,), ass, ass_bool)
    measure("Eq", lambda: pi_prieq_2pc(xs, ys, bA, bbits, l, ass, ass_bool, off_b, counting_pi_mul),
            lambda r: np.array_equal(np.mod(sum(r), 2), (x == y).astype(np.int64)))
    return out


# ----------------------------------------------------------------------------
# Reporting helpers
# ----------------------------------------------------------------------------
STAGE_ORDER = ["encrypt", "prediction", "error", "abs_value", "histogram", "embed",
               "receiver_combine", "extract_sequential"]


def fmt_psnr(v):
    return "inf" if v == float("inf") else "%.2f" % v


def fmt_time(s):
    if s < 120:
        return "%.2fs" % s
    if s < 7200:
        return "%.1f min" % (s / 60)
    if s < 172800:
        return "%.1f h" % (s / 3600)
    return "%.1f days" % (s / 86400)


def fmt_bits(b):
    if b >= 8e6:
        return "%.1f MB" % (b / 8e6)
    if b >= 8e3:
        return "%.1f KB" % (b / 8e3)
    return "%d b" % b


def table(headers, rows):
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    out = [line, "  ".join("-" * w for w in widths)]
    for r in rows:
        out.append("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    return out


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        o = float(o)
        return "inf" if o == float("inf") else o
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[16, 24, 32, 48],
                    help="image side lengths to test (sequential extraction is slow: keep small)")
    ap.add_argument("--images", type=int, default=2, help="max images taken from --image-dir")
    ap.add_argument("--image-dir", default="test_images")
    ap.add_argument("--parties", type=int, default=2, help="number of parties n (n=2 is the validated setup; n=3 passed a small smoke test)")
    ap.add_argument("--l", type=int, default=10)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()

    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    n, l = args.parties, args.l
    paths = find_images(args.image_dir, args.images)
    sources = [(os.path.basename(p), p) for p in paths] or [("synthetic", None)]
    if not paths:
        say("[note] no images found in '%s' -- using a synthetic smooth+noise image." % args.image_dir)

    all_runs = []
    for size in args.sizes:
        for name, path in sources:
            img = load_gray(path, size) if path else synthetic_image(size)
            print("\n>> %s @ %dx%d  (n=%d, l=%d)" % (name, size, size, n, l))
            r = run_pipeline(img, n, l, verbose=not args.quiet)
            r["image"], r["size"] = name, size
            all_runs.append(r)
            print("   done: total MPC time %s, lossless=%s, bits_correct=%s" % (
                fmt_time(sum(s["time_sec"] for s in r["stages"].values())),
                r["psnr4_mpc_extract_on_shares"] == float("inf"), r["bits_correct"]))

    bench = None
    if not args.skip_bench:
        print("\n>> protocol micro-benchmark ...")
        bench = bench_protocols(n, l)

    # ------------------------------------------------------------------ report
    say("=" * 100)
    say("MPC PIPELINE METRICS   (n=%d parties, ring Z_2^%d, SGAP predictor, PEHS)" % (n, l))
    say("=" * 100)

    say("\nA. CORRECTNESS  (paper Table II format; PSNR in dB)")
    say("   PSNR1 = plaintext PEHS marked | PSNR2 = marked image reconstructed from MPC-embedded shares")
    say("   PSNR3 = plaintext-side extraction on MI | PSNR4 = MPC extraction on shares, then reconstruct")
    rows = []
    for r in all_runs:
        rows.append([r["image"][:16], "%dx%d" % (r["size"], r["size"]), r["capacity_bits"],
                     "%.3f" % r["embedding_rate_bpp"], fmt_psnr(r["psnr1_plain_marked"]),
                     fmt_psnr(r["psnr2_mpc_marked"]), fmt_psnr(r["psnr3_plain_extract_on_MI"]),
                     fmt_psnr(r["psnr4_mpc_extract_on_shares"]),
                     "yes" if r["embed_bit_exact_vs_plaintext"] else "NO",
                     "yes" if r["bits_correct"] and r["extraction_paths_agree"] else "NO",
                     r["overflow_pixels_in_marked"]])
    for ln in table(["Image", "Size", "EC(bits)", "ER(bpp)", "PSNR1", "PSNR2", "PSNR3", "PSNR4",
                     "Embed==plain", "Bits OK", "Overflow px"], rows):
        say("   " + ln)
    ok = all(r["embed_bit_exact_vs_plaintext"] and r["bits_correct"] and r["extraction_paths_agree"]
             and r["ME_LE_match_plaintext"]
             and r["psnr3_plain_extract_on_MI"] == float("inf")
             and r["psnr4_mpc_extract_on_shares"] == float("inf") for r in all_runs)
    say("\n   Runs: %d | all bit-exact & lossless & both extraction paths agree: %s" % (
        len(all_runs), "YES" if ok else "NO -- inspect rows above"))
    say("   Data expansion: total n*l/8 = %.2fx, per hider l/8 = %.2fx" % (n * l / 8.0, l / 8.0))

    sizes = sorted(set(r["size"] for r in all_runs))

    def avg(size, getter):
        vals = [getter(r) for r in all_runs if r["size"] == size]
        return sum(vals) / len(vals)

    say("\nB. TIME PER STAGE  (seconds, mean over images; includes inline offline-randomness generation)")
    rows = []
    for st in STAGE_ORDER:
        rows.append([st] + ["%.3f" % avg(s, lambda r: r["stages"][st]["time_sec"]) for s in sizes])
    tot = [avg(s, lambda r: sum(v["time_sec"] for v in r["stages"].values())) for s in sizes]
    pl = [avg(s, lambda r: r["plain_embed_sec"] + r["plain_extract_sec"]) for s in sizes]
    rows.append(["TOTAL (MPC)"] + ["%.3f" % t for t in tot])
    rows.append(["plaintext embed+extract"] + ["%.4f" % t for t in pl])
    rows.append(["slowdown (x)"] + ["%.0f" % (t / p) if p > 0 else "-" for t, p in zip(tot, pl)])
    for ln in table(["Stage"] + ["%dx%d" % (s, s) for s in sizes], rows):
        say("   " + ln)

    say("\nC. COMMUNICATION PER STAGE  (rounds = sequential reveals; bits = all parties combined)")
    rows = []
    for st in STAGE_ORDER:
        if st in ("encrypt", "error", "receiver_combine"):
            continue
        cells = []
        for s in sizes:
            rd = avg(s, lambda r: r["stages"][st]["rounds"])
            bt = avg(s, lambda r: r["stages"][st]["comm_bits"])
            cells.append("%d rounds / %s" % (round(rd), fmt_bits(bt)))
        rows.append([st] + cells)
    for ln in table(["Stage"] + ["%dx%d" % (s, s) for s in sizes], rows):
        say("   " + ln)
    say("   (encrypt / error / receiver_combine need no protocol rounds: local ASS operations)")

    if bench:
        say("\nD. PER-PROTOCOL MICRO-BENCHMARK vs PAPER TABLE VI   (n=%d, l=%d, 2000 elements)" % (n, l))
        theory = paper_table6(n, l)
        rows = []
        for name in ["Mul", "RS", "B2A", "Cmp", "Eq"]:
            b = bench[name]
            tr, tb = theory[name]
            rows.append([name, "yes" if b["correct"] else "NO", b["rounds"], tr,
                         "%.0f" % b["comm_bits_per_elem_all_parties"], tb,
                         "%.0f" % b["time_us_per_elem"]])
        for ln in table(["Protocol", "Correct", "Rounds(ours)", "Rounds(paper)",
                         "Bits/elem(ours)", "Bits(paper)", "us/elem"], rows):
            say("   " + ln)
        say("   Rounds are directly comparable. Our bit counts are TOTAL across all parties and include the")
        say("   bit-decomposition helper reveals, so they are not one-to-one with the paper's formulas.")
        say("   Our Cmp/Eq use a ripple-carry / sequential-AND structure and our RS uses bit decomposition,")
        say("   so they need more rounds than the paper's carry-lookahead / N-fan-in (N-BTE) versions.")

    # ------------------------------------------------------------------ extrapolation
    big = max(sizes)
    valid = avg(big, lambda r: r["valid_pixels"])
    bins = avg(big, lambda r: r["hist_bins"])
    t_hist = avg(big, lambda r: r["stages"]["histogram"]["time_sec"])
    t_ext = avg(big, lambda r: r["stages"]["extract_sequential"]["time_sec"])
    t_emb = avg(big, lambda r: r["stages"]["embed"]["time_sec"])
    t_pred = avg(big, lambda r: r["stages"]["prediction"]["time_sec"])
    t_abs = avg(big, lambda r: r["stages"]["abs_value"]["time_sec"])
    full_valid = 511 * 510
    hist_per_bin_px = t_hist / (bins * valid)
    ext_per_px = t_ext / valid
    vec_per_px = (t_pred + t_abs + t_emb) / valid  # vectorised stages scale ~linearly with pixels
    proj = {
        "extract_sequential": ext_per_px * full_valid,
        "histogram_320_bins": hist_per_bin_px * 320 * full_valid,
        "histogram_measured_bins": hist_per_bin_px * bins * full_valid,
        "prediction_abs_embed": vec_per_px * full_valid,
    }
    say("\nE. EXTRAPOLATION TO A FULL 512x512 IMAGE   (linear, from the measured %dx%d runs)" % (big, big))
    say("   measured: extraction %.3f s/pixel | histogram %.2e s per (bin x pixel) | %d bins here" % (
        ext_per_px, hist_per_bin_px, bins))
    rows = [
        ["sequential extraction (261k px, one protocol chain per pixel)", fmt_time(proj["extract_sequential"])],
        ["histogram, paper's SGAP sample space (320 bins)", fmt_time(proj["histogram_320_bins"])],
        ["histogram, only as many bins as this image needs", fmt_time(proj["histogram_measured_bins"])],
        ["prediction + |e| + embed (vectorised)", fmt_time(proj["prediction_abs_embed"])],
    ]
    for ln in table(["Stage @ 512x512", "Projected"], rows):
        say("   " + ln)
    say("   Order-of-magnitude only: assumes per-pixel cost stays constant as the image grows.")
    say("   Note: the histogram's bin range here is taken from plaintext ground truth to bound the loop;")
    say("   a deployment would need a public constant (e.g. the paper's 320 for SGAP).")

    say("\nF. KNOWN GAPS AFFECTING THE NUMBERS ABOVE")
    say("   - Overflow/underflow map (paper eq. 34) is not implemented: 'Overflow px' counts marked pixels")
    say("     that leave [0,255]. Recovery is still lossless in the ring Z_2^%d, but an 8-bit stego file would wrap." % l)
    say("   - ME/LE are returned as values, not recorded in LSBs (paper eq. 30-33), so EC is not reduced by that overhead.")
    say("   - Embedding reveals which pixels sit at the peak bin (r1) to assign bits sequentially; extraction")
    say("     reveals r1/r1b per pixel. This is a documented relaxation of the paper's design.")
    say("   - n=2 is the validated configuration; n=3 passed a small smoke test; larger n is untested.")

    with open("mpc_metrics_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open("mpc_metrics_results.json", "w") as f:
        json.dump(jsonable({"runs": all_runs, "protocol_bench": bench, "projection_512": proj}), f, indent=2)
    print("\nSaved mpc_metrics_report.txt and mpc_metrics_results.json")


if __name__ == "__main__":
    main()