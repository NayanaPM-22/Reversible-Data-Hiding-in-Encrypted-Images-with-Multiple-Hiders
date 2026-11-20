"""
run_full_mpc.py -- ONE full-size run of the true MPC-protocol PEHS pipeline
(SGAP prediction -> |e| -> histogram -> histogram-shift embed -> sequential extract),
all on additive shares via Pi_Mul / Pi_RS / Pi_PriCmp / Pi_PriEq / Pi_B2A.

Built for a long unattended run (about 1.5 h at 512x512 on one machine):
  * live progress with elapsed time and ETA
  * checkpoints after every stage, and every --ckpt-every pixels during extraction
  * re-run the SAME command to resume after a crash / Ctrl+C / reboot
  * on Windows, asks the OS not to sleep while it runs
  * at the end: compares against plaintext PEHS (bit-exact check), writes results + images

Usage (from backend/, venv active):
    python run_full_mpc.py --image test_images/4.2.04.tiff           # 512x512, n=2
    python run_full_mpc.py --image test_images/lena.png --size 128   # quicker trial run
    python run_full_mpc.py --image <img> --fresh                     # discard old checkpoints

Scope: grayscale, n=2 parties, one machine simulating both parties (no network latency).
"""
import sys
import os
import time
import json
import argparse
import platform
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.core.ass import ASS
from app.core.pehs import pehs_embed_plaintext, pehs_extract_plaintext, psnr, sgap_predict
from app.protocols.offline import Offline
from app.protocols import pehs_mpc
from app.protocols.mul import pi_mul
from app.protocols.pricmp import gen_beta, pi_pricmp_2pc
from app.protocols.prieq import pi_prieq_2pc
from app.protocols.b2a import gen_random_bit_both_fields, pi_b2a


# ----------------------------------------------------------------------------- utilities
def log(msg=""):
    print(msg, flush=True)


def fmt_time(s):
    s = float(s)
    if s < 90:
        return "%.1fs" % s
    if s < 5400:
        return "%.1f min" % (s / 60)
    return "%.2f h" % (s / 3600)


def keep_awake():
    """Windows: ask the OS not to sleep while this process runs. No-op elsewhere."""
    if platform.system() == "Windows":
        try:
            import ctypes
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
            log("[power] requested: do not sleep while running")
        except Exception as e:  # pragma: no cover
            log("[power] could not set no-sleep flag (%s) -- disable sleep manually" % e)
    else:
        log("[power] not Windows: disable sleep manually for long runs")


def atomic_savez(path, **arrays):
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def pack(shares):
    return {"s%d" % i: s for i, s in enumerate(shares)}


def unpack(z, n):
    return [z["s%d" % i].copy() for i in range(n)]


def pack_named(prefix, shares):
    return {"%s%d" % (prefix, i): s for i, s in enumerate(shares)}


def unpack_named(z, prefix, n):
    return [z["%s%d" % (prefix, i)].copy() for i in range(n)]


class Progress:
    """Prints a line at most every `every_sec` seconds with elapsed time and ETA."""

    def __init__(self, label, total, every_sec=20, already_done=0, base_elapsed=0.0):
        self.label, self.total = label, total
        self.every = every_sec
        self.t0 = time.time()
        self.last = 0.0
        self.done0 = already_done
        self.base = base_elapsed

    def update(self, done, force=False):
        now = time.time()
        if not force and now - self.last < self.every:
            return
        self.last = now
        run = now - self.t0
        fresh = done - self.done0
        rate = fresh / run if run > 0 else 0
        eta = (self.total - done) / rate if rate > 0 else float("inf")
        log("   [%s] %d/%d (%.1f%%)  elapsed %s  ETA %s" % (
            self.label, done, self.total, 100.0 * done / self.total,
            fmt_time(self.base + run), fmt_time(eta) if eta != float("inf") else "?"))


# ----------------------------------------------------------------------------- pipeline pieces
def histogram_with_progress(abs_e_shares, valid_mask, l, ass, ass_bool, off_bool, bins, hist_path):
    """Same logic as pehs_mpc.histogram_on_shares, with progress + per-bin resume."""
    n = ass.n
    flat = [s[valid_mask] for s in abs_e_shares]
    shape = flat[0].shape
    # Counts can reach the number of valid pixels, far beyond the 10-bit pixel ring (wraps above 511).
    # The paper shares the histogram h over a larger ring Z_{M*N*2}; we do the same: B2A into Z_2^l_cnt.
    l_cnt = max(l, int(np.ceil(np.log2(shape[0] + 1))) + 2)
    ass_cnt = ASS(l=l_cnt, n=n)
    log("   histogram counts are shared over Z_2^%d (pixels use Z_2^%d)" % (l_cnt, l))
    hist = np.zeros(bins, dtype=np.int64)
    start = 0
    elapsed0 = 0.0
    if os.path.exists(hist_path):
        z = np.load(hist_path)
        hist, start, elapsed0 = z["hist"].copy(), int(z["next_bin"]), float(z["elapsed"])
        log("   resuming histogram at bin %d/%d" % (start, bins))
    prog = Progress("histogram bin", bins, already_done=start, base_elapsed=elapsed0)
    t_run = time.time()
    for b in range(start, bins):
        bin_shares = [np.full(shape, b, dtype=np.int64) if i == 0 else np.zeros(shape, dtype=np.int64)
                      for i in range(n)]
        beta_A, beta_bits = gen_beta(l, shape, ass, ass_bool)
        eq_bool = pi_prieq_2pc(flat, bin_shares, beta_A, beta_bits, l, ass, ass_bool, off_bool, pi_mul)
        rb_B, rb_A = gen_random_bit_both_fields(shape, ass_bool, ass_cnt)
        eq_arith = pi_b2a(eq_bool, rb_B, rb_A, ass_bool, ass_cnt)
        hist[b] = int(ass_cnt.reconstruct([np.array([np.sum(s)]) for s in eq_arith])[0])  # only the count is revealed
        prog.update(b + 1)
        if (b + 1) % 10 == 0 or b + 1 == bins:
            atomic_savez(hist_path, hist=hist, next_bin=b + 1, elapsed=elapsed0 + time.time() - t_run)
    ME = int(np.argmax(hist))
    LE = None
    for cand in range(ME + 1, bins + 1):
        if cand >= bins or hist[cand] == 0:
            LE = cand
            break
    return hist, ME, LE, elapsed0 + time.time() - t_run


def extract_pixel(i, j, recovered, ME, LE, l, ass, ass_bool, off_bool, off_arith):
    """One pixel of sequential extraction. Mutates `recovered`; returns the extracted bit or None."""
    n = ass.n

    def const1(v):
        return [np.array([v], dtype=np.int64) if p == 0 else np.array([0], dtype=np.int64) for p in range(n)]

    N = [recovered[p][i - 1, j:j + 1] for p in range(n)]
    W = [recovered[p][i, j - 1:j] for p in range(n)]
    NE = [recovered[p][i - 1, j + 1:j + 2] for p in range(n)]
    NW = [recovered[p][i - 1, j - 1:j] for p in range(n)]

    sum_NW = ass.add_shares(N, W)
    diff = ass.add_shares(NE, ass.scal_mul(NW, -1))
    rs_sum = pehs_mpc.rs_on_shares(sum_NW, l, ass, ass_bool, off_bool)
    rs_d1 = pehs_mpc.rs_on_shares(diff, l, ass, ass_bool, off_bool)
    rs_d2 = pehs_mpc.rs_on_shares(rs_d1, l, ass, ass_bool, off_bool)
    pred = ass.add_shares(rs_sum, rs_d2)

    x_prime = [recovered[p][i, j:j + 1] for p in range(n)]
    e = ass.add_shares(x_prime, ass.scal_mul(pred, -1))
    abs_e, sign = pehs_mpc.abs_value_on_shares(e, l, ass, ass_bool, off_bool, off_arith)

    bA, bb = gen_beta(l, (1,), ass, ass_bool)
    r1_bool = pi_prieq_2pc(abs_e, const1(ME), bA, bb, l, ass, ass_bool, off_bool, pi_mul)
    rbB, rbA = gen_random_bit_both_fields((1,), ass_bool, ass)
    r1 = int(ass.reconstruct(pi_b2a(r1_bool, rbB, rbA, ass_bool, ass))[0])

    bA, bb = gen_beta(l, (1,), ass, ass_bool)
    r1b_bool = pi_prieq_2pc(abs_e, const1(ME + 1), bA, bb, l, ass, ass_bool, off_bool, pi_mul)
    rbB, rbA = gen_random_bit_both_fields((1,), ass_bool, ass)
    r1b = int(ass.reconstruct(pi_b2a(r1b_bool, rbB, rbA, ass_bool, ass))[0])

    bit, undo = None, 0
    if r1 == 1:
        bit = 0
    elif r1b == 1:
        bit, undo = 1, 1
    else:
        bA, bb = gen_beta(l, (1,), ass, ass_bool)
        c1 = pi_pricmp_2pc(abs_e, const1(ME + 2), bA, bb, l, ass, ass_bool, off_bool, pi_mul)
        bA, bb = gen_beta(l, (1,), ass, ass_bool)
        c2 = pi_pricmp_2pc(abs_e, const1(LE + 1), bA, bb, l, ass, ass_bool, off_bool, pi_mul)
        not_c2 = [c.copy() for c in c2]
        not_c2[0] = (1 - not_c2[0]) % 2
        r2_bool = pi_mul([c1, not_c2], off_bool.gen_nbte(N=2, shape=(1,)), ass_bool)
        rbB, rbA = gen_random_bit_both_fields((1,), ass_bool, ass)
        undo = int(ass.reconstruct(pi_b2a(r2_bool, rbB, rbA, ass_bool, ass))[0])

    if undo != 0:
        amount = pi_mul([sign, const1(undo)], off_arith.gen_nbte(N=2, shape=(1,)), ass)
        new_val = ass.add_shares(x_prime, ass.scal_mul(amount, -1))
        for p in range(n):
            recovered[p][i, j] = new_val[p][0]
    return bit


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--n", type=int, default=2, help="parties (n=2 is the validated configuration)")
    ap.add_argument("--l", type=int, default=10)
    ap.add_argument("--bins", type=int, default=320,
                    help="public histogram bin count; 320 is the paper's SGAP sample space [0,319]")
    ap.add_argument("--bits", type=int, default=0, help="secret bits to embed (0 = more than capacity, i.e. maximum)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--ckpt-every", type=int, default=2000, help="extraction checkpoint interval (pixels)")
    ap.add_argument("--fresh", action="store_true", help="ignore existing checkpoints and start over")
    ap.add_argument("--continue-on-mismatch", action="store_true",
                    help="keep going even if ME/LE or the embedded image differ from plaintext PEHS")
    ap.add_argument("--debug-stop-after", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args()

    n, l, size = args.n, args.l, args.size
    name = os.path.splitext(os.path.basename(args.image))[0]
    workdir = args.workdir or "mpc_full_run_%s_%d" % (name, size)
    os.makedirs(workdir, exist_ok=True)
    P = lambda f: os.path.join(workdir, f)

    config = {"image": os.path.basename(args.image), "size": size, "n": n, "l": l,
              "bins": args.bins, "bits": args.bits, "seed": args.seed}
    meta_path = P("meta.json")
    if args.fresh:
        for f in os.listdir(workdir):
            os.remove(P(f))
    meta = {"config": config, "timings": {}}
    if os.path.exists(meta_path):
        old = json.load(open(meta_path))
        if old.get("config") != config:
            sys.exit("Checkpoints in '%s' belong to a different configuration:\n  saved: %s\n  now:   %s\n"
                     "Use --fresh to discard them, or --workdir for a separate run." % (workdir, old.get("config"), config))
        meta = old
        log("[resume] found checkpoints in %s (stages done: %s)" % (workdir, ", ".join(meta["timings"]) or "none"))
    T = meta["timings"]

    def save_meta():
        atomic_json(meta_path, meta)

    keep_awake()
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    ass = ASS(l=l, n=n)
    ass_bool = ASS(l=1, n=n)
    off_arith = Offline(ass)
    off_bool = Offline(ass_bool)

    img = np.array(Image.open(args.image).convert("L").resize((size, size)), dtype=np.int64)
    h, w = img.shape
    log("=" * 78)
    log("FULL MPC RUN  image=%s  %dx%d  n=%d  l=%d  bins=%d" % (name, h, w, n, l, args.bins))
    log("=" * 78)
    run_t0 = time.time()

    # ---- plaintext reference + baseline timing (cheap, always recomputed) ----
    pred_p, mask_p = sgap_predict(img)
    max_e = int(np.abs(img - pred_p)[mask_p].max())
    if max_e >= args.bins:
        sys.exit("max |e| = %d needs at least %d bins; raise --bins" % (max_e, max_e + 1))
    nbits = args.bits if args.bits > 0 else h * w
    data_bits = rng.integers(0, 2, size=nbits).tolist()
    t0 = time.time()
    marked_p, ME_p, LE_p, n_emb_p = pehs_embed_plaintext(img, data_bits)
    t_plain_embed = time.time() - t0
    t0 = time.time()
    rec_p, bits_p = pehs_extract_plaintext(marked_p, ME_p, LE_p)
    t_plain_extract = time.time() - t0
    log("plaintext PEHS reference: ME=%d LE=%d capacity=%d bits, PSNR(marked)=%.2f dB, lossless=%s" % (
        ME_p, LE_p, n_emb_p, psnr(img, marked_p), np.array_equal(rec_p, img)))

    # ---- stage 1: encrypt, predict, error, |e| ----
    if os.path.exists(P("stage1.npz")):
        z = np.load(P("stage1.npz"))
        img_shares, abs_e, sign = unpack_named(z, "x", n), unpack_named(z, "a", n), unpack_named(z, "g", n)
        mask = z["mask"].astype(bool)
        log("[stage 1] loaded from checkpoint")
    else:
        log("[stage 1] encrypt, SGAP prediction, error, |e| on shares ...")
        t0 = time.time()
        img_shares = ass.share(img)
        T["encrypt"] = time.time() - t0
        t0 = time.time()
        pred_shares, mask = pehs_mpc.sgap_predict_on_shares(img_shares, l, ass, ass_bool, off_bool)
        T["prediction"] = time.time() - t0
        e_shares = pehs_mpc.compute_prediction_error_on_shares(img_shares, pred_shares, ass)
        t0 = time.time()
        abs_e, sign = pehs_mpc.abs_value_on_shares(e_shares, l, ass, ass_bool, off_bool, off_arith)
        T["abs_value"] = time.time() - t0
        atomic_savez(P("stage1.npz"), mask=mask, **pack_named("x", img_shares),
                     **pack_named("a", abs_e), **pack_named("g", sign))
        save_meta()
        log("   done: prediction %s, |e| %s" % (fmt_time(T["prediction"]), fmt_time(T["abs_value"])))

    # ---- stage 2: histogram -> ME, LE ----
    if "histogram" in T and os.path.exists(P("hist.npz")):
        z = np.load(P("hist.npz"))
        hist = z["hist"]
        ME = int(np.argmax(hist))
        LE = next((c for c in range(ME + 1, args.bins + 1) if c >= args.bins or hist[c] == 0), args.bins)
        log("[stage 2] histogram loaded from checkpoint: ME=%d LE=%d" % (ME, LE))
    else:
        log("[stage 2] histogram over %d public bins (one Pi_PriEq + Pi_B2A pass per bin) ..." % args.bins)
        hist, ME, LE, t_hist = histogram_with_progress(abs_e, mask, l, ass, ass_bool, off_bool, args.bins, P("hist.npz"))
        T["histogram"] = t_hist
        save_meta()
        log("   done in %s: ME=%d LE=%d" % (fmt_time(t_hist), ME, LE))
    if (ME, LE) != (ME_p, LE_p):
        log("   !! MPC histogram gives ME=%d LE=%d but plaintext PEHS gives ME=%d LE=%d" % (ME, LE, ME_p, LE_p))
        if not args.continue_on_mismatch:
            sys.exit("Aborting before the long extraction stage (use --fresh after fixing, or --continue-on-mismatch).")

    # ---- stage 3: embed ----
    if os.path.exists(P("stage3.npz")):
        z = np.load(P("stage3.npz"))
        marked_shares = unpack_named(z, "m", n)
        n_emb = int(z["n_emb"])
        data_bits = z["data_bits"].tolist()
        log("[stage 3] embedded shares loaded from checkpoint (%d bits)" % n_emb)
    else:
        log("[stage 3] histogram-shift embedding on shares ...")
        t0 = time.time()
        marked_shares, n_emb = pehs_mpc.histogram_shift_embed_on_shares(
            img_shares, abs_e, sign, ME, LE, data_bits, mask, l, ass, ass_bool, off_bool, off_arith)
        T["embed"] = time.time() - t0
        atomic_savez(P("stage3.npz"), n_emb=n_emb, data_bits=np.array(data_bits, dtype=np.int8),
                     **pack_named("m", marked_shares))
        save_meta()
        log("   done in %s: %d bits embedded" % (fmt_time(T["embed"]), n_emb))

    marked_mpc = ass.reconstruct(marked_shares)   # receiver-side combine, used only for verification
    embed_ok = np.array_equal(marked_mpc, marked_p)
    log("   check: MPC-embedded image == plaintext-embedded image: %s" % embed_ok)
    if not embed_ok and not args.continue_on_mismatch:
        sys.exit("Embedded image differs from plaintext PEHS -- aborting before the long extraction stage.")

    # ---- stage 4: sequential extraction ----
    rows, cols = np.where(mask)
    total = len(rows)
    if os.path.exists(P("extract.npz")):
        z = np.load(P("extract.npz"))
        recovered = unpack_named(z, "r", n)
        extracted = z["bits"].tolist()
        start = int(z["next_idx"])
        ext_elapsed0 = float(z["elapsed"])
        log("[stage 4] resuming sequential extraction at pixel %d/%d" % (start, total))
    else:
        recovered = [s.copy() for s in marked_shares]
        extracted, start, ext_elapsed0 = [], 0, 0.0
        log("[stage 4] sequential extraction, %d pixels, one protocol chain per pixel (this is the long part)" % total)

    prog = Progress("extract pixel", total, already_done=start, base_elapsed=ext_elapsed0)
    t_run = time.time()
    for idx in range(start, total):
        bit = extract_pixel(int(rows[idx]), int(cols[idx]), recovered, ME, LE, l, ass, ass_bool, off_bool, off_arith)
        if bit is not None:
            extracted.append(bit)
        prog.update(idx + 1)
        if (idx + 1) % args.ckpt_every == 0:
            atomic_savez(P("extract.npz"), bits=np.array(extracted, dtype=np.int8), next_idx=idx + 1,
                         elapsed=ext_elapsed0 + time.time() - t_run, **pack_named("r", recovered))
        if args.debug_stop_after and idx + 1 == args.debug_stop_after:
            atomic_savez(P("extract.npz"), bits=np.array(extracted, dtype=np.int8), next_idx=idx + 1,
                         elapsed=ext_elapsed0 + time.time() - t_run, **pack_named("r", recovered))
            sys.exit("[debug] stopped after %d pixels (checkpoint written)" % (idx + 1))
    prog.update(total, force=True)
    T["extract_sequential"] = ext_elapsed0 + time.time() - t_run
    save_meta()

    # ---- verification + report ----
    rec_mpc = ass.reconstruct(recovered)
    rec3, bits3 = pehs_extract_plaintext(marked_mpc, ME, LE)
    ov = int(np.sum((marked_mpc < 0) | (marked_mpc > 255)))
    res = {
        "image": name, "size": [h, w], "n": n, "l": l, "bins": args.bins,
        "valid_pixels": int(total), "ME": ME, "LE": LE,
        "ME_LE_match_plaintext": bool(ME == ME_p and LE == LE_p),
        "capacity_bits": int(n_emb), "embedding_rate_bpp": n_emb / float(h * w),
        "embed_bit_exact_vs_plaintext": bool(np.array_equal(marked_mpc, marked_p)),
        "psnr1_plain_marked": psnr(img, marked_p),
        "psnr2_mpc_marked": psnr(img, marked_mpc),
        "psnr3_plain_extract_on_MI": psnr(img, rec3),
        "psnr4_mpc_extract_on_shares": psnr(img, rec_mpc),
        "bits_correct": bool(extracted[:n_emb] == data_bits[:n_emb]),
        "extraction_paths_agree": bool(extracted == bits3),
        "overflow_pixels_in_marked": ov,
        "timings_sec": {k: float(v) for k, v in T.items()},
        "total_mpc_sec": float(sum(T.values())),
        "plain_embed_sec": t_plain_embed, "plain_extract_sec": t_plain_extract,
        "this_session_wall_sec": time.time() - run_t0,
        "note": "single process simulating both parties; no network latency; grayscale; n=%d" % n,
    }

    def jf(v):
        return "inf" if v == float("inf") else round(v, 4)
    out = {k: (jf(v) if isinstance(v, float) else v) for k, v in res.items()}
    atomic_json(P("full_run_results.json"), out)
    Image.fromarray(np.clip(marked_mpc, 0, 255).astype(np.uint8)).save(P("marked_image.png"))
    Image.fromarray(np.clip(rec_mpc, 0, 255).astype(np.uint8)).save(P("recovered_image.png"))
    Image.fromarray(img.astype(np.uint8)).save(P("original_image.png"))

    ok = (res["embed_bit_exact_vs_plaintext"] and res["bits_correct"] and res["extraction_paths_agree"]
          and res["ME_LE_match_plaintext"] and res["psnr4_mpc_extract_on_shares"] == float("inf"))
    log("")
    log("=" * 78)
    log("RESULT  %s %dx%d  n=%d" % (name, h, w, n))
    log("=" * 78)
    log("  capacity            : %d bits (%.3f bpp)" % (n_emb, res["embedding_rate_bpp"]))
    log("  PSNR1 / PSNR2       : %s / %s dB   (plaintext vs MPC-embedded marked image)" % (jf(res["psnr1_plain_marked"]), jf(res["psnr2_mpc_marked"])))
    log("  PSNR3 / PSNR4       : %s / %s dB   (plaintext-side vs MPC extraction)" % (jf(res["psnr3_plain_extract_on_MI"]), jf(res["psnr4_mpc_extract_on_shares"])))
    log("  bit-exact vs plain  : embed=%s  bits=%s  both extraction paths agree=%s" % (
        res["embed_bit_exact_vs_plaintext"], res["bits_correct"], res["extraction_paths_agree"]))
    log("  overflow pixels     : %d (paper eq. 34 overflow map not implemented)" % ov)
    for k, v in T.items():
        log("  time %-18s: %s" % (k, fmt_time(v)))
    log("  TOTAL MPC time      : %s   (plaintext embed+extract: %s)" % (
        fmt_time(res["total_mpc_sec"]), fmt_time(t_plain_embed + t_plain_extract)))
    log("  verdict             : %s" % ("ALL CHECKS PASS" if ok else "MISMATCH -- inspect the numbers above"))
    log("  saved to            : %s/ (full_run_results.json, original/marked/recovered PNGs)" % workdir)


if __name__ == "__main__":
    main()