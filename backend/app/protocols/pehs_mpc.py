"""
MPC-protocol version of PEHS -- Hiders operate directly on shares,
never reconstructing the plaintext image during embedding.
Built incrementally, each piece tested against the plaintext
equivalent in app/core/pehs.py before composing further.
"""
import numpy as np
from app.protocols.pricmp import gen_beta, pi_pricmp_2pc
from app.protocols.rshift import pi_rs_2pc_correct
from app.protocols.b2a import gen_random_bit_both_fields, pi_b2a
from app.protocols.mul import pi_mul


def rs_on_shares(x_shares, l, ass, ass_bool, offline_bool):
    """Convenience wrapper: generates fresh beta randomness and calls Pi_RS."""
    shape = x_shares[0].shape
    beta_A, beta_bits = gen_beta(l, shape, ass, ass_bool)
    return pi_rs_2pc_correct(
        x_shares, beta_A, beta_bits, l, ass, ass_bool, offline_bool,
        pi_mul, gen_random_bit_both_fields, pi_b2a
    )


def sgap_predict_on_shares(img_shares, l, ass, ass_bool, offline_bool):
    """
    Protocol version of sgap_predict (eq. 18-19):
    pred = RS(N + W) + RS(RS(NE - NW))
    Only uses Pi_RS (already validated) plus local ASS addition/subtraction
    (free under additive homomorphism -- no protocol call needed).

    img_shares: list of n share arrays, each shape (H, W)
    Returns: (full_pred_shares, valid_mask)
    """
    n = ass.n
    h, w = img_shares[0].shape

    def slice_shares(shares, r0, r1, c0, c1):
        return [s[r0:r1, c0:c1] for s in shares]

    N_shares  = slice_shares(img_shares, 0, h-1, 1, w-1)
    W_shares  = slice_shares(img_shares, 1, h,   0, w-2)
    NE_shares = slice_shares(img_shares, 0, h-1, 2, w)
    NW_shares = slice_shares(img_shares, 0, h-1, 0, w-2)

    # Local operations -- free, no communication round needed
    sum_NW = ass.add_shares(N_shares, W_shares)
    diff_NENW = ass.add_shares(NE_shares, ass.scal_mul(NW_shares, -1))

    # RS(N+W) -- 1 round via Pi_RS
    rs_sum = rs_on_shares(sum_NW, l, ass, ass_bool, offline_bool)

    # RS(RS(NE-NW)) -- 2 rounds via Pi_RS (matches eq. 18's >>2 / divide-by-4)
    rs_diff_once = rs_on_shares(diff_NENW, l, ass, ass_bool, offline_bool)
    rs_diff_twice = rs_on_shares(rs_diff_once, l, ass, ass_bool, offline_bool)

    pred_shares = ass.add_shares(rs_sum, rs_diff_twice)

    valid_mask = np.zeros((h, w), dtype=bool)
    valid_mask[1:h, 1:w-1] = True

    # Pad prediction back to full (h,w) shape
    full_pred_shares = []
    for s in pred_shares:
        full = np.zeros((h, w), dtype=np.int64)
        full[1:h, 1:w-1] = s
        full_pred_shares.append(full)

    return full_pred_shares, valid_mask


def compute_prediction_error_on_shares(img_shares, pred_shares, ass):
    """e = x - pred. Free local operation, no protocol call needed."""
    return ass.add_shares(img_shares, ass.scal_mul(pred_shares, -1))


def abs_value_on_shares(e_shares, l, ass, ass_bool, offline_bool, offline_arith):
    """
    |e| via eq. 24: |e| = (2 * B2A(PriCmp(e, 0)) - 1) * e
    PriCmp(e,0) = 1 if e>=0 else 0 -- this becomes the sign multiplier.
    Returns (abs_e_shares, sign_shares) -- sign is needed again later for shifting.
    """
    shape = e_shares[0].shape
    zero_shares = [np.zeros(shape, dtype=np.int64) for _ in range(ass.n)]

    beta_A, beta_bits = gen_beta(l, shape, ass, ass_bool)
    ge_zero_bool = pi_pricmp_2pc(e_shares, zero_shares, beta_A, beta_bits, l,
                                   ass, ass_bool, offline_bool, pi_mul)

    rb_B, rb_A = gen_random_bit_both_fields(shape, ass_bool, ass)
    ge_zero_arith = pi_b2a(ge_zero_bool, rb_B, rb_A, ass_bool, ass)

    # sign = 2*ge_zero - 1  (i.e. +1 if e>=0, -1 if e<0)
    sign_shares = ass.add_const(ass.scal_mul(ge_zero_arith, 2), -1)

    # |e| = sign * e -- genuine multiplication of two SECRET values, needs Pi_Mul
    # with ARITHMETIC-field Beaver triples (not boolean), since both operands
    # live in Z_{2^l}.
    nbte_arith = offline_arith.gen_nbte(N=2, shape=shape)
    abs_e_shares = pi_mul([sign_shares, e_shares], nbte_arith, ass)

    return abs_e_shares, sign_shares

def histogram_on_shares(abs_e_shares, valid_mask, l, ass, ass_bool, offline_bool, offline_arith, max_e_estimate):
    """
    Protocol version of Alg. 6 (Pi_HIST): for each candidate bin value,
    count how many valid pixels have that |e|, without revealing which
    pixel matched which bin. Only the final counts (the histogram) are public.

    abs_e_shares: full (h,w) shares of |e| (only entries under valid_mask matter)
    max_e_estimate: search bins from 0 to this value (caller picks a safe upper bound)

    Returns: hist (plain numpy array of counts, public), ME (int), LE (int)
    """
    from app.protocols.prieq import pi_prieq_2pc

    n = ass.n
    valid_flat_shares = [s[valid_mask] for s in abs_e_shares]
    shape = valid_flat_shares[0].shape

    hist = np.zeros(max_e_estimate + 1, dtype=np.int64)

    for bin_val in range(max_e_estimate + 1):
        bin_shares = [np.full(shape, bin_val, dtype=np.int64) if i == 0
                      else np.zeros(shape, dtype=np.int64) for i in range(n)]

        beta_A, beta_bits = gen_beta(l, shape, ass, ass_bool)
        eq_bool = pi_prieq_2pc(valid_flat_shares, bin_shares, beta_A, beta_bits, l,
                                 ass, ass_bool, offline_bool, pi_mul)

        rb_B, rb_A = gen_random_bit_both_fields(shape, ass_bool, ass)
        eq_arith = pi_b2a(eq_bool, rb_B, rb_A, ass_bool, ass)

        # Sum of eq_arith across all pixels = count of pixels matching this bin.
        # Summing shares locally (free), THEN reconstructing = revealing only
        # the aggregate count, never any individual pixel's match.
        count_shares = [np.array([np.sum(s)]) for s in eq_arith]
        count = int(ass.reconstruct(count_shares)[0])
        hist[bin_val] = count

    ME = int(np.argmax(hist))
    LE = None
    for candidate in range(ME + 1, max_e_estimate + 2):
        if candidate > max_e_estimate or hist[candidate] == 0:
            LE = candidate
            break
    if LE is None:
        LE = max_e_estimate + 1

    return hist, ME, LE

def histogram_shift_embed_on_shares(img_shares, abs_e_shares, sign_shares, ME, LE,
                                      data_bits, valid_mask, l, ass, ass_bool,
                                      offline_bool, offline_arith):
    """
    Protocol version of eq. 26-29 (histogram shifting).

    SIMPLIFICATION (documented): r1 (whether |e|==ME per pixel) is revealed to
    the Hiders so they can sequentially assign data bits in raster order, exactly
    matching the plaintext algorithm's bit ordering. This discloses WHICH pixels
    sit at the peak point, not their true pixel/error VALUES -- a modest,
    explicit relaxation beyond the paper's stated design, needed for practical
    sequential bitstream assignment without additional oblivious-selection
    machinery beyond the 5 core protocols.
    """
    from app.protocols.prieq import pi_prieq_2pc

    n = ass.n
    valid_flat_x = [s[valid_mask] for s in img_shares]
    valid_flat_e = [s[valid_mask] for s in abs_e_shares]
    valid_flat_sign = [s[valid_mask] for s in sign_shares]
    shape = valid_flat_e[0].shape

    def const_shares(value):
        return [np.full(shape, value, dtype=np.int64) if i == 0
                else np.zeros(shape, dtype=np.int64) for i in range(n)]

    # r1 = PriEq(|e|, ME)
    beta_A1, beta_bits1 = gen_beta(l, shape, ass, ass_bool)
    r1_bool = pi_prieq_2pc(valid_flat_e, const_shares(ME), beta_A1, beta_bits1, l,
                             ass, ass_bool, offline_bool, pi_mul)
    rb_B1, rb_A1 = gen_random_bit_both_fields(shape, ass_bool, ass)
    r1_arith = pi_b2a(r1_bool, rb_B1, rb_A1, ass_bool, ass)
    r1_revealed = ass.reconstruct(r1_arith)   # <-- the documented simplification

    # r2 = PriCmp(|e|, ME+1) AND NOT PriCmp(|e|, LE)  -->  ME < |e| < LE
    beta_A2, beta_bits2 = gen_beta(l, shape, ass, ass_bool)
    cmp1_bool = pi_pricmp_2pc(valid_flat_e, const_shares(ME + 1), beta_A2, beta_bits2,
                                l, ass, ass_bool, offline_bool, pi_mul)
    beta_A3, beta_bits3 = gen_beta(l, shape, ass, ass_bool)
    cmp2_bool = pi_pricmp_2pc(valid_flat_e, const_shares(LE), beta_A3, beta_bits3,
                                l, ass, ass_bool, offline_bool, pi_mul)
    not_cmp2_bool = [c.copy() for c in cmp2_bool]
    not_cmp2_bool[0] = (1 - not_cmp2_bool[0]) % 2
    nbte_bool = offline_bool.gen_nbte(N=2, shape=shape)
    r2_bool = pi_mul([cmp1_bool, not_cmp2_bool], nbte_bool, ass_bool)
    rb_B2, rb_A2 = gen_random_bit_both_fields(shape, ass_bool, ass)
    r2_arith = pi_b2a(r2_bool, rb_B2, rb_A2, ass_bool, ass)

    # Sequential bit assignment, in raster order, wherever r1_revealed==1 --
    # exactly mirrors the plaintext algorithm's bit_idx/data_bits consumption.
    b_array = np.zeros(shape, dtype=np.int64)
    bit_idx = 0
    for k in range(shape[0]):
        if r1_revealed[k] == 1 and bit_idx < len(data_bits):
            b_array[k] = data_bits[bit_idx]
            bit_idx += 1
    bits_embedded = bit_idx

    # x' = x + b*sign*r1 + sign*r2
    nbte1 = offline_arith.gen_nbte(N=2, shape=shape)
    sign_r1 = pi_mul([valid_flat_sign, r1_arith], nbte1, ass)
    b_sign_r1 = [ass._wrap_signed(b_array * s) for s in sign_r1]   # b is public -> free local scalar mult

    nbte2 = offline_arith.gen_nbte(N=2, shape=shape)
    sign_r2 = pi_mul([valid_flat_sign, r2_arith], nbte2, ass)

    delta = ass.add_shares(b_sign_r1, sign_r2)
    marked_valid = ass.add_shares(valid_flat_x, delta)

    full_marked_shares = []
    for i in range(n):
        full = img_shares[i].copy()
        full[valid_mask] = marked_valid[i]
        full_marked_shares.append(full)

    return full_marked_shares, bits_embedded

def histogram_extract_on_shares(marked_shares, ME, LE, valid_mask, l, ass, ass_bool,
                                  offline_bool, offline_arith):
    """
    Protocol version of eq. 36-38 (extraction + recovery).
    Same documented simplification as embed: r1 (|e'|==ME) is revealed per
    pixel to allow sequential bit extraction in raster order.
    """
    from app.protocols.prieq import pi_prieq_2pc

    n = ass.n

    # Recompute prediction + error on the MARKED shares (context for extraction
    # must come from the marked image, matching the plaintext algorithm exactly)
    pred_shares, _ = sgap_predict_on_shares(marked_shares, l, ass, ass_bool, offline_bool)
    e_shares = compute_prediction_error_on_shares(marked_shares, pred_shares, ass)
    abs_e_shares, sign_shares = abs_value_on_shares(e_shares, l, ass, ass_bool, offline_bool, offline_arith)

    valid_flat_x = [s[valid_mask] for s in marked_shares]
    valid_flat_e = [s[valid_mask] for s in abs_e_shares]
    valid_flat_sign = [s[valid_mask] for s in sign_shares]
    shape = valid_flat_e[0].shape

    def const_shares(value):
        return [np.full(shape, value, dtype=np.int64) if i == 0
                else np.zeros(shape, dtype=np.int64) for i in range(n)]

    # r1 = PriEq(|e'|, ME)  -> bit=0 was embedded here (or nothing, if capacity exceeded)
    beta_A1, beta_bits1 = gen_beta(l, shape, ass, ass_bool)
    r1_bool = pi_prieq_2pc(valid_flat_e, const_shares(ME), beta_A1, beta_bits1, l,
                             ass, ass_bool, offline_bool, pi_mul)
    rb_B1, rb_A1 = gen_random_bit_both_fields(shape, ass_bool, ass)
    r1_arith = pi_b2a(r1_bool, rb_B1, rb_A1, ass_bool, ass)
    r1_revealed = ass.reconstruct(r1_arith)

    # r1b = PriEq(|e'|, ME+1)  -> bit=1 was embedded here
    beta_A1b, beta_bits1b = gen_beta(l, shape, ass, ass_bool)
    r1b_bool = pi_prieq_2pc(valid_flat_e, const_shares(ME + 1), beta_A1b, beta_bits1b, l,
                              ass, ass_bool, offline_bool, pi_mul)
    rb_B1b, rb_A1b = gen_random_bit_both_fields(shape, ass_bool, ass)
    r1b_arith = pi_b2a(r1b_bool, rb_B1b, rb_A1b, ass_bool, ass)
    r1b_revealed = ass.reconstruct(r1b_arith)

    # r2 = ME+1 < |e'| <= LE  -> generic shift-by-1 zone (needs to be undone,
    # no bit extracted here)
    beta_A2, beta_bits2 = gen_beta(l, shape, ass, ass_bool)
    cmp1_bool = pi_pricmp_2pc(valid_flat_e, const_shares(ME + 2), beta_A2, beta_bits2,
                                l, ass, ass_bool, offline_bool, pi_mul)
    beta_A3, beta_bits3 = gen_beta(l, shape, ass, ass_bool)
    cmp2_bool = pi_pricmp_2pc(valid_flat_e, const_shares(LE + 1), beta_A3, beta_bits3,
                                l, ass, ass_bool, offline_bool, pi_mul)
    not_cmp2_bool = [c.copy() for c in cmp2_bool]
    not_cmp2_bool[0] = (1 - not_cmp2_bool[0]) % 2
    nbte_bool = offline_bool.gen_nbte(N=2, shape=shape)
    r2_bool = pi_mul([cmp1_bool, not_cmp2_bool], nbte_bool, ass_bool)
    rb_B2, rb_A2 = gen_random_bit_both_fields(shape, ass_bool, ass)
    r2_arith = pi_b2a(r2_bool, rb_B2, rb_A2, ass_bool, ass)

    # Extract bits sequentially, raster order, wherever r1 or r1b fired
    extracted_bits = []
    b_undo_array = np.zeros(shape, dtype=np.int64)   # -1 * (bit that was added), to undo
    for k in range(shape[0]):
        if r1_revealed[k] == 1:
            extracted_bits.append(0)
            # nothing to undo -- bit=0 meant no shift happened
        elif r1b_revealed[k] == 1:
            extracted_bits.append(1)
            b_undo_array[k] = 1   # undo the +sign*1 shift

    # x = x' - sign*1*(r1b) - sign*1*r2   [both undo cases are "shift back by sign"]
    undo_r1b = [ass._wrap_signed(b_undo_array * s) for s in
                pi_mul([valid_flat_sign, r1b_arith], offline_arith.gen_nbte(N=2, shape=shape), ass)]
    undo_r2 = pi_mul([valid_flat_sign, r2_arith], offline_arith.gen_nbte(N=2, shape=shape), ass)

    total_undo = ass.add_shares(undo_r1b, undo_r2)
    recovered_valid = ass.add_shares(valid_flat_x, ass.scal_mul(total_undo, -1))

    n_parties = ass.n
    full_recovered_shares = []
    for i in range(n_parties):
        full = marked_shares[i].copy()
        full[valid_mask] = recovered_valid[i]
        full_recovered_shares.append(full)

    return full_recovered_shares, extracted_bits

def histogram_extract_sequential_on_shares(marked_shares, ME, LE, valid_mask, l,
                                             ass, ass_bool, offline_bool, offline_arith,
                                             verbose=False):
    """
    Sequential, pixel-by-pixel extraction -- correctness-first version.

    Unlike the batched version, this recomputes each pixel's prediction using
    the shares array AS IT IS BEING RECOVERED, matching the plaintext algorithm's
    causal dependency (later pixels' context depends on earlier pixels' already-
    restored values). This is inherently sequential and does not vectorize --
    a genuine architectural cost of this class of order-dependent RDH scheme
    under MPC, not something batching can shortcut without breaking correctness.
    """
    from app.protocols.prieq import pi_prieq_2pc

    n = ass.n
    h, w = marked_shares[0].shape

    # Working copy we mutate pixel-by-pixel as we go
    recovered_shares = [s.copy() for s in marked_shares]
    extracted_bits = []

    rows, cols = np.where(valid_mask)
    total = len(rows)

    def const_shares_scalar(value):
        return [np.array([value], dtype=np.int64) if i == 0
                else np.array([0], dtype=np.int64) for i in range(n)]

    for idx in range(total):
        i, j = rows[idx], cols[idx]

        # ---- Predict this single pixel using recovered_shares (causal context) ----
        N  = [recovered_shares[p][i-1, j:j+1]   for p in range(n)]
        W  = [recovered_shares[p][i,   j-1:j]   for p in range(n)]
        NE = [recovered_shares[p][i-1, j+1:j+2] for p in range(n)]
        NW = [recovered_shares[p][i-1, j-1:j]   for p in range(n)]

        sum_NW = ass.add_shares(N, W)
        diff_NENW = ass.add_shares(NE, ass.scal_mul(NW, -1))
        rs_sum = rs_on_shares(sum_NW, l, ass, ass_bool, offline_bool)
        rs_diff_once = rs_on_shares(diff_NENW, l, ass, ass_bool, offline_bool)
        rs_diff_twice = rs_on_shares(rs_diff_once, l, ass, ass_bool, offline_bool)
        pred = ass.add_shares(rs_sum, rs_diff_twice)

        x_prime = [recovered_shares[p][i, j:j+1] for p in range(n)]
        e = ass.add_shares(x_prime, ass.scal_mul(pred, -1))
        abs_e, sign = abs_value_on_shares(e, l, ass, ass_bool, offline_bool, offline_arith)

        # ---- r1: |e'|==ME  (bit=0 embedded) ----
        beta_A1, beta_bits1 = gen_beta(l, (1,), ass, ass_bool)
        r1_bool = pi_prieq_2pc(abs_e, const_shares_scalar(ME), beta_A1, beta_bits1, l,
                                 ass, ass_bool, offline_bool, pi_mul)
        rb_B1, rb_A1 = gen_random_bit_both_fields((1,), ass_bool, ass)
        r1_arith = pi_b2a(r1_bool, rb_B1, rb_A1, ass_bool, ass)
        r1 = int(ass.reconstruct(r1_arith)[0])

        # ---- r1b: |e'|==ME+1  (bit=1 embedded) ----
        beta_A2, beta_bits2 = gen_beta(l, (1,), ass, ass_bool)
        r1b_bool = pi_prieq_2pc(abs_e, const_shares_scalar(ME + 1), beta_A2, beta_bits2, l,
                                  ass, ass_bool, offline_bool, pi_mul)
        rb_B2, rb_A2 = gen_random_bit_both_fields((1,), ass_bool, ass)
        r1b_arith = pi_b2a(r1b_bool, rb_B2, rb_A2, ass_bool, ass)
        r1b = int(ass.reconstruct(r1b_arith)[0])

        undo = 0
        if r1 == 1:
            extracted_bits.append(0)
            undo = 0
        elif r1b == 1:
            extracted_bits.append(1)
            undo = 1
        else:
            # ---- r2: ME+1 < |e'| <= LE  (generic shift-by-1 zone, undo, no bit) ----
            beta_A3, beta_bits3 = gen_beta(l, (1,), ass, ass_bool)
            cmp1_bool = pi_pricmp_2pc(abs_e, const_shares_scalar(ME + 2), beta_A3, beta_bits3,
                                        l, ass, ass_bool, offline_bool, pi_mul)
            beta_A4, beta_bits4 = gen_beta(l, (1,), ass, ass_bool)
            cmp2_bool = pi_pricmp_2pc(abs_e, const_shares_scalar(LE + 1), beta_A4, beta_bits4,
                                        l, ass, ass_bool, offline_bool, pi_mul)
            not_cmp2 = [c.copy() for c in cmp2_bool]
            not_cmp2[0] = (1 - not_cmp2[0]) % 2
            nbte_bool = offline_bool.gen_nbte(N=2, shape=(1,))
            r2_bool = pi_mul([cmp1_bool, not_cmp2], nbte_bool, ass_bool)
            r2 = int(ass.reconstruct(pi_b2a(r2_bool, *gen_random_bit_both_fields((1,), ass_bool, ass),
                                              ass_bool, ass))[0])
            undo = r2  # 1 if in shift zone, else 0 (untouched pixel)

        if undo != 0:
            nbte_undo = offline_arith.gen_nbte(N=2, shape=(1,))
            undo_amount = pi_mul([sign, [np.array([undo], dtype=np.int64) if p == 0
                                          else np.zeros(1, dtype=np.int64) for p in range(n)]],
                                   nbte_undo, ass)
            new_val = ass.add_shares(x_prime, ass.scal_mul(undo_amount, -1))
        else:
            new_val = x_prime

        for p in range(n):
            recovered_shares[p][i, j] = new_val[p][0]

        if verbose and (idx % 20 == 0 or idx == total - 1):
            print(f"   [extract] pixel {idx}/{total} ({100*(idx+1)/total:.0f}%)")

    return recovered_shares, extracted_bits