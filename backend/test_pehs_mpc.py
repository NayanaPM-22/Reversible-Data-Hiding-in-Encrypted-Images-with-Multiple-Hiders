"""Test the MPC-protocol PEHS pieces against the plaintext versions,
one building block at a time."""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from app.core.ass import ASS
from app.core.pehs import sgap_predict
from app.protocols.offline import Offline
from app.protocols.pehs_mpc import (
    sgap_predict_on_shares,
    compute_prediction_error_on_shares,
    abs_value_on_shares,
)

np.random.seed(10)
l = 10
n = 2
ass = ASS(l=l, n=n)
ass_bool = ASS(l=1, n=n)
offline_bool = Offline(ass_bool)
offline_arith = Offline(ass)

# Small test image (fast for iterating; scale up to 512x512 once everything passes)
h, w = 20, 20
img = np.random.randint(0, 256, size=(h, w), dtype=np.int64)

# ---- 1. SGAP prediction ----
pred_plain, valid_mask_plain = sgap_predict(img)

img_shares = ass.share(img)
pred_shares, valid_mask_mpc = sgap_predict_on_shares(img_shares, l, ass, ass_bool, offline_bool)
pred_reconstructed = ass.reconstruct(pred_shares)

mask_match = np.array_equal(valid_mask_plain, valid_mask_mpc)
pred_match = np.array_equal(pred_plain[valid_mask_plain], pred_reconstructed[valid_mask_plain])
print(f"Valid mask matches:       {mask_match}")
print(f"Predictions match:        {pred_match}")

# ---- 2. Prediction error ----
e_shares = compute_prediction_error_on_shares(img_shares, pred_shares, ass)
e_reconstructed = ass.reconstruct(e_shares)
e_plain = img - pred_plain
e_match = np.array_equal(e_plain[valid_mask_plain], e_reconstructed[valid_mask_plain])
print(f"Prediction error matches: {e_match}")

# ---- 3. Absolute value ----
abs_e_shares, sign_shares = abs_value_on_shares(e_shares, l, ass, ass_bool, offline_bool, offline_arith)
abs_e_reconstructed = ass.reconstruct(abs_e_shares)
abs_e_plain = np.abs(e_plain)
abs_match = np.array_equal(abs_e_plain[valid_mask_plain], abs_e_reconstructed[valid_mask_plain])
print(f"Absolute value matches:   {abs_match}")

if not abs_match:
    diff = abs_e_plain[valid_mask_plain] != abs_e_reconstructed[valid_mask_plain]
    print(f"  Mismatches: {np.sum(diff)} / {diff.size}")
    idx = np.where(diff)[0][:5]
    flat_plain = abs_e_plain[valid_mask_plain]
    flat_mpc = abs_e_reconstructed[valid_mask_plain]
    for i in idx:
        print(f"    plain={flat_plain[i]}, mpc={flat_mpc[i]}")

print(f"\n{'ALL CHECKS PASS' if all([mask_match, pred_match, e_match, abs_match]) else 'SOME FAILED -- see above'}")
from app.protocols.pehs_mpc import histogram_on_shares

abs_errors_plain = abs_e_plain[valid_mask_plain]
max_e_plain = int(abs_errors_plain.max())

hist_mpc, ME_mpc, LE_mpc = histogram_on_shares(
    abs_e_shares, valid_mask_plain, l, ass, ass_bool, offline_bool, offline_arith,
    max_e_estimate=max_e_plain
)

hist_plain = np.bincount(abs_errors_plain, minlength=max_e_plain + 1)
ME_plain = int(np.argmax(hist_plain))
LE_plain = None
for candidate in range(ME_plain + 1, max_e_plain + 2):
    if candidate > max_e_plain or hist_plain[candidate] == 0:
        LE_plain = candidate; break
if LE_plain is None:
    LE_plain = max_e_plain + 1

hist_match = np.array_equal(hist_mpc, hist_plain)
me_match = (ME_mpc == ME_plain)
le_match = (LE_mpc == LE_plain)

print(f"Histogram matches:        {hist_match}")
print(f"ME matches:                {me_match}  (mpc={ME_mpc}, plain={ME_plain})")
print(f"LE matches:                {le_match}  (mpc={LE_mpc}, plain={LE_plain})")

from app.protocols.pehs_mpc import histogram_shift_embed_on_shares
from app.core.pehs import pehs_embed_plaintext

np.random.seed(20)
test_bits = np.random.randint(0, 2, size=200).tolist()

marked_plain, ME_p, LE_p, n_emb_plain = pehs_embed_plaintext(img, test_bits)

marked_shares_mpc, n_emb_mpc = histogram_shift_embed_on_shares(
    img_shares, abs_e_shares, sign_shares, ME_mpc, LE_mpc, test_bits,
    valid_mask_plain, l, ass, ass_bool, offline_bool, offline_arith
)
marked_reconstructed = ass.reconstruct(marked_shares_mpc)

embed_match = np.array_equal(marked_plain[valid_mask_plain], marked_reconstructed[valid_mask_plain])
count_match = (n_emb_plain == n_emb_mpc)

print(f"\nHistogram shift embed matches: {embed_match}")
print(f"Bits embedded count matches:   {count_match}  (plain={n_emb_plain}, mpc={n_emb_mpc})")

if not embed_match:
    diff = marked_plain[valid_mask_plain] != marked_reconstructed[valid_mask_plain]
    print(f"  Mismatches: {np.sum(diff)} / {diff.size}")

from app.protocols.pehs_mpc import histogram_extract_on_shares
from app.core.pehs import pehs_extract_plaintext

recovered_plain, extracted_bits_plain = pehs_extract_plaintext(marked_plain, ME_p, LE_p)

recovered_shares_mpc, extracted_bits_mpc = histogram_extract_on_shares(
    marked_shares_mpc, ME_mpc, LE_mpc, valid_mask_plain, l, ass, ass_bool,
    offline_bool, offline_arith
)
recovered_reconstructed = ass.reconstruct(recovered_shares_mpc)

recovery_match = np.array_equal(recovered_plain[valid_mask_plain], recovered_reconstructed[valid_mask_plain])
bits_match = (extracted_bits_plain == extracted_bits_mpc)

print(f"\nRecovery matches (lossless): {recovery_match}")
print(f"Extracted bits match:        {bits_match}")
if bits_match:
    print(f"  Extracted: {extracted_bits_mpc}")

print(f"\n{'FULL MPC PIPELINE ROUND TRIP PASSES' if (recovery_match and bits_match) else 'MISMATCH -- see above'}")
