"""Test the sequential (correctness-first) extraction against a DENSE
embedding case -- small enough to run in reasonable time, dense enough
to trigger neighbor-context dependencies the batched version got wrong."""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from app.core.ass import ASS
from app.core.pehs import pehs_embed_plaintext, pehs_extract_plaintext
from app.protocols.offline import Offline
from app.protocols.pehs_mpc import (
    sgap_predict_on_shares, compute_prediction_error_on_shares,
    abs_value_on_shares, histogram_on_shares, histogram_shift_embed_on_shares,
    histogram_extract_sequential_on_shares,
)

l = 10
n = 2
ass = ASS(l=l, n=n)
ass_bool = ASS(l=1, n=n)
offline_bool = Offline(ass_bool)
offline_arith = Offline(ass)

np.random.seed(7)
h, w = 15, 15   # small on purpose -- sequential extraction is slow by design
img = np.random.randint(0, 256, size=(h, w), dtype=np.int64)

# Force DENSE embedding: as many bits as the image can possibly hold
dense_bits = np.random.randint(0, 2, size=1000).tolist()

marked_plain, ME_p, LE_p, n_emb_plain = pehs_embed_plaintext(img, dense_bits)
recovered_plain, extracted_plain = pehs_extract_plaintext(marked_plain, ME_p, LE_p)
print(f"Plaintext: bits_embedded={n_emb_plain}, lossless={np.array_equal(img, recovered_plain)}")

img_shares = ass.share(img)
pred_shares, valid_mask = sgap_predict_on_shares(img_shares, l, ass, ass_bool, offline_bool)
e_shares = compute_prediction_error_on_shares(img_shares, pred_shares, ass)
abs_e_shares, sign_shares = abs_value_on_shares(e_shares, l, ass, ass_bool, offline_bool, offline_arith)
max_e = int(ass.reconstruct(abs_e_shares)[valid_mask].max())
hist, ME_mpc, LE_mpc = histogram_on_shares(abs_e_shares, valid_mask, l, ass, ass_bool,
                                             offline_bool, offline_arith, max_e_estimate=max_e)
marked_shares, n_emb_mpc = histogram_shift_embed_on_shares(
    img_shares, abs_e_shares, sign_shares, ME_mpc, LE_mpc, dense_bits,
    valid_mask, l, ass, ass_bool, offline_bool, offline_arith
)
print(f"MPC embed: bits_embedded={n_emb_mpc}, ME={ME_mpc}, LE={LE_mpc}")

print("\nRunning SEQUENTIAL extraction (this will print progress)...")
recovered_shares, extracted_mpc = histogram_extract_sequential_on_shares(
    marked_shares, ME_mpc, LE_mpc, valid_mask, l, ass, ass_bool, offline_bool,
    offline_arith, verbose=True
)
recovered_reconstructed = ass.reconstruct(recovered_shares)

recovery_match = np.array_equal(img[valid_mask], recovered_reconstructed[valid_mask])
bits_match = (extracted_plain == extracted_mpc)

print(f"\nRecovery matches (lossless): {recovery_match}")
print(f"Bits match:                  {bits_match}")
print(f"\n{'SEQUENTIAL EXTRACTION FIXES THE BUG' if (recovery_match and bits_match) else 'STILL MISMATCHED -- see below'}")

if not recovery_match:
    diff = img[valid_mask] != recovered_reconstructed[valid_mask]
    print(f"Mismatches: {np.sum(diff)} / {diff.size}")