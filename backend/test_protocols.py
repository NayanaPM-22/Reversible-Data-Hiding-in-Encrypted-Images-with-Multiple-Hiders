"""Quick sanity check: do the 5 MPC protocols still work correctly
when imported as real project files, not notebook cells?"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from app.core.ass import ASS
from app.protocols.offline import Offline
from app.protocols.mul import pi_mul
from app.protocols.pricmp import pi_pricmp_2pc, gen_beta
from app.protocols.prieq import pi_prieq_2pc
from app.protocols.b2a import pi_b2a, gen_random_bit_both_fields
from app.protocols.rshift import pi_rs_2pc_correct

np.random.seed(42)
l = 10
ass = ASS(l=l, n=2)
ass_bool = ASS(l=1, n=2)
offline = Offline(ass)
offline_bool = Offline(ass_bool)

n_trials = 100
shape = (n_trials,)

# Test Pi_Mul
x = np.random.randint(-20, 20, size=shape, dtype=np.int64)
y = np.random.randint(-20, 20, size=shape, dtype=np.int64)
xs, ys = ass.share(x), ass.share(y)
nbte = offline.gen_nbte(N=2, shape=shape)
prod = pi_mul([xs, ys], nbte, ass)
ok_mul = np.array_equal(ass.reconstruct(prod), ass._wrap_signed(x * y))
print(f"Pi_Mul:    {'PASS' if ok_mul else 'FAIL'}")

# Test Pi_PriCmp
xv = np.random.randint(-100, 100, size=shape, dtype=np.int64)
yv = np.random.randint(-100, 100, size=shape, dtype=np.int64)
xvs, yvs = ass.share(xv), ass.share(yv)
beta_A, beta_bits = gen_beta(l, shape, ass, ass_bool)
r = pi_pricmp_2pc(xvs, yvs, beta_A, beta_bits, l, ass, ass_bool, offline_bool, pi_mul)
rec = (r[0] + r[1]) % 2
ok_cmp = np.array_equal(rec, (xv >= yv).astype(np.int64))
print(f"Pi_PriCmp: {'PASS' if ok_cmp else 'FAIL'}")

# Test Pi_PriEq
beta_A2, beta_bits2 = gen_beta(l, shape, ass, ass_bool)
r2 = pi_prieq_2pc(xvs, yvs, beta_A2, beta_bits2, l, ass, ass_bool, offline_bool, pi_mul)
rec2 = (r2[0] + r2[1]) % 2
ok_eq = np.array_equal(rec2, (xv == yv).astype(np.int64))
print(f"Pi_PriEq:  {'PASS' if ok_eq else 'FAIL'}")

# Test Pi_B2A
bits_in = np.random.randint(0, 2, size=shape, dtype=np.int64)
b_shares = ass_bool.share(bits_in)
rb_B, rb_A = gen_random_bit_both_fields(shape, ass_bool, ass)
b2a_out = pi_b2a(b_shares, rb_B, rb_A, ass_bool, ass)
ok_b2a = np.array_equal(ass.reconstruct(b2a_out), bits_in)
print(f"Pi_B2A:    {'PASS' if ok_b2a else 'FAIL'}")

# Test Pi_RS
xv2 = np.random.randint(-200, 200, size=shape, dtype=np.int64)
xvs2 = ass.share(xv2)
beta_A3, beta_bits3 = gen_beta(l, shape, ass, ass_bool)
rs_out = pi_rs_2pc_correct(xvs2, beta_A3, beta_bits3, l, ass, ass_bool, offline_bool,
                             pi_mul, gen_random_bit_both_fields, pi_b2a)
ok_rs = np.array_equal(ass.reconstruct(rs_out), xv2 >> 1)
print(f"Pi_RS:     {'PASS' if ok_rs else 'FAIL'}")

print(f"\n{'ALL PROTOCOLS PASS' if all([ok_mul, ok_cmp, ok_eq, ok_b2a, ok_rs]) else 'SOME FAILED -- see above'}")