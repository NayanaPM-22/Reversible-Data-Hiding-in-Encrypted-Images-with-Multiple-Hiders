
import numpy as np

def gen_random_bit_both_fields(shape, ass_bool, ass_arith):
    rb = np.random.randint(0, 2, size=shape, dtype=np.int64)
    rb_B_shares = ass_bool.share(rb)
    rb_A_shares = ass_arith.share(rb)
    return rb_B_shares, rb_A_shares

def pi_b2a(b_shares_B, rb_B, rb_A, ass_bool, ass_arith):
    n = ass_bool.n
    a_shares = [(b_shares_B[i] + rb_B[i]) % 2 for i in range(n)]
    a = ass_bool.reconstruct(a_shares)
    out = []
    for i in range(n):
        if_zero = rb_A[i]
        if i == 0:
            if_one = 1 - rb_A[i]
        else:
            if_one = -rb_A[i]
        out.append(np.where(a == 0, if_zero, if_one))
    return [ass_arith._wrap_signed(o) for o in out]
