
import numpy as np

def bit_of(v, j):
    return (v.astype(np.int64) >> j) & 1

def pi_rs_2pc_correct(x_shares, beta_A_shares, beta_bits_B, l, ass_arith, ass_bool,
                        offline_bool, pi_mul_fn, gen_random_bit_both_fields_fn, pi_b2a_fn):
    n = ass_arith.n
    alpha_shares = ass_arith.add_shares(x_shares, ass_arith.scal_mul(beta_A_shares, -1))
    alpha = ass_arith.reconstruct(alpha_shares)

    alpha0 = bit_of(alpha, 0)
    beta0 = beta_bits_B[0]
    C = [(alpha0 * beta0[i]) % 2 for i in range(n)]

    x_bits_A = []
    for j in range(1, l):
        alpha_j = bit_of(alpha, j)
        beta_j = beta_bits_B[j]
        t1 = [beta_j[i].copy() for i in range(n)]
        t1[0] = (t1[0] + alpha_j) % 2
        bit_x_j = [(t1[i] + C[i]) % 2 for i in range(n)]

        rb_B, rb_A = gen_random_bit_both_fields_fn(alpha.shape, ass_bool, ass_arith)
        bit_x_j_A = pi_b2a_fn(bit_x_j, rb_B, rb_A, ass_bool, ass_arith)
        x_bits_A.append((j - 1, bit_x_j_A))

        if j < l - 1:
            ab = [(alpha_j * beta_j[i]) % 2 for i in range(n)]
            xor_ab = [beta_j[i].copy() for i in range(n)]
            xor_ab[0] = (xor_ab[0] + alpha_j) % 2
            nbte_c = offline_bool.gen_nbte(N=2, shape=alpha.shape)
            c_and = pi_mul_fn([C, xor_ab], nbte_c, ass_bool)
            C = [(ab[i] + c_and[i]) % 2 for i in range(n)]

    result = [np.zeros_like(alpha) for _ in range(n)]
    top_bit_pos = l - 2
    for pos, bit_shares in x_bits_A:
        weight = -(1 << pos) if pos == top_bit_pos else (1 << pos)
        for i in range(n):
            result[i] = result[i] + weight * bit_shares[i]

    return [ass_arith._wrap_signed(r) for r in result]
