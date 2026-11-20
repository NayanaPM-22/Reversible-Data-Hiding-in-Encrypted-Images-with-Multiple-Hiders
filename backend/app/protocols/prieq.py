
import numpy as np

def bit_of(v, j):
    return (v.astype(np.int64) >> j) & 1

def pi_prieq_2pc(x_shares, y_shares, beta_A_shares, beta_bits_B, l, ass, ass_bool, offline_bool, pi_mul_fn):
    n = ass.n
    d_shares = ass.add_shares(x_shares, ass.scal_mul(y_shares, -1))
    alpha_shares = ass.add_shares(d_shares, ass.scal_mul(beta_A_shares, -1))
    alpha = ass.reconstruct(alpha_shares)
    neg_alpha = (-alpha).astype(np.int64)

    result = None
    for j in range(l):
        neg_alpha_j = bit_of(neg_alpha, j)
        beta_j = beta_bits_B[j]
        P_j = [beta_j[i].copy() for i in range(n)]
        P_j[0] = (P_j[0] + neg_alpha_j) % 2
        not_P_j = [P_j[i].copy() for i in range(n)]
        not_P_j[0] = (1 - not_P_j[0]) % 2
        if result is None:
            result = not_P_j
        else:
            nbte_and = offline_bool.gen_nbte(N=2, shape=alpha.shape)
            result = pi_mul_fn([result, not_P_j], nbte_and, ass_bool)
    return result
