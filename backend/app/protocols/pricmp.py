
import numpy as np

def bit_of(v, j):
    return (v.astype(np.int64) >> j) & 1

def gen_beta(l, shape, ass_arith, ass_bool):
    beta = np.random.randint(0, 1 << l, size=shape, dtype=np.int64)
    beta_A_shares = ass_arith.share(beta)
    bit_shares = []
    for j in range(l):
        bit = (beta >> j) & 1
        bit_shares.append(ass_bool.share(bit))
    return beta_A_shares, bit_shares

def pi_pricmp_2pc(x_shares, y_shares, beta_A_shares, beta_bits_B, l, ass, ass_bool, offline_bool, pi_mul):
    n = ass.n
    d_shares = ass.add_shares(x_shares, ass.scal_mul(y_shares, -1))
    alpha_shares = ass.add_shares(d_shares, ass.scal_mul(beta_A_shares, -1))
    alpha = ass.reconstruct(alpha_shares)

    C = [np.zeros_like(alpha) for _ in range(n)]
    for j in range(0, l - 1):
        alpha_j = bit_of(alpha, j)
        beta_j = beta_bits_B[j]
        G_j = [(alpha_j * beta_j[i]) % 2 for i in range(n)]
        P_j = [beta_j[i].copy() for i in range(n)]
        P_j[0] = (P_j[0] + alpha_j) % 2
        nbte_and = offline_bool.gen_nbte(N=2, shape=alpha.shape)
        PC_shares = pi_mul([P_j, C], nbte_and, ass_bool)
        C = [(G_j[i] + PC_shares[i]) % 2 for i in range(n)]

    top_alpha_bit = bit_of(alpha, l - 1)
    beta_top = beta_bits_B[l - 1]
    d_top = [beta_top[i].copy() for i in range(n)]
    d_top[0] = (d_top[0] + top_alpha_bit) % 2
    d_top = [(d_top[i] + C[i]) % 2 for i in range(n)]

    r = [d_top[i].copy() for i in range(n)]
    r[0] = (1 - r[0]) % 2
    return r
