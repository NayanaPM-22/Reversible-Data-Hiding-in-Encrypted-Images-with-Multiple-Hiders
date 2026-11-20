
import itertools
import numpy as np

def pi_mul(x_shares_list, nbte, ass):
    N = len(x_shares_list)
    n = ass.n

    d = []
    for j in range(1, N + 1):
        xj = x_shares_list[j - 1]
        aj = nbte[(j,)]
        dj_shares = ass.add_shares(xj, ass.scal_mul(aj, -1))
        d.append(ass.reconstruct(dj_shares))

    f = [np.zeros_like(d[0]) for _ in range(n)]
    all_idx = set(range(1, N + 1))
    for size in range(1, N + 1):
        for I in itertools.combinations(range(1, N + 1), size):
            prod = np.ones_like(d[0])
            for j in (all_idx - set(I)):
                prod = prod * d[j - 1]
            aI = nbte[I]
            for i in range(n):
                f[i] = f[i] + prod * aI[i]

    prod_all = np.ones_like(d[0])
    for dj in d:
        prod_all = prod_all * dj
    f[0] = f[0] + prod_all

    return [ass._wrap_signed(fi) for fi in f]
