import numpy as np

class ASS:
    def __init__(self, l, n):
        self.l = l
        self.n = n
        self.mod = 1 << l
        self.half = 1 << (l - 1)

    def _wrap_signed(self, x):
        x = np.mod(x, self.mod)
        return np.where(x >= self.half, x - self.mod, x)

    def share(self, secret):
        secret = np.asarray(secret, dtype=np.int64)
        shares = []
        acc = np.zeros_like(secret, dtype=np.int64)
        for i in range(self.n - 1):
            r = np.random.randint(-self.half, self.half, size=secret.shape, dtype=np.int64)
            shares.append(r)
            acc = self._wrap_signed(acc + r)
        last = self._wrap_signed(secret - acc)
        shares.append(last)
        return shares

    def reconstruct(self, shares):
        acc = np.zeros_like(shares[0], dtype=np.int64)
        for s in shares:
            acc = self._wrap_signed(acc + s)
        return acc

    def add_shares(self, a_shares, b_shares):
        return [self._wrap_signed(a + b) for a, b in zip(a_shares, b_shares)]

    def add_const(self, shares, c):
        out = [s.copy() for s in shares]
        out[0] = self._wrap_signed(out[0] + c)
        return out

    def scal_mul(self, shares, k):
        return [self._wrap_signed(k * s) for s in shares]