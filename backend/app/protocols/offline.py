
import itertools
import numpy as np

class Offline:
    def __init__(self, ass):
        self.ass = ass

    def gen_nbte(self, N, shape):
        a = {j: np.random.randint(-self.ass.half, self.ass.half, size=shape, dtype=np.int64)
             for j in range(1, N + 1)}
        out = {}
        for size in range(1, N + 1):
            for subset in itertools.combinations(range(1, N + 1), size):
                val = np.ones(shape, dtype=np.int64)
                for j in subset:
                    val = self.ass._wrap_signed(val * a[j])
                out[subset] = self.ass.share(val)
        return out
