import math
from itertools import product

import numpy as np
import torch


class BaseAgent(object):
    def __init__(self, estimator: torch.nn.Module = None):
        """
        BaseAgent initialization.

        Parameters:

            estimator: torch.nn.Module
                Neural network instance.

        """

        self.estimator = estimator

    def take_action(self, env_state):
        raise NotImplementedError


class RandomAgent(BaseAgent):
    _forbidden_bigrams = {"bk", "fq", "jc", "jt", "mj", "qh", "qx", "vj", "wz", "zh",
                          "bq", "fv", "jd", "jv", "mq", "qj", "qy", "vk", "xb", "zj",
                          "bx", "fx", "jf", "jw", "mx", "qk", "qz", "vm", "xg", "zn",
                          "cb", "fz", "jg", "jx", "mz", "ql", "sx", "vn", "xj", "zq",
                          "cf", "gq", "jh", "jy", "pq", "qm", "sz", "vp", "xk", "zr",
                          "cg", "gv", "jk", "jz", "pv", "qn", "tq", "vq", "xv", "zs",
                          "cj", "gx", "jl", "kq", "px", "qo", "tx", "vt", "xz", "zx",
                          "cp", "hk", "jm", "kv", "qb", "qp", "vb", "vw", "yq", "cv",
                          "hv", "jn", "kx", "qc", "qr", "vc", "vx", "yv", "cw", "hx",
                          "jp", "kz", "qd", "qs", "vd", "vz", "yz", "cx", "hz", "jq",
                          "lq", "qe", "qt", "vf", "wq", "zb", "dx", "iy", "jr", "lx",
                          "qf", "qv", "vg", "wv", "zc", "fk", "jb", "js", "mg", "qg",
                          "qw", "vh", "wx", "zg"}

    def take_action(self, env_state):
        """Choose random action depending on env_state."""

        return self.choose_symbol(env_state)

    def choose_symbol(self, env_state):
        sym_col = env_state.env.sym_col.table
        last_col = len(env_state.result) - 1

        if last_col < 0:
            last_col += 1

            top_1 = list(np.random.choice(sym_col[last_col],
                                          2 * int(math.sqrt(len(sym_col[last_col]))), replace=False))
            top_2 = list(np.random.choice(sym_col[last_col + 1],
                                          2 * int(math.sqrt(len(sym_col[last_col + 1]))), replace=False))

            for i, j in product(top_1, top_2):
                b = i + j
                if self.is_not_forbidden(bigram=b):
                    env_state.result += b

                    return env_state

            return env_state

        else:
            i = env_state.result[len(env_state.result) - 1]
            top_2 = list(np.random.choice(sym_col[last_col + 1],
                                          2 * int(math.sqrt(len(sym_col[last_col + 1]))), replace=False))
            for j in top_2:
                if self.is_not_forbidden(bigram=i + j):
                    env_state.result += j
                    return env_state

            env_state.result = env_state.result[:len(env_state.result) - 1]
            return env_state

    @classmethod
    def is_not_forbidden(cls, bigram: str) -> bool:
        return bigram not in cls._forbidden_bigrams


# TODO implement
class QAgent(BaseAgent):
    def __init__(self, estimator: torch.nn.Module, epsilon: float = 0.0):
        super().__init__(estimator)

        self.epsilon = epsilon

    def take_action(self, env_state):
        pass

    def update_estimator(self, estimator):
        self.estimator = estimator

    def update_epsilon(self, epsilon: float):
        self.epsilon = epsilon
