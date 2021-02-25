import math

import numpy as np


class ActionsSpace(object):

    @classmethod
    def choose_symbol(cls, env_state):
        sym_col = env_state.env.sym_col.table
        last_col = len(env_state.result) - 1

        if last_col < 0:
            last_col += 1

            top_1 = list(np.random.choice(sym_col[last_col],
                         2 * int(math.sqrt(len(sym_col[last_col]))), replace=False))
            top_2 = list(np.random.choice(sym_col[last_col + 1],
                         2 * int(math.sqrt(len(sym_col[last_col + 1]))), replace=False))

            for i in top_1:
                for j in top_2:
                    if Politics.check_bigrams(i, j):
                        env_state.result += str(i) + str(j)
                        return env_state

            return env_state

        else:
            i = env_state.result[len(env_state.result) - 1]
            top_2 = list(np.random.choice(sym_col[last_col + 1],
                         2 * int(math.sqrt(len(sym_col[last_col + 1]))), replace=False))
            for j in top_2:
                if Politics.check_bigrams(i, j):
                    env_state.result += str(j)
                    return env_state

            env_state.result = env_state.result[:len(env_state.result) - 1]
            return env_state


class Politics(object):
    _unused_bigrams = ["bk", "fq", "jc", "jt", "mj", "qh", "qx", "vj", "wz", "zh",
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
                       "qw", "vh", "wx", "zg"]

    @classmethod
    def check_bigrams(cls, sym1: str, sym2: str):
        if cls._unused_bigrams.count(sym1 + sym2) == 0:
            return True
        if cls._unused_bigrams.count(sym1 + sym2) == 1:
            return False
