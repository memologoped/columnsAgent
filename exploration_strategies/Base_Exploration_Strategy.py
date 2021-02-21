import numpy as np
import math


class ActionsSpace(object):

    @classmethod
    def choose_symbol(cls, col_state, symbol_col):
        last_column, last_position = col_state.get_last_position()

        if last_column < 0:
            last_column += 1
            top_1 = list(np.random.choice(symbol_col.table[last_column],
                                          int(math.sqrt(len(symbol_col.table[last_column]))), replace=False))
            top_2 = list(np.random.choice(symbol_col.table[last_column + 1],
                                          int(math.sqrt(len(symbol_col.table[last_column + 1]))), replace=False))

            for i in top_1:
                for j in top_2:
                    if Politics.check_bigrams(i, j):
                        col_state.state.table[last_column][symbol_col.table[last_column].index(i)] = 1
                        col_state.state.table[last_column + 1][symbol_col.table[last_column + 1].index(j)] = 1
                        return col_state.state

            return col_state.state

        else:
            top_1 = symbol_col.table[last_column][last_position]
            top_2 = list(np.random.choice(symbol_col.table[last_column + 1],
                                          int(math.sqrt(len(symbol_col.table[last_column + 1]))), replace=False))

            for i in top_2:
                if Politics.check_bigrams(top_1, i):
                    col_state.state.table[last_column + 1][symbol_col.table[last_column + 1].index(i)] = 1
                    return col_state.state

            col_state.state.table[last_column][last_position] = 0
            return col_state.state


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


class Reward(object):
    pass
