import math

import numpy as np

import rules
import state


class Moves(object):

    @classmethod
    def choose_symbol(cls, sym_col, answ_col):
        last_column, last_position = state.ColumnsState.get_last_position(answ_col)

        if last_column < 0:
            last_column += 1
            top_1 = list(np.random.choice(sym_col.table[last_column],
                                          int(math.sqrt(len(sym_col.table[last_column]))), replace=False))
            top_2 = list(np.random.choice(sym_col.table[last_column + 1],
                                          int(math.sqrt(len(sym_col.table[last_column + 1]))), replace=False))

            for i in top_1:
                for j in top_2:
                    if rules.Politics.check_bigrams(i, j):
                        answ_col.table[last_column][sym_col.table[last_column].index(i)] = 1
                        answ_col.table[last_column + 1][sym_col.table[last_column + 1].index(j)] = 1
                        return answ_col

            return answ_col

        else:
            top_1 = sym_col.table[last_column][last_position]
            top_2 = list(np.random.choice(sym_col.table[last_column + 1],
                                          int(math.sqrt(len(sym_col.table[last_column + 1]))), replace=False))

            for i in top_2:
                if rules.Politics.check_bigrams(top_1, i):
                    answ_col.table[last_column + 1][sym_col.table[last_column + 1].index(i)] = 1
                    return answ_col

            answ_col.table[last_column][last_position] = 0
            return answ_col
