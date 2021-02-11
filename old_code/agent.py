import actions
import state
from columns import *


class Agent:
    def __init__(self):
        pass

    def select_move(self, table_state):
        raise NotImplementedError


class RandomBot(object):
    def __init__(self, env: Environment):
        self.sym_col = env.sym_col
        self.answ_col = BinColumns(env.depth_col)

    def _filling_binary_table(self):
        while state.ColumnsState.is_not_over(self.answ_col):
            self.answ_col = actions.Moves.choose_symbol(self.sym_col, self.answ_col)

    def create_sentence(self):
        self._filling_binary_table()
        sentence = str()
        for i in range(len(self.answ_col.table)):
            for j in range(len(self.answ_col.table[i])):
                if self.answ_col.table[i][j] == 1:
                    sentence += str(self.sym_col.table[i][j])
        return sentence
