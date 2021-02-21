from columns_gym import envs_utils
from columns_gym import columns


class Environment(object):
    def __init__(self, file):
        """Receive some data for columns generation and rewards politics for estimating actions."""
        # text with possible mistakes
        self.text, self.mistakes = envs_utils.text_prepare(file)
        depth_col = envs_utils.gen_depth(self.text)
        pos_col = envs_utils.gen_pos(depth_col)

        # main environment
        self.symbol_col = columns.SymColumns(depth_col, pos_col, self.text)

        # bin table with answers
        self.answer_col = columns.BinColumns.true_col(pos_col)

    def step(self, action=None):
        """Applies action to the current state and returns new state and reward."""

        return state, reward


class EnvironmentState(object):

    def __init__(self, envs):
        self.state = columns.BinColumns(envs.dept_col)
        self.reward = 0

    def get_last_position(self):
        self.state.table.reverse()
        for i in range(len(self.state.table)):
            for j in range(len(self.state.table[i])):
                if self.state.table[i][j] == 1:
                    self.state.table.reverse()
                    return len(self.state.table) - 1 - i, j
        self.state.table.reverse()
        return -1, -1

    def is_not_over(self):
        if self.state.table[len(self.state.table) - 1].count(1) == 1:
            return False
        else:
            return True