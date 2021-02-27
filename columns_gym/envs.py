import numpy as np

from columns_gym import envs_utils


# TODO объеденить класс EnvironmentState и Environment в один

class Environment(object):
    def __init__(self, file):
        """Receive some data for columns generation and rewards politics for estimating actions."""

        self.true_text, self.mistakes = envs_utils.text_prepare(file)

        col_depth = envs_utils.gen_depth(self.true_text)
        pos_col = envs_utils.gen_pos(col_depth)

        self.sym_col = SymColumns(col_depth, pos_col, self.true_text)

    def step(self, action=None):
        """
        Applies action to the current state and returns new state and reward.

        """
        pass


class EnvironmentState(object):

    def __init__(self, env):
        self.env = env
        self.reward = 0
        self.result = str()

    def is_over(self) -> bool:
        return len(self.result) == len(self.env.true_text)

    def show_res(self):
        return self.result


class BaseColumns(object):
    # constructor of empty columns table filled with zeros
    def __init__(self, depth_columns: list):
        self.depth_columns = depth_columns

        table = [0] * len(self.depth_columns)

        for i in range(len(self.depth_columns)):
            table[i] = [0] * self.depth_columns[i]
        self.table = table

    # formatted output columns table
    def __str__(self):
        print("=" * 50)

        max_len = max([len(col) for col in self.table])

        for i in range(max_len):
            for j in range(len(self.table)):
                if len(self.table[j]) - 1 < i:
                    print(" ", end="|")
                    continue

                print(self.table[j][i], end="|")

            print()

        return str("=" * 50)


class SymColumns(BaseColumns):
    def __init__(self, depth_columns: list, position_array: list, text: str):
        super().__init__(depth_columns)
        assert len(self.depth_columns) == len(position_array), "Size error 1"  # TODO очень информативно ...
        assert len(text) == len(self.depth_columns), "Size error 2"

        self.text = text
        self.position_array = position_array

        self._fill_col()

    def __str__(self):
        super().__str__()
        return ""

    # filling column symbols
    def _fill_col(self):
        for i in range(len(self.table)):
            sym_array = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                         'n', 'o', 'p', 'q', 'r', 's', 't', 'v', 'u', 'w', 'x', 'y', 'z']

            del sym_array[sym_array.index(self.text[i])]

            rand_sym = list(np.random.choice(sym_array, len(self.table[i]) - 1))
            rand_sym.insert(self.position_array[i], self.text[i])

            assert len(rand_sym) == len(self.table[i]), "Size error 3"

            for j in range(len(self.table[i])):
                self.table[i][j] = rand_sym[j]
