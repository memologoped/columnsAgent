import numpy as np


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
        assert len(self.depth_columns) == len(position_array), "Size error 1"
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


class BinColumns(BaseColumns):
    def __init__(self, depth_columns: list):
        super().__init__(depth_columns)

    def __str__(self):
        super().__str__()
        return ""

    # method for checking the result
    # true letters are denoted by 1 in the matrix consisting of 0 and 1
    def true_col(self, pos_arr: list):
        for i in range(len(self.table)):
            self.table[i][pos_arr[i]] = 1

    # method for environment
    # matrix filled with zeros
    def bot_col(self):
        pass
