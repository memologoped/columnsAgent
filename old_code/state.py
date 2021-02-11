class ColumnsState(object):

    @classmethod
    def get_last_position(cls, table):
        table.table.reverse()
        for i in range(len(table.table)):
            for j in range(len(table.table[i])):
                if table.table[i][j] == 1:
                    table.table.reverse()
                    return len(table.table) - 1 - i, j
        table.table.reverse()
        return -1, -1

    @classmethod
    def is_not_over(cls, table):
        if table.table[len(table.table) - 1].count(1) == 1:
            return False
        else:
            return True
