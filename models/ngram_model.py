import copy

from numpy import array

from utils import NgramScore, beam_search


def z_read(line: str, num_solutions: int) -> None:
    columns = line.lower().split()

    loader = NgramScore("../data/statistic/english_1grams.txt")
    # list for probabilities as monograms of characters in first column
    first_layer = [loader.score(i) for i in columns[0]]

    # list for probabilities as bigrams of characters in first and second columns
    second_layer = [copy.deepcopy([first_layer]) for _ in range(len(columns[0]))]  # extend of first layer
    loader = NgramScore("../data/statistic/english_2grams.txt")
    for i, j in enumerate(second_layer, start=0):
        second_layer[i][0][i] *= 0.1  # activation first layer
        j.append(copy.deepcopy([loader.score(columns[0][i] + sym) for sym in columns[1]]))

    third_layer = list()  # list for probabilities as trigrams of characters in first, second and third columns
    for i in range(len(columns[0])):
        for _ in range(len(columns[1])):
            third_layer.append(copy.deepcopy(second_layer[i]))  # extend of second layer
    loader = NgramScore("../data/statistic/english_3grams.txt")
    for i, j in enumerate(third_layer, start=0):
        k = i % len(columns[1])
        t = 0
        third_layer[i][1][k] *= 0.1  # activation second layer
        probabilities = list()
        for l in columns[2]:
            probabilities.append(loader.score(columns[0][t] + columns[1][k] + l))
        j.append(copy.deepcopy(probabilities))
        if k % len(columns[0]) == 0 and k != 0:
            t += 1

    best_decision = list()
    for table in third_layer:
        table = array(table, dtype=object)
        best_decision.append(beam_search(table, 2))

    best_decision = array(best_decision, dtype=object)
    best_decision = best_decision.reshape(-1, 2)
    best_decision = sorted(best_decision, key=lambda list_: list_[1])
    best_decision = best_decision[:num_solutions]

    sent_starts = list()
    for i in range(len(best_decision)):
        pos = best_decision[i][0]
        sent = str()
        for i, j in enumerate(pos, start=0):
            sent += columns[i][j]
        sent_starts.append(sent)

    gen_sent = list()
    loader = NgramScore("../data/statistic/english_4grams.txt")

    for i in sent_starts:
        for line in columns[len(i):]:
            prob = {}
            for sym in line:
                prob[sym] = loader.score(i + sym)
            new_sym = max(prob, key=prob.get)
            i += new_sym
        gen_sent.append(i)

    print(f"{num_solutions} best solutions:")
    for i in gen_sent:
        print(i)


def main() -> None:
    # str_ = "hyjs eacfn fdl glb uto".upper()
    str_true = "thehighwayroundsacurveandtransitionstondstreetrunningeastwardthroughmorefarmlandasthetrunklineappro" \
               "achesneway"
    str_ = "twer hvb rye hfj idf g fdhh ghw kja ghjy r rtyo u nfgh dhjk s a cghfhf u r vfgh e a fn d t r afgh n " \
           "s i tfgh i ghjo n srt t o nghj d smn t rkl e edfg t fdr u fdn n iret hn rtyg e adfsg s t wdfg a r vbd " \
           "t hvbcnv r o u iopg xcvh zxm sdo qwr dfge frety a dfgr m l kla ern drt auio jks vbnt bvnh e fght r u " \
           "dsfn ikk bnl gfi kbn fe ea hgp dsfp feir bnco ajkl etc dfh ehjd s dgn e dfw dfka yghp"
    data = [str_]

    with open("ztext.txt", mode="r") as file:
        data = file.readlines()

    num_solutions = 5

    for line in data:
        z_read(line, num_solutions)


if __name__ == '__main__':
    main()

