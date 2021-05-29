from math import log
import pickle
import numpy
from numpy import array
from math import log10


# beam_search
def beam_search(data: numpy.array, k: int) -> numpy.array:
    sequences = [[list(), 0.0]]
    for row in data:
        all_candidates = list()
        for i in range(len(sequences)):
            seq, score = sequences[i]
            for j in range(len(row)):
                candidate = [seq + [j], score + log(-row[j])]
                all_candidates.append(candidate)
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        sequences = ordered[:k]
    return array(sequences, dtype=object)


def main_beam_search() -> None:
    best_decision = 3
    data = [[-0.1, -0.2, -0.3, -0.4, -0.5],
            [-0.5, -0.4, -0.3, -0.2, -0.1],
            [-0.1, -0.2, -0.3, -0.4, -0.5],
            [-0.5, -0.4, -0.3, -0.2, -0.1],
            [-0.1, -0.2, -0.3, -0.4, -0.5],
            [-0.5, -0.4, -0.3, -0.2, -0.1],
            [-0.1, -0.2, -0.3, -0.4, -0.5],
            [-0.5, -0.4, -0.3, -0.2, -0.1],
            [-0.1, -0.2, -0.3, -0.4, -0.5],
            [-0.5, -0.4, -0.3, -0.2, -0.1]]
    data = array(data, dtype=object)
    result = beam_search(data, best_decision)
    print(f"{best_decision} best decision is:\n")
    for seq in result:
        print(seq)


# markov chain
def generate_freq_table(data: str, k: int) -> dict:
    table = {}
    for i in range(len(data) - k):
        x = data[i:i + k]
        y = data[i + k]

        if table.get(x) is None:
            table[x] = {}
            table[x][y] = 1
        else:
            if table[x].get(y) is None:
                table[x][y] = 1
            else:
                table[x][y] += 1
    return table


def freq_into_prob(table: dict) -> dict:
    for symbs in table.keys():
        s = float(sum(table[symbs].values()))
        for freq in table[symbs].keys():
            table[symbs][freq] = table[symbs][freq] / s
    return table


def markov_chain(train_filename: str, k: int) -> dict:
    with open(train_filename, mode="r") as file:
        data = file.read()
    freq_table = generate_freq_table(data, k)
    prob_table = freq_into_prob(freq_table)
    return prob_table


def save_chain(filename, model) -> None:
    with open(filename, mode="wb") as file:
        pickle.dump(model, file)


def load_chain(filename: str) -> dict:
    with open(filename, mode="rb") as file:
        model = pickle.load(file)
    return model


def sample_next(text: str, model: dict, k: int) -> dict:
    text = text[-k:]
    prob = dict(zip(list(model[text].keys()), list(model[text].values())))
    prob = dict(sorted(prob.items(), key=lambda item: item[1]))
    return prob


def main_markov_chain() -> None:
    text = "chec"
    model = load_chain("./data/statistic/model_3.pkl")
    res = sample_next(text, model, 3)
    print(f"Possible continuation for {text} is: {res}")


# n-gram statistic
class NgramScore(object):
    def __init__(self, ngramfile: str, sep=' '):
        self.ngrams = {}
        key = None
        for line in open(ngramfile, 'r'):
            key, count = line.split(sep)
            self.ngrams[key] = int(count)
        self.L = len(key)
        self.N = sum(self.ngrams.values())
        for key in self.ngrams.keys():
            self.ngrams[key] = log10(float(self.ngrams[key]) / self.N)
        self.floor = log10(0.01 / self.N)

    def score(self, text: str) -> float:
        score = 0
        ngrams = self.ngrams.__getitem__
        for i in range(len(text) - self.L + 1):
            if text[i:i + self.L] in self.ngrams:
                score += ngrams(text[i:i + self.L])
            else:
                score += self.floor
        return score


def main_ngram() -> None:
    text = "chec"
    loader = NgramScore("./data/statistic/english_4grams.txt")
    value = loader.score(text)
    print(f"For 'chec' value = {value}")


if __name__ == "__main__":
    main_beam_search()
    main_markov_chain()
    main_ngram()
