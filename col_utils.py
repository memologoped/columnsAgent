import random
import string

import numpy as np


def get_text(file):
    text = file.readline()
    if not text:
        print("\n\nTHE END OF WORK.\nFile is read to the End or is empty.")
        raise SystemExit
    return text


def del_punct(sentence):
    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    sentence = sentence.replace(" ", "")
    sentence = sentence.rstrip()
    return sentence


def gen_sym(length):
    letters = string.ascii_lowercase
    gen_symbol = ''.join((random.choice(letters) for _ in range(length)))
    return gen_symbol


def skip_mist(text_list: list, position: int):
    # print(text_list[position], " ", position)
    text_list[position] = ""


def insert_mist(text_list: list, position):
    few_sym = np.random.choice(2, 1, p=[0.99, 0.01])[0]
    if few_sym == 1:
        digits = [2, 3, 4]
        length = random.choice(digits)
        new_symbol = gen_sym(int(length))
    else:
        new_symbol = gen_sym(1)

    text_list[position] = text_list[position] + new_symbol


def replace_mist(text_list: list, position: int):
    new_symbol = gen_sym(1)
    text_list[position] = new_symbol


def poss_mist(text: str):
    mistakes = dict()  # save insert and replace mistakes (without skip mistakes)
    text_list = list(text)
    mist_text = ""
    shift = 0

    for i in range(len(text_list)):
        type_mist = np.random.choice(4, 1, p=[0.965, 0.01, 0.01, 0.015])[0]

        if type_mist == 1:
            skip_mist(text_list, i)
            shift -= 1

        if type_mist == 2:
            insert_mist(text_list, i)
            diff = len(text_list[i]) - 1

            for pos in range(i + 1, i + 1 + diff):
                mistakes[pos + shift] = "insert"

            shift += len(text_list[i]) - 1

        if type_mist == 3:
            replace_mist(text_list, i)
            mistakes[i + shift] = "replace"

        mist_text += "".join(text_list[i])
    return mist_text, mistakes


def gen_depth(text: str):
    depth_array = list()
    depths = np.random.choice(26, 26, replace=False)
    gen_sum = depths.sum()
    depths = list(depths)
    prob_array = list()

    for i in range(len(depths)):
        prob_array.append(depths[i] / gen_sum)
    depth_probability_mass = np.round(prob_array, decimals=5)

    for j in range(len(text)):
        depth_column = np.random.choice(26, 1, p=depth_probability_mass) + 1
        depth_array.append(depth_column[0])

    return depth_array


def gen_pos(depth_array):
    pos_array = list()

    for k in range(len(depth_array)):
        pos_array.append((np.random.choice(depth_array[k], 1))[0])

    return pos_array


def text_prepare(file):
    text = get_text(file)
    solid_text = del_punct(text)
    # truth_text = solid_text  # if it's necessary
    res_text, mistakes = poss_mist(solid_text)
    return res_text, mistakes
