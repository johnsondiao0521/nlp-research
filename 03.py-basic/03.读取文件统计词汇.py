import os


def read_data(file):
    with open(file, encoding='utf-8') as f:
        all_data = f.read().split('\n')

    all_chn, all_eng = [], []
    for data in all_data:
        data_s = data.split(' ')
        if len(data_s) != 2:
            continue
        chn, eng = data_s
        all_chn.append(chn)
        all_eng.append(eng)

    return all_chn, all_eng


def get_word_2_index(all_chn):
    word_2_index = {}
    for chn in all_chn:
        for w in chn:
            word_2_index[w] = word_2_index.get(w, len(word_2_index))

    return word_2_index


def get_eng_2_count(all_eng):
    eng_2_count = {}
    for eng in all_eng:
        for w in eng:
            eng_2_count[w] = eng_2_count.get(w, 1) + 1
    return eng_2_count


if __name__ == '__main__':

    all_chn, all_eng = read_data(os.path.join('data', 'text.txt'))

    chn_2_index = get_word_2_index(all_chn)
    eng_2_index = get_word_2_index(all_eng)

    eng_2_count = get_eng_2_count(all_eng)

    print("")
