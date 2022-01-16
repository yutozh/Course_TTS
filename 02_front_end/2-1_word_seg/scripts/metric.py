import sys

def words2ranges(word_list):
    res = []
    count = 0
    for w in word_list:
        res.append((count, count + len(w) - 1))
        count += len(w)
    return res


def metric(s1, s2):
    """
    Calculate the metric of two sentences
    :param s1: Prediction sentence
    :param s2: Ground truth sentence
    :return: TP, TP+FP, TP+FN
    """
    w1 = s1.strip().split(' ')
    w2 = s2.strip().split(' ')
    wr1 = set(["{}-{}".format(r[0], r[1]) for r in words2ranges(w1)])
    wr2 = set(["{}-{}".format(r[0], r[1]) for r in words2ranges(w2)])
    TP = len(wr1 & wr2)
    TP_FP = len(w1)
    TP_FN = len(w2)
    return TP, TP_FP, TP_FN

def statistic(path1, path2):
    f1 = open(path1, 'r', encoding='utf8')
    f2 = open(path2, 'r', encoding='utf8')
    TP, TP_FP, TP_FN = 0, 0, 0
    while True:
        l1 = f1.readline()
        l2 = f2.readline()
        if l1 == "":
            break
        m = metric(l1, l2)
        TP += m[0]
        TP_FP += m[1]
        TP_FN += m[2]

    f1.close()
    f2.close()
    return TP, TP_FP, TP_FN

def format_output(path, output_path):
    with open(path, 'r', encoding='utf8') as f1, open(output_path, 'w+', encoding='utf8') as f2:
        sentence = []
        temp = []
        for line in f1:
            line = line.strip().replace("\n", "")
            if line == "":
                if len(temp):
                    sentence.append("".join(temp))
                    temp = []
                f2.write(" ".join(sentence) + '\n')
                sentence = []
                continue
            char, char_type = line.split('\t')
            if char_type == "S":
                if len(temp):
                    sentence.append("".join(temp))
                    temp = []
                sentence.append(char)
            elif char_type in ["B", "M"]:
                temp.append(char)
            elif char_type == "E":
                temp.append(char)
                sentence.append("".join(temp))
                temp = []



if __name__ == '__main__':
    # prediction_file  = sys.argv[1]
    # gt_file = sys.argv[2]
    # output_raw_file = r"\\wsl$\Ubuntu-20.04\root\speech\TTS_Course\02_front_end\2-1_word_seg\data\people_daliy_10W\test.out"
    # prediction_file = r"\\wsl$\Ubuntu-20.04\root\speech\TTS_Course\02_front_end\2-1_word_seg\data\people_daliy_10W\test.out.raw"
    # gt_file = r"\\wsl$\Ubuntu-20.04\root\speech\TTS_Course\02_front_end\2-1_word_seg\data\people_daliy_10W\test.raw"
    output_raw_file = r"data/people_daliy_10W/test.out"
    prediction_file = r"data/people_daliy_10W/test.out.raw"
    gt_file = r"data/people_daliy_10W/test.raw"

    format_output(output_raw_file, prediction_file)

    TP, TP_FP, TP_FN = statistic(prediction_file, gt_file)
    # print(TP, TP_FP, TP_FN)

    # gold = '结婚 的 和 尚未 结婚 的 都 应该 好好 考虑 一下 人生 大事'
    # pred = '结婚 的 和尚 未结婚 的 都 应该 好好考虑 一下 人生大事'
    # TP, TP_FP, TP_FN = metric(gold, gold)

    precision = TP / TP_FP
    recall = TP / TP_FN
    f1 = 2 * precision*recall / (precision + recall)

    print("Precision: {}\nRecall: {}\nF1: {}".format(precision, recall, f1))
