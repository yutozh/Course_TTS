import sys



def align(path, output_path):
    with open(path, 'r', encoding='utf8') as f1, open(output_path, 'w+', encoding='utf8') as f2:
        for line in f1:
            line = line.strip().replace("\n", "")
            grapheme, phoneme = line.split('\t')
            phoneme_list = phoneme.split(' ')
            assert len(grapheme) == len(phoneme_list)

            corp_line = []
            for g, p in zip(grapheme, phoneme_list):
                corp_line.append(g+"}"+p)
            f2.write(" ".join(corp_line) + "\n")

if __name__ == '__main__':
    input_file  = sys.argv[1]
    output_file = sys.argv[2]
    # align("dataset/train.dict", "output/train.corpus")
    align(input_file, output_file)

