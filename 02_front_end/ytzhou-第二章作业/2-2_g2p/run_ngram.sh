#!/usr/bin/env bash

# 将原始训练数据转换成corpus格式
python3 align.py dataset/train.dict output/train.corpus

# 使用ngram方法，训练模型
estimate-ngram -o 8 -t output/train.corpus -wl output/train.o8.arpa

# 将arpa格式的模型转成fst
phonetisaurus-arpa2wfst --lm=output/train.o8.arpa --ofile=output/train.o8.fst

# 利用fst模型，对测试集进行标注
phonetisaurus-apply --model output/train.o8.fst --word_list dataset/test.dict > output/test.out

# 计算标注结果的准确率
python3 acc.py --src_path output/test.out --gt_path dataset/gt.dict
