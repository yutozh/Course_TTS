#!/usr/bin/env bash

# 将原始训练数据转换成corpus格式
python3 align.py dataset/train.dict output/train.corpus

# 进入RnnLMG2P文件夹
cd RnnLMG2P || exit

# 执行训练命令,生成模型
python3 script/train-g2p-rnnlm.py -c script/train.corpus -p mymode

# 使用训练好的模型，对测试数据进行解码
phonetisaurus-g2prnn --rnnlm=mymode.rnnlm --test=testRnnlm.dict --nbest=1 | ./prettify.pl > tmp.txt

# 格式化输出内容
awk -F '\\t' '{print \$1"\\t"\$2}' tmp.txt > test.rnnlm.out

# 切换文件夹
cd 02_front_end/2-2_g2p || exit

# 计算标注结果的准确率
python3 acc.py --src_path output/test.rnnlm.out --gt_path dataset/gtRnnlm.dict
