# Voice cloning
## 1. 实验要求
1. 随机挑选之前课程中提供的抄本数据进行录音。
2. 以few-shot的方式refine声学模型的部分/全部参数。
3. 以同样的方式对WaveRNN声码器部分进行refine。
4. 尝试对自适应效果进行改进。

## 2. 实验步骤
### 2.1 录音
1. 从之前课程提供的标贝科技开源数据中的label中随机挑选50句左右的文本进行录音。选取集内抄本的原因是我们已经提供了对应的文本特征。
2. 尽量挑选安静场所进行录音，避免音量过大或过小。可以利用Audition等相关软件检查录音质量，也可以进行一定的音频处理操作，比如：降噪处理。

### 2.2 数据准备
1. 调整录音的采样率，注意和之前实验设置保持统一。
2. 声学模型部分的数据准备请参照04_seq2seq_tts/README.md
3. 声码器部分的数据准备请参照06_nn_vocoder/README.md

### 2.3 声学模型自适应
1. 修改声学模型部分，加入模型restore的相关代码。
2. 在之前课程中训练好的声学模型的基础上，固定encoder部分参数，尝试对attention/decoder的部分参数或全部参数进行训练。
3. 尝试不同的训练步数和学习率大小，利用griffin-lim方法评估训练效果。

### 2.4 声码器自适应
1. 声码器代码部分已经提供restore的相关代码。
2. 在之前课程中训练好的声码器的基础上，尝试对整体参数/部分参数进行refine，看看能否在保持声码器稳定的基础上改善音色的相似度。
3. 和声学模型对接，验证自适应效果。

## 3. 实验说明
1. 以多说话人数据训练的基础源模型(source model)的训练耗费时间和算力较大，本次实验直接使用之前课程训练的好的声学模型及声码器作为基础模型。因此课程作业仅以实验为目的。
2. 由于缺乏多说话人数据训练的源模型，模型部分和课程中的方法略有出入，主要体现在无需speaker encoder/embedding部分。因此，实验以few-shot的方式进行部分refine。
3. 有时间和余力的同学也可以利用一些论文中给出的公开数据，例如使用数据集 (http://www.openslr.org/38/) 进行自己的声学模型及声码器的源模型训练。（ps.在此基础上也可以尝试zero-shot的方式)
4. 声码器部分的自适应相对没有声学模型部分稳定，特别是在源模型数据量和覆盖度不足的情况下。有条件的同学可以利用libritts等多说话人数据（不限语言），进行vocoder部分的源模型训练。通常情况下，无需在录音上refine也能够达到一个相对不错的音质。