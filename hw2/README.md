# ADL2019/hw2

Sequence Classification with Contextual Embeddings

* [Homework 2 Website](https://www.csie.ntu.edu.tw/~miulab/s107-adl/A2)
* [Homework 2 Slide](https://docs.google.com/presentation/d/1dK1IubKXqseagzMlDEEFO4M0Gp7qedv78le3cw5rKcM/edit#slide=id.g52160a344b_3_8)
* [Kaggle Competition](https://www.kaggle.com/c/adl2019-homework-2)
    * Public Leaderboard Rank: 18/67
    * Private Leaderboard Rank: 15/67
* [Guide for Part 1](https://docs.google.com/presentation/d/1h24s8ZLErwcBK42yLwGP9cc2mIQqeO_uLHeV3WewXmo/edit#slide=id.p)
* [data.zip](https://drive.google.com/open?id=1mFCnbIE0-vM5coBmRHPo21Cvj_CCGEfu)
* TA's [README.md](https://github.com/JasonYao81000/ADL2019/blob/master/hw2/TA_README.md)

## 0. Requirements
```shell
#!/bin/bash
apt-get update
apt-get install -y python-software-properties
apt-get install -y software-properties-common

add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get install -y python3.6
apt-get install -y python3.6-dev libmysqlclient-dev

wget https://bootstrap.pypa.io/get-pip.py
python3.6 get-pip.py

python3.6 -m pip install --upgrade setuptools
python3.6 -m pip install torch torchvision
python3.6 -m pip install spacy pyyaml python-box tqdm ipdb
python3.6 -m spacy download en
python3.6 -m pip install allennlp
python3.6 -m pip install flair
```

## Part 1. Train an ELMo to beat the simple baseline

### 1. Train your own ELMo
The codes for training ELMo is base on this [repo](https://github.com/HIT-SCIR/ELMoForManyLangs).
```python
python -m ELMo.biLM train \
    --seed 9487 \
    --gpu 0 \
    --train_path ./data/language_model/train.txt \
    --valid_path ./data/language_model/valid.txt \
    --config_path ./ELMo/configs/cnn_50_100_512_4096_sample.json \
    --model ./ELMo/output \
    --optimizer adam \
    --lr 0.001 \
    --lr_decay 0.8 \
    --batch_size 64 \
    --max_epoch 10 \
    --max_sent_len 64 \
    --max_vocab_size 150000 \
    --min_count 3
```

### 2. Train BCN based on ELMo for classification task
1. Modify the `MY_ELMo` in `/hw2/ELMo/embedder.py` to `True`.
2. Train BCN based on ELMo for classification task 
```python
python -m BCN.train model/bcn_my_elmo
```

### 3. Make prediction for BCN based on ELMo
Based on the development set performance, you can choose which epoch's model checkpoint to use to generate prediction.Optionally, you can specify the batch size.
```python
python -m BCN.predict model/bcn_my_elmo 7 --batch_size 512
```
You will then have a prediction as a csv file that can be uploaded to kaggle under `model/bcn_my_elmo/predictions/`.

We have beat the simple baseline with `0.49049` on the private and `0.49954` on the public.

## Part 2. Beat the strong baseline with nearly no limitation

### 1. Contextualized Embeddings
* We used the embeddings provided by [zalandoresearch/flair](https://github.com/zalandoresearch/flair), such as `FlairEmbeddings`, `ELMoEmbeddings`, and `BertEmbeddings`.
* That is, you need to install `allennlp` and `flair` via `pip`.
* The used embeddings are listed as followings:
    * `FlairEmbeddings('news-forward')`
    * `FlairEmbeddings('news-backward')`
    * `ELMoEmbeddings('original')`
    * `BertEmbeddings('bert-large-cased')`
    * `BertEmbeddings('bert-large-uncased')`

### 2. Train BCN based on Contextualized Embeddings for classification task
```python
python -m BCN.train model/bcn_bert
python -m BCN.train model/bcn_bert_elmo
python -m BCN.train model/bcn_bert_elmo_un
python -m BCN.train model/bcn_bert_un
python -m BCN.train model/bcn_elmo
python -m BCN.train model/bcn_flair_bert_elmo
python -m BCN.train model/bcn_flair_bert_elmo_un
```

### 3. Predict on the top 5 best Eval. accuracy for each model
```python
python -m BCN.predict model/bcn_bert 2 --batch_size 8
python -m BCN.predict model/bcn_bert 3 --batch_size 8
python -m BCN.predict model/bcn_bert 5 --batch_size 8
python -m BCN.predict model/bcn_bert 7 --batch_size 8
python -m BCN.predict model/bcn_bert 16 --batch_size 8
```
```python
python -m BCN.predict model/bcn_bert_elmo 1 --batch_size 8
python -m BCN.predict model/bcn_bert_elmo 4 --batch_size 8
python -m BCN.predict model/bcn_bert_elmo 5 --batch_size 8
python -m BCN.predict model/bcn_bert_elmo 6 --batch_size 8
python -m BCN.predict model/bcn_bert_elmo 9 --batch_size 8
```
```python
python -m BCN.predict model/bcn_bert_elmo_un 7 --batch_size 8
python -m BCN.predict model/bcn_bert_elmo_un 8 --batch_size 8
python -m BCN.predict model/bcn_bert_elmo_un 10 --batch_size 8
python -m BCN.predict model/bcn_bert_elmo_un 11 --batch_size 8
python -m BCN.predict model/bcn_bert_elmo_un 12 --batch_size 8
```
```python
python -m BCN.predict model/bcn_bert_un 7 --batch_size 8
python -m BCN.predict model/bcn_bert_un 8 --batch_size 8
python -m BCN.predict model/bcn_bert_un 10 --batch_size 8
python -m BCN.predict model/bcn_bert_un 11 --batch_size 8
python -m BCN.predict model/bcn_bert_un 12 --batch_size 8
```
```python
python -m BCN.predict model/bcn_elmo 3 --batch_size 8
python -m BCN.predict model/bcn_elmo 5 --batch_size 8
python -m BCN.predict model/bcn_elmo 12 --batch_size 8
python -m BCN.predict model/bcn_elmo 14 --batch_size 8
python -m BCN.predict model/bcn_elmo 16 --batch_size 8
```
```python
python -m BCN.predict model/bcn_flair_bert_elmo 2 --batch_size 8
python -m BCN.predict model/bcn_flair_bert_elmo 5 --batch_size 8
python -m BCN.predict model/bcn_flair_bert_elmo 6 --batch_size 8
python -m BCN.predict model/bcn_flair_bert_elmo 9 --batch_size 8
python -m BCN.predict model/bcn_flair_bert_elmo 16 --batch_size 8
```
```python
python -m BCN.predict model/bcn_flair_bert_elmo_un 6 --batch_size 8
python -m BCN.predict model/bcn_flair_bert_elmo_un 9 --batch_size 8
python -m BCN.predict model/bcn_flair_bert_elmo_un 10 --batch_size 8
python -m BCN.predict model/bcn_flair_bert_elmo_un 11 --batch_size 8
python -m BCN.predict model/bcn_flair_bert_elmo_un 16 --batch_size 8
```

### 4. Ensemble Performance for each model with 5 .csv files

| Model Name | Flair | BERT | BERT_UN | ELMo | Private | Public |
| ---------- | :---: | :--: | :-----: | :--: | ------- | ------ |
| bcn_bert | | V | | | 0.53755 | 0.49864 |
| bcn_bert_elmo | | V | | V | 0.55022 | 0.53031 |
| bcn_bert_elmo_un | | | V | V | 0.55022 | 0.53031 |
| bcn_bert_un | | | V | | 0.52669 | 0.49140 |
| bcn_elmo | | | | V | 0.51131 | 0.50407 |
| bcn_flair_bert_elmo | V | V | | V | 0.54027 | 0.51402 |
| bcn_flair_bert_elmo_un | V | | V | V | 0.53484 | 0.50769 |

### 5. Ensemble Performance over 35 .csv files

| Top N .Files | Private | Public |
| ------------ | ------- | ------ |
| 35 | 0.56561 | 0.52579 |
| 30 | 0.55837 | 0.52941 |
| 25 | 0.55746 | 0.53303 |
| 20 | 0.56018 | 0.52941 |
| 15 | 0.55475 | 0.53484 |
| 10 | 0.55022 | 0.53031 |
