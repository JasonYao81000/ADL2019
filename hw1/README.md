# ADL2019/hw1
Dialogue Modeling
* [Homework 1 Website](https://www.csie.ntu.edu.tw/~miulab/s107-adl/A1)
* [Homework 1 Slide](https://docs.google.com/presentation/d/15LCy7TkJXl2pdz394gSPKY-fkwA3wFusv7h01lWOMSw/edit#slide=id.p)
* [Kaggle Competition](https://www.kaggle.com/c/adl2019-homework-1)
    * Public Leaderboard Rank: 1/99
    * Private Leaderboard Rank: 2/99
* [Example Code](https://drive.google.com/file/d/1KLOEg7x64BAIk8nFJwXaf9eczGjV667e/view)
* [Data](https://www.kaggle.com/c/13262/download-all)

## 0. Requirements
```
torch==1.0.1
tqdm==4.28.1
nltk==3.4
numpy==1.15.4
```

## 1. Data Preprocessing
### 1. Prepare the dataset and pre-trained embeddings (FastText is used here) in `./data`.
```
./data/train.json
./data/valid.json
./data/test.json
./data/crawl-300d-2M.vec
```

### 2. Preprocess the data
```
cd ./ADL2019/hw1/src
python make_dataset.py ../data/
```

### 3. How we Preprocess the Dataset
- [x] Load pre-trained embedding `FastText`
- [x] Tokenize the sentences using `NLTK`
- [x] Convert token to word indices
- [x] Sample batch and negative candidates (`positive:negative=1:4`)
- [x] Pad samples to the same length (`context:option=300:50`)
- [x] Simply concatenate them into single sequence
- [x] Separate them with special tokens (`participant_1`, `participant_2`)
- [ ] Concatenate (or add) "speaker embedding" after the embeddings

## 2. Training and Prediction
```
python train.py ../models/bigru_batt_5_max_focal/
python predict.py ../models/bigru_batt_5_max_focal/ --epoch -1
```

## 3. Results (Recall@10)

| RNN | Attention | Concat | Pooling | Similarity | Loss | Valid Score | Test Score | 
| --- | --------- | ------ | ------- | ---------- | ---- | ----------- | ---------- |
| BiGRU      | None             | 1 | Max  | Cosine | Focal | 0.5202 | 9.76666 |
| BiGRU      | Bahdanau         | 4 | Max  | MLP | BCE   | 0.7512 | 9.36666 |
| BiGRU      | Bahdanau         | 4 | Max  | MLP | Focal | 0.7524 | 9.35333 |
| BiGRU      | Bahdanau         | 5 | Max  | MLP | Focal | 0.7466 | 9.43333 |
| BiGRU      | Bahdanau w/ drop | 4 | Max  | MLP | Focal | 0.7458 | 9.41333 |
| BiGRU      | Bahdanau         | 4 | Mean | MLP | Focal | 0.7474 | 9.40000 |
| BiGRU      | Bahdanau w/ norm | 4 | Max  | MLP | Focal | 0.7458 | 9.42666 |
| BiGRU      | Luong            | 4 | Max  | MLP | Focal | 0.7162 | 9.48666 |
| BiGRU      | Luong w/ norm    | 4 | Max  | MLP | Focal | 0.7418 | 9.41333 |
| Deep BiGRU | Bahdanau         | 4 | Max  | MLP | Focal | 0.7286 | 9.40666 |
| Fat BiGRU  | Bahdanau         | 4 | Max  | MLP | Focal | 0.7354 | 9.46000 |
| Thin BiGRU  | Bahdanau         | 4 | Max  | MLP | Focal | 0.7516 | 9.43333 |
| BiLSTM     | Bahdanau         | 4 | Max  | MLP | BCE | 0.7554 | 9.44000 |
| BiLSTM     | Bahdanau         | 4 | Max  | MLP | Focal | 0.7522 | 9.37333 |
| BiLSTM     | Bahdanau         | 5 | Max  | MLP | Focal | 0.7490 | 9.43333 |
| BiLSTM     | Bahdanau         | 4 | Mean | MLP | Focal | 0.7426 | 9.40666 |

###### tags: `NTU` `ADL` `2019`
