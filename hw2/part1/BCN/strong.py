import argparse
import csv
import random
import sys
from pathlib import Path
import pickle
import re

import ipdb
import spacy
from box import Box
from tqdm import tqdm

from dataset import Part1Dataset
import numpy as np
import torch

import os
import numpy as np
import pandas as pd
from glob import glob

from dataset import create_data_loader
from train import Model
from common.utils import load_pkl
from ELMo.embedder import Embedder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', dest='data_dir', type=str, help='Data directory')
    parser.add_argument('--save_model_dir', dest='save_model_dir', type=str, help='Model directory')
    args = parser.parse_args()

    return vars(args)


def load_data(mode, data_path, nlp):
    print('[*] Loading {} data from {}'.format(mode, data_path))
    with data_path.open(encoding="utf8") as f:
        reader = csv.DictReader(f)
        data = [r for r in reader]

    for d in tqdm(data, desc='[*] Tokenizing', dynamic_ncols=True):
        text = re.sub('-+', ' ', d['text'])
        text = re.sub('\s+', ' ', text)
        doc = nlp(text)
        d['text'] = [token.text for token in doc]
    print('[-] {} data loaded\n'.format(mode.capitalize()))

    return data


def create_dataset(data, word_vocab, char_vocab, dataset_dir):
    for m, d in data.items():
        print('[*] Creating {} dataset'.format(m))
        dataset = Part1Dataset(d, word_vocab, char_vocab)
        dataset_path = (dataset_dir / '{}.pkl'.format(m))
        with dataset_path.open(mode='wb') as f:
            pickle.dump(dataset, f)
        print('[-] {} dataset saved to {}\n'.format(m.capitalize(), dataset_path))


def test(data_dir, dataset_dir, word_vocab, char_vocab):
    try:
#        print(dataset_dir / 'config.yaml')
        cfg = Box.from_yaml(filename=dataset_dir / 'config.yaml')
    except FileNotFoundError:
        print('[!] Dataset directory({}) must contain config.yaml'.format(dataset_dir))
        exit(1)
    print('[-] Vocabs and datasets will be saved to {}\n'.format(dataset_dir))

#    output_files = ['word.pkl', 'char.pkl', 'test.pkl']
#    if any([(dataset_dir+p).exists() for p in output_files]):
#        print('[!] Directory already contains saved vocab/dataset')
#        exit(1)

    nlp = spacy.load('en')
    nlp.disable_pipes(*nlp.pipe_names)

    data_dir = Path(cfg.data_dir)
    data = {m: load_data(m, data_dir / '{}.csv'.format(m), nlp)
            for m in ['test']}
    create_dataset(data, word_vocab, char_vocab, dataset_dir)



def main(data_dir, save_model_dir):
    model_dir = Path('./code/data/model/MODEL_NAME')
    prediction_dir = model_dir / 'predictions'
            
    if not prediction_dir.exists():
        prediction_dir.mkdir()
        print('[-] Directory {} created'.format(prediction_dir))
    model_path(data_dir, save_model_dir, model_dir, prediction_dir)


def model_path(data_dir, save_model_dir, model_dir, prediction_dir):
    dataset_dir = Path('./code/data/classification')
    batch_size = 8
    word_vocab = load_pkl(dataset_dir / 'word.pkl')
    char_vocab = load_pkl(dataset_dir / 'char.pkl')
    test(data_dir, dataset_dir, word_vocab, char_vocab)
    try:
        cfg = Box.from_yaml(filename=model_dir / 'config.yaml')
    except FileNotFoundError:
        print('[!] Model directory({}) must contain config.yaml'.format(model_dir))
        exit(1)
        
#    dataset_dir = Path('./dataset/classification')
    test_dataset_path = dataset_dir / 'test.pkl'
    ckpt_path = model_dir / 'ckpts' / 'predict.ckpt'
    print('[-] Test dataset: {}'.format(test_dataset_path))
#    print('[-] Model checkpoint: {}\n'.format(ckpt_path))

    print('[*] Loading vocabs and test dataset from {}'.format(dataset_dir))
    test_dataset = load_pkl(test_dataset_path)
    
    model_predict(cfg, save_model_dir, ckpt_path, test_dataset, word_vocab, char_vocab, prediction_dir, batch_size)


def model_predict(cfg, save_model_dir, ckpt_path, test_dataset, word_vocab, char_vocab, prediction_dir, batch_size):
    device = torch.device('{}:{}'.format(cfg.device.type, cfg.device.ordinal))
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed_all(cfg.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print('[*] Creating test data loader\n')
    if batch_size:
        cfg.data_loader.batch_size = batch_size
    data_loader = create_data_loader(
        test_dataset, word_vocab, char_vocab, **cfg.data_loader, shuffle=False)

    if cfg.use_elmo:
        print('[*] Creating ELMo embedder')
        elmo_embedder = Embedder(**cfg.elmo_embedder)
    else:
        elmo_embedder = None

    print('[*] Creating model\n')
    cfg.net.n_ctx_embs = cfg.elmo_embedder.n_ctx_embs if cfg.use_elmo else 0
    cfg.net.ctx_emb_dim = cfg.elmo_embedder.ctx_emb_dim if cfg.use_elmo else 0
    model = Model(device, word_vocab, char_vocab, cfg.net, cfg.optim)
    print("1")
    print(ckpt_path)
    model.load_state(ckpt_path)
    print("2")
    Ids, predictions = predict(
        device, data_loader, cfg.data_loader.max_sent_len, elmo_embedder, model)
    print("predict_done")
    save_predictions(Ids, predictions, Path(save_model_dir))


def predict(device, data_loader, max_sent_len, elmo_embedder, model):
    model.set_eval()
    with torch.no_grad():
        Ids = []
        predictions = []
        bar = tqdm(data_loader, desc='[Predict]', leave=False, dynamic_ncols=True, ascii=True)
        for batch in bar:
            Ids += batch['Id']
            text_word = batch['text_word'].to(device=device)
            text_char = batch['text_char'].to(device=device)
            if elmo_embedder and elmo_embedder.ctx_emb_dim > 0:
                text_ctx_emb = elmo_embedder(batch['text_orig'], max_sent_len)
                text_ctx_emb = torch.tensor(text_ctx_emb, device=device)
            else:
                text_ctx_emb = torch.empty(
                    (*text_word.shape, 0), dtype=torch.float32, device=device)
            text_pad_mask = batch['text_pad_mask'].to(device=device)
            logits = model(text_word, text_char, text_ctx_emb, text_pad_mask)
            label = logits.max(dim=1)[1]
            predictions += label.tolist()
        bar.close()

    return Ids, predictions


def save_predictions(Ids, predictions, output_path):
    with output_path.open(mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Id', 'label'])
        writer.writeheader()
        writer.writerows(
            [{'Id': Id, 'label': p + 1} for Id, p in zip(Ids, predictions)])
    print('[-] Output saved to {}'.format(output_path))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)
