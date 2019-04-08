import argparse
import csv
import random
import sys
from pathlib import Path

import ipdb
import numpy as np
import torch
from box import Box
from tqdm import tqdm

from .dataset import create_data_loader
from .train import Model
from common.utils import load_pkl
from ELMo.embedder import Embedder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=Path, help='Model directory')
    parser.add_argument('epoch', type=int, help='Model checkpoint number')
    parser.add_argument('--batch_size', type=int, help='Inference batch size')
    args = parser.parse_args()

    return vars(args)


def main(model_dir, epoch, batch_size):
    try:
        cfg = Box.from_yaml(filename=model_dir / 'config.yaml')
    except FileNotFoundError:
        print('[!] Model directory({}) must contain config.yaml'.format(model_dir))
        exit(1)

    device = torch.device('{}:{}'.format(cfg.device.type, cfg.device.ordinal))
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed_all(cfg.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    prediction_dir = model_dir / 'predictions'
    if not prediction_dir.exists():
        prediction_dir.mkdir()
        print('[-] Directory {} created'.format(prediction_dir))

    dataset_dir = Path(cfg.dataset_dir)
    test_dataset_path = dataset_dir / 'test.pkl'
    ckpt_path = model_dir / 'ckpts' / 'epoch-{}.ckpt'.format(epoch)
    print('[-] Test dataset: {}'.format(test_dataset_path))
    print('[-] Model checkpoint: {}\n'.format(ckpt_path))

    print('[*] Loading vocabs and test dataset from {}'.format(dataset_dir))
    word_vocab = load_pkl(dataset_dir / 'word.pkl')
    char_vocab = load_pkl(dataset_dir / 'char.pkl')
    test_dataset = load_pkl(test_dataset_path)

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
    cfg.net.ctx_emb_dim = cfg.elmo_embedder.ctx_emb_dim if cfg.use_elmo else 0
    model = Model(device, word_vocab, char_vocab, cfg.net, cfg.optim)
    model.load_state(ckpt_path)

    Ids, predictions = predict(
        device, data_loader, cfg.data_loader.max_sent_len, elmo_embedder, model)
    save_predictions(Ids, predictions, prediction_dir / 'epoch-{}.csv'.format(epoch))


def predict(device, data_loader, max_sent_len, elmo_embedder, model):
    model.set_eval()
    with torch.no_grad():
        Ids = []
        predictions = []
        bar = tqdm(data_loader, desc='[Predict]', leave=False, dynamic_ncols=True)
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
    with output_path.open(mode='w') as f:
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
