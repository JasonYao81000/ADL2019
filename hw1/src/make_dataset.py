import argparse
import logging
import os
import pdb
import sys
import traceback
import pickle
import json
from embedding import Embedding
from preprocessor import Preprocessor


def main(args):
    config_path = os.path.join(args.dest_dir, 'config.json')
    logging.info('loading configuration from {}'.format(config_path))
    with open(config_path) as f:
        config = json.load(f)

    preprocessor = Preprocessor(None)

    # collect words appear in the data
    words = set()
    logging.info('collecting words from {}'.format(config['valid_json_path']))
    words |= preprocessor.collect_words(config['test_json_path'],
                                        n_workers=args.n_workers)
    logging.info('collecting words from {}'.format(config['train_json_path']))
    words |= preprocessor.collect_words(config['train_json_path'],
                                        n_workers=args.n_workers)
    logging.info('collecting words from {}'.format(config['test_json_path']))
    words |= preprocessor.collect_words(config['valid_json_path'],
                                        n_workers=args.n_workers)

    # load embedding only for words in the data
    logging.info(
        'loading embedding from {}'.format(config['embedding_vec_path'])
    )
    embedding = Embedding(config['embedding_vec_path'], words)
    embedding_pkl_path = os.path.join(args.dest_dir, 'embedding.pkl')
    logging.info('Saving embedding to {}'.format(embedding_pkl_path))
    with open(embedding_pkl_path, 'wb') as f:
        pickle.dump(embedding, f)

    # update embedding used by preprocessor
    preprocessor.embedding = embedding

    # valid
    logging.info('Processing valid from {}'.format(config['valid_json_path']))
    valid = preprocessor.get_dataset(
        config['valid_json_path'], args.n_workers,
        {'n_positive': -1, 'n_negative': -1, 'shuffle': False}
    )
    valid_pkl_path = os.path.join(args.dest_dir, 'valid.pkl')
    logging.info('Saving valid to {}'.format(valid_pkl_path))
    with open(valid_pkl_path, 'wb') as f:
        pickle.dump(valid, f)

    # train
    logging.info('Processing train from {}'.format(config['train_json_path']))
    train = preprocessor.get_dataset(config['train_json_path'], args.n_workers)
    train_pkl_path = os.path.join(args.dest_dir, 'train.pkl')
    logging.info('Saving train to {}'.format(train_pkl_path))
    with open(train_pkl_path, 'wb') as f:
        pickle.dump(train, f)

    # test
    logging.info('Processing test from {}'.format(config['test_json_path']))
    test = preprocessor.get_dataset(
        config['test_json_path'], args.n_workers,
        {'n_positive': -1, 'n_negative': -1, 'shuffle': False}
    )
    test_pkl_path = os.path.join(args.dest_dir, 'test.pkl')
    logging.info('Saving test to {}'.format(test_pkl_path))
    with open(test_pkl_path, 'wb') as f:
        pickle.dump(test, f)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess and generate preprocessed pickle.")
    parser.add_argument('dest_dir', type=str,
                        help='[input] Path to the directory that .')
    parser.add_argument('--n_workers', type=int, default=4)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s: %(message)s',
        level=logging.INFO, datefmt='%m-%d %H:%M:%S'
    )
    args = _parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
