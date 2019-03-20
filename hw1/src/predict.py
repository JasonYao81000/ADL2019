import argparse
import logging
import os
import pdb
import pickle
import sys
import traceback
import json
from metrics import Recall


def main(args):
    # load config
    config_path = os.path.join(args.model_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    # load embedding
    logging.info('loading embedding...')
    with open(config['model_parameters']['embedding'], 'rb') as f:
        embedding = pickle.load(f)
        config['model_parameters']['embedding'] = embedding.vectors

    # make model
    if config['arch'] == 'ExampleNet':
        from example_predictor import ExamplePredictor
        PredictorClass = ExamplePredictor

    predictor = PredictorClass(metrics=[],
                               **config['model_parameters'])
    model_path = os.path.join(
        args.model_dir,
        'model.pkl.{}'.format(args.epoch))
    logging.info('loading model from {}'.format(model_path))
    predictor.load(model_path)

    # predict test
    logging.info('loading test data...')
    with open(config['test'], 'rb') as f:
        test = pickle.load(f)
        test.shuffle = False
    logging.info('predicting...')
    print(test)
    print(test.collate_fn)
    predicts = predictor.predict_dataset(test, test.collate_fn)

    output_path = os.path.join(args.model_dir,
                               'predict-{}.csv'.format(args.epoch))
    write_predict_csv(predicts, test, output_path)


def write_predict_csv(predicts, data, output_path, n=10):
    outputs = []
    for predict, sample in zip(predicts, data):
        candidate_ranking = [
            {
                'candidate-id': oid,
                'confidence': score.item()
            }
            for score, oid in zip(predict, sample['option_ids'])
        ]

        candidate_ranking = sorted(candidate_ranking,
                                   key=lambda x: -x['confidence'])
        best_ids = [candidate_ranking[i]['candidate-id']
                    for i in range(n)]
        outputs.append(
            ''.join(
                ['1-' if oid in best_ids
                 else '0-'
                 for oid in sample['option_ids']])
        )

    logging.info('Writing output to {}'.format(output_path))
    with open(output_path, 'w') as f:
        f.write('Id,Predict\n')
        for output, sample in zip(outputs, data):
            f.write(
                '{},{}\n'.format(
                    sample['id'],
                    output
                )
            )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train.")
    parser.add_argument('model_dir', type=str,
                        help='Directory to the model checkpoint.')
    parser.add_argument('--device', default=None,
                        help='Device used to train. Can be cpu or cuda:0,'
                        ' cuda:1, etc.')
    parser.add_argument('--not_load', action='store_true',
                        help='Do not load any model.')
    parser.add_argument('--epoch', type=int, default=10)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
