# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:49:18 2019

@author: hb2506
"""
import json
import logging
from multiprocessing.dummy import Pool
from dataset import DialogDataset
from tqdm import tqdm
import re
import pickle
import json
#from embedding import Embedding
#from preprocessor import Preprocessor


    config_path = os.path.join(args.dest_dir, 'config.json')
    logging.info('loading configuration from {}'.format(config_path))
    with open(config_path) as f:
        config = json.load(f)


    # collect words appear in the data
    words = set()
    words |= collect_words(config['test_json_path'],
                                        n_workers=args.n_workers)
    words |= collect_words(config['train_json_path'],
                                        n_workers=args.n_workers)
    words |= collect_words(config['valid_json_path'],
                                        n_workers=args.n_workers)
   
    embedding = Embedding(config['embedding_vec_path'], words)


    # valid
    valid = preprocessor.get_dataset(
        config['valid_json_path'], args.n_workers,
        {'n_positive': -1, 'n_negative': -1, 'shuffle': True}
    )
    valid_pkl_path = os.path.join(args.dest_dir, 'valid.pkl')
    logging.info('Saving valid to {}'.format(valid_pkl_path))
    with open(valid_pkl_path, 'wb') as f:
        pickle.dump(valid, f)

    # train
    train = preprocessor.get_dataset(config['train_json_path'], args.n_workers,
        {'n_positive': -1, 'n_negative': -1, 'shuffle': True}                                 
    )
    train_pkl_path = os.path.join(args.dest_dir, 'train.pkl')
    logging.info('Saving train to {}'.format(train_pkl_path))
    with open(train_pkl_path, 'wb') as f:
        pickle.dump(train, f)

    # test
    test = preprocessor.get_dataset(
        config['test_json_path'], args.n_workers,
        {'n_positive': -1, 'n_negative': -1, 'shuffle': False}
    )
    test_pkl_path = os.path.join(args.dest_dir, 'test.pkl')
    logging.info('Saving test to {}'.format(test_pkl_path))
    with open(test_pkl_path, 'wb') as f:
        pickle.dump(test, f)






    def tokenize(sentence):
        filtrate = re.compile(u'[^\u0041-\u005A | \u0061-\u007A]')#非中文
        filtered_str = filtrate.sub(r' ', sentence)#replace
        querywords = filtered_str.split()     
        return querywords

    def sentence_to_indices(sentence):
        new_sentence = list(tokenize(sentence))
        for i in range(len(new_sentence)):
            indices.append(embedding.to_index(new_sentence[i]))          
        return indices
    
    def collect_words(data_path, n_workers=5):
        with open(data_path) as f:
            data = json.load(f)
        utterances = []
        for sample in data:
            utterances += (
                [message['utterance']
                 for message in sample['messages-so-far']]
                + [option['utterance']
                   for option in sample['options-for-next']]
            )
        utterances = list(set(utterances))
        chunks = [
            ' '.join(utterances[i:i + len(utterances) // n_workers])
            for i in range(0, len(utterances), len(utterances) // n_workers)
        ]
        wordlist = []
        words = list()
        filtrate = re.compile(u'[^\u0041-\u005A | \u0061-\u007A]')#非中文
        for i in range(len(chunks)):
            filtered_str = filtrate.sub(r' ', chunks[i])#replace
            wordlist.append(filtered_str)
        for i in range(len(wordlist)):
            querywords = wordlist[i].split()
            words = words + querywords
        return set(words)

    def get_dataset(data_path, n_workers=5, dataset_args={}):
        with open(data_path) as f:
            dataset = json.load(f)
        results = [None] * n_workers
        with Pool(processes=n_workers) as pool:
            for i in range(n_workers):
                batch_start = (len(dataset) // n_workers) * i
                if i == n_workers - 1:
                    batch_end = len(dataset)
                else:
                    batch_end = (len(dataset) // n_workers) * (i + 1)
                batch = dataset[batch_start: batch_end]
                results[i] = pool.apply_async(self.preprocess_samples, [batch])
            pool.close()
            pool.join()
        processed = []
        for result in results:
            processed += result.get()
        padding = to_index('</s>')
        return DialogDataset(processed, padding=padding, **dataset_args)

    def preprocess_samples(dataset):
        processed = []
        for sample in tqdm(dataset):
            processed.append(preprocess_sample(sample))
        return processed

    def preprocess_sample(data):
        processed = {}
        processed['id'] = data['example-id']
        processed['context'] = []
        processed['speaker'] = []
        for message in data['messages-so-far']:
            processed['context'].append(
                sentence_to_indices(message['utterance'].lower())
            )
        processed['options'] = []
        processed['option_ids'] = []
        if 'options-for-correct-answers' in data:
            processed['n_corrects'] = len(data['options-for-correct-answers'])
            for option in data['options-for-correct-answers']:
                processed['options'].append(
                    sentence_to_indices(option['utterance'].lower())
                )
                processed['option_ids'].append(option['candidate-id'])
        else:
            processed['n_corrects'] = 0
        for option in data['options-for-next']:
            if option['candidate-id'] in processed['option_ids']:
                continue
            processed['options'].append(
                sentence_to_indices(option['utterance'].lower())
            )
            processed['option_ids'].append(option['candidate-id'])
        return processed
    
    def Embedding(words, oov_as_unk=True, lower=True, rand_seed=524):
        from gensim.models import Word2Vec
        w2v_model = []
        word_dict = {}
        embedding_dim = 50
        window_ = 1
        min_count_ = 0
        sample_ = 1e-1
        iter_ = 10
        w2v_model = Word2Vec(texts, size=embedding_dim, window=window_, min_count=min_count_,sample=sample_,iter=iter_,  workers=12)
        text_weights = w2v_model.wv.syn0 
        vocab = dict([(k, v.index) for k,v in w2v_model.wv.vocab.items()])  
        vocab_list = [k for k,v in w2v_model.wv.vocab.items()]
        word_dict = dict([(k, w2v_model.wv[k]) for k in vocab_list])
        embedding_matrix = word_dict.get()
        return vocab_list, embedding_matrix, word_dict


    def to_index(word):
        word = word.lower()
        if word in self.word_dict:
            return self.word_dict[word]

    