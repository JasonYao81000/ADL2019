import unittest
import os
import json

import numpy as np
import tensorflow as tf

from .bin.bilm.model import BidirectionalLanguageModel
from .bin.bilm.data import Batcher
from .bin.bilm.elmo import weight_layers

FIXTURES = './corpus_tokenized/'



class TestWeightedLayers(object):
    def _check_weighted_layer(sentences, l2_coef, do_layer_norm, use_top_only):
        print("1")
#        tf.global_variables_initializer()
#        sess = tf.Session()
        with tf.Session() as sess:
            print("2")
            FIXTURES = './ELMo/corpus_tokenized/'
            # create the Batcher
            vocab_file = os.path.join(FIXTURES, 'train.txt')
            print(vocab_file)
            batcher = Batcher(vocab_file, 50)
    
            # load the model
            options_file = os.path.join(FIXTURES, 'options.json')
            weight_file = os.path.join(FIXTURES, 'allennlp.hdf5')
            print("3")
            character_ids = tf.placeholder('int32', (None, None, 50))
            model = BidirectionalLanguageModel(
                options_file, weight_file, max_batch_size=4)
            bilm_ops = model(character_ids)
            print("4")
            weighted_ops = []
            for k in range(2):
                ops = weight_layers(str(k), bilm_ops, l2_coef=l2_coef, 
                                         do_layer_norm=do_layer_norm,
                                         use_top_only=use_top_only)
                weighted_ops.append(ops)
    
            # initialize
            print("5")
            sess.run(tf.global_variables_initializer())


            # make some data
            print(sentences)        
    #        X_chars = batcher.batch_sentences(sentences)
            print("7")
            ops = model(character_ids)
            print("8")
            print(ops)
            print(character_ids)
    #        print(X_chars)
    #        lm_embeddings, mask, weighted0, weighted1 = sess.run(
    #            [ops['lm_embeddings'], ops['mask'],
    #             weighted_ops[0]['weighted_op'], weighted_ops[1]['weighted_op']],
    #            feed_dict={character_ids: X_chars}
    #        )
            char_ids = batcher.batch_sentences(sentences)
            lm_embeddings = sess.run(
                ops['lm_embeddings'], feed_dict={character_ids: char_ids}
            )
            print("9")
#            tf.reset_default_graph()
#            sess.close()
        return lm_embeddings