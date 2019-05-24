'''
Modified from https://github.com/tsc2017/Frechet-Inception-Distance &
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/gan/python/eval/python/classifier_metrics_impl.py
'''

import tensorflow as tf
import os
import functools
import numpy as np
import glob
import sys
from tqdm import tqdm
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
tfgan = tf.contrib.gan

INCEPTION_FINAL_POOL = 'pool_3:0'
INCEPTION_DEFAULT_IMAGE_SIZE = 299
ACTIVATION_DIM = 2048

BATCH_SIZE = 32
TEST_SIZE = 5000

assert len(sys.argv)==2, "Input the directory of the generated images [gen_samples/]"

def load_and_preprocess_image(path):
    image = tf.read_file(path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.cast(image, tf.float32)
    image = image / 255. * 2 - 1  # [0, 255] -> [-1, 1]

    return image


def inception_activations(images, height=INCEPTION_DEFAULT_IMAGE_SIZE, width=INCEPTION_DEFAULT_IMAGE_SIZE, num_splits = 1):
    images = tf.image.resize_bilinear(images, [height, width])
    generated_images_list = array_ops.split(images, num_or_size_splits = num_splits)
    activations = functional_ops.map_fn(
        fn = functools.partial(tfgan.eval.run_inception, output_tensor = INCEPTION_FINAL_POOL),
        elems = array_ops.stack(generated_images_list),
        parallel_iterations = 1,
        back_prop = False,
        swap_memory = True,
        name = 'RunClassifier')
    activations = array_ops.concat(array_ops.unstack(activations), 0)
    return activations


def get_fid(sess, fake_dir):
    real_activations = np.load('real_activations.npy')

    # load dumped generated images
    fns = glob.glob(os.path.join(fake_dir, '*.png'))    
    assert len(fns) == TEST_SIZE, "Number of generated images do not equal to TEST_SIZE"
    
    dataset = tf.data.Dataset.from_tensor_slices(fns)
    dataset = dataset.map(load_and_preprocess_image).batch(BATCH_SIZE)
    iterator = tf.data.make_one_shot_iterator(dataset)
#    iterator = dataset.make_one_shot_iterator() # for TF version < 1.13
    
    images = iterator.get_next() # [N, H, W, C]
    activations = inception_activations(images)
    real_acts = tf.placeholder(tf.float32, [None, ACTIVATION_DIM], name = 'real_activations')
    fake_acts = tf.placeholder(tf.float32, [None, ACTIVATION_DIM], name = 'fake_activations')
    fid = tfgan.eval.frechet_classifier_distance_from_activations(real_acts, fake_acts)
    sess.graph.finalize()
    
    print("Extracting features from Inception ...")
    fake_activations = np.zeros([TEST_SIZE, ACTIVATION_DIM])
    for i in tqdm(range(0, TEST_SIZE, BATCH_SIZE)):
        fake_activations[i: i+BATCH_SIZE] = sess.run(activations)
        
    print("Calculating FID scores ...")
    fid_score = sess.run(fid, {real_acts: real_activations, fake_acts: fake_activations})
    
    return fid_score


if __name__ == '__main__':
    with tf.Session() as sess:
        fid_score = get_fid(sess, sys.argv[1])
        print("================================================")
        print("FID score: {:.3f}".format(fid_score))
        print("================================================")
