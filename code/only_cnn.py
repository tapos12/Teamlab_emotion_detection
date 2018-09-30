""""
Usage: nn.py <IMAGE_PATH> <IMAGE_INDEX_PATH> <TRAIN_PATH> <EVAL_PATH>
"""

import logging
import math
import numpy as np
import os
import random
import sys

import lasagne
import theano
import theano.tensor as T
import docopt

from lib.reader import Corpus
from lib.eval import pformat

# theano.config.compute_test_value = 'warn'


class Lookup(dict):
    def __missing__(self, key):
        return self['__unk__']


def prepare_data(corpus, emotion_lookup):
    images = []
    tweet_ids = []
    emotions = []
    for tweet in corpus:
        if tweet.image is None:
            continue
        images.append(tweet.image)
        tweet_ids.append(tweet.tweet_id)
        emotions.append(emotion_lookup[tweet.emotion])

    images = np.asarray(images, dtype='float32')
    emotions = np.asarray(emotions, dtype='int8')

    return images, tweet_ids, emotions


def generate_batches(xi, ti, e, batch_size=500):
    indices = list(range(math.ceil(len(xi)/batch_size)))
    random.shuffle(indices)
    for i in indices:
        images = xi[i*batch_size:(i+1)*batch_size]
        tweet_ids = ti[i*batch_size:(i+1)*batch_size]
        emotions = e[i*batch_size:(i+1)*batch_size]

        yield images, tweet_ids, emotions


def build_network(image_input_var, num_target):
    image_network = lasagne.layers.InputLayer((None, 128, 128, 3), input_var=image_input_var)
    print('post InputLayer:\t%s' %str(image_network.output_shape))
    image_network = lasagne.layers.DimshuffleLayer(image_network, (0, 3, 1, 2))
    print('post DimshuffleLayer:\t%s' %str(image_network.output_shape))

    print()

    filter1 = lasagne.layers.Conv2DLayer(image_network, num_filters=20, filter_size=(1, 1), W=lasagne.init.Uniform(), nonlinearity=lasagne.nonlinearities.rectify)
    print('post Conv2DLayer:\t%s' %str(filter1.output_shape))
    filter1 = lasagne.layers.MaxPool2DLayer(filter1, pool_size=(2, 2))
    print('post MaxPool2DLayer:\t%s' %str(filter1.output_shape))
    filter1 = lasagne.layers.DropoutLayer(filter1, p=.5)
    print('post DropoutLayer:\t%s' %str(filter1.output_shape))
    _, *shape = filter1.output_shape
    # filter5 = lasagne.layers.ReshapeLayer(filter5, (-1, max_sent, int(np.prod(shape))))
    # print('post ReshapeLayer:\t%s' %str(filter5.output_shape))
    filter1 = lasagne.layers.FlattenLayer(filter1, 2)
    print('post FlattenLayer:\t%s' %str(filter1.output_shape))
    # image_network = lasagne.layers.DenseLayer(filter5, word_embedding_length*2)
    # print('post DenseLayer:\t%s' %str(image_network.output_shape))

    filter3 = lasagne.layers.Conv2DLayer(image_network, num_filters=16, filter_size=(3, 3), W=lasagne.init.Uniform(), nonlinearity=lasagne.nonlinearities.rectify)
    print('post Conv2DLayer:\t%s' %str(filter3.output_shape))
    filter3 = lasagne.layers.MaxPool2DLayer(filter3, pool_size=(2, 2))
    print('post MaxPool2DLayer:\t%s' %str(filter3.output_shape))
    filter3 = lasagne.layers.DropoutLayer(filter3, p=.5)
    print('post DropoutLayer:\t%s' %str(filter3.output_shape))
    _, *shape = filter3.output_shape
    # filter5 = lasagne.layers.ReshapeLayer(filter5, (-1, max_sent, int(np.prod(shape))))
    # print('post ReshapeLayer:\t%s' %str(filter5.output_shape))
    filter3 = lasagne.layers.FlattenLayer(filter3, 2)
    print('post FlattenLayer:\t%s' %str(filter3.output_shape))
    # image_network = lasagne.layers.DenseLayer(filter5, word_embedding_length*2)
    # print('post DenseLayer:\t%s' %str(image_network.output_shape))

    filter5 = lasagne.layers.Conv2DLayer(image_network, num_filters=12, filter_size=(5, 5), W=lasagne.init.Uniform(), nonlinearity=lasagne.nonlinearities.rectify)
    print('post Conv2DLayer:\t%s' %str(filter5.output_shape))
    filter5 = lasagne.layers.MaxPool2DLayer(filter5, pool_size=(2, 2))
    print('post MaxPool2DLayer:\t%s' %str(filter5.output_shape))
    filter5 = lasagne.layers.DropoutLayer(filter5, p=.5)
    print('post DropoutLayer:\t%s' %str(filter5.output_shape))
    _, *shape = filter5.output_shape
    # filter5 = lasagne.layers.ReshapeLayer(filter5, (-1, max_sent, int(np.prod(shape))))
    # print('post ReshapeLayer:\t%s' %str(filter5.output_shape))
    filter5 = lasagne.layers.FlattenLayer(filter5, 2)
    print('post FlattenLayer:\t%s' %str(filter5.output_shape))
    # image_network = lasagne.layers.DenseLayer(filter5, word_embedding_length*2)
    # print('post DenseLayer:\t%s' %str(image_network.output_shape))

    print()

    network = lasagne.layers.ConcatLayer([filter1, filter3, filter5], 1)
    print('post ConcatLayer:\t%s' %str(network.output_shape))

    # print()

    # network = lasagne.layers.ReshapeLayer(network, (-1, 2*num_hidden))
    # print('post ReshapeLayer:\t%s' %str(network.output_shape))
    network = lasagne.layers.DenseLayer(network, num_units=num_target, nonlinearity=lasagne.nonlinearities.softmax)
    print('post DenseLayer:\t%s' %str(network.output_shape))
    return network


def main(image_path: str, image_index_path: str, pred_path: str, gold_path: str, epochs: int) -> None:
    # train_longest_tweet = max((len(x) for tweet in train_corpus for x in tweet.body))
    # eval_longest_tweet = max((len(x) for tweet in eval_corpus for x in tweet.body))
    # longest_tweet = max(train_longest_tweet, eval_longest_tweet)

    train_corpus = Corpus(train_path, image_path, image_index_path, image_res=(128, 128))
    eval_corpus = Corpus(eval_path, image_path, image_index_path, image_res=(128, 128))

    emotion_lookup = Lookup()
    j = 0
    for tweet in train_corpus:
        tag = tweet.emotion
        if tag not in emotion_lookup:
            emotion_lookup[tag] = j
            j += 1

    emotion_lookup['__pad__'] = len(emotion_lookup)
    emotion_lookup['__unk__'] = len(emotion_lookup)

    reverse_emotion_lookup = {i: x for x, i in emotion_lookup.items()}

    x = T.ftensor4('x')
    y = T.ivector('y')

    network = build_network(
            image_input_var=x,
            num_target=len(emotion_lookup))

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, y)
    # loss = lasagne.objectives.categorical_crossentropy(prediction, y) * T.reshape(T.cast(mask, 'float32'), [-1])
    # loss = T.sum(loss)/T.sum(mask)
    loss = T.mean(loss)

    params = lasagne.layers.get_all_params(network)
    updates = lasagne.updates.adam(loss, params)

    train = theano.function(
            inputs=[x,y],
            outputs=[prediction, loss],
            updates=updates,
            allow_input_downcast=True
            )

    predict = theano.function(
            inputs=[x],
            outputs=[prediction],
            allow_input_downcast=True
            )

    prev_error = -1
    epoch = 0

    xis, tis, es = prepare_data(train_corpus, emotion_lookup)
    test_xis, test_tis, test_es = prepare_data(eval_corpus, emotion_lookup)

    try:
        prev_error = -1
        while True:
            epoch += 1
            errors = []
            for images, _, emotions in generate_batches(xis, tis, es):
                ys, err = train(images, emotions)
                errors.append(err)
            error = sum(errors)/len(errors)

            for images, tweet_ids, _ in generate_batches(xis, tis, es):
                ys = predict(images)[0]
                for y, ti in zip(ys, tweet_ids):
                    pred = reverse_emotion_lookup[int(np.argmax(y))]
                    train_corpus.get_tweet(ti).prediction = pred

            for images, tweet_ids, _ in generate_batches(test_xis, test_tis, test_es):
                ys = predict(images)[0]
                for y, ti in zip(ys, tweet_ids):
                    pred = reverse_emotion_lookup[int(np.argmax(y))]
                    eval_corpus.get_tweet(ti).prediction = pred

            train_results = train_corpus.evaluate()
            eval_results = eval_corpus.evaluate()

            eval_width = 10
            emotion_width = len(max(eval_results['emotions'], key=lambda x: len(x)))

            if prev_error == -1:
                print('\rEpoch %i error: %.6f' %(epoch, error))
            else:
                err_delta = error-prev_error
                print('\rEpoch %i error: %.6f error_delta: %.6f' %(epoch, error, err_delta))
            prev_error = error

            print(pformat(eval_results)+'\n')
            print(' == Train Macro Avg F_Score: %.2f ==' %(train_results['macro_avg_f_score']))
            print(' == Eval Macro Avg F_Score: %.2f ==' %(eval_results['macro_avg_f_score']))
            print(' == Train Accuracy: %.2f ==' %(train_results['accuracy']))
            print(' == Eval  Accuracy: %.2f ==\n' %(eval_results['accuracy']))
            # print(' Time elapsed: Train: %.2f Eval %.2f' %(train_elapsed, eval_elapsed))
            print(' %s\n' %('-'*78))

    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    args = docopt.docopt(__doc__)

    image_path = args['<IMAGE_PATH>']
    image_index_path = args['<IMAGE_INDEX_PATH>']
    train_path = args['<TRAIN_PATH>']
    eval_path = args['<EVAL_PATH>']

    main(image_path, image_index_path, train_path, eval_path, 20)
