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


class DocEmbeddingLayer(lasagne.layers.Layer):
    def get_output_for(self, input, **kwargs):
        min = T.min(input, 1)
        max = T.max(input, 1)
        return T.concatenate([min, max], 1)

    def get_output_shape_for(self, input_shape):
        return (None, 60)


class Lookup(dict):
    def __missing__(self, key):
        return self['__unk__']


def prepare_data(corpus, word_lookup, emotion_lookup):
    texts = []
    images = []
    tweet_ids = []
    emotions = []
    for tweet in corpus:
        if tweet.image is None:
            continue
        text = [word_lookup[word] for word in tweet.content]
        # text = tweet.body + [word_lookup['__pad__']]*(longest_tweet-len(tweet.body))
        texts.append(text)
        images.append(tweet.image)
        tweet_ids.append(tweet.tweet_id)
        emotions.append(emotion_lookup[tweet.emotion])

    # print(texts)

    # texts = np.asarray(texts, dtype='int32')
    images = np.asarray(images, dtype='float32')
    emotions = np.asarray(emotions, dtype='int8')

    return texts, images, tweet_ids, emotions


def generate_batches(xt, xi, ti, e, word_lookup, batch_size=500):
    indices = list(range(math.ceil(len(xt)/batch_size)))
    random.shuffle(indices)
    for i in indices:
        texts = xt[i*batch_size:(i+1)*batch_size]
        # masks = []
        images = xi[i*batch_size:(i+1)*batch_size]
        tweet_ids = ti[i*batch_size:(i+1)*batch_size]
        emotions = e[i*batch_size:(i+1)*batch_size]

        longest_tweet = max([len(x) for x in texts])

        for i, text in enumerate(texts):
            # masks.append([1]*len(text)+[0]*(longest_tweet-len(text)))
            texts[i] += [word_lookup['__pad__']]*(longest_tweet-len(text))

        texts = np.asarray(texts, dtype='int32')
        # masks = np.asarray(masks, dtype='int8')

        yield texts, images, tweet_ids, emotions


def build_network(text_input_var, image_input_var, number_of_words, word_embedding_length, num_target):
    # mask = lasagne.layers.InputLayer((None, None), input_var=mask_var)
    text_network = lasagne.layers.InputLayer((None, None), input_var=text_input_var)
    print('post InputLayer:\t%s' %str(text_network.output_shape))
    text_network = lasagne.layers.EmbeddingLayer(text_network, number_of_words, word_embedding_length)
    print('post EmbeddingLayer:\t%s' %str(text_network.output_shape))
    text_network = lasagne.layers.DropoutLayer(text_network, p=.5)
    print('post DropoutLayer:\t%s' %str(text_network.output_shape))
    text_network = DocEmbeddingLayer(text_network)
    print('post DocEmbeddingLayer:\t%s' %str(text_network.output_shape))

    print()

    image_network = lasagne.layers.InputLayer((None, 128, 128, 3), input_var=image_input_var)
    print('post InputLayer:\t%s' %str(image_network.output_shape))
    image_network = lasagne.layers.DimshuffleLayer(image_network, (0, 3, 1, 2))
    print('post DimshuffleLayer:\t%s' %str(image_network.output_shape))

    print()

    filter1 = lasagne.layers.Conv2DLayer(image_network, num_filters=24, filter_size=(1, 1), W=lasagne.init.Uniform(), nonlinearity=lasagne.nonlinearities.rectify)
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

    print()

    filter3 = lasagne.layers.Conv2DLayer(image_network, num_filters=20, filter_size=(3, 3), W=lasagne.init.Uniform(), nonlinearity=lasagne.nonlinearities.rectify)
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

    print()

    filter5 = lasagne.layers.Conv2DLayer(image_network, num_filters=16, filter_size=(5, 5), W=lasagne.init.Uniform(), nonlinearity=lasagne.nonlinearities.rectify)
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

    image_network = lasagne.layers.ConcatLayer([filter1, filter3, filter5], 1)
    print('post ConcatLayer:\t%s' %str(image_network.output_shape))

    image_network = lasagne.layers.DenseLayer(image_network, word_embedding_length*2, nonlinearity=lasagne.nonlinearities.rectify)
    print('post DenseLayer:\t%s' %str(image_network.output_shape))

    print()

    # image_network = lasagne.layers.ConcatLayer([filter3, filter5], 2)
    # print('post ConcatLayer:\t%s' %str(image_network.output_shape))

    # print()

    network = lasagne.layers.ConcatLayer([text_network, image_network])
    print('post ConcatLayer:\t%s' %str(network.output_shape))
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

    word_lookup = Lookup()
    emotion_lookup = Lookup()
    i = 1
    j = 0
    for tweet in train_corpus:
        for word in tweet.content:
            if word not in word_lookup:
                word_lookup[word] = i
                i += 1
        tag = tweet.emotion
        if tag not in emotion_lookup:
            emotion_lookup[tag] = j
            j += 1

    word_lookup['__pad__'] = len(word_lookup)
    word_lookup['__unk__'] = len(word_lookup)

    emotion_lookup['__pad__'] = len(emotion_lookup)
    emotion_lookup['__unk__'] = len(emotion_lookup)

    reverse_emotion_lookup = {i: x for x, i in emotion_lookup.items()}

    x1 = T.imatrix('x1')
    # m = T.imatrix('y')
    x2 = T.ftensor4('x2')
    y = T.ivector('y')

    network = build_network(
            text_input_var=x1,
            image_input_var=x2,
            number_of_words=len(word_lookup),
            word_embedding_length=30,
            num_target=len(emotion_lookup))

    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, y)
    # loss = lasagne.objectives.categorical_crossentropy(prediction, y) * T.reshape(T.cast(m, 'float32'), [-1])
    # loss = T.sum(loss)/T.sum(m)
    loss = T.mean(loss)

    params = lasagne.layers.get_all_params(network)
    updates = lasagne.updates.adam(loss, params)

    train = theano.function(
            inputs=[x1,x2,y],
            outputs=[prediction,loss],
            updates=updates,
            allow_input_downcast=True,
            on_unused_input='warn'
            )

    predict = theano.function(
            inputs=[x1,x2],
            outputs=[prediction],
            allow_input_downcast=True,
            on_unused_input='warn'
            )

    prev_error = -1
    epoch = 0

    xts, xis, tis, es = prepare_data(train_corpus, word_lookup, emotion_lookup)
    test_xts, test_xis, test_tis, test_es = prepare_data(eval_corpus, word_lookup, emotion_lookup)

    try:
        prev_error = -1
        while True:
            epoch += 1
            errors = []
            for texts, images, _, emotions in generate_batches(xts, xis, tis, es, word_lookup):
                ys, err = train(texts, images, emotions)
                errors.append(err)
            error = sum(errors)/len(errors)

            for texts, images, tweet_ids, _ in generate_batches(xts, xis, tis, es, word_lookup):
                ys = predict(texts, images)[0]
                for y, ti in zip(ys, tweet_ids):
                    pred = reverse_emotion_lookup[int(np.argmax(y))]
                    train_corpus.get_tweet(ti).prediction = pred

            for texts, images, tweet_ids, _ in generate_batches(test_xts, test_xis, test_tis, test_es, word_lookup):
                ys = predict(texts, images)[0]
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
