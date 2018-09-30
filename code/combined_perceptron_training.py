#!/usr/bin/env python3

"""
Usage: perceptron.py <IMAGE_PATH> <IMAGE_INDEX_PATH> <TRAIN_PATH> <EVAL_PATH>
"""

# standard library
import logging
from operator import itemgetter
import sys
import time

# third-party
from PIL import Image
import docopt

# package-level
from lib.reader import Corpus
from lib.feature_extractors import TextualFeatureExtractor, ImageFeatureExtractor
from lib.perceptron import Perceptron
from lib.eval import pformat

logging.basicConfig(
    format="[%(asctime)s] <%(name)s> %(levelname)s: %(message)s",
    datefmt='%H:%M:%S',
    level=logging.DEBUG
)

logger = logging.getLogger('main')

logger.info('logging started')

def main(image_path: str, image_index_path: str, pred_path: str, gold_path: str, epochs: int) -> None:
    text_train_corpus = Corpus(train_path)
    text_eval_corpus = Corpus(eval_path)

    text_feature_extractor = TextualFeatureExtractor(
        [text_train_corpus, text_eval_corpus], caching=True)

    classes = text_feature_extractor.classes
    text_classifier = Perceptron(classes)

    predictions = {}

    epoch = 0
    while epoch < 10:
        epoch += 1
        train_start_timestamp = time.time()
        text_train_corpus.shuffle()
        for i, tweet in enumerate(text_train_corpus):
            feature_vector = text_feature_extractor.get_feature_vector(tweet)
            prediction = text_classifier.train(feature_vector, tweet.emotion)
            # tweet.full_prediction = prediction
            tweet.prediction = max(prediction.items(), key=itemgetter(1))[0]
            predictions[tweet.tweet_id] = prediction

            if i%1000 == 0 and not i==0:
                print('\r%i' %i, file=sys.stderr, end='')

        train_elapsed = time.time()-train_start_timestamp
        train_results = text_train_corpus.evaluate()

        eval_start_timestamp = time.time()
        text_eval_corpus.shuffle()
        for i, tweet in enumerate(text_eval_corpus):
            feature_vector = text_feature_extractor.get_feature_vector(tweet)
            prediction = text_classifier.predict(feature_vector)
            # tweet.full_prediction = prediction
            tweet.prediction = max(prediction.items(), key=itemgetter(1))[0]
            predictions[tweet.tweet_id] = prediction

        eval_elapsed = time.time()-eval_start_timestamp
        eval_results = text_eval_corpus.evaluate()

        eval_width = 10
        emotion_width = len(max(eval_results['emotions'], key=lambda x: len(x)))

        print('\rEpoch %i:' %epoch)
        print(pformat(eval_results)+'\n')
        print(' == Macro Avg F_Score: %.2f ==' %(train_results['macro_avg_f_score']))
        print(' == Train Accuracy:    %.2f ==' %(train_results['accuracy']))
        print(' == Eval  Accuracy:    %.2f ==\n' %(eval_results['accuracy']))
        print(' Time elapsed: Train: %.2f Eval %.2f' %(train_elapsed, eval_elapsed))
        print(' %s\n' %('-'*78))


    image_train_corpus = Corpus(train_path, image_path, image_index_path, image_res=(12, 12))
    image_eval_corpus = Corpus(eval_path, image_path, image_index_path, image_res=(12, 12))

    image_feature_extractor = ImageFeatureExtractor(
        [image_train_corpus, image_eval_corpus], [predictions], caching=True)

    classes = image_feature_extractor.classes
    image_classifier = Perceptron(classes)

    epoch = 0
    while True:
        epoch += 1
        train_start_timestamp = time.time()
        image_train_corpus.shuffle()
        for i, tweet in enumerate(image_train_corpus):
            feature_vector = image_feature_extractor.get_feature_vector(tweet)
            prediction = image_classifier.train(feature_vector, tweet.emotion)
            # tweet.full_prediction = prediction
            tweet.prediction = max(prediction.items(), key=itemgetter(1))[0]

            if i%1000 == 0 and not i==0:
                print('\r%i' %i, file=sys.stderr, end='')

        train_elapsed = time.time()-train_start_timestamp
        train_results = image_train_corpus.evaluate()

        eval_start_timestamp = time.time()
        image_eval_corpus.shuffle()
        for i, tweet in enumerate(image_eval_corpus):
            feature_vector = image_feature_extractor.get_feature_vector(tweet)
            prediction = image_classifier.predict(feature_vector)
            # tweet.full_prediction = prediction
            tweet.prediction = max(prediction.items(), key=itemgetter(1))[0]

        eval_elapsed = time.time()-eval_start_timestamp
        eval_results = image_eval_corpus.evaluate()

        eval_width = 10
        emotion_width = len(max(eval_results['emotions'], key=lambda x: len(x)))

        print('\rEpoch %i:' %epoch)
        print(pformat(eval_results)+'\n')
        print(' == Train Macro Avg F_Score: %.2f ==' %(train_results['macro_avg_f_score']))
        print(' == Eval Macro Avg F_Score: %.2f ==' %(eval_results['macro_avg_f_score']))
        print(' == Train Accuracy: %.2f ==' % (train_results['accuracy']))
        print(' == Eval  Accuracy: %.2f ==\n' % (eval_results['accuracy']))
        print(' %s\n' %('-'*78))

        # print(' eval: %.2f eval_avg: %.2f' %((correct_num/len(eval_corpus)*100), correct_num_avg/len(eval_corpus)*100), end='')
        # print(' \ttime elapsed train: %.2f eval: %.2f total: %.2f' %(train_elapsed, eval_elapsed, train_elapsed+eval_elapsed))

if __name__ == '__main__':
    args = docopt.docopt(__doc__)

    image_path = args['<IMAGE_PATH>']
    image_index_path = args['<IMAGE_INDEX_PATH>']
    train_path = args['<TRAIN_PATH>']
    eval_path = args['<EVAL_PATH>']

    main(image_path, image_index_path, train_path, eval_path, 20)
